# Azir

Azir is a lightweight multi-provider LLM gateway built with Python and FastAPI.

It exposes a single OpenAI-style chat completions endpoint and forwards requests to different model providers behind the scenes. Each provider adapter translates the shared Azir request format into the provider-specific wire format, sends the request using raw HTTP, and normalizes the provider response back into one consistent response shape.

## Current Status

Azir currently supports:

- FastAPI HTTP server
- `POST /v1/chat/completions`
- OpenAI-style request validation with Pydantic
- Anthropic and OpenAI providers, both using raw `httpx`
- Shared `Provider` interface (`complete()` and `stream()`)
- Shared `httpx.AsyncClient`
- A model registry (`model_registry.py`) as the single source of truth for routable models
- Explicit model routing (e.g. `"model": "claude-sonnet-4-6"`)
- Capability-, cost-, and latency-aware automatic routing (`"model": "azir-auto"` plus `"task"`, optional `"max_cost_usd"` budget, optional `"routing_policy"`: `cheap`, `fast`, or `balanced` (default)), skipping models that have been failing recently (health-aware)
- Fallback across providers on transient upstream failures (non-streaming)
- Clean, normalized provider error handling
- Per-attempt telemetry for non-streaming requests (provider, model, latency, token usage, status, estimated cost)
- Anthropic system-message translation, stop-reason normalization, and token-usage normalization
- Streaming chat completions (`stream: true`) for both Anthropic and OpenAI, as Server-Sent Events
- API keys loaded from `.env`

## Request Flows

### Model resolution (shared by every flow)

```text
request.model
  |
  +-- "azir-auto" --> task missing?                         -> 400
  |                   no enabled model has task?            -> 400
  |                   none within max_cost_usd (if given)?  -> 400
  |                   drop unhealthy models (unless all are unhealthy)
  |                   rank the remaining models by routing_policy
  |                        (default "balanced"; ties -> registry order):
  |                          cheap    -> lowest estimated cost
  |                          fast     -> lowest latency estimate
  |                          balanced -> lowest normalized cost+latency score
  |
  +-- anything else --> not in registry?            -> 400 Unknown model
                        disabled?                   -> 400 Model is disabled
                        else that registry entry (task / max_cost_usd /
                        routing_policy / health don't affect it)
  |
  v
concrete ModelConfig (name + provider)
```

### Routing cost estimate (`azir-auto` only)

Before execution, each candidate model's request cost is estimated from the registry's per-1K-token prices:

```text
input_tokens  = ceil(total characters of all message content / 4)
output_tokens = max_tokens, or 200 if not set (the same default AnthropicProvider sends)
cost          = input_tokens/1000 * input_cost_per_1k + output_tokens/1000 * output_cost_per_1k
```

This is a deterministic routing heuristic only -- no tokenizer or provider API is involved, and it is not what providers will bill. Telemetry's `estimated_cost_usd` is computed separately, after the request, from the provider's actual returned token usage.

Example with the current registry, for a 2-character prompt and no `max_tokens` (1 input token, 200 output tokens), `task: "chat"`:

```text
claude-sonnet-4-6:  0.001 * 0.003   + 0.2 * 0.015  = 0.003003    USD
gpt-4o-mini:        0.001 * 0.00015 + 0.2 * 0.0006 = 0.00012015  USD  <- cheaper
```

The cost estimate filters candidates by `max_cost_usd` for every policy, and is what `cheap` and `balanced` rank by (see "Routing policies" below).

### Routing latency estimate (`azir-auto` only)

`latency.py` keeps an in-memory exponentially weighted moving average (EWMA) of observed latency per concrete model:

```text
first sample:  estimate = sample
afterwards:    estimate = 0.3 * sample + 0.7 * previous_estimate
untried model: 1000 ms (DEFAULT_LATENCY_ESTIMATE_MS)
```

Samples come from non-streaming provider attempts in `route_request()` -- the same `latency_ms` telemetry reports:

- successful attempts are recorded
- timed-out attempts are recorded (the elapsed time is a lower bound on how slow the model was)
- fast error responses (429, 5xx, 4xx) and connection failures are **not** recorded -- an instant 503 says nothing about how fast the model answers, and recording it would make a failing model look fast
- requests rejected before any provider is called, and streams, record nothing

An untried model sits at the 1000 ms default: it is neither assumed instant nor never tried -- a model observed slower than 1000 ms loses to it under `fast`, and it gets sampled.

Example: after `claude-sonnet-4-6` has been observed at 400 ms then 600 ms, and `gpt-4o-mini` at 1500 ms then 1100 ms:

```text
claude-sonnet-4-6:  0.3 * 600  + 0.7 * 400  = 460 ms
gpt-4o-mini:        0.3 * 1100 + 0.7 * 1500 = 1380 ms
```

State is per process and lost on restart; it is not persisted or shared between workers.

### Routing policies (`azir-auto` only)

Every policy ranks the **same** candidate set -- enabled models with the `task` capability that fit `max_cost_usd`, minus unhealthy ones (see "Model health" below) -- and the lowest score wins. Exact ties go to registry order. No `routing_policy` means `balanced`.

| Policy     | Score                                                                    |
|------------|--------------------------------------------------------------------------|
| `cheap`    | estimated request cost (latency is ignored)                              |
| `fast`     | latency estimate, 1000 ms for untried models (cost is ignored beyond the budget filter) |
| `balanced` | `0.5 * normalized_cost + 0.5 * normalized_latency`                       |

`balanced` never adds dollars to milliseconds. Each metric is min-max normalized across the current candidates first:

```text
normalized = (value - min) / (max - min)      # 0 = best candidate, 1 = worst
           = 0 for every candidate if all values are equal
```

The weights are `BALANCED_COST_WEIGHT` / `BALANCED_LATENCY_WEIGHT` in `router.py` (not configurable per request).

**Cold start.** Untried models use the same 1000 ms default under `fast` and `balanced`. With no history at all, every latency is equal: `fast` falls back to registry order, and `balanced`'s latency term is 0 for everyone, so it picks the cheapest.

**Two candidates.** With exactly two candidates each normalized metric is 0 or 1. If one model is both cheaper and faster, `balanced` picks it; if one is cheaper and the other faster, both score 0.5 and registry order decides. `balanced` only picks a "middle" model when there are three or more candidates.

Example: three eligible models, a 2-character prompt with no `max_tokens` (estimated cost = 0.2 * output price):

```text
model   output $/1K  cost   latency   norm cost  norm latency  balanced
A       1.0          0.2    900 ms    0.0        1.0           0.5
B       2.0          0.4    400 ms    0.5        0.1667        0.3333
C       3.0          0.6    300 ms    1.0        0.0           0.5

cheap -> A      fast -> C      balanced -> B
```

### Model health (`azir-auto` only)

`health.py` keeps the last 20 provider outcomes per concrete model, in memory:

```text
success_rate = successes in window / outcomes in window
unhealthy    = outcomes >= 5 (MIN_HEALTH_SAMPLES)
               and success_rate < 0.6 (HEALTH_SUCCESS_THRESHOLD)
```

A model with fewer than 5 recorded outcomes is never unhealthy. Exactly 0.6 is healthy. Old outcomes fall out of the window as new ones arrive, so a recovered model becomes eligible again on its own -- but only once it gets traffic, e.g. as a fallback or an explicit request.

What is recorded, at the same per-attempt point as telemetry and fallback:

| Outcome | Recorded as |
|---|---|
| Non-streaming attempt succeeds (primary or fallback) | success |
| Non-streaming attempt fails transiently (429, upstream 5xx, timeout, connection failure) | failure |
| Stream opens successfully | success (stream duration and mid-stream errors are not considered) |
| Stream fails transiently before it opens | failure |
| Non-transient provider error (400, 401, 403, 404, other upstream 4xx) | nothing -- request or configuration problem, not model health |
| Rejected before any provider is called (unknown/disabled model, missing/unsupported task, over budget, invalid request) | nothing |

How it affects routing:

- **`azir-auto`:** unhealthy models are removed after the budget filter, before the routing policy ranks what is left. `balanced` normalizes across the healthy candidates only.
- **Every eligible model unhealthy:** the filter is skipped and the policy ranks the full eligible set, exactly as if there were no health state. Azir never returns a 400 just because of health; a failing primary still falls back normally.
- **Explicit models** are always used, healthy or not.
- **Fallbacks** keep their existing rule (first enabled model per other provider that fits `task` and budget). Health does not filter them.

Example, `task: "chat"`, `routing_policy: "cheap"`, after `gpt-4o-mini` recorded 2 successes and 4 failures (2/6 = 0.33 < 0.6, unhealthy) and `claude-sonnet-4-6` 3 successes (fewer than 5 samples, healthy):

```text
eligible:   claude-sonnet-4-6, gpt-4o-mini
healthy:    claude-sonnet-4-6
cheap ->    claude-sonnet-4-6   (gpt-4o-mini would win on cost if it were healthy)
```

Resolution always happens before any provider is called, so providers, fallback, telemetry, and streaming only ever see concrete model names -- never `azir-auto`.

### Non-streaming

```text
Client
  |
  v
POST /v1/chat/completions  (FastAPI + Pydantic validation)
  |
  v
router.route_request(...)
  |
  v
router.plan_attempts(...)  --> [resolved model, one fallback per other provider]
  |
  +--> attempt 1: provider.complete(concrete model)
  |       |
  |       +-- success ----------------> emit telemetry, record latency + health success --> ChatResponse
  |       +-- transient failure ------> emit telemetry, record health failure
  |                                     (+ latency if timeout) --> next attempt
  |       +-- non-recoverable failure -> emit telemetry --> raise immediately
  |
  +--> attempt 2 ... (same rules); if all fail transiently, re-raise the last error
```

### Streaming (`stream: true`)

```text
Client
  |
  v
POST /v1/chat/completions
  |
  v
router.stream_chat_completion(...)
  |
  v
router.resolve_model(...)              <-- same resolution as above; 400s happen here
  |
  v
<Provider>.stream(concrete model)      <-- opens the upstream connection and
  |                                        reads only its headers/status
  +-- upstream failure --> HTTPException (clean error, no stream started)
  |
  +-- success --> async generator handed back to main.py
                    |
                    v
             StreamingResponse (200, text/event-stream)
                    |
                    v
   <Provider>._translate_stream(...) translates each upstream SSE event
   into an Azir SSE chunk as it arrives, no buffering
                    |
                    v
                 Client (incremental `data: {...}` chunks, ending in `data: [DONE]`)
```

Streams read the current latency estimates when resolving `azir-auto` but never update them. They do update health: a success when the stream opens, a failure on a transient pre-stream error. There is no cross-provider fallback and no telemetry on the streaming path -- see "Streaming" under Important Design Decisions and Current Limitations.

## Why Azir Exists

LLM providers expose different APIs.

For example, Anthropic and OpenAI differ in areas such as:

- system-message handling
- response content structure
- stop / finish reason names
- token usage field names
- authentication headers
- supported request parameters

Azir hides those differences from the caller.

The caller sends one common request format and receives one common response format regardless of which provider handles the request.

## Architecture

```text
main.py
   |
   v
router.py
   |
   +--> model_registry.py
   |
   +--> latency.py
   |
   +--> health.py
   |
   +--> AnthropicProvider
   |
   +--> OpenAIProvider
   |
   v
telemetry / normalized responses
```

```text
azir/
├── main.py
├── router.py
├── model_registry.py
├── latency.py
├── health.py
├── telemetry.py
├── schemas.py
├── config.py
├── providers/
│   ├── base.py
│   ├── errors.py
│   ├── anthropic.py
│   └── openai.py
├── tests/
├── .github/workflows/tests.yml
├── .env.example        (copy to .env, which is git-ignored)
├── .gitignore
├── pyproject.toml
├── uv.lock
└── LICENSE
```

### `main.py`

Owns the HTTP layer only:

- create the FastAPI application
- create one shared `httpx.AsyncClient` and the provider instances during startup (lifespan)
- expose `/v1/chat/completions`
- return a `StreamingResponse` for `stream: true`, otherwise the `ChatResponse` model
- delegate everything else to `router.route_request()` / `router.stream_chat_completion()`

`main.py` contains no provider-specific logic and no routing policy.

### `schemas.py`

Defines Azir's request and response contracts using Pydantic: `Message`, `ChatRequest`, `ChatResponse`, `Choice`, `Usage`.

`ChatRequest` fields: `model`, `messages`, `max_tokens`, `temperature`, `stream`, and the optional routing fields `task`, `max_cost_usd`, and `routing_policy`. No routing field is ever forwarded to a provider.

- `task` is required when `model` is `azir-auto`; when given, it also restricts which models may be used as fallbacks.
- `max_cost_usd` (>= 0) caps the *estimated* request cost of every model Azir picks itself -- the `azir-auto` choice and any fallback. An explicitly named model is always honored regardless of it.
- `routing_policy` (`cheap` | `fast` | `balanced`, default `balanced`) chooses how `azir-auto` ranks eligible models. It is ignored for explicitly named models. Any other value is rejected by validation (FastAPI's standard 422).

### `model_registry.py`

The single source of truth for which concrete models Azir can route to. Each `ModelConfig` has:

- `name`
- `provider`
- `capabilities` (e.g. `chat`, `coding`, `reasoning`, `classification`)
- `enabled`
- `input_cost_per_1k` / `output_cost_per_1k` (rough static USD rates, used for `azir-auto` cost estimates and telemetry estimates)

`ModelConfig.estimate_cost_usd(input_tokens, output_tokens)` is the one place the pricing formula lives; both the router and telemetry use it.

Currently registered:

| Model               | Provider    | Capabilities                 | USD / 1K in | USD / 1K out |
|---------------------|-------------|------------------------------|-------------|--------------|
| `claude-sonnet-4-6` | `anthropic` | chat, coding, reasoning      | 0.003       | 0.015        |
| `gpt-4o-mini`       | `openai`    | chat, classification         | 0.00015     | 0.0006       |

`find_models(capability=..., provider=...)` returns enabled models matching the filters, **in registry order**. That order picks each provider's fallback model and breaks ties between equally cheap `azir-auto` candidates, so selection is always deterministic.

Models not in the registry are rejected with a 400 -- there is no `claude-*` / `gpt-*` prefix-based routing. To route to a new model, register it.

### `latency.py`

In-memory latency state keyed by concrete model name (see "Routing latency estimate" above for the formula):

- `record_latency(model, latency_ms)` -- fold one observation into the model's EWMA
- `get_latency_estimate(model)` -- current estimate in ms, or `None` if never observed
- `get_latency_stats(model)` -- estimate plus sample count
- `reset_latency()` -- clear everything (used by tests)

Updates are synchronous and lock-guarded. Nothing is persisted.

### `health.py`

In-memory rolling window of recent provider outcomes per concrete model (see "Model health" above):

- `record_success(model)` / `record_failure(model)` -- append one outcome to the model's last-20 window
- `get_success_rate(model)` -- successes / outcomes in the window, or `None` if none recorded
- `get_sample_count(model)` -- outcomes in the window
- `is_healthy(model)` -- `True` below 5 samples; otherwise success rate >= 0.6
- `reset_health()` -- clear everything (used by tests)

Health state is the router's current signal; telemetry is the separate historical event log. Neither is derived from the other.

### `router.py`

Routing and orchestration:

- `estimate_request_tokens(request)` / `estimate_request_cost_usd(request, config)` -- the pre-execution routing estimate described above
- `routing_latency_ms(config)` -- the model's latency estimate, or the cold-start default
- `_eligible_candidates(request)` -- the shared `azir-auto` eligibility phase (task required, capability + enabled filter, budget filter, clean 400s)
- `_prefer_healthy(candidates)` -- drop unhealthy candidates, unless that would drop all of them
- `normalize(values)` / `select_by_policy(request, candidates)` -- rank the eligible candidates by `cheap`, `fast`, or `balanced` (see "Routing policies")
- `resolve_model(request)` -- explicit model as-is, or for `azir-auto` the policy-selected eligible model -> one `ModelConfig` (or a clean 400; a registry entry naming a provider Azir doesn't implement is a 500 configuration error)
- `plan_attempts(request)` -- the resolved model, then for each other provider in `PROVIDER_ORDER` the first enabled registry model of that provider (that also supports `task` and fits `max_cost_usd`, if given)
- `route_request(app, request)` -- runs the plan for non-streaming requests, with fallback, per-attempt telemetry, and latency and health recording
- `stream_chat_completion(app, request)` -- resolves the model, then calls that provider's `stream()`; no fallback

**Fallback policy.** `route_request()` moves to the next attempt only when `providers.errors.is_transient_provider_error()` says the failure is transient: rate limiting (429), provider unavailability / unexpected 5xx (500/502/503), and timeouts (504) or connection failures. Anything else is raised immediately without trying another provider:

- malformed requests (upstream 400) and other request-level upstream 4xx
- credential / permission errors (401 / 403) -- these indicate misconfiguration and should surface, not be masked
- model/resource not found (404)
- Azir's own routing errors (unknown model, disabled model, missing or unsupported task, nothing within `max_cost_usd`) -- these are rejected before any provider is called

### `providers/base.py`

Defines the common provider contract:

```python
complete(request: ChatRequest) -> ChatResponse
stream(request: ChatRequest) -> AsyncIterator[str]
```

`request.model` is always a concrete, registry-resolved model name by the time a provider sees it. `stream()` must open the upstream connection and raise an `HTTPException` for pre-stream failures before returning its async iterator of Azir SSE chunks (ending in `data: [DONE]`).

`providers/base.py` also defines `sse_chunk(delta, *, finish_reason)`, a small shared helper that formats one Azir-style `data: {...}\n\n` SSE line, so both providers produce identically shaped streams.

### `providers/errors.py`

- `raise_provider_error(exc, provider)` translates `httpx` failures into clean `HTTPException`s (no raw provider bodies, headers, or keys reach the client): 400/401/403/404/429 pass through, upstream 5xx and unrecognized statuses become 502, timeouts 504, connection failures 502.
- `is_transient_provider_error(exc)` classifies those exceptions for the router's fallback policy (see above).
- `is_timeout_provider_error(exc)` identifies timeouts, the only failures whose elapsed time is recorded as a latency sample.

### `providers/anthropic.py`

Translates between Azir's shared format and Anthropic's Messages API, for both `complete()` and `stream()`:

- moving `system` messages to Anthropic's top-level `system` field
- authenticating with Anthropic (`x-api-key`, `anthropic-version`)
- extracting text from Anthropic `content` blocks
- mapping Anthropic `stop_reason` values to Azir/OpenAI-style `finish_reason` values
- normalizing Anthropic token usage
- translating Anthropic's streaming SSE events into Azir's streaming SSE chunks

```text
Anthropic                 Azir
----------------------------------------
end_turn               -> stop
stop_sequence          -> stop
max_tokens             -> length
input_tokens           -> prompt_tokens
output_tokens          -> completion_tokens
```

`_translate_stream()` is the only place that knows Anthropic's event names (`message_start`, `content_block_delta`, `message_delta`, `ping`, `message_stop`, `error`, ...).

### `providers/openai.py`

Calls OpenAI's Chat Completions API using the same `ChatRequest` and returns the same `ChatResponse`, for both `complete()` and `stream()`.

Because Azir's public format is already OpenAI-like, this provider needs less translation. For streaming, OpenAI's chunks already carry `choices[0] = {index, delta, finish_reason}`, so `_translate_stream()` just strips OpenAI's outer wrapper (`id`/`object`/`created`/`model`) via `sse_chunk()`. OpenAI's own `data: [DONE]` line is treated as "stop reading"; Azir always emits its own `[DONE]`, including when the connection drops before OpenAI's arrives.

### `telemetry.py`

Structured telemetry types and emission. `router.route_request()` emits one `RequestTelemetry` record per provider attempt, success or failure:

- `provider` and `model` -- the concrete provider/model Azir attempted (never `azir-auto`)
- `status` (`success` / `error`) and `status_code`
- `latency_ms` for that attempt
- `prompt_tokens` / `completion_tokens` / `total_tokens` (success only)
- `estimated_cost_usd`, computed from the registry's static per-1K-token rates (`None` for unregistered models)

Records are emitted as single-line JSON via a dedicated `azir.telemetry` logger (its own `StreamHandler`, so they appear on stdout without extra logging setup). Nothing is persisted or aggregated.

## Configuration

Copy the example file and fill in your keys:

```bash
cp .env.example .env
```

```env
ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_API_KEY=your_openai_key
```

`.env` is git-ignored; never commit it. Configuration is loaded through `pydantic-settings` (real environment variables take priority over `.env`); a missing key fails at startup rather than at request time.

## Setup

Requires [uv](https://docs.astral.sh/uv/) (Python 3.13 is picked up from `.python-version`).

```bash
git clone <your-repo-url> azir
cd azir
uv sync
cp .env.example .env   # then add your keys
```

Run the server:

```bash
uv run uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000` (interactive docs at `/docs`).

Run the tests:

```bash
uv run pytest
```

Tests need no `.env` and no API keys: `tests/conftest.py` sets dummy keys, and every provider call is mocked. The same command runs in CI (`.github/workflows/tests.yml`) on pushes to `main` and on pull requests.

## Security

Azir holds your provider API keys and spends money on every request it forwards. What it does today:

- API keys are read only from the environment / `.env` and sent only to the matching provider's API over HTTPS.
- Client-facing errors use fixed, templated messages -- raw provider response bodies, headers, and keys are never returned to the caller.
- Telemetry logs metadata only (provider, model, status, latency, token counts, cost estimate) -- never prompts, completions, or keys.
- Request validation rejects malformed input, including non-positive `max_tokens` and negative `max_cost_usd`.

What it does **not** do yet -- keep this in mind before exposing it beyond your machine:

- **No authentication.** Anyone who can reach the server can use your provider keys. `uvicorn` binds to `127.0.0.1` by default; don't run it with `--host 0.0.0.0` or put it behind a public endpoint without adding auth in front of it (e.g. a reverse proxy).
- **No rate limiting or request-size limits** beyond `max_tokens`, so a client can send arbitrarily large prompts.
- **`max_cost_usd` is a routing estimate, not a spending cap.** It only filters which model `azir-auto` picks; it does not enforce actual spend. Use provider-side spend limits for that.

## Example Requests

Explicit model:

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-6",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 50
  }'
```

Automatic routing by task:

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "azir-auto",
    "task": "classification",
    "messages": [{"role": "user", "content": "Is this review positive? \"Great product!\""}],
    "max_tokens": 50
  }'
```

Automatic routing with a cost budget (returns 400 if no capable model's estimated cost fits):

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "azir-auto",
    "task": "chat",
    "max_cost_usd": 0.001,
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 50
  }'
```

Automatic routing with an explicit policy (`cheap`, `fast`, or `balanced`):

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "azir-auto",
    "task": "chat",
    "routing_policy": "fast",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 50
  }'
```

Every provider returns the same response shape (`model` is the name the provider reports):

```json
{
  "model": "provider-model-name",
  "choices": [
    {
      "message": {"role": "assistant", "content": "..."},
      "finish_reason": "stop"
    }
  ],
  "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
}
```

Streaming (works with explicit models or `azir-auto`; use `-N` so curl shows chunks as they arrive):

```bash
curl -N -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 50,
    "stream": true
  }'
```

Both providers produce the same shape of Server-Sent Events:

```text
data: {"choices":[{"delta":{"role":"assistant"},"index":0,"finish_reason":null}]}

data: {"choices":[{"delta":{"content":"Hel"},"index":0,"finish_reason":null}]}

data: {"choices":[{"delta":{"content":"lo"},"index":0,"finish_reason":null}]}

data: {"choices":[{"delta":{},"index":0,"finish_reason":"stop"}]}

data: [DONE]
```

This intentionally omits the `id` / `object` / `created` fields real OpenAI streaming chunks carry, matching the minimal non-streaming `ChatResponse`.

## Important Design Decisions

### Raw `httpx` instead of provider SDKs

Raw HTTP keeps provider-specific wire formats explicit: URLs, headers, JSON bodies, response parsing, and error handling are all visible in `providers/`.

### One registry, one routing path

All model knowledge -- which models exist, their provider, capabilities, enabled state, and pricing -- lives in `model_registry.py`. Both streaming and non-streaming requests resolve through `router.resolve_model()`, so they accept and reject exactly the same models.

### Shared `httpx.AsyncClient`

One `AsyncClient` is created during FastAPI startup and reused across requests for connection pooling.

### Streaming

**Why `stream()` opens the connection before returning.** Once a path function returns a `StreamingResponse`, Starlette sends the HTTP status line *before* pulling the first chunk. If opening the upstream connection were deferred into the generator, a pre-output failure (bad API key, connection refused) would happen after the client already received "200 OK". So each provider's `stream()` is a plain coroutine that sends the request and checks the status itself, and only then returns the chunk-producing async generator.

**Why there's no fallback for streaming.** Non-streaming fallback works because nothing has been sent to the client when a provider fails. For a stream, the 200 status and possibly some content are already with the client, so switching providers mid-response isn't something a client could sensibly reassemble. `stream_chat_completion()` therefore uses only the resolved model's provider. This is a deliberate scope limit.

**Mid-stream failures.** If the upstream connection fails *after* streaming has started (an Anthropic `error` event, or a dropped connection), Azir stops producing content, emits one chunk with `"finish_reason": "error"` (never raw upstream error text), then `data: [DONE]`. Clients should treat `finish_reason: "error"` as "this response is incomplete."

**Client disconnects.** If the client goes away mid-stream, the generator is closed and the upstream connection is released; no further chunks are emitted.

## Current Limitations

Azir does not yet support:

- authentication or rate limiting (see Security)
- cross-provider fallback for streaming requests
- telemetry for streaming requests
- retries within a single provider (fallback moves to the *next provider*)
- telemetry persistence or aggregation (records are logged, not stored)
- billing-accurate cost tracking (only rough static rates from the registry)
- accurate pre-execution token counts (routing uses a characters/4 heuristic)
- cost- or latency-based ordering of fallbacks (fallbacks use registry order, filtered by task and budget)
- persisted or shared latency or health history (both live in one process's memory and reset on restart)
- health-filtered fallbacks, active health checks, or time-based circuit breaking (an unhealthy model only recovers as new outcomes for it are recorded)
- latency samples from streaming requests
- routing to models that aren't in the registry

## Future Work

Not implemented today:

1. Richer health handling (time-based circuit breaking, active probes, health-filtered fallbacks)
2. Telemetry persistence and aggregation, including streaming telemetry
3. More advanced routing/fallback policies (per-provider retries, streaming fallback, richer task selection, quality-aware selection)

## License

[MIT](LICENSE)
