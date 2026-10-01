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
- Capability-aware automatic routing (`"model": "azir-auto"` plus `"task"`)
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
  +-- "azir-auto" --> task missing?                 -> 400
  |                   no enabled model has task?    -> 400
  |                   else first enabled model (registry order) with that capability
  |
  +-- anything else --> not in registry?            -> 400 Unknown model
                        disabled?                   -> 400 Model is disabled
                        else that registry entry
  |
  v
concrete ModelConfig (name + provider)
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
  |       +-- success ----------------> emit telemetry --> ChatResponse
  |       +-- transient failure ------> emit telemetry --> next attempt
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

There is no cross-provider fallback and no telemetry on the streaming path -- see "Streaming" under Important Design Decisions and Current Limitations.

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
├── telemetry.py
├── schemas.py
├── config.py
├── providers/
│   ├── base.py
│   ├── errors.py
│   ├── anthropic.py
│   └── openai.py
├── tests/
├── .env
├── .gitignore
└── pyproject.toml
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

`ChatRequest` fields: `model`, `messages`, `max_tokens`, `temperature`, `stream`, and an optional `task`. `task` is required when `model` is `azir-auto`; when given, it also restricts which models may be used as fallbacks. It is never forwarded to a provider.

### `model_registry.py`

The single source of truth for which concrete models Azir can route to. Each `ModelConfig` has:

- `name`
- `provider`
- `capabilities` (e.g. `chat`, `coding`, `reasoning`, `classification`)
- `enabled`
- `input_cost_per_1k` / `output_cost_per_1k` (rough static rates, used only for telemetry estimates)

Currently registered:

| Model               | Provider    | Capabilities                 |
|---------------------|-------------|------------------------------|
| `claude-sonnet-4-6` | `anthropic` | chat, coding, reasoning      |
| `gpt-4o-mini`       | `openai`    | chat, classification         |

`find_models(capability=..., provider=...)` returns enabled models matching the filters, **in registry order**. That order is what makes `azir-auto` and fallback selection deterministic.

Models not in the registry are rejected with a 400 -- there is no `claude-*` / `gpt-*` prefix-based routing. To route to a new model, register it.

### `router.py`

Routing and orchestration:

- `resolve_model(request)` -- explicit model or `azir-auto` + `task` -> one enabled, registered `ModelConfig` (or a clean 400; a registry entry naming a provider Azir doesn't implement is a 500 configuration error)
- `plan_attempts(request)` -- the resolved model, then for each other provider in `PROVIDER_ORDER` the first enabled registry model of that provider (that also supports `task`, if given)
- `route_request(app, request)` -- runs the plan for non-streaming requests, with fallback and per-attempt telemetry
- `stream_chat_completion(app, request)` -- resolves the model, then calls that provider's `stream()`; no fallback

**Fallback policy.** `route_request()` moves to the next attempt only when `providers.errors.is_transient_provider_error()` says the failure is transient: rate limiting (429), provider unavailability / unexpected 5xx (500/502/503), and timeouts (504) or connection failures. Anything else is raised immediately without trying another provider:

- malformed requests (upstream 400) and other request-level upstream 4xx
- credential / permission errors (401 / 403) -- these indicate misconfiguration and should surface, not be masked
- model/resource not found (404)
- Azir's own routing errors (unknown model, disabled model, missing or unsupported task) -- these are rejected before any provider is called

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

Create a `.env` file:

```env
ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_API_KEY=your_openai_key
```

Never commit `.env`. Configuration is loaded through `pydantic-settings`; a missing key fails at startup rather than at request time.

## Setup

Install dependencies with `uv`:

```bash
uv sync
```

Run the server:

```bash
uv run uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000` (interactive docs at `/docs`).

Run the tests (no real provider calls are made):

```bash
uv run pytest
```

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

- cross-provider fallback for streaming requests
- telemetry for streaming requests
- retries within a single provider (fallback moves to the *next provider*)
- telemetry persistence or aggregation (records are logged, not stored)
- billing-accurate cost tracking (only rough static rates from the registry)
- routing to models that aren't in the registry

## Future Work

Not implemented today:

1. Cost-aware routing
2. Latency-aware routing
3. Health-aware routing (provider health scoring, circuit breaking)
4. Telemetry persistence and aggregation, including streaming telemetry
5. More advanced routing/fallback policies (per-provider retries, streaming fallback, richer task selection)
