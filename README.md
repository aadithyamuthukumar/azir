# Azir

Azir is a lightweight multi-provider LLM gateway built with Python and FastAPI.

It exposes a single OpenAI-style chat completions endpoint and forwards requests to different model providers behind the scenes. Each provider adapter translates the shared Azir request format into the provider-specific wire format, sends the request using raw HTTP, and normalizes the provider response back into one consistent response shape.

## Current Status

Azir currently supports:

- FastAPI HTTP server
- `POST /v1/chat/completions`
- OpenAI-style request validation with Pydantic
- Anthropic provider support
- OpenAI provider support
- Shared provider interface
- Provider selection through `router.py`
- Anthropic system-message translation
- Anthropic stop-reason normalization
- Normalized token usage
- Shared `httpx.AsyncClient`
- API keys loaded from `.env`
- Non-streaming chat completions
- Explicit rejection of `stream=true`

Current request flow:

```text
Client
  |
  v
POST /v1/chat/completions
  |
  v
FastAPI + Pydantic validation
  |
  v
router.select_provider(...)
  |
  +--> AnthropicProvider
  |
  +--> OpenAIProvider
  |
  v
Provider API
  |
  v
Normalize provider response
  |
  v
ChatResponse
  |
  v
Client
```

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
azir/
├── main.py
├── router.py
├── schemas.py
├── config.py
├── providers/
│   ├── base.py
│   ├── anthropic.py
│   └── openai.py
├── .env
├── .gitignore
└── pyproject.toml
```

### `main.py`

Owns the HTTP layer.

Responsibilities:

- create the FastAPI application
- create one shared `httpx.AsyncClient`
- initialize provider instances during application startup
- expose `/v1/chat/completions`
- reject unsupported streaming requests
- ask the router which provider should handle a request

`main.py` should not contain provider-specific translation logic.

### `schemas.py`

Defines Azir's request and response contracts using Pydantic.

The main models are:

- `Message`
- `ChatRequest`
- `ChatResponse`
- `Choice`
- `Usage`

The public request format is intentionally OpenAI-like so clients can use one familiar shape across providers.

### `providers/base.py`

Defines the common provider contract.

Conceptually:

```python
complete(request: ChatRequest) -> ChatResponse
```

Every provider must accept the same Azir request model and return the same Azir response model.

This allows the rest of the application to work with providers without knowing their internal API details.

### `providers/anthropic.py`

Translates between Azir's shared format and Anthropic's Messages API.

Current Anthropic-specific responsibilities include:

- moving `system` messages to Anthropic's top-level `system` field
- supplying an Anthropic-compatible request body
- authenticating with Anthropic
- extracting text from Anthropic `content` blocks
- mapping Anthropic `stop_reason` values to Azir/OpenAI-style `finish_reason` values
- normalizing Anthropic token usage

Example mapping:

```text
Anthropic                 Azir
----------------------------------------
end_turn               -> stop
stop_sequence          -> stop
max_tokens             -> length
input_tokens           -> prompt_tokens
output_tokens          -> completion_tokens
```

### `providers/openai.py`

Calls OpenAI using the same Azir `ChatRequest` and returns the same `ChatResponse`.

Because Azir's internal/public format is already OpenAI-like, this provider requires less translation than Anthropic.

### `router.py`

Chooses which provider handles a request.

Current routing is intentionally simple and based on the requested model name.

Example:

```text
claude-* -> AnthropicProvider
gpt-*    -> OpenAIProvider
```

Unknown models should be rejected instead of silently being sent to the wrong provider.

The long-term goal is to evolve this into a smarter routing layer based on factors such as:

- cost
- latency
- capability
- provider health
- reliability
- fallback policies

## Configuration

Create a `.env` file:

```env
ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_API_KEY=your_openai_key
```

Never commit `.env`.

Make sure `.gitignore` contains:

```text
.env
.venv/
__pycache__/
```

Configuration is loaded through `pydantic-settings`.

## Setup

Initialize and install dependencies with `uv`:

```bash
uv init azir
cd azir
uv add fastapi "uvicorn[standard]" httpx pydantic pydantic-settings
```

Run the server:

```bash
uv run uvicorn main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

FastAPI documentation:

```text
http://127.0.0.1:8000/docs
```

## Example Request

Anthropic:

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-6",
    "messages": [
      {
        "role": "user",
        "content": "Say hello in one sentence."
      }
    ],
    "max_tokens": 50
  }'
```

OpenAI:

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [
      {
        "role": "user",
        "content": "Say hello in one sentence."
      }
    ],
    "max_tokens": 50
  }'
```

Both providers should return the same general response shape:

```json
{
  "model": "provider-model-name",
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
  }
}
```

Actual token values depend on the request and provider.

## Important Design Decisions

### Raw `httpx` instead of provider SDKs

Azir uses raw HTTP calls so provider-specific wire formats remain explicit.

This gives the project direct control over:

- request URLs
- headers
- JSON bodies
- response parsing
- error handling
- provider translation

It also avoids hiding important provider differences behind SDK abstractions.

### Shared `Provider` interface

Every provider exposes the same operation:

```text
ChatRequest -> complete() -> ChatResponse
```

This keeps routing separate from provider implementation details and makes future providers easier to add.

### Shared `httpx.AsyncClient`

Azir creates one `AsyncClient` during FastAPI startup and reuses it across requests.

This enables connection pooling and avoids repeatedly creating and tearing down HTTP clients for every model call.

## Current Limitations

Azir does not yet support:

- streaming
- retries
- provider fallbacks
- normalized provider error responses
- rate-limit handling
- timeout policies
- automated tests
- telemetry persistence
- cost tracking
- latency-based routing
- capability-based routing

## Next Milestones

Near-term work:

1. Provider error handling
2. Automated provider/router tests
3. Fallback routing
4. Streaming responses
5. Request telemetry
6. Token and cost tracking
7. Smarter model routing

Longer-term direction:

```text
Request
   |
   v
Azir Router
   |
   +--> evaluate cost
   +--> evaluate latency
   +--> evaluate capability
   +--> evaluate provider health
   |
   v
Select model/provider
   |
   v
Execute request
   |
   v
Record outcome
```

The goal is for Azir to evolve from a multi-provider compatibility layer into an intelligent LLM routing and control plane.
