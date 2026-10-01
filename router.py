import time
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException

from model_registry import ModelConfig, find_models, get_model
from providers.errors import is_transient_provider_error
from schemas import ChatRequest, ChatResponse
from telemetry import RequestTelemetry, estimate_cost_usd

# Virtual model name: Azir picks the cheapest capable model from the
# registry based on the request's `task`.
AUTO_MODEL = "azir-auto"

# Providers Azir has an implementation for, in the order they are tried
# when falling back.
PROVIDER_ORDER = ["anthropic", "openai"]

# Rough characters-per-token ratio used to estimate input size for routing.
CHARS_PER_TOKEN = 4

# Output-token estimate when the request has no `max_tokens`. Matches the
# default AnthropicProvider sends upstream in that case.
DEFAULT_OUTPUT_TOKENS_ESTIMATE = 200


def estimate_request_tokens(request: ChatRequest) -> int:
    """Estimated input tokens: total message characters / 4, rounded up.

    A deterministic routing heuristic only -- not what any provider will
    actually count or bill.
    """
    chars = sum(len(message.content) for message in request.messages)
    return -(-chars // CHARS_PER_TOKEN)


def estimate_request_cost_usd(request: ChatRequest, config: ModelConfig) -> float:
    """Pre-execution cost estimate for running `request` on `config`, using
    estimated input tokens and `max_tokens` (or a default) as output.
    """
    output_tokens = request.max_tokens or DEFAULT_OUTPUT_TOKENS_ESTIMATE
    return config.estimate_cost_usd(estimate_request_tokens(request), output_tokens)


def _within_budget(request: ChatRequest, candidates: list[ModelConfig]) -> list[ModelConfig]:
    if request.max_cost_usd is None:
        return candidates

    return [
        config
        for config in candidates
        if estimate_request_cost_usd(request, config) <= request.max_cost_usd
    ]


def resolve_model(request: ChatRequest) -> ModelConfig:
    """Resolve the request's model to one enabled, registered concrete
    model. The registry is the only source of truth: unknown or disabled
    models are rejected rather than guessed at.

    An explicit model is used as-is. `azir-auto` picks, among enabled
    models with the `task` capability that fit `max_cost_usd` (if given),
    the one with the lowest estimated request cost; ties go to registry
    order.
    """
    if request.model == AUTO_MODEL:
        if not request.task:
            raise HTTPException(
                status_code=400,
                detail=f"'task' is required when model is '{AUTO_MODEL}'.",
            )

        candidates = find_models(capability=request.task)

        if not candidates:
            raise HTTPException(
                status_code=400,
                detail=f"No enabled model supports task: {request.task}",
            )

        candidates = _within_budget(request, candidates)

        if not candidates:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"No enabled model supports task '{request.task}' within "
                    f"max_cost_usd={request.max_cost_usd}"
                ),
            )

        # min() keeps the first of equal keys, so ties fall to registry order.
        config = min(candidates, key=lambda c: estimate_request_cost_usd(request, c))
    else:
        config = get_model(request.model)

        if config is None:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model: {request.model}",
            )

        if not config.enabled:
            raise HTTPException(
                status_code=400,
                detail=f"Model is disabled: {request.model}",
            )

    if config.provider not in PROVIDER_ORDER:
        raise HTTPException(
            status_code=500,
            detail=f"Unsupported provider configuration: {config.provider}",
        )

    return config


def plan_attempts(request: ChatRequest) -> list[ModelConfig]:
    """The resolved primary model, followed by one fallback model per
    remaining provider in `PROVIDER_ORDER` -- the first enabled registry
    model for that provider that also supports `request.task`, if given,
    and fits `request.max_cost_usd`, if given.
    """
    primary = resolve_model(request)
    attempts = [primary]

    for name in PROVIDER_ORDER:
        if name == primary.provider:
            continue

        fallbacks = _within_budget(request, find_models(capability=request.task, provider=name))
        if fallbacks:
            attempts.append(fallbacks[0])

    return attempts


def _get_provider(app: FastAPI, name: str):
    return getattr(app.state, f"{name}_provider")


def _with_model(request: ChatRequest, config: ModelConfig) -> ChatRequest:
    return request.model_copy(update={"model": config.name})


async def route_request(app: FastAPI, request: ChatRequest) -> ChatResponse:
    """Run a non-streaming request against the planned concrete models in
    order (see `plan_attempts`), moving to the next one only when a
    provider fails transiently (rate limit, 5xx, timeout, connection
    error). Any other provider error is raised immediately. If every
    attempt fails transiently, the last error is re-raised.

    Each attempt (success or failure) emits a RequestTelemetry record for
    the concrete provider/model that was actually attempted.
    """
    last_error: HTTPException | None = None

    for config in plan_attempts(request):
        provider = _get_provider(app, config.provider)
        started_at = time.perf_counter()

        try:
            response = await provider.complete(_with_model(request, config))
        except HTTPException as exc:
            RequestTelemetry(
                provider=config.provider,
                model=config.name,
                status="error",
                status_code=exc.status_code,
                latency_ms=(time.perf_counter() - started_at) * 1000,
            ).emit()

            if not is_transient_provider_error(exc):
                raise

            last_error = exc
            continue

        usage = response.usage
        RequestTelemetry(
            provider=config.provider,
            model=config.name,
            status="success",
            status_code=200,
            latency_ms=(time.perf_counter() - started_at) * 1000,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            estimated_cost_usd=estimate_cost_usd(
                config.provider, config.name, usage.prompt_tokens, usage.completion_tokens
            ),
        ).emit()

        return response

    raise last_error


async def stream_chat_completion(app: FastAPI, request: ChatRequest) -> AsyncIterator[str]:
    """Resolve the request to a concrete model (same rules as
    `route_request`), open a streaming connection via its provider, and
    return an async iterator of ready-to-send SSE chunks.

    There is no cross-provider fallback here: once the StreamingResponse
    starts, the 200 status is already committed. Providers validate the
    upstream connection before returning, so pre-stream failures still
    surface as a normal HTTPException.

    No RequestTelemetry is emitted for streams yet -- latency and usage
    are only known after the response has started. See README.
    """
    config = resolve_model(request)
    provider = _get_provider(app, config.provider)

    return await provider.stream(_with_model(request, config))
