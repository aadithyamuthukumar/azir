import time
from typing import AsyncIterator

from fastapi import BackgroundTasks, FastAPI, HTTPException

from health import is_healthy, record_failure, record_success
from judge import schedule_evaluation
from latency import DEFAULT_LATENCY_ESTIMATE_MS, get_latency_estimate, record_latency
from model_registry import ModelConfig, find_models, get_model
from providers.errors import is_timeout_provider_error, is_transient_provider_error
from schemas import ChatRequest, ChatResponse
from telemetry import RequestTelemetry, estimate_cost_usd, publish

# Virtual model name: Azir picks a capable model from the registry based on
# the request's `task` and `routing_policy`.
AUTO_MODEL = "azir-auto"

# Policy used when an `azir-auto` request doesn't name one.
DEFAULT_ROUTING_POLICY = "balanced"

# Weights of the `balanced` score. Cost and latency are each min-max
# normalized across the eligible candidates first, so these are unitless.
BALANCED_COST_WEIGHT = 0.5
BALANCED_LATENCY_WEIGHT = 0.5

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


def routing_latency_ms(config: ModelConfig) -> float:
    """The model's observed latency estimate, or the neutral cold-start
    default if it has never been observed.
    """
    estimate = get_latency_estimate(config.name)
    return DEFAULT_LATENCY_ESTIMATE_MS if estimate is None else estimate


def _within_budget(request: ChatRequest, candidates: list[ModelConfig]) -> list[ModelConfig]:
    if request.max_cost_usd is None:
        return candidates

    return [
        config
        for config in candidates
        if estimate_request_cost_usd(request, config) <= request.max_cost_usd
    ]


def _eligible_candidates(request: ChatRequest) -> list[ModelConfig]:
    """The `azir-auto` candidate set every routing policy ranks: enabled
    models with the `task` capability that fit `max_cost_usd` (if given),
    in registry order. Raises a clean 400 if there are none.
    """
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

    return candidates


def _prefer_healthy(candidates: list[ModelConfig]) -> list[ModelConfig]:
    """Drop candidates the health tracker currently marks unhealthy. If that
    would drop all of them, keep the full set instead: stale health state
    must not turn into a hard outage, and a failing primary still falls
    back through `route_request` as usual.
    """
    healthy = [config for config in candidates if is_healthy(config.name)]
    return healthy or candidates


def normalize(values: list[float]) -> list[float]:
    """Min-max scale to [0, 1]: (value - min) / (max - min). If every value
    is equal, each normalizes to 0 (no spread, so no preference).
    """
    low, high = min(values), max(values)

    if high == low:
        return [0.0] * len(values)

    return [(value - low) / (high - low) for value in values]


def select_by_policy(request: ChatRequest, candidates: list[ModelConfig]) -> ModelConfig:
    """Pick one of the eligible `candidates` (registry order) by the
    request's routing policy; lower score wins:

    - cheap:    estimated request cost
    - fast:     latency estimate (cold-start default for unobserved models)
    - balanced: BALANCED_COST_WEIGHT * normalized cost
                + BALANCED_LATENCY_WEIGHT * normalized latency
    """
    policy = request.routing_policy or DEFAULT_ROUTING_POLICY

    if policy == "cheap":
        scores = [estimate_request_cost_usd(request, c) for c in candidates]
    elif policy == "fast":
        scores = [routing_latency_ms(c) for c in candidates]
    else:
        costs = normalize([estimate_request_cost_usd(request, c) for c in candidates])
        latencies = normalize([routing_latency_ms(c) for c in candidates])
        scores = [
            BALANCED_COST_WEIGHT * cost + BALANCED_LATENCY_WEIGHT * latency
            for cost, latency in zip(costs, latencies)
        ]

    # min() keeps the first of equal scores, so ties fall to registry order.
    best = min(range(len(candidates)), key=lambda i: scores[i])
    return candidates[best]


def resolve_model(request: ChatRequest) -> ModelConfig:
    """Resolve the request's model to one enabled, registered concrete
    model. The registry is the only source of truth: unknown or disabled
    models are rejected rather than guessed at.

    An explicit model is used as-is (`routing_policy` and health are
    ignored). `azir-auto` ranks the eligible candidates -- minus unhealthy
    ones, unless all are unhealthy -- with `select_by_policy`.
    """
    if request.model == AUTO_MODEL:
        candidates = _prefer_healthy(_eligible_candidates(request))
        config = select_by_policy(request, candidates)
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


def _telemetry_sink(app: FastAPI):
    # None when persistence isn't configured: telemetry is then only logged.
    return getattr(app.state, "telemetry_store", None)


def _with_model(request: ChatRequest, config: ModelConfig) -> ChatRequest:
    return request.model_copy(update={"model": config.name})


async def route_request(
    app: FastAPI,
    request: ChatRequest,
    background_tasks: BackgroundTasks | None = None,
) -> ChatResponse:
    """Run a non-streaming request against the planned concrete models in
    order (see `plan_attempts`), moving to the next one only when a
    provider fails transiently (rate limit, 5xx, timeout, connection
    error). Any other provider error is raised immediately. If every
    attempt fails transiently, the last error is re-raised.

    Each attempt (success or failure) publishes a RequestTelemetry record
    for the concrete provider/model that was actually attempted -- logged,
    and persisted if a telemetry store is configured. Its latency
    also feeds that model's routing estimate on success or timeout; fast
    error responses and connection failures don't, since their elapsed
    time says nothing about how fast the model answers. Each attempt also
    updates that model's health: a success, or a failure if it failed
    transiently. Non-transient errors (bad request, credentials, not
    found) aren't model-health signals and are not recorded.

    If `background_tasks` is given and LLM judging is enabled, the
    successful response is also queued for a quality evaluation that runs
    after it is sent (see judge.py). It never changes the response.
    """
    last_error: HTTPException | None = None

    for config in plan_attempts(request):
        provider = _get_provider(app, config.provider)
        started_at = time.perf_counter()

        try:
            response = await provider.complete(_with_model(request, config))
        except HTTPException as exc:
            latency_ms = (time.perf_counter() - started_at) * 1000
            await publish(
                RequestTelemetry(
                    provider=config.provider,
                    model=config.name,
                    status="error",
                    status_code=exc.status_code,
                    latency_ms=latency_ms,
                ),
                _telemetry_sink(app),
            )

            if is_timeout_provider_error(exc):
                record_latency(config.name, latency_ms)

            if not is_transient_provider_error(exc):
                raise

            record_failure(config.name)
            last_error = exc
            continue

        latency_ms = (time.perf_counter() - started_at) * 1000
        record_latency(config.name, latency_ms)
        record_success(config.name)

        usage = response.usage
        telemetry_id = await publish(
            RequestTelemetry(
                provider=config.provider,
                model=config.name,
                status="success",
                status_code=200,
                latency_ms=latency_ms,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                estimated_cost_usd=estimate_cost_usd(
                    config.provider, config.name, usage.prompt_tokens, usage.completion_tokens
                ),
            ),
            _telemetry_sink(app),
        )
        schedule_evaluation(background_tasks, app, request, config, response, telemetry_id)

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
    are only known after the response has started. See README. For the
    same reason streams read the latency estimates but never update them.

    Health is updated from whether the stream *opened*: a success once the
    provider accepted the connection, a failure on a transient pre-stream
    error. What happens mid-stream is not recorded.
    """
    config = resolve_model(request)
    provider = _get_provider(app, config.provider)

    try:
        event_stream = await provider.stream(_with_model(request, config))
    except HTTPException as exc:
        if is_transient_provider_error(exc):
            record_failure(config.name)
        raise

    record_success(config.name)
    return event_stream
