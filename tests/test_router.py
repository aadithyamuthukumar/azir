import json
import logging
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

import model_registry
import router
from health import MIN_HEALTH_SAMPLES, get_sample_count, get_success_rate, is_healthy, record_failure
from latency import DEFAULT_LATENCY_ESTIMATE_MS, get_latency_stats, record_latency
from model_registry import MODEL_REGISTRY, ModelConfig
from router import (
    DEFAULT_OUTPUT_TOKENS_ESTIMATE,
    estimate_request_cost_usd,
    estimate_request_tokens,
    normalize,
    plan_attempts,
    resolve_model,
    route_request,
    stream_chat_completion,
)
from schemas import ChatRequest, ChatResponse, Choice, Message, Usage

ANTHROPIC_MODEL = "claude-sonnet-4-6"
OPENAI_MODEL = "gpt-4o-mini"


class StubProvider:
    def __init__(self, result=None, error=None, stream_result=None, stream_error=None):
        self.result = result
        self.error = error
        self.requests = []
        self.stream_result = stream_result
        self.stream_error = stream_error
        self.stream_requests = []

    async def complete(self, request):
        self.requests.append(request)

        if self.error is not None:
            raise self.error

        return self.result

    async def stream(self, request):
        self.stream_requests.append(request)

        if self.stream_error is not None:
            raise self.stream_error

        return self.stream_result


def make_app(anthropic=None, openai=None, telemetry_store=None):
    return SimpleNamespace(
        state=SimpleNamespace(
            anthropic_provider=anthropic or StubProvider(),
            openai_provider=openai or StubProvider(),
            telemetry_store=telemetry_store,
        )
    )


def make_request(model: str, task: str | None = None, **extra) -> ChatRequest:
    return ChatRequest(model=model, task=task, messages=[Message(role="user", content="hi")], **extra)


def make_response(model: str) -> ChatResponse:
    return ChatResponse(
        model=model,
        choices=[Choice(message=Message(role="assistant", content="hello"), finish_reason="stop")],
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


def upstream_error(status_code: int, mapped_status: int) -> HTTPException:
    """An HTTPException shaped like raise_provider_error()'s output: the
    client-facing status, chained from the original upstream httpx error."""
    request = httpx.Request("POST", "https://example.invalid")
    response = httpx.Response(status_code, request=request)
    exc = HTTPException(status_code=mapped_status, detail="upstream failure")
    exc.__cause__ = httpx.HTTPStatusError("error", request=request, response=response)
    return exc


@pytest.fixture
def disable_model(monkeypatch):
    def _disable(name: str):
        monkeypatch.setattr(MODEL_REGISTRY[name], "enabled", False)

    return _disable


# --- Model resolution ---


def test_resolve_explicit_anthropic_model():
    config = resolve_model(make_request(ANTHROPIC_MODEL))

    assert (config.provider, config.name) == ("anthropic", ANTHROPIC_MODEL)


def test_resolve_explicit_openai_model():
    config = resolve_model(make_request(OPENAI_MODEL))

    assert (config.provider, config.name) == ("openai", OPENAI_MODEL)


@pytest.mark.parametrize("model", ["mystery-model", "claude-unregistered", "gpt-unregistered"])
def test_resolve_rejects_unknown_model_even_with_known_prefix(model):
    with pytest.raises(HTTPException) as exc_info:
        resolve_model(make_request(model))

    assert exc_info.value.status_code == 400
    assert "Unknown model" in exc_info.value.detail


def test_resolve_rejects_disabled_explicit_model(disable_model):
    disable_model(ANTHROPIC_MODEL)

    with pytest.raises(HTTPException) as exc_info:
        resolve_model(make_request(ANTHROPIC_MODEL))

    assert exc_info.value.status_code == 400


@pytest.mark.parametrize(
    "task, expected",
    [
        ("coding", ANTHROPIC_MODEL),
        ("reasoning", ANTHROPIC_MODEL),
        ("classification", OPENAI_MODEL),
        # both can chat; gpt-4o-mini is cheaper despite coming second in the registry
        ("chat", OPENAI_MODEL),
    ],
)
def test_resolve_auto_picks_cheapest_capable_model(task, expected):
    assert resolve_model(make_request("azir-auto", task)).name == expected


def test_resolve_auto_requires_task():
    with pytest.raises(HTTPException) as exc_info:
        resolve_model(make_request("azir-auto"))

    assert exc_info.value.status_code == 400
    assert "task" in exc_info.value.detail


def test_resolve_auto_rejects_unsupported_task():
    with pytest.raises(HTTPException) as exc_info:
        resolve_model(make_request("azir-auto", "image-generation"))

    assert exc_info.value.status_code == 400


def test_resolve_auto_skips_disabled_models(disable_model):
    # the cheaper model is disabled, so the pricier capable one is chosen
    disable_model(OPENAI_MODEL)

    assert resolve_model(make_request("azir-auto", "chat")).name == ANTHROPIC_MODEL


def test_resolve_auto_rejects_task_when_only_capable_model_is_disabled(disable_model):
    disable_model(ANTHROPIC_MODEL)

    with pytest.raises(HTTPException) as exc_info:
        resolve_model(make_request("azir-auto", "coding"))

    assert exc_info.value.status_code == 400


def test_resolve_surfaces_misconfigured_provider(monkeypatch):
    monkeypatch.setitem(
        MODEL_REGISTRY,
        "mystery-model",
        ModelConfig("mystery-model", "mystery", 0.0, 0.0, {"chat"}),
    )

    with pytest.raises(HTTPException) as exc_info:
        resolve_model(make_request("mystery-model"))

    assert exc_info.value.status_code == 500


# --- Cost estimation and cost-aware azir-auto selection ---


def cost_request(task="coding", content="hi", **extra) -> ChatRequest:
    return ChatRequest(
        model="azir-auto", task=task, messages=[Message(role="user", content=content)], **extra
    )


@pytest.fixture
def registry(monkeypatch):
    """Replace the registry with the given models, in the given order."""

    def _install(*models: ModelConfig):
        monkeypatch.setattr(model_registry, "MODEL_REGISTRY", {m.name: m for m in models})

    return _install


def model(name, provider="anthropic", input_cost=0.001, output_cost=0.001, capabilities=("chat", "coding"), enabled=True):
    return ModelConfig(name, provider, input_cost, output_cost, set(capabilities), enabled)


def test_estimate_request_tokens_is_chars_over_four_rounded_up():
    request = ChatRequest(
        model="azir-auto",
        messages=[Message(role="system", content="hello"), Message(role="user", content="world!!")],
    )

    # 5 + 7 = 12 characters across all messages -> 3 tokens
    assert estimate_request_tokens(request) == 3
    assert estimate_request_tokens(request) == 3
    assert estimate_request_tokens(cost_request(content="x" * 13)) == 4
    assert estimate_request_tokens(cost_request(content="")) == 0


def test_estimate_request_cost_uses_max_tokens_or_default_output_budget():
    config = model("m", input_cost=1.0, output_cost=10.0)

    # 400 chars -> 100 input tokens; 50 output tokens
    assert estimate_request_cost_usd(cost_request(content="x" * 400, max_tokens=50), config) == pytest.approx(
        0.1 * 1.0 + 0.05 * 10.0
    )
    assert estimate_request_cost_usd(cost_request(content="x" * 400), config) == pytest.approx(
        0.1 * 1.0 + (DEFAULT_OUTPUT_TOKENS_ESTIMATE / 1000) * 10.0
    )


def test_auto_picks_cheapest_capable_model_regardless_of_registry_order(registry):
    registry(
        model("pricey", input_cost=0.01, output_cost=0.05),
        model("cheap", provider="openai", input_cost=0.001, output_cost=0.002),
    )

    assert resolve_model(cost_request()).name == "cheap"


def test_auto_ignores_cheaper_incapable_model(registry):
    registry(
        model("capable", input_cost=0.01, output_cost=0.05),
        model("cheap-chat-only", provider="openai", input_cost=0.0, output_cost=0.0, capabilities=("chat",)),
    )

    assert resolve_model(cost_request(task="coding")).name == "capable"


def test_auto_ignores_cheaper_disabled_model(registry):
    registry(
        model("enabled", input_cost=0.01, output_cost=0.05),
        model("cheap-disabled", provider="openai", input_cost=0.0, output_cost=0.0, enabled=False),
    )

    assert resolve_model(cost_request()).name == "enabled"


def test_auto_breaks_cost_ties_by_registry_order(registry):
    registry(
        model("first", provider="openai"),
        model("second", provider="anthropic"),
    )

    assert resolve_model(cost_request()).name == "first"


def test_auto_cost_weighs_both_input_and_output(registry):
    registry(
        model("cheap-input", input_cost=0.001, output_cost=0.1),
        model("cheap-output", provider="openai", input_cost=0.1, output_cost=0.001),
    )

    # long prompt, short answer -> input price dominates
    assert resolve_model(cost_request(content="x" * 40_000, max_tokens=10)).name == "cheap-input"
    # short prompt, long answer -> output price dominates
    assert resolve_model(cost_request(content="hi", max_tokens=4000)).name == "cheap-output"


def test_explicit_model_ignores_cost_and_budget():
    request = ChatRequest(
        model=ANTHROPIC_MODEL,
        task="chat",
        max_cost_usd=0.0,
        messages=[Message(role="user", content="hi")],
    )

    # gpt-4o-mini would be cheaper, but an explicit model is always honored
    assert resolve_model(request).name == ANTHROPIC_MODEL


def test_auto_excludes_candidates_over_budget(registry):
    # with "hi" (1 input token) + 200 default output tokens, output price dominates:
    # "cheap" ~= 0.2 * 1.0 = 0.2, "pricey" ~= 0.2 * 2.0 = 0.4
    registry(
        model("pricey", provider="anthropic", input_cost=0.0, output_cost=2.0),
        model("cheap", provider="openai", input_cost=0.0, output_cost=1.0),
    )

    # within budget: both are candidates, so "pricey" remains the fallback
    assert [c.name for c in plan_attempts(cost_request(max_cost_usd=1.0))] == ["cheap", "pricey"]
    # budget is inclusive; "pricey" is excluded from selection and fallback
    assert [c.name for c in plan_attempts(cost_request(max_cost_usd=0.2))] == ["cheap"]

    with pytest.raises(HTTPException) as exc_info:
        resolve_model(cost_request(max_cost_usd=0.19))

    assert exc_info.value.status_code == 400
    assert "max_cost_usd" in exc_info.value.detail


def test_auto_budget_with_real_registry_returns_400_when_nothing_fits():
    # claude-sonnet-4-6 is the only coding model: ~0.003 USD for "hi" + 200 output tokens
    with pytest.raises(HTTPException) as exc_info:
        resolve_model(cost_request(task="coding", max_cost_usd=0.001))

    assert exc_info.value.status_code == 400


def test_plan_fallbacks_exclude_models_over_budget():
    # gpt-4o-mini ~0.00012 fits; claude-sonnet-4-6 ~0.003 does not
    request = cost_request(task="chat", max_cost_usd=0.001)

    assert [c.name for c in plan_attempts(request)] == [OPENAI_MODEL]


def test_max_cost_usd_rejects_negative_values():
    with pytest.raises(ValidationError):
        cost_request(max_cost_usd=-1)


# --- Routing policies ---

POLICIES = ["cheap", "fast", "balanced"]


@pytest.fixture
def three_models(registry):
    """Three eligible coding models. With "hi" + 200 default output tokens
    (input is free), estimated cost is 0.2 * output_cost:

        model    cost   latency   norm cost   norm latency   balanced
        cheap    0.2    900 ms    0.0         1.0            0.5
        middle   0.4    400 ms    0.5         0.1667         0.3333
        quick    0.6    300 ms    1.0         0.0            0.5
    """
    registry(
        model("cheap", provider="openai", input_cost=0.0, output_cost=1.0),
        model("middle", input_cost=0.0, output_cost=2.0),
        model("quick", input_cost=0.0, output_cost=3.0),
    )
    record_latency("cheap", 900.0)
    record_latency("middle", 400.0)
    record_latency("quick", 300.0)


@pytest.fixture
def cheap_slow_vs_pricey_fast(registry):
    registry(
        model("cheap-slow", provider="openai", input_cost=0.001, output_cost=0.001),
        model("pricey-fast", input_cost=0.01, output_cost=0.05),
    )
    record_latency("cheap-slow", 900.0)
    record_latency("pricey-fast", 300.0)


# General


@pytest.mark.parametrize("policy", [None, *POLICIES])
@pytest.mark.parametrize("explicit", [ANTHROPIC_MODEL, OPENAI_MODEL])
def test_explicit_model_ignores_policy_and_latency(explicit, policy):
    record_latency(ANTHROPIC_MODEL, 5000.0)
    record_latency(OPENAI_MODEL, 5000.0)

    assert resolve_model(make_request(explicit, "chat", routing_policy=policy)).name == explicit


def test_missing_policy_defaults_to_balanced(three_models):
    request = cost_request()

    assert request.routing_policy is None
    assert resolve_model(request).name == resolve_model(cost_request(routing_policy="balanced")).name == "middle"


@pytest.mark.parametrize("policy", ["fastest", "BALANCED", ""])
def test_invalid_policy_is_rejected(policy):
    with pytest.raises(ValidationError):
        cost_request(routing_policy=policy)


# cheap


def test_cheap_picks_lowest_estimated_cost(three_models):
    assert resolve_model(cost_request(routing_policy="cheap")).name == "cheap"


def test_cheap_ignores_faster_but_pricier_model(cheap_slow_vs_pricey_fast):
    assert resolve_model(cost_request(routing_policy="cheap")).name == "cheap-slow"


def test_cheap_breaks_ties_by_registry_order_not_latency(registry):
    registry(model("first", provider="openai"), model("second"))
    record_latency("first", 900.0)
    record_latency("second", 100.0)

    assert resolve_model(cost_request(routing_policy="cheap")).name == "first"


# fast


def test_fast_picks_lowest_latency(three_models):
    assert resolve_model(cost_request(routing_policy="fast")).name == "quick"


def test_fast_ignores_cheaper_but_slower_model(cheap_slow_vs_pricey_fast):
    assert resolve_model(cost_request(routing_policy="fast")).name == "pricey-fast"


def test_fast_with_real_registry():
    record_latency(ANTHROPIC_MODEL, 900.0)
    record_latency(OPENAI_MODEL, 300.0)

    assert resolve_model(make_request("azir-auto", "chat", routing_policy="fast")).name == OPENAI_MODEL


def test_fast_breaks_ties_by_registry_order_not_cost(registry):
    registry(
        model("first", input_cost=0.01),
        model("second", provider="openai", input_cost=0.001),
    )
    record_latency("first", 500.0)
    record_latency("second", 500.0)

    assert resolve_model(cost_request(routing_policy="fast")).name == "first"


def test_fast_cold_start_is_deterministic():
    # no history: every model sits at the same default, so registry order decides
    request = make_request("azir-auto", "chat", routing_policy="fast")

    assert [resolve_model(request).name for _ in range(3)] == [ANTHROPIC_MODEL] * 3


def test_fast_cold_start_unknown_model_uses_neutral_default(registry):
    registry(
        model("known", provider="openai"),
        model("unknown"),
    )
    request = cost_request(routing_policy="fast")

    # observed faster than the default: beats the untried model
    record_latency("known", DEFAULT_LATENCY_ESTIMATE_MS - 600)
    assert resolve_model(request).name == "known"

    # observed slower than the default: the untried model gets a turn
    for _ in range(10):
        record_latency("known", DEFAULT_LATENCY_ESTIMATE_MS + 2000)
    assert resolve_model(request).name == "unknown"


# balanced


def test_balanced_weighs_both_cost_and_latency(three_models):
    # neither the cheapest nor the fastest, but the best of both
    assert resolve_model(cost_request(routing_policy="balanced")).name == "middle"


def test_normalize_scales_to_unit_range():
    assert normalize([0.2, 0.4, 0.6]) == pytest.approx([0.0, 0.5, 1.0])
    assert normalize([900.0, 400.0, 300.0]) == pytest.approx([1.0, 1 / 6, 0.0])


@pytest.mark.parametrize("values", [[5.0, 5.0, 5.0], [7.0]])
def test_normalize_equal_values_are_zero(values):
    assert normalize(values) == [0.0] * len(values)


def test_balanced_equal_costs_compare_latency_only(registry):
    registry(model("slow", provider="openai"), model("quick"))
    record_latency("slow", 900.0)
    record_latency("quick", 300.0)

    assert resolve_model(cost_request(routing_policy="balanced")).name == "quick"


def test_balanced_equal_latency_compares_cost_only(registry):
    # cold start: both at the default latency
    registry(
        model("pricey", input_cost=0.01),
        model("cheap", provider="openai", input_cost=0.001),
    )

    assert resolve_model(cost_request(routing_policy="balanced")).name == "cheap"


def test_balanced_cold_start_with_real_registry_picks_cheapest():
    assert resolve_model(make_request("azir-auto", "chat")).name == OPENAI_MODEL


def test_balanced_ties_go_to_registry_order(cheap_slow_vs_pricey_fast, registry):
    # two candidates, one cheaper and one faster: each normalizes to
    # 0.5 * 0 + 0.5 * 1 = 0.5, so registry order decides
    assert resolve_model(cost_request(routing_policy="balanced")).name == "cheap-slow"

    registry(
        model("pricey-fast", input_cost=0.01, output_cost=0.05),
        model("cheap-slow", provider="openai", input_cost=0.001, output_cost=0.001),
    )
    assert resolve_model(cost_request(routing_policy="balanced")).name == "pricey-fast"


# Shared eligibility


@pytest.mark.parametrize("policy", POLICIES)
def test_policies_ignore_incapable_model(registry, policy):
    registry(
        model("capable", input_cost=0.01),
        model("chat-only", provider="openai", input_cost=0.0, output_cost=0.0, capabilities=("chat",)),
    )
    record_latency("capable", 2000.0)
    record_latency("chat-only", 10.0)

    assert resolve_model(cost_request(task="coding", routing_policy=policy)).name == "capable"


@pytest.mark.parametrize("policy", POLICIES)
def test_policies_ignore_disabled_model(registry, policy):
    registry(
        model("enabled", input_cost=0.01),
        model("disabled", provider="openai", input_cost=0.0, output_cost=0.0, enabled=False),
    )
    record_latency("enabled", 2000.0)
    record_latency("disabled", 10.0)

    assert resolve_model(cost_request(routing_policy=policy)).name == "enabled"


@pytest.mark.parametrize("policy", POLICIES)
def test_budget_filter_applies_to_every_policy(three_models, policy):
    # only "cheap" (0.2 USD) fits, however fast the others are
    assert resolve_model(cost_request(max_cost_usd=0.2, routing_policy=policy)).name == "cheap"

    with pytest.raises(HTTPException) as exc_info:
        resolve_model(cost_request(max_cost_usd=0.19, routing_policy=policy))

    assert exc_info.value.status_code == 400


# --- Health-aware azir-auto selection ---


def make_unhealthy(name: str) -> None:
    for _ in range(MIN_HEALTH_SAMPLES):
        record_failure(name)
    assert not is_healthy(name)


@pytest.mark.parametrize("explicit", [ANTHROPIC_MODEL, OPENAI_MODEL])
def test_explicit_model_is_used_even_when_unhealthy(explicit):
    make_unhealthy(explicit)
    request = make_request(explicit, "chat")

    assert resolve_model(request).name == explicit
    assert plan_attempts(request)[0].name == explicit


def test_auto_excludes_unhealthy_model():
    # cold start: balanced picks the cheaper gpt-4o-mini...
    assert resolve_model(make_request("azir-auto", "chat")).name == OPENAI_MODEL

    # ...until it is unhealthy
    make_unhealthy(OPENAI_MODEL)
    assert resolve_model(make_request("azir-auto", "chat")).name == ANTHROPIC_MODEL


def test_auto_keeps_model_with_insufficient_history():
    for _ in range(MIN_HEALTH_SAMPLES - 1):
        record_failure(OPENAI_MODEL)

    assert resolve_model(make_request("azir-auto", "chat")).name == OPENAI_MODEL


@pytest.mark.parametrize("policy", POLICIES)
def test_auto_with_every_candidate_unhealthy_uses_full_eligible_set(policy):
    request = make_request("azir-auto", "chat", routing_policy=policy)
    expected = resolve_model(request).name

    make_unhealthy(ANTHROPIC_MODEL)
    make_unhealthy(OPENAI_MODEL)

    # no 400: same choice as if there were no health state at all
    assert resolve_model(request).name == expected


def test_auto_only_capable_model_unhealthy_is_still_routed():
    make_unhealthy(ANTHROPIC_MODEL)

    assert resolve_model(make_request("azir-auto", "coding")).name == ANTHROPIC_MODEL


def test_cheap_picks_cheapest_healthy_model(three_models):
    make_unhealthy("cheap")

    assert resolve_model(cost_request(routing_policy="cheap")).name == "middle"


def test_fast_picks_fastest_healthy_model(three_models):
    make_unhealthy("quick")

    assert resolve_model(cost_request(routing_policy="fast")).name == "middle"


def test_balanced_scores_only_healthy_candidates(registry):
    # "dominant" is cheapest and fastest (score 0) until it is unhealthy;
    # the rest are then normalized among themselves, as in `three_models`
    registry(
        model("cheap", provider="openai", input_cost=0.0, output_cost=1.0),
        model("middle", input_cost=0.0, output_cost=2.0),
        model("quick", input_cost=0.0, output_cost=3.0),
        model("dominant", input_cost=0.0, output_cost=0.5),
    )
    for name, latency_ms in [("cheap", 900.0), ("middle", 400.0), ("quick", 300.0), ("dominant", 100.0)]:
        record_latency(name, latency_ms)

    request = cost_request(routing_policy="balanced")
    assert resolve_model(request).name == "dominant"

    make_unhealthy("dominant")
    assert resolve_model(request).name == "middle"


# --- Fallback planning ---


def test_plan_explicit_model_then_one_fallback_per_other_provider():
    assert [c.name for c in plan_attempts(make_request(ANTHROPIC_MODEL))] == [ANTHROPIC_MODEL, OPENAI_MODEL]
    assert [c.name for c in plan_attempts(make_request(OPENAI_MODEL))] == [OPENAI_MODEL, ANTHROPIC_MODEL]


def test_plan_fallbacks_must_support_task():
    # gpt-4o-mini has no "coding" capability, so there is nothing to fall back to
    assert [c.name for c in plan_attempts(make_request("azir-auto", "coding"))] == [ANTHROPIC_MODEL]


def test_plan_skips_disabled_fallback(disable_model):
    disable_model(OPENAI_MODEL)

    assert [c.name for c in plan_attempts(make_request(ANTHROPIC_MODEL))] == [ANTHROPIC_MODEL]


# --- Non-streaming routing ---


@pytest.mark.anyio
async def test_route_request_uses_primary_provider_on_success():
    anthropic = StubProvider(result=make_response(ANTHROPIC_MODEL))
    openai = StubProvider()
    app = make_app(anthropic, openai)

    response = await route_request(app, make_request(ANTHROPIC_MODEL))

    assert response.model == ANTHROPIC_MODEL
    assert len(anthropic.requests) == 1
    assert not openai.requests


@pytest.mark.anyio
async def test_route_request_sends_concrete_model_for_azir_auto():
    openai = StubProvider(result=make_response(OPENAI_MODEL))
    app = make_app(StubProvider(), openai)

    await route_request(app, make_request("azir-auto", "classification"))

    assert openai.requests[0].model == OPENAI_MODEL


@pytest.mark.anyio
@pytest.mark.parametrize(
    "error",
    [
        upstream_error(429, 429),
        upstream_error(503, 502),
        HTTPException(status_code=504, detail="Timed out."),
        HTTPException(status_code=502, detail="Could not reach Anthropic."),
    ],
)
async def test_route_request_falls_back_on_transient_failure(error):
    anthropic = StubProvider(error=error)
    openai = StubProvider(result=make_response(OPENAI_MODEL))
    app = make_app(anthropic, openai)

    response = await route_request(app, make_request(ANTHROPIC_MODEL))

    assert response.model == OPENAI_MODEL
    assert len(anthropic.requests) == 1
    assert openai.requests[0].model == OPENAI_MODEL


@pytest.mark.anyio
async def test_route_request_falls_back_to_anthropic_when_openai_fails():
    anthropic = StubProvider(result=make_response(ANTHROPIC_MODEL))
    openai = StubProvider(error=upstream_error(429, 429))
    app = make_app(anthropic, openai)

    response = await route_request(app, make_request(OPENAI_MODEL))

    assert response.model == ANTHROPIC_MODEL
    assert anthropic.requests[0].model == ANTHROPIC_MODEL


@pytest.mark.anyio
async def test_route_request_azir_auto_falls_back_from_cost_selected_primary():
    anthropic = StubProvider(result=make_response(ANTHROPIC_MODEL))
    openai = StubProvider(error=upstream_error(503, 502))
    app = make_app(anthropic, openai)

    await route_request(app, make_request("azir-auto", "chat"))

    # cheapest chat model (gpt-4o-mini) is tried first, then the fallback
    assert openai.requests[0].model == OPENAI_MODEL
    assert anthropic.requests[0].model == ANTHROPIC_MODEL


@pytest.mark.anyio
@pytest.mark.parametrize(
    "error",
    [
        upstream_error(400, 400),
        upstream_error(401, 401),
        upstream_error(403, 403),
        upstream_error(404, 404),
        # unrecognized upstream 4xx is surfaced as 502 but is still a request error
        upstream_error(422, 502),
    ],
)
async def test_route_request_does_not_fall_back_on_non_recoverable_error(error):
    anthropic = StubProvider(error=error)
    openai = StubProvider(result=make_response(OPENAI_MODEL))
    app = make_app(anthropic, openai)

    with pytest.raises(HTTPException) as exc_info:
        await route_request(app, make_request(ANTHROPIC_MODEL))

    assert exc_info.value is error
    assert not openai.requests


@pytest.mark.anyio
async def test_route_request_raises_last_error_when_all_providers_fail():
    anthropic = StubProvider(error=upstream_error(503, 502))
    openai = StubProvider(error=upstream_error(429, 429))
    app = make_app(anthropic, openai)

    with pytest.raises(HTTPException) as exc_info:
        await route_request(app, make_request(ANTHROPIC_MODEL))

    assert exc_info.value.status_code == 429
    assert len(anthropic.requests) == 1
    assert len(openai.requests) == 1


@pytest.mark.anyio
@pytest.mark.parametrize(
    "request_",
    [make_request("mystery-model"), make_request("azir-auto"), make_request("azir-auto", "image-generation")],
)
async def test_route_request_rejects_bad_routing_without_calling_providers(request_):
    anthropic, openai = StubProvider(), StubProvider()
    app = make_app(anthropic, openai)

    with pytest.raises(HTTPException) as exc_info:
        await route_request(app, request_)

    assert exc_info.value.status_code == 400
    assert not anthropic.requests
    assert not openai.requests


@pytest.mark.anyio
async def test_route_request_emits_telemetry_for_each_attempt(caplog):
    # the provider may report a more specific model name than was requested
    anthropic = StubProvider(result=make_response("claude-sonnet-4-6-20260101"))
    openai = StubProvider(error=upstream_error(503, 502))
    app = make_app(anthropic, openai)

    with caplog.at_level(logging.INFO, logger="azir.telemetry"):
        await route_request(app, make_request("azir-auto", "chat"))

    records = [json.loads(r.message) for r in caplog.records]
    assert len(records) == 2

    assert records[0]["provider"] == "openai"
    assert records[0]["model"] == OPENAI_MODEL
    assert records[0]["status"] == "error"
    assert records[0]["status_code"] == 502

    # telemetry records the concrete model Azir attempted, never "azir-auto"
    assert records[1]["provider"] == "anthropic"
    assert records[1]["model"] == ANTHROPIC_MODEL
    assert records[1]["status"] == "success"
    assert records[1]["status_code"] == 200
    assert records[1]["total_tokens"] == 2
    # cost comes from actual returned usage (1 + 1 tokens), not the routing estimate
    assert records[1]["estimated_cost_usd"] == pytest.approx(0.001 * 0.003 + 0.001 * 0.015)


# --- Telemetry persistence ---


def store_with_pool(**pool_kwargs):
    from tests.test_telemetry_store import FakePool
    from telemetry_store import TelemetryStore

    pool = FakePool(**pool_kwargs)
    return TelemetryStore(pool), pool


@pytest.mark.anyio
async def test_route_request_persists_success_row_for_concrete_model():
    store, pool = store_with_pool()
    app = make_app(openai=StubProvider(result=make_response(OPENAI_MODEL)), telemetry_store=store)

    await route_request(app, make_request("azir-auto", "chat"))

    [row] = pool.inserts
    provider, model_name, status, status_code, latency_ms, *usage_and_cost = row
    # concrete provider/model, never "azir-auto"
    assert (provider, model_name, status, status_code) == ("openai", OPENAI_MODEL, "success", 200)
    assert latency_ms >= 0
    assert usage_and_cost == [1, 1, 2, pytest.approx(0.001 * 0.00015 + 0.001 * 0.0006), "user"]


@pytest.mark.anyio
async def test_route_request_fallback_persists_one_row_per_attempt():
    store, pool = store_with_pool()
    anthropic = StubProvider(error=upstream_error(503, 502))
    openai = StubProvider(result=make_response(OPENAI_MODEL))

    await route_request(make_app(anthropic, openai, store), make_request(ANTHROPIC_MODEL))

    assert [row[:4] for row in pool.inserts] == [
        ("anthropic", ANTHROPIC_MODEL, "error", 502),
        ("openai", OPENAI_MODEL, "success", 200),
    ]
    # failed attempt: no usage, no cost
    assert pool.inserts[0][5:9] == (None, None, None, None)


@pytest.mark.anyio
async def test_route_request_persists_non_transient_failure_before_raising():
    store, pool = store_with_pool()
    error = upstream_error(401, 401)

    with pytest.raises(HTTPException) as exc_info:
        await route_request(make_app(StubProvider(error=error), telemetry_store=store), make_request(ANTHROPIC_MODEL))

    assert exc_info.value is error
    assert [row[:4] for row in pool.inserts] == [("anthropic", ANTHROPIC_MODEL, "error", 401)]


@pytest.mark.anyio
@pytest.mark.parametrize("db_error", [ConnectionRefusedError(), OSError("db down"), RuntimeError("boom")])
async def test_database_failure_does_not_break_successful_response(caplog, db_error):
    store, pool = store_with_pool(error=db_error)
    app = make_app(openai=StubProvider(result=make_response(OPENAI_MODEL)), telemetry_store=store)

    with caplog.at_level(logging.INFO, logger="azir.telemetry"):
        response = await route_request(app, make_request(OPENAI_MODEL))

    assert response.model == OPENAI_MODEL
    assert pool.inserts == []
    assert any("Failed to persist telemetry" in r.message for r in caplog.records)
    # routing state is still updated
    assert get_sample_count(OPENAI_MODEL) == 1
    assert get_latency_stats(OPENAI_MODEL).samples == 1


@pytest.mark.anyio
async def test_database_failure_does_not_mask_provider_error_or_stop_fallback():
    store, _pool = store_with_pool(error=ConnectionRefusedError())
    anthropic = StubProvider(error=upstream_error(503, 502))
    openai = StubProvider(result=make_response(OPENAI_MODEL))

    response = await route_request(make_app(anthropic, openai, store), make_request(ANTHROPIC_MODEL))

    assert response.model == OPENAI_MODEL


@pytest.mark.anyio
async def test_slow_database_is_bounded_and_does_not_break_response(monkeypatch):
    import telemetry_store

    monkeypatch.setattr(telemetry_store, "WRITE_TIMEOUT_SECONDS", 0.01)
    store, pool = store_with_pool(delay=1.0)
    app = make_app(openai=StubProvider(result=make_response(OPENAI_MODEL)), telemetry_store=store)

    response = await route_request(app, make_request(OPENAI_MODEL))

    assert response.model == OPENAI_MODEL
    assert pool.inserts == []


@pytest.mark.anyio
@pytest.mark.parametrize("request_", [make_request("mystery-model"), make_request("azir-auto")])
async def test_pre_provider_errors_persist_nothing(request_):
    store, pool = store_with_pool()

    with pytest.raises(HTTPException):
        await route_request(make_app(telemetry_store=store), request_)

    assert pool.executed == []


@pytest.mark.anyio
async def test_streaming_persists_nothing():
    store, pool = store_with_pool()
    openai = StubProvider(stream_result=object())

    await stream_chat_completion(make_app(openai=openai, telemetry_store=store), make_request(OPENAI_MODEL))

    assert pool.executed == []


# --- Latency recording ---


def fake_clock(monkeypatch, *seconds):
    """Make router's perf_counter() return the given values in order (two
    calls per provider attempt: start, then end)."""
    ticks = iter(seconds)
    monkeypatch.setattr(router, "time", SimpleNamespace(perf_counter=lambda: next(ticks)))


def timeout_error() -> HTTPException:
    exc = HTTPException(status_code=504, detail="Timed out.")
    exc.__cause__ = httpx.ReadTimeout("timed out")
    return exc


def connect_error() -> HTTPException:
    exc = HTTPException(status_code=502, detail="Could not reach provider.")
    exc.__cause__ = httpx.ConnectError("connection refused")
    return exc


@pytest.mark.anyio
async def test_route_request_records_latency_after_success(monkeypatch, caplog):
    fake_clock(monkeypatch, 10.0, 10.25)
    app = make_app(openai=StubProvider(result=make_response(OPENAI_MODEL)))

    with caplog.at_level(logging.INFO, logger="azir.telemetry"):
        await route_request(app, make_request("azir-auto", "chat"))

    # recorded under the concrete model, never "azir-auto"
    stats = get_latency_stats(OPENAI_MODEL)
    assert stats.samples == 1
    assert stats.estimate_ms == pytest.approx(250.0)
    assert get_latency_stats("azir-auto") is None
    # telemetry reports the same measurement
    assert json.loads(caplog.records[0].message)["latency_ms"] == pytest.approx(250.0)


@pytest.mark.anyio
async def test_route_request_records_latency_after_timeout(monkeypatch):
    fake_clock(monkeypatch, 0.0, 30.0, 30.0, 30.1)
    anthropic = StubProvider(error=timeout_error())
    openai = StubProvider(result=make_response(OPENAI_MODEL))

    await route_request(make_app(anthropic, openai), make_request(ANTHROPIC_MODEL))

    # the timed-out attempt is a (lower-bound) latency sample, and fallback still ran
    assert get_latency_stats(ANTHROPIC_MODEL).estimate_ms == pytest.approx(30_000.0)
    assert get_latency_stats(OPENAI_MODEL).estimate_ms == pytest.approx(100.0)


@pytest.mark.anyio
@pytest.mark.parametrize("error", [upstream_error(429, 429), upstream_error(503, 502), connect_error()])
async def test_route_request_does_not_record_fast_failures(monkeypatch, error):
    fake_clock(monkeypatch, 0.0, 0.01, 0.01, 0.5)
    anthropic = StubProvider(error=error)
    openai = StubProvider(result=make_response(OPENAI_MODEL))

    await route_request(make_app(anthropic, openai), make_request(ANTHROPIC_MODEL))

    # a fast 429/5xx or a refused connection must not make the model look fast
    assert get_latency_stats(ANTHROPIC_MODEL) is None
    assert get_latency_stats(OPENAI_MODEL).samples == 1


@pytest.mark.anyio
async def test_route_request_records_nothing_when_no_provider_is_called():
    with pytest.raises(HTTPException):
        await route_request(make_app(), make_request("azir-auto", "image-generation"))

    assert get_latency_stats(ANTHROPIC_MODEL) is None
    assert get_latency_stats(OPENAI_MODEL) is None


# claude-sonnet-4-6 observed at 200 ms, gpt-4o-mini at 900 ms; gpt-4o-mini is cheaper.
# balanced: one model is cheaper, the other faster -> 0.5 each -> registry order (claude).
POLICY_PRIMARY = [
    ("cheap", OPENAI_MODEL, ANTHROPIC_MODEL),
    ("fast", ANTHROPIC_MODEL, OPENAI_MODEL),
    ("balanced", ANTHROPIC_MODEL, OPENAI_MODEL),
]


@pytest.mark.anyio
@pytest.mark.parametrize("policy, primary, fallback", POLICY_PRIMARY)
async def test_route_request_falls_back_from_policy_selected_primary(policy, primary, fallback):
    record_latency(ANTHROPIC_MODEL, 200.0)
    record_latency(OPENAI_MODEL, 900.0)
    providers = {
        ANTHROPIC_MODEL: StubProvider(result=make_response(ANTHROPIC_MODEL)),
        OPENAI_MODEL: StubProvider(result=make_response(OPENAI_MODEL)),
    }
    providers[primary].error = upstream_error(503, 502)
    request = make_request("azir-auto", "chat", routing_policy=policy)

    assert [c.name for c in plan_attempts(request)] == [primary, fallback]

    response = await route_request(make_app(providers[ANTHROPIC_MODEL], providers[OPENAI_MODEL]), request)

    assert providers[primary].requests[0].model == primary
    assert response.model == fallback


@pytest.mark.anyio
@pytest.mark.parametrize("policy, primary, _fallback", POLICY_PRIMARY)
async def test_route_request_telemetry_records_policy_selected_concrete_model(caplog, policy, primary, _fallback):
    record_latency(ANTHROPIC_MODEL, 200.0)
    record_latency(OPENAI_MODEL, 900.0)
    app = make_app(
        StubProvider(result=make_response(ANTHROPIC_MODEL)),
        StubProvider(result=make_response(OPENAI_MODEL)),
    )

    with caplog.at_level(logging.INFO, logger="azir.telemetry"):
        await route_request(app, make_request("azir-auto", "chat", routing_policy=policy))

    [record] = [json.loads(r.message) for r in caplog.records]
    assert record["model"] == primary
    assert record["provider"] == MODEL_REGISTRY[primary].provider
    assert record["status"] == "success"


# --- Health recording ---


@pytest.mark.anyio
async def test_route_request_records_success():
    app = make_app(openai=StubProvider(result=make_response(OPENAI_MODEL)))

    await route_request(app, make_request("azir-auto", "chat"))

    assert (get_sample_count(OPENAI_MODEL), get_success_rate(OPENAI_MODEL)) == (1, 1.0)
    assert get_sample_count(ANTHROPIC_MODEL) == 0
    assert get_sample_count("azir-auto") == 0


@pytest.mark.anyio
@pytest.mark.parametrize(
    "error", [upstream_error(429, 429), upstream_error(503, 502), timeout_error(), connect_error()]
)
async def test_route_request_records_transient_failure_and_fallback_success_independently(error):
    anthropic = StubProvider(error=error)
    openai = StubProvider(result=make_response(OPENAI_MODEL))

    await route_request(make_app(anthropic, openai), make_request(ANTHROPIC_MODEL))

    assert (get_sample_count(ANTHROPIC_MODEL), get_success_rate(ANTHROPIC_MODEL)) == (1, 0.0)
    assert (get_sample_count(OPENAI_MODEL), get_success_rate(OPENAI_MODEL)) == (1, 1.0)


@pytest.mark.anyio
async def test_route_request_records_failure_for_every_failed_attempt():
    anthropic = StubProvider(error=upstream_error(503, 502))
    openai = StubProvider(error=upstream_error(429, 429))

    with pytest.raises(HTTPException):
        await route_request(make_app(anthropic, openai), make_request(ANTHROPIC_MODEL))

    assert get_success_rate(ANTHROPIC_MODEL) == 0.0
    assert get_success_rate(OPENAI_MODEL) == 0.0


@pytest.mark.anyio
@pytest.mark.parametrize(
    "error",
    [
        upstream_error(400, 400),
        upstream_error(401, 401),
        upstream_error(403, 403),
        upstream_error(404, 404),
        upstream_error(422, 502),
    ],
)
async def test_route_request_does_not_record_non_transient_failures(error):
    anthropic = StubProvider(error=error)

    with pytest.raises(HTTPException):
        await route_request(make_app(anthropic, StubProvider()), make_request(ANTHROPIC_MODEL))

    assert get_sample_count(ANTHROPIC_MODEL) == 0


@pytest.mark.anyio
@pytest.mark.parametrize(
    "request_",
    [
        make_request("mystery-model"),
        make_request("azir-auto"),
        make_request("azir-auto", "image-generation"),
        make_request("azir-auto", "coding", max_cost_usd=0.0001),
    ],
)
async def test_route_request_pre_provider_errors_do_not_affect_health(request_):
    with pytest.raises(HTTPException):
        await route_request(make_app(), request_)

    assert get_sample_count(ANTHROPIC_MODEL) == 0
    assert get_sample_count(OPENAI_MODEL) == 0


@pytest.mark.anyio
async def test_repeated_failures_move_azir_auto_off_the_failing_model():
    # gpt-4o-mini is the cheap choice but keeps returning 503; each request
    # falls back to claude. After MIN_HEALTH_SAMPLES failures gpt-4o-mini
    # is unhealthy and claude becomes the primary. (`cheap` keeps latency
    # samples from claude's fallbacks out of the picture.)
    anthropic = StubProvider(result=make_response(ANTHROPIC_MODEL))
    openai = StubProvider(error=upstream_error(503, 502))
    app = make_app(anthropic, openai)
    request = make_request("azir-auto", "chat", routing_policy="cheap")

    for _ in range(MIN_HEALTH_SAMPLES + 2):
        await route_request(app, request)

    assert len(openai.requests) == MIN_HEALTH_SAMPLES
    assert len(anthropic.requests) == MIN_HEALTH_SAMPLES + 2
    assert not is_healthy(OPENAI_MODEL)
    assert plan_attempts(request)[0].name == ANTHROPIC_MODEL


@pytest.mark.anyio
async def test_observed_latency_steers_later_fast_requests(monkeypatch):
    # request 1: cold start -> both at the default -> registry order (claude), observed at 3000 ms
    # request 2: gpt-4o-mini is untried (1000 ms default) and now looks faster, observed at 500 ms
    # request 3: gpt-4o-mini at 500 ms vs claude at 3000 ms -> gpt-4o-mini again
    fake_clock(monkeypatch, 0.0, 3.0, 0.0, 0.5, 0.0, 0.5)
    anthropic = StubProvider(result=make_response(ANTHROPIC_MODEL))
    openai = StubProvider(result=make_response(OPENAI_MODEL))
    app = make_app(anthropic, openai)
    request = make_request("azir-auto", "chat", routing_policy="fast")

    chosen = [(await route_request(app, request)).model for _ in range(3)]

    assert chosen == [ANTHROPIC_MODEL, OPENAI_MODEL, OPENAI_MODEL]


# --- Streaming routing ---


@pytest.mark.anyio
@pytest.mark.parametrize("model, provider_name", [(ANTHROPIC_MODEL, "anthropic"), (OPENAI_MODEL, "openai")])
async def test_stream_chat_completion_uses_provider_for_explicit_model(model, provider_name):
    sentinel = object()
    providers = {"anthropic": StubProvider(), "openai": StubProvider()}
    providers[provider_name].stream_result = sentinel
    app = make_app(providers["anthropic"], providers["openai"])

    result = await stream_chat_completion(app, make_request(model))

    assert result is sentinel
    assert providers[provider_name].stream_requests[0].model == model


@pytest.mark.anyio
async def test_stream_chat_completion_resolves_azir_auto_before_streaming():
    openai = StubProvider(stream_result=object())
    app = make_app(StubProvider(), openai)

    await stream_chat_completion(app, make_request("azir-auto", "classification"))

    assert openai.stream_requests[0].model == OPENAI_MODEL


@pytest.mark.anyio
async def test_stream_chat_completion_uses_cost_selected_model():
    anthropic = StubProvider(stream_result=object())
    openai = StubProvider(stream_result=object())
    app = make_app(anthropic, openai)
    request = make_request("azir-auto", "chat")

    await stream_chat_completion(app, request)

    # same concrete model the non-streaming path would pick, chosen before streaming
    assert openai.stream_requests[0].model == resolve_model(request).name == OPENAI_MODEL
    assert not anthropic.stream_requests


@pytest.mark.anyio
@pytest.mark.parametrize("policy, primary, fallback", POLICY_PRIMARY)
async def test_stream_chat_completion_uses_policy_selected_model(policy, primary, fallback):
    record_latency(ANTHROPIC_MODEL, 200.0)
    record_latency(OPENAI_MODEL, 900.0)
    providers = {
        ANTHROPIC_MODEL: StubProvider(stream_result=object()),
        OPENAI_MODEL: StubProvider(stream_result=object()),
    }
    request = make_request("azir-auto", "chat", routing_policy=policy, stream=True)

    await stream_chat_completion(make_app(providers[ANTHROPIC_MODEL], providers[OPENAI_MODEL]), request)

    assert providers[primary].stream_requests[0].model == resolve_model(request).name == primary
    assert not providers[fallback].stream_requests
    # stream duration is never recorded
    assert get_latency_stats(primary).samples == 1


@pytest.mark.anyio
async def test_stream_chat_completion_records_success_once_stream_opens():
    openai = StubProvider(stream_result=object())

    await stream_chat_completion(make_app(StubProvider(), openai), make_request(OPENAI_MODEL))

    assert (get_sample_count(OPENAI_MODEL), get_success_rate(OPENAI_MODEL)) == (1, 1.0)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "error, recorded",
    [(upstream_error(503, 502), 1), (timeout_error(), 1), (upstream_error(401, 401), 0)],
)
async def test_stream_chat_completion_records_only_transient_pre_stream_failures(error, recorded):
    openai = StubProvider(stream_error=error)

    with pytest.raises(HTTPException):
        await stream_chat_completion(make_app(StubProvider(), openai), make_request(OPENAI_MODEL))

    assert get_sample_count(OPENAI_MODEL) == recorded
    assert get_sample_count(ANTHROPIC_MODEL) == 0


@pytest.mark.anyio
async def test_stream_chat_completion_skips_unhealthy_model():
    make_unhealthy(OPENAI_MODEL)
    anthropic = StubProvider(stream_result=object())
    openai = StubProvider(stream_result=object())

    await stream_chat_completion(make_app(anthropic, openai), make_request("azir-auto", "chat", stream=True))

    assert anthropic.stream_requests[0].model == ANTHROPIC_MODEL
    assert not openai.stream_requests


@pytest.mark.anyio
async def test_stream_chat_completion_rejects_auto_over_budget():
    anthropic = StubProvider(stream_result=object())
    app = make_app(anthropic, StubProvider())
    request = ChatRequest(
        model="azir-auto", task="coding", max_cost_usd=0.0001, stream=True,
        messages=[Message(role="user", content="hi")],
    )

    with pytest.raises(HTTPException) as exc_info:
        await stream_chat_completion(app, request)

    assert exc_info.value.status_code == 400
    assert not anthropic.stream_requests


@pytest.mark.anyio
async def test_stream_chat_completion_propagates_pre_stream_provider_error():
    openai = StubProvider(stream_error=upstream_error(401, 401))
    app = make_app(StubProvider(), openai)

    with pytest.raises(HTTPException) as exc_info:
        await stream_chat_completion(app, make_request(OPENAI_MODEL))

    assert exc_info.value.status_code == 401


@pytest.mark.anyio
async def test_stream_chat_completion_does_not_fall_back_on_failure():
    anthropic = StubProvider(stream_error=upstream_error(503, 502))
    openai = StubProvider(stream_result=object())
    app = make_app(anthropic, openai)

    with pytest.raises(HTTPException) as exc_info:
        await stream_chat_completion(app, make_request(ANTHROPIC_MODEL))

    assert exc_info.value.status_code == 502
    assert not openai.stream_requests


@pytest.mark.anyio
@pytest.mark.parametrize("request_", [make_request("mystery-model"), make_request("azir-auto")])
async def test_stream_chat_completion_rejects_bad_routing(request_):
    anthropic, openai = StubProvider(), StubProvider()
    app = make_app(anthropic, openai)

    with pytest.raises(HTTPException) as exc_info:
        await stream_chat_completion(app, request_)

    assert exc_info.value.status_code == 400
    assert not anthropic.stream_requests
    assert not openai.stream_requests


@pytest.mark.anyio
async def test_stream_chat_completion_rejects_disabled_model(disable_model):
    disable_model(OPENAI_MODEL)
    openai = StubProvider(stream_result=object())
    app = make_app(StubProvider(), openai)

    with pytest.raises(HTTPException) as exc_info:
        await stream_chat_completion(app, make_request(OPENAI_MODEL))

    assert exc_info.value.status_code == 400
    assert not openai.stream_requests
