import logging

import pytest
from fastapi import BackgroundTasks, HTTPException

import quality
import router
from config import settings
from latency import record_latency
from quality import DEFAULT_QUALITY_SCORE, MIN_QUALITY_SAMPLES, load_quality_estimates, quality_from_history
from router import (
    BALANCED_COST_WEIGHT,
    BALANCED_LATENCY_WEIGHT,
    BALANCED_QUALITY_WEIGHT,
    load_routing_quality,
    plan_attempts,
    resolve_model,
    route_request,
    stream_chat_completion,
)
from telemetry_store import QUALITY_HISTORY_SQL, TelemetryStore
from tests.test_analytics import SqlitePool
from tests.test_judge import SequenceProvider, judge_reply
from tests.test_router import (  # noqa: F401  (registry is a fixture)
    ANTHROPIC_MODEL,
    OPENAI_MODEL,
    StubProvider,
    cost_request,
    make_app,
    make_request,
    make_response,
    make_unhealthy,
    model,
    registry,
    upstream_error,
)


def add_evaluations(pool: SqlitePool, model_name: str, scores, task=None, provider="anthropic"):
    for score in scores:
        pool.db.execute(
            "INSERT INTO response_evaluations (provider, model, task, judge_provider, judge_model, score, reason) "
            "VALUES (?, ?, ?, 'openai', 'gpt-4o-mini', ?, 'r')",
            (provider, model_name, task, score),
        )


def store_with(pool: SqlitePool) -> TelemetryStore:
    return TelemetryStore(pool)


# --- Quality lookup ---


@pytest.mark.anyio
async def test_lookup_uses_task_specific_average_with_enough_samples():
    pool = SqlitePool()
    add_evaluations(pool, "m", [0.9] * MIN_QUALITY_SAMPLES, task="coding")
    add_evaluations(pool, "m", [0.1] * MIN_QUALITY_SAMPLES, task="chat")

    estimates = await load_quality_estimates(store_with(pool), ["m"], "coding")

    assert estimates == {"m": pytest.approx(0.9)}  # not the 0.5 overall average


@pytest.mark.anyio
async def test_lookup_falls_back_to_overall_average_when_task_history_is_thin():
    pool = SqlitePool()
    add_evaluations(pool, "m", [0.9] * (MIN_QUALITY_SAMPLES - 1), task="coding")
    add_evaluations(pool, "m", [0.3, 0.3], task="chat")
    add_evaluations(pool, "m", [0.6])  # legacy row without a task still counts

    estimates = await load_quality_estimates(store_with(pool), ["m"], "coding")

    assert estimates["m"] == pytest.approx((0.9 * 4 + 0.3 * 2 + 0.6) / 7)


@pytest.mark.anyio
async def test_lookup_without_task_uses_overall_average():
    pool = SqlitePool()
    add_evaluations(pool, "m", [0.8] * MIN_QUALITY_SAMPLES, task="coding")

    assert await load_quality_estimates(store_with(pool), ["m"], None) == {"m": pytest.approx(0.8)}


@pytest.mark.anyio
async def test_lookup_uses_neutral_default_without_enough_history():
    pool = SqlitePool()
    add_evaluations(pool, "thin", [1.0] * (MIN_QUALITY_SAMPLES - 1), task="coding")

    estimates = await load_quality_estimates(store_with(pool), ["thin", "never-judged"], "coding")

    assert estimates == {"thin": DEFAULT_QUALITY_SCORE, "never-judged": DEFAULT_QUALITY_SCORE}
    assert DEFAULT_QUALITY_SCORE == 0.5


@pytest.mark.anyio
async def test_lookup_is_one_query_for_the_whole_candidate_set():
    pool = SqlitePool()
    add_evaluations(pool, "a", [0.7] * 5)
    add_evaluations(pool, "b", [0.2] * 5)
    add_evaluations(pool, "not-a-candidate", [1.0] * 5)

    estimates = await load_quality_estimates(store_with(pool), ["a", "b"], None)

    assert estimates == {"a": pytest.approx(0.7), "b": pytest.approx(0.2)}
    assert pool.queries == [QUALITY_HISTORY_SQL]


@pytest.mark.anyio
async def test_lookup_without_store_is_neutral_and_queries_nothing():
    assert await load_quality_estimates(None, ["a"], "chat") == {"a": DEFAULT_QUALITY_SCORE}


@pytest.mark.anyio
async def test_lookup_database_failure_returns_neutral_estimates(caplog):
    pool = SqlitePool(error=ConnectionRefusedError())

    with caplog.at_level(logging.WARNING, logger="azir.quality"):
        estimates = await load_quality_estimates(store_with(pool), ["a", "b"], "chat")

    assert estimates == {"a": DEFAULT_QUALITY_SCORE, "b": DEFAULT_QUALITY_SCORE}
    assert "Quality history lookup failed" in caplog.text


@pytest.mark.anyio
async def test_lookup_timeout_returns_neutral_estimates(monkeypatch):
    monkeypatch.setattr(quality, "QUALITY_LOOKUP_TIMEOUT_SECONDS", 0.01)
    pool = SqlitePool(delay=1.0)

    assert await load_quality_estimates(store_with(pool), ["a"], None) == {"a": DEFAULT_QUALITY_SCORE}


def test_quality_from_history_hierarchy():
    enough = MIN_QUALITY_SAMPLES
    row = {"overall_count": enough, "overall_average": 0.7, "task_count": enough, "task_average": 0.9}

    assert quality_from_history(row, "coding") == 0.9
    assert quality_from_history({**row, "task_count": enough - 1}, "coding") == 0.7
    assert quality_from_history(row, None) == 0.7
    assert quality_from_history({**row, "overall_count": enough - 1, "task_count": 0}, "coding") == 0.5
    assert quality_from_history(None, "coding") == 0.5


# --- Worked example: four candidates, each policy picks a different one ---


@pytest.fixture
def four_models(registry):
    """"hi" + 200 default output tokens, free input: cost = 0.2 * output_cost.

        model       cost  latency  quality   norm cost  norm latency  penalty  balanced
        budget      0.2   900 ms   0.40      0.00       0.875         0.60     0.4928
        speedy      0.8   200 ms   0.50      0.75       0.000         0.50     0.4175
        premium     1.0   1000 ms  0.95      1.00       1.000         0.05     0.6770
        allrounder  0.4   400 ms   0.80      0.25       0.250         0.20     0.2330
    """
    registry(
        model("budget", provider="openai", input_cost=0.0, output_cost=1.0),
        model("speedy", input_cost=0.0, output_cost=4.0),
        model("premium", input_cost=0.0, output_cost=5.0),
        model("allrounder", input_cost=0.0, output_cost=2.0),
    )
    for name, latency_ms in [("budget", 900.0), ("speedy", 200.0), ("premium", 1000.0), ("allrounder", 400.0)]:
        record_latency(name, latency_ms)
    return {"budget": 0.40, "speedy": 0.50, "premium": 0.95, "allrounder": 0.80}


@pytest.mark.parametrize(
    "policy, winner",
    [("cheap", "budget"), ("fast", "speedy"), ("quality", "premium"), ("balanced", "allrounder")],
)
def test_worked_example_each_policy_winner(four_models, policy, winner):
    assert resolve_model(cost_request(routing_policy=policy), four_models).name == winner


def test_balanced_weights_are_the_documented_v1_values():
    assert (BALANCED_COST_WEIGHT, BALANCED_LATENCY_WEIGHT, BALANCED_QUALITY_WEIGHT) == (0.33, 0.33, 0.34)
    assert BALANCED_COST_WEIGHT + BALANCED_LATENCY_WEIGHT + BALANCED_QUALITY_WEIGHT == pytest.approx(1.0)


# --- quality policy ---


def test_quality_cheaper_lower_quality_model_loses(registry):
    registry(
        model("cheap-meh", provider="openai", input_cost=0.0, output_cost=0.1),
        model("pricey-good", input_cost=0.0, output_cost=9.0),
    )

    assert resolve_model(cost_request(routing_policy="quality"), {"cheap-meh": 0.3, "pricey-good": 0.8}).name == "pricey-good"


def test_quality_faster_lower_quality_model_loses(registry):
    registry(model("fast-meh", provider="openai"), model("slow-good"))
    record_latency("fast-meh", 100.0)
    record_latency("slow-good", 5000.0)

    assert resolve_model(cost_request(routing_policy="quality"), {"fast-meh": 0.3, "slow-good": 0.8}).name == "slow-good"


def test_quality_ties_break_by_registry_order(registry):
    registry(model("first", provider="openai", output_cost=9.0), model("second", output_cost=0.1))

    assert resolve_model(cost_request(routing_policy="quality"), {"first": 0.7, "second": 0.7}).name == "first"
    # cold start: everything is neutral
    assert resolve_model(cost_request(routing_policy="quality")).name == "first"


def test_quality_ignores_incapable_model(registry):
    registry(model("capable"), model("chat-only", provider="openai", capabilities=("chat",)))

    assert resolve_model(cost_request(task="coding", routing_policy="quality"), {"capable": 0.2, "chat-only": 0.99}).name == "capable"


def test_quality_ignores_disabled_model(registry):
    registry(model("enabled"), model("disabled", provider="openai", enabled=False))

    assert resolve_model(cost_request(routing_policy="quality"), {"enabled": 0.2, "disabled": 0.99}).name == "enabled"


def test_quality_ignores_unhealthy_model(registry):
    registry(model("ok"), model("best-but-failing", provider="openai"))
    make_unhealthy("best-but-failing")

    assert resolve_model(cost_request(routing_policy="quality"), {"ok": 0.2, "best-but-failing": 0.99}).name == "ok"


def test_quality_respects_budget(registry):
    registry(
        model("affordable", provider="openai", input_cost=0.0, output_cost=1.0),
        model("too-pricey", input_cost=0.0, output_cost=5.0),
    )
    estimates = {"affordable": 0.2, "too-pricey": 0.99}

    assert resolve_model(cost_request(max_cost_usd=0.5, routing_policy="quality"), estimates).name == "affordable"

    with pytest.raises(HTTPException):
        resolve_model(cost_request(max_cost_usd=0.1, routing_policy="quality"), estimates)


# --- balanced ---


def test_balanced_cost_affects_ranking(registry):
    registry(model("pricey", output_cost=5.0), model("cheap", provider="openai", output_cost=1.0))

    assert resolve_model(cost_request(routing_policy="balanced"), {"pricey": 0.8, "cheap": 0.8}).name == "cheap"


def test_balanced_latency_affects_ranking(registry):
    registry(model("slow", provider="openai"), model("quick"))
    record_latency("slow", 900.0)
    record_latency("quick", 300.0)

    assert resolve_model(cost_request(routing_policy="balanced"), {"slow": 0.8, "quick": 0.8}).name == "quick"


def test_balanced_quality_affects_ranking_with_equal_cost_and_latency(registry):
    # equal cost and latency normalize to 0 (no division by zero): quality decides
    registry(model("worse", provider="openai"), model("better"))

    assert resolve_model(cost_request(routing_policy="balanced"), {"worse": 0.4, "better": 0.6}).name == "better"


def test_balanced_high_quality_expensive_model_can_beat_cheap_low_quality(registry):
    # norm cost 0 / 0.25 / 1, equal latency:
    #   cheap-bad   0.33 * 0    + 0.34 * 0.8  = 0.272
    #   pricey-good 0.33 * 0.25 + 0.34 * 0.1  = 0.1165   <- wins
    #   luxury      0.33 * 1    + 0.34 * 0.5  = 0.500
    registry(
        model("cheap-bad", provider="openai", input_cost=0.0, output_cost=1.0),
        model("pricey-good", input_cost=0.0, output_cost=2.0),
        model("luxury", input_cost=0.0, output_cost=5.0),
    )
    estimates = {"cheap-bad": 0.2, "pricey-good": 0.9, "luxury": 0.5}

    assert resolve_model(cost_request(routing_policy="balanced"), estimates).name == "pricey-good"


def test_balanced_quality_gap_must_outweigh_cost_gap(registry):
    # two candidates: the cost gap is the full 0.33, the quality gap only
    # 0.34 * 0.8 = 0.272, so the cheap model still wins
    registry(
        model("cheap-bad", provider="openai", input_cost=0.0, output_cost=1.0),
        model("pricey-good", input_cost=0.0, output_cost=2.0),
    )

    assert resolve_model(cost_request(routing_policy="balanced"), {"cheap-bad": 0.1, "pricey-good": 0.9}).name == "cheap-bad"


def test_balanced_cold_start_quality_counts_as_neutral(registry):
    registry(model("judged", provider="openai"), model("untried"))
    request = cost_request(routing_policy="balanced")

    # untried is at 0.5: beats a model judged below that, loses to one above
    assert resolve_model(request, {"judged": 0.4}).name == "untried"
    assert resolve_model(request, {"judged": 0.6}).name == "judged"


def test_balanced_complete_tie_is_deterministic_registry_order(registry):
    registry(model("first", provider="openai"), model("second"), model("third"))
    request = cost_request(routing_policy="balanced")

    assert [resolve_model(request, {"first": 0.7, "second": 0.7, "third": 0.7}).name for _ in range(3)] == ["first"] * 3


# --- Existing policies and explicit models ignore quality ---


def test_cheap_ignores_quality(four_models):
    assert resolve_model(cost_request(routing_policy="cheap"), {**four_models, "budget": 0.0}).name == "budget"


def test_fast_ignores_quality(four_models):
    assert resolve_model(cost_request(routing_policy="fast"), {**four_models, "speedy": 0.0}).name == "speedy"


@pytest.mark.parametrize("policy", [None, "quality", "balanced"])
def test_explicit_model_ignores_quality(policy):
    request = make_request(OPENAI_MODEL, "chat", routing_policy=policy)

    assert resolve_model(request, {ANTHROPIC_MODEL: 1.0, OPENAI_MODEL: 0.0}).name == OPENAI_MODEL


# --- When the lookup happens ---


@pytest.mark.anyio
@pytest.mark.parametrize(
    "request_",
    [
        make_request(ANTHROPIC_MODEL, "chat", routing_policy="quality"),
        make_request("azir-auto", "chat", routing_policy="cheap"),
        make_request("azir-auto", "chat", routing_policy="fast"),
    ],
)
async def test_no_quality_lookup_for_explicit_models_cheap_or_fast(request_):
    pool = SqlitePool()

    assert await load_routing_quality(make_app(telemetry_store=store_with(pool)), request_) is None
    assert pool.queries == []


@pytest.mark.anyio
@pytest.mark.parametrize("policy", [None, "balanced", "quality"])
async def test_quality_and_balanced_look_up_the_healthy_candidates_once(policy):
    pool = SqlitePool()
    make_unhealthy(ANTHROPIC_MODEL)  # with a healthy alternative, it is not a candidate

    estimates = await load_routing_quality(
        make_app(telemetry_store=store_with(pool)), make_request("azir-auto", "chat", routing_policy=policy)
    )

    assert estimates == {OPENAI_MODEL: DEFAULT_QUALITY_SCORE}
    assert len(pool.queries) == 1


# --- Integration ---


def quality_app(pool, anthropic=None, openai=None):
    return make_app(
        anthropic or StubProvider(result=make_response(ANTHROPIC_MODEL)),
        openai or StubProvider(result=make_response(OPENAI_MODEL)),
        store_with(pool),
    )


@pytest.mark.anyio
async def test_judge_scores_from_earlier_requests_steer_a_later_azir_auto_request(monkeypatch):
    monkeypatch.setattr(settings, "llm_judge_enabled", True)
    monkeypatch.setattr(settings, "llm_judge_model", OPENAI_MODEL)
    pool = SqlitePool()
    anthropic = StubProvider(result=make_response(ANTHROPIC_MODEL))
    weak = judge_reply('{"score": 0.2, "reason": "Mostly wrong."}')
    # openai serves the judge for the earlier requests, then the later user request
    openai = SequenceProvider(*[weak] * MIN_QUALITY_SAMPLES, make_response(OPENAI_MODEL))
    app = quality_app(pool, anthropic, openai)
    later = make_request("azir-auto", "chat", routing_policy="quality")

    # cold start: both chat models are neutral, registry order picks claude
    assert resolve_model(later, await load_routing_quality(app, later)).name == ANTHROPIC_MODEL

    # requests A: explicit claude, each judged (0.2) and persisted with task "chat"
    for _ in range(MIN_QUALITY_SAMPLES):
        tasks = BackgroundTasks()
        await route_request(app, make_request(ANTHROPIC_MODEL, "chat"), tasks)
        await tasks()

    assert tuple(
        pool.db.execute(
            "SELECT COUNT(*), AVG(score) FROM response_evaluations WHERE model = ? AND task = 'chat'", (ANTHROPIC_MODEL,)
        ).fetchone()
    ) == (MIN_QUALITY_SAMPLES, pytest.approx(0.2))

    # request B: claude's history (0.2) is now below gpt-4o-mini's neutral 0.5
    response = await route_request(app, later)

    assert response.model == OPENAI_MODEL
    assert openai.requests[-1].model == OPENAI_MODEL  # concrete model, never azir-auto
    assert len(anthropic.requests) == MIN_QUALITY_SAMPLES  # B never touched claude


@pytest.mark.anyio
async def test_fallback_starts_from_the_quality_selected_model():
    pool = SqlitePool()
    add_evaluations(pool, OPENAI_MODEL, [0.9] * 5, task="chat", provider="openai")
    anthropic = StubProvider(result=make_response(ANTHROPIC_MODEL))
    openai = StubProvider(error=upstream_error(503, 502))
    app = quality_app(pool, anthropic, openai)
    request = make_request("azir-auto", "chat", routing_policy="quality")

    assert [c.name for c in plan_attempts(request, await load_routing_quality(app, request))] == [
        OPENAI_MODEL,
        ANTHROPIC_MODEL,
    ]

    response = await route_request(app, request)

    assert openai.requests[0].model == OPENAI_MODEL
    assert response.model == ANTHROPIC_MODEL  # existing fallback, unchanged
    # one lookup for the plan above, one for the request: none per fallback attempt
    assert pool.queries.count(QUALITY_HISTORY_SQL) == 2


@pytest.mark.anyio
async def test_streaming_uses_the_same_quality_aware_selection():
    pool = SqlitePool()
    add_evaluations(pool, OPENAI_MODEL, [0.9] * 5, task="chat", provider="openai")
    anthropic = StubProvider(stream_result=object())
    openai = StubProvider(stream_result=object())
    request = make_request("azir-auto", "chat", routing_policy="quality", stream=True)

    await stream_chat_completion(quality_app(pool, anthropic, openai), request)

    assert openai.stream_requests[0].model == OPENAI_MODEL
    assert anthropic.stream_requests == []


@pytest.mark.anyio
@pytest.mark.parametrize("policy", ["quality", "balanced"])
async def test_quality_lookup_failure_does_not_fail_routing(caplog, policy):
    app = quality_app(SqlitePool(error=ConnectionRefusedError()))
    request = make_request("azir-auto", "chat", routing_policy=policy)

    with caplog.at_level(logging.WARNING, logger="azir.quality"):
        response = await route_request(app, request)

    # same choice as with no quality history at all
    assert response.model == resolve_model(request).name
    assert "Quality history lookup failed" in caplog.text


@pytest.mark.anyio
async def test_slow_quality_lookup_is_bounded(monkeypatch):
    monkeypatch.setattr(quality, "QUALITY_LOOKUP_TIMEOUT_SECONDS", 0.01)
    app = quality_app(SqlitePool(delay=1.0))

    response = await route_request(app, make_request("azir-auto", "chat", routing_policy="quality"))

    assert response.model == ANTHROPIC_MODEL  # neutral quality -> registry order


@pytest.mark.anyio
async def test_routing_quality_reads_the_store_on_app_state(monkeypatch):
    # the router reads the persistence layer directly, never an HTTP endpoint
    calls = []

    class RecordingStore:
        async def fetch_quality_history(self, models, task):
            calls.append((models, task))
            return {}

    app = make_app(telemetry_store=RecordingStore())
    await load_routing_quality(app, make_request("azir-auto", "chat", routing_policy="quality"))

    assert calls == [([ANTHROPIC_MODEL, OPENAI_MODEL], "chat")]
    assert router.QUALITY_POLICIES == {"quality", "balanced"}
