import asyncio
import json
import logging
import re
import sqlite3

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import main
import telemetry_store
from main import app
from schemas import AnalyticsSummary, ModelAnalytics, ProviderAnalytics
from telemetry import RequestTelemetry
from telemetry_store import INSERT_SQL, SCHEMA_PATH, TelemetryStore
from tests.test_router import StubProvider, make_response

DSN = "postgresql://azir:s3cret-pw@db.internal:5432/azir"


class SqlitePool:
    """Stands in for an asyncpg.Pool, backed by an in-memory SQLite copy of
    `request_telemetry`, so the store's real aggregation SQL is executed
    (no production database needed). The analytics queries stick to SQL
    both engines accept; Postgres-only behavior (e.g. SUM(integer) ->
    bigint) is covered by schema/type assertions, not by this fake."""

    def __init__(self, error: Exception | None = None, delay: float = 0.0):
        self.error = error
        self.delay = delay
        self.queries: list[str] = []
        self.closed = False
        # TestClient serves the app from its own thread
        self.db = sqlite3.connect(":memory:", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute(
            """
            CREATE TABLE request_telemetry (
                id                 INTEGER PRIMARY KEY,
                provider           TEXT NOT NULL,
                model              TEXT NOT NULL,
                status             TEXT NOT NULL,
                status_code        INTEGER NOT NULL,
                latency_ms         DOUBLE PRECISION NOT NULL,
                prompt_tokens      INTEGER,
                completion_tokens  INTEGER,
                total_tokens       INTEGER,
                estimated_cost_usd DOUBLE PRECISION,
                traffic            TEXT NOT NULL DEFAULT 'user',
                created_at         TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.db.execute(
            """
            CREATE TABLE response_evaluations (
                id                 INTEGER PRIMARY KEY,
                telemetry_id       INTEGER REFERENCES request_telemetry (id),
                provider           TEXT NOT NULL,
                model              TEXT NOT NULL,
                task               TEXT,
                judge_provider     TEXT NOT NULL,
                judge_model        TEXT NOT NULL,
                score              DOUBLE PRECISION NOT NULL CHECK (score >= 0 AND score <= 1),
                reason             TEXT NOT NULL,
                judge_telemetry_id INTEGER REFERENCES request_telemetry (id),
                created_at         TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

    def insert(self, *records: RequestTelemetry) -> "SqlitePool":
        for record in records:
            self._run(
                INSERT_SQL,
                (
                    record.provider,
                    record.model,
                    record.status,
                    record.status_code,
                    record.latency_ms,
                    record.prompt_tokens,
                    record.completion_tokens,
                    record.total_tokens,
                    record.estimated_cost_usd,
                    record.traffic,
                ),
            ).fetchone()
        return self

    def _run(self, query: str, args: tuple = ()):
        # Postgres array match -> SQLite JSON-array membership (the one
        # non-portable construct, used by the routing quality lookup)
        query = query.replace("= ANY($1)", "IN (SELECT value FROM json_each($1))")
        args = tuple(json.dumps(arg) if isinstance(arg, list) else arg for arg in args)
        # asyncpg placeholders ($1, $2, ...) -> sqlite's numbered "?NNN"
        return self.db.execute(re.sub(r"\$(\d+)", r"?\1", query), args)

    async def _before(self, query: str):
        self.queries.append(query)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error

    async def execute(self, query, *args):
        await self._before(query)
        if query != SCHEMA_PATH.read_text():  # table already exists here
            self._run(query, args)

    async def fetchval(self, query, *args):
        await self._before(query)
        return self._run(query, args).fetchone()[0]

    async def fetchrow(self, query, *args):
        await self._before(query)
        return self._run(query, args).fetchone()

    async def fetch(self, query, *args):
        await self._before(query)
        return self._run(query, args).fetchall()

    async def close(self):
        self.closed = True


def attempt(provider, model, latency_ms, tokens=None, cost=None) -> RequestTelemetry:
    """A success when `tokens` (prompt, completion) is given, else a failure."""
    if tokens is None:
        return RequestTelemetry(
            provider=provider, model=model, status="error", status_code=502, latency_ms=latency_ms
        )
    prompt, completion = tokens
    return RequestTelemetry(
        provider=provider,
        model=model,
        status="success",
        status_code=200,
        latency_ms=latency_ms,
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        estimated_cost_usd=cost,
    )


ALL_SUCCESS = [
    attempt("openai", "gpt-4o-mini", 100.0, (10, 5), 0.001),
    attempt("openai", "gpt-4o-mini", 200.0, (20, 10), 0.002),
    attempt("anthropic", "claude-sonnet-4-6", 300.0, (100, 50), 0.01),
]

MIXED = [
    attempt("openai", "gpt-4o-mini", 100.0, (10, 5), 0.001),
    attempt("openai", "gpt-4o-mini", 200.0, (20, 10), 0.002),
    attempt("openai", "gpt-4o-mini", 30.0),
    attempt("anthropic", "claude-sonnet-4-6", 400.0, (100, 50), 0.01),
    attempt("anthropic", "claude-sonnet-4-6", 50.0),
    attempt("openai", "gpt-4o", 300.0, (40, 20), 0.005),
]


@pytest.fixture
def client():
    # Real lifespan (DATABASE_URL is cleared in conftest, so no store);
    # tests install a store backed by SqlitePool as needed.
    with TestClient(app) as test_client:
        yield test_client


def use_pool(pool: SqlitePool) -> SqlitePool:
    app.state.telemetry_store = TelemetryStore(pool)
    return pool


# --- Summary ---


def test_summary_empty_database_returns_zeros_and_null_averages(client):
    use_pool(SqlitePool())

    response = client.get("/v1/analytics/summary")

    assert response.status_code == 200
    assert response.json() == {
        "total_attempts": 0,
        "successful_attempts": 0,
        "failed_attempts": 0,
        "success_rate": None,
        "average_latency_ms": None,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
        "total_estimated_cost_usd": 0.0,
    }


def test_summary_all_success(client):
    use_pool(SqlitePool().insert(*ALL_SUCCESS))

    data = client.get("/v1/analytics/summary").json()

    assert data["total_attempts"] == 3
    assert data["successful_attempts"] == 3
    assert data["failed_attempts"] == 0
    assert data["success_rate"] == 1.0
    assert data["average_latency_ms"] == 200.0


def test_summary_mixed_success_and_failure(client):
    use_pool(SqlitePool().insert(*MIXED))

    data = client.get("/v1/analytics/summary").json()

    assert data["total_attempts"] == 6
    assert data["successful_attempts"] == 4
    assert data["failed_attempts"] == 2
    assert data["success_rate"] == 0.6667  # 4/6, 4 decimals


def test_summary_token_totals_skip_failed_attempts(client):
    use_pool(SqlitePool().insert(*MIXED))

    data = client.get("/v1/analytics/summary").json()

    assert data["total_prompt_tokens"] == 10 + 20 + 100 + 40
    assert data["total_completion_tokens"] == 5 + 10 + 50 + 20
    assert data["total_tokens"] == 255


def test_summary_total_cost(client):
    use_pool(SqlitePool().insert(*MIXED))

    data = client.get("/v1/analytics/summary").json()

    assert data["total_estimated_cost_usd"] == pytest.approx(0.018)


def test_summary_cost_ignores_unpriced_attempts(client):
    use_pool(SqlitePool().insert(attempt("openai", "unregistered", 10.0, (1, 1), None)))

    data = client.get("/v1/analytics/summary").json()

    assert data["total_tokens"] == 2
    assert data["total_estimated_cost_usd"] == 0.0


def test_summary_average_latency_includes_failures(client):
    use_pool(SqlitePool().insert(*MIXED))

    data = client.get("/v1/analytics/summary").json()

    assert data["average_latency_ms"] == 180.0  # (100+200+30+400+50+300) / 6


def test_summary_average_latency_is_rounded_to_two_decimals(client):
    use_pool(
        SqlitePool().insert(
            attempt("openai", "gpt-4o-mini", 1.0, (1, 1)),
            attempt("openai", "gpt-4o-mini", 1.0, (1, 1)),
            attempt("openai", "gpt-4o-mini", 2.0, (1, 1)),
        )
    )

    assert client.get("/v1/analytics/summary").json()["average_latency_ms"] == 1.33


# --- Models ---


def test_models_empty_database_returns_empty_list(client):
    use_pool(SqlitePool())

    response = client.get("/v1/analytics/models")

    assert response.status_code == 200
    assert response.json() == []


def test_models_groups_by_model_with_counts_rates_latency_and_totals(client):
    use_pool(SqlitePool().insert(*MIXED))

    data = client.get("/v1/analytics/models").json()

    assert data == [
        {
            "model": "gpt-4o-mini",
            "provider": "openai",
            "attempt_count": 3,
            "success_count": 2,
            "failure_count": 1,
            "success_rate": 0.6667,
            "average_latency_ms": 110.0,
            "total_tokens": 45,
            "total_estimated_cost_usd": pytest.approx(0.003),
        },
        {
            "model": "claude-sonnet-4-6",
            "provider": "anthropic",
            "attempt_count": 2,
            "success_count": 1,
            "failure_count": 1,
            "success_rate": 0.5,
            "average_latency_ms": 225.0,
            "total_tokens": 150,
            "total_estimated_cost_usd": pytest.approx(0.01),
        },
        {
            "model": "gpt-4o",
            "provider": "openai",
            "attempt_count": 1,
            "success_count": 1,
            "failure_count": 0,
            "success_rate": 1.0,
            "average_latency_ms": 300.0,
            "total_tokens": 60,
            "total_estimated_cost_usd": pytest.approx(0.005),
        },
    ]


def test_models_all_failures_report_zero_rate_tokens_and_cost(client):
    use_pool(SqlitePool().insert(attempt("anthropic", "claude-sonnet-4-6", 40.0)))

    [row] = client.get("/v1/analytics/models").json()

    assert row["success_rate"] == 0.0
    assert row["total_tokens"] == 0
    assert row["total_estimated_cost_usd"] == 0.0
    assert row["average_latency_ms"] == 40.0


def test_models_ordering_is_attempts_desc_then_model_name(client):
    use_pool(
        SqlitePool().insert(
            attempt("openai", "zeta", 1.0, (1, 1)),
            attempt("openai", "beta", 1.0, (1, 1)),
            attempt("anthropic", "alpha", 1.0, (1, 1)),
            attempt("openai", "gamma", 1.0, (1, 1)),
            attempt("openai", "gamma", 1.0, (1, 1)),
        )
    )

    first = [row["model"] for row in client.get("/v1/analytics/models").json()]
    second = [row["model"] for row in client.get("/v1/analytics/models").json()]

    assert first == ["gamma", "alpha", "beta", "zeta"]
    assert first == second


# --- Providers ---


def test_providers_empty_database_returns_empty_list(client):
    use_pool(SqlitePool())

    response = client.get("/v1/analytics/providers")

    assert response.status_code == 200
    assert response.json() == []


def test_providers_groups_by_provider_with_counts_rates_latency_and_totals(client):
    use_pool(SqlitePool().insert(*MIXED))

    data = client.get("/v1/analytics/providers").json()

    assert data == [
        {
            "provider": "openai",
            "attempt_count": 4,
            "success_count": 3,
            "failure_count": 1,
            "success_rate": 0.75,
            "average_latency_ms": 157.5,  # (100+200+30+300) / 4
            "total_tokens": 105,
            "total_estimated_cost_usd": pytest.approx(0.008),
        },
        {
            "provider": "anthropic",
            "attempt_count": 2,
            "success_count": 1,
            "failure_count": 1,
            "success_rate": 0.5,
            "average_latency_ms": 225.0,
            "total_tokens": 150,
            "total_estimated_cost_usd": pytest.approx(0.01),
        },
    ]


def test_providers_ordering_ties_break_on_provider_name(client):
    use_pool(
        SqlitePool().insert(
            attempt("openai", "gpt-4o-mini", 1.0, (1, 1)),
            attempt("anthropic", "claude-sonnet-4-6", 1.0, (1, 1)),
        )
    )

    data = client.get("/v1/analytics/providers").json()

    assert [row["provider"] for row in data] == ["anthropic", "openai"]


def test_judge_traffic_is_excluded_from_attempt_analytics(client):
    judge_call = attempt("openai", "gpt-4o-mini", 999.0, (500, 50), 0.5)
    judge_call.traffic = "judge"
    use_pool(SqlitePool().insert(*ALL_SUCCESS, judge_call))

    summary = client.get("/v1/analytics/summary").json()
    models = client.get("/v1/analytics/models").json()
    providers = client.get("/v1/analytics/providers").json()

    assert summary["total_attempts"] == 3
    assert summary["average_latency_ms"] == 200.0
    assert summary["total_estimated_cost_usd"] == pytest.approx(0.013)
    assert sum(row["attempt_count"] for row in models) == 3
    assert sum(row["attempt_count"] for row in providers) == 3


# --- Quality ---


def add_evaluation(pool: SqlitePool, model, provider, score):
    pool.db.execute(
        "INSERT INTO response_evaluations (provider, model, judge_provider, judge_model, score, reason) "
        "VALUES (?, ?, 'openai', 'gpt-4o-mini', ?, 'ok')",
        (provider, model, score),
    )


def test_quality_empty_database_returns_empty_list(client):
    use_pool(SqlitePool())

    response = client.get("/v1/analytics/quality")

    assert response.status_code == 200
    assert response.json() == []


def test_quality_averages_scores_per_evaluated_model(client):
    pool = use_pool(SqlitePool())
    for score in (0.9, 0.8, 0.6):
        add_evaluation(pool, "claude-sonnet-4-6", "anthropic", score)
    add_evaluation(pool, "gpt-4o-mini", "openai", 1.0)

    assert client.get("/v1/analytics/quality").json() == [
        {
            "model": "claude-sonnet-4-6",
            "provider": "anthropic",
            "evaluation_count": 3,
            "average_quality_score": 0.7667,  # 2.3 / 3, 4 decimals
        },
        {
            "model": "gpt-4o-mini",
            "provider": "openai",
            "evaluation_count": 1,
            "average_quality_score": 1.0,
        },
    ]


# --- Failures ---


ALL_PATHS = [
    "/v1/analytics/summary",
    "/v1/analytics/models",
    "/v1/analytics/providers",
    "/v1/analytics/quality",
]


@pytest.mark.parametrize("path", ALL_PATHS)
def test_database_failure_is_clean_503_without_leaking_credentials(client, caplog, path):
    use_pool(SqlitePool(error=OSError(f"could not connect to {DSN}")))

    with caplog.at_level(logging.ERROR, logger="azir.analytics"):
        response = client.get(path)

    assert response.status_code == 503
    assert response.json() == {"detail": "Telemetry analytics are temporarily unavailable."}
    assert "s3cret-pw" not in response.text
    assert "postgresql://" not in response.text
    assert "Telemetry analytics query failed" in caplog.text


def test_slow_database_read_times_out_as_503(client, monkeypatch):
    monkeypatch.setattr(telemetry_store, "READ_TIMEOUT_SECONDS", 0.01)
    use_pool(SqlitePool(delay=1.0))

    assert client.get("/v1/analytics/summary").status_code == 503


@pytest.mark.parametrize("path", ALL_PATHS)
def test_analytics_without_database_url_is_503(client, path):
    assert app.state.telemetry_store is None

    response = client.get(path)

    assert response.status_code == 503
    assert "not configured" in response.json()["detail"]


@pytest.mark.anyio
async def test_store_read_methods_propagate_database_errors():
    store = TelemetryStore(SqlitePool(error=ConnectionRefusedError()))

    for read in (
        store.fetch_summary,
        store.fetch_model_stats,
        store.fetch_provider_stats,
        store.fetch_quality_stats,
    ):
        with pytest.raises(ConnectionRefusedError):
            await read()


# --- End to end through the shared pool ---


def test_analytics_reuse_the_lifespan_pool_and_report_concrete_models(monkeypatch):
    pool = SqlitePool()
    create_pool_calls = []

    async def fake_create_pool(dsn, **kwargs):
        create_pool_calls.append(dsn)
        return pool

    monkeypatch.setattr(main.settings, "database_url", DSN)
    monkeypatch.setattr(telemetry_store.asyncpg, "create_pool", fake_create_pool)

    with TestClient(app) as test_client:
        app.state.anthropic_provider = StubProvider(result=make_response("claude-sonnet-4-6"))
        app.state.openai_provider = StubProvider(result=make_response("gpt-4o-mini"))

        chat = {"model": "azir-auto", "task": "coding", "messages": [{"role": "user", "content": "hi"}]}
        assert test_client.post("/v1/chat/completions", json=chat).status_code == 200

        for path in ["/v1/analytics/summary", "/v1/analytics/models", "/v1/analytics/providers"]:
            assert test_client.get(path).status_code == 200
        models = test_client.get("/v1/analytics/models").json()

    assert create_pool_calls == [DSN]  # one pool, never one per request
    assert [(row["model"], row["provider"]) for row in models] == [("claude-sonnet-4-6", "anthropic")]
    assert pool.closed


# --- Response schemas ---


def test_response_schemas_validate_store_rows_and_round_for_presentation():
    summary = AnalyticsSummary(
        total_attempts=3,
        successful_attempts=2,
        failed_attempts=1,
        success_rate=2 / 3,
        average_latency_ms=123.456789,
        total_prompt_tokens=1,
        total_completion_tokens=2,
        total_tokens=3,
        total_estimated_cost_usd=0.0000045,
    )
    model = ModelAnalytics(
        model="gpt-4o-mini",
        provider="openai",
        attempt_count=1,
        success_count=1,
        failure_count=0,
        success_rate=1.0,
        average_latency_ms=1.005,
        total_tokens=2,
        total_estimated_cost_usd=0.0000045,
    )

    assert summary.success_rate == 0.6667
    assert summary.average_latency_ms == 123.46
    # cost is never rounded: per-request estimates are fractions of a cent
    assert summary.total_estimated_cost_usd == 0.0000045
    assert model.total_estimated_cost_usd == 0.0000045


def test_response_schemas_reject_incomplete_rows():
    with pytest.raises(ValidationError):
        ProviderAnalytics(provider="openai", attempt_count=1)
