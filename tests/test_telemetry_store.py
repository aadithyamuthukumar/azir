import asyncio
import re
from dataclasses import fields

import pytest

import telemetry_store
from telemetry import RequestTelemetry
from telemetry_store import INSERT_SQL, SCHEMA_PATH, TelemetryStore, open_telemetry_store


class FakePool:
    """Stands in for an asyncpg.Pool: records every execute() call."""

    def __init__(self, error: Exception | None = None, delay: float = 0.0):
        self.error = error
        self.delay = delay
        self.executed: list[tuple[str, tuple]] = []
        self.closed = False

    async def execute(self, query, *args):
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        self.executed.append((query, args))

    async def fetchval(self, query, *args):
        # INSERT ... RETURNING id: telemetry ids count up from 1 like BIGSERIAL
        await self.execute(query, *args)
        return len(self.inserts)

    async def close(self):
        self.closed = True

    @property
    def inserts(self) -> list[tuple]:
        return [args for query, args in self.executed if query == INSERT_SQL]


def success_record(**overrides) -> RequestTelemetry:
    values = dict(
        provider="openai",
        model="gpt-4o-mini",
        status="success",
        status_code=200,
        latency_ms=250.0,
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        estimated_cost_usd=0.0000045,
    )
    return RequestTelemetry(**{**values, **overrides})


def error_record() -> RequestTelemetry:
    return RequestTelemetry(
        provider="anthropic", model="claude-sonnet-4-6", status="error", status_code=502, latency_ms=12.5
    )


# --- Schema and SQL ---


def test_insert_columns_match_request_telemetry_fields():
    columns = re.search(r"\((.*?)\)", INSERT_SQL, re.S).group(1)

    assert [c.strip() for c in columns.split(",")] == [f.name for f in fields(RequestTelemetry)]


def test_schema_defines_request_telemetry_with_nullable_usage():
    schema = SCHEMA_PATH.read_text()
    assert "CREATE TABLE IF NOT EXISTS request_telemetry" in schema

    column_lines = {line.split()[0]: line for line in schema.splitlines() if line.startswith("    ")}

    for required in ["provider", "model", "status", "status_code", "latency_ms", "created_at"]:
        assert "NOT NULL" in column_lines[required]
    for nullable in ["prompt_tokens", "completion_tokens", "total_tokens", "estimated_cost_usd"]:
        assert "NOT NULL" not in column_lines[nullable]
    assert "DEFAULT NOW()" in column_lines["created_at"]


# --- TelemetryStore ---


@pytest.mark.anyio
async def test_save_inserts_one_row_with_the_record_values():
    pool = FakePool()

    row_id = await TelemetryStore(pool).save(success_record())

    assert pool.inserts == [
        ("openai", "gpt-4o-mini", "success", 200, 250.0, 10, 5, 15, 0.0000045, "user")
    ]
    assert row_id == 1


@pytest.mark.anyio
async def test_save_failed_attempt_stores_null_usage_and_cost():
    pool = FakePool()

    await TelemetryStore(pool).save(error_record())

    assert pool.inserts == [
        ("anthropic", "claude-sonnet-4-6", "error", 502, 12.5, None, None, None, None, "user")
    ]


@pytest.mark.anyio
async def test_save_reuses_the_same_pool():
    pool = FakePool()
    store = TelemetryStore(pool)

    await store.save(success_record())
    await store.save(error_record())

    assert len(pool.inserts) == 2


@pytest.mark.anyio
async def test_save_propagates_database_errors():
    # publish() is what makes persistence best-effort; the store itself
    # reports failures
    with pytest.raises(ConnectionRefusedError):
        await TelemetryStore(FakePool(error=ConnectionRefusedError())).save(success_record())


@pytest.mark.anyio
async def test_save_is_bounded_by_write_timeout(monkeypatch):
    monkeypatch.setattr(telemetry_store, "WRITE_TIMEOUT_SECONDS", 0.01)

    with pytest.raises(asyncio.TimeoutError):
        await TelemetryStore(FakePool(delay=1.0)).save(success_record())


@pytest.mark.anyio
async def test_init_schema_runs_schema_file_and_close_closes_pool():
    pool = FakePool()
    store = TelemetryStore(pool)

    await store.init_schema()
    await store.close()

    assert pool.executed == [(SCHEMA_PATH.read_text(), ())]
    assert pool.closed


# --- open_telemetry_store (startup) ---


@pytest.fixture
def fake_create_pool(monkeypatch):
    calls = []

    def _install(pool=None, error=None):
        async def create_pool(dsn, **kwargs):
            calls.append((dsn, kwargs))
            if error is not None:
                raise error
            return pool

        monkeypatch.setattr(telemetry_store.asyncpg, "create_pool", create_pool)
        return calls

    return _install


@pytest.mark.anyio
@pytest.mark.parametrize("url", [None, ""])
async def test_open_without_database_url_disables_persistence(fake_create_pool, url):
    calls = fake_create_pool(FakePool())

    assert await open_telemetry_store(url) is None
    assert calls == []


@pytest.mark.anyio
async def test_open_creates_one_lazy_pool_and_applies_schema(fake_create_pool):
    pool = FakePool()
    calls = fake_create_pool(pool)

    store = await open_telemetry_store("postgresql://u:p@db/azir")

    assert isinstance(store, TelemetryStore)
    assert len(calls) == 1
    assert calls[0][1]["min_size"] == 0
    assert pool.executed == [(SCHEMA_PATH.read_text(), ())]


@pytest.mark.anyio
async def test_open_keeps_store_when_database_unreachable_at_startup(fake_create_pool):
    fake_create_pool(FakePool(error=ConnectionRefusedError()))

    # schema setup fails, but startup continues and later writes can succeed
    assert isinstance(await open_telemetry_store("postgresql://u:p@db/azir"), TelemetryStore)


@pytest.mark.anyio
async def test_open_returns_none_when_pool_cannot_be_created(fake_create_pool):
    fake_create_pool(error=ValueError("invalid DSN"))

    assert await open_telemetry_store("not-a-dsn") is None
