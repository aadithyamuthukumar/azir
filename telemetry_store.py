import asyncio
import logging
from pathlib import Path

import asyncpg

from judge import ResponseEvaluation
from telemetry import RequestTelemetry

logger = logging.getLogger("azir.telemetry_store")

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

# Connections are opened lazily, on the first write (min_size=0), so an
# unreachable database never blocks startup.
POOL_MAX_SIZE = 5

# Upper bound on one telemetry write, including acquiring/opening a
# connection. Writes are awaited inline after each provider attempt, so this
# is the most a slow or unreachable database can add to a request.
WRITE_TIMEOUT_SECONDS = 2.0

INSERT_SQL = """
INSERT INTO request_telemetry (
    provider, model, status, status_code, latency_ms,
    prompt_tokens, completion_tokens, total_tokens, estimated_cost_usd, traffic
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
RETURNING id
"""

INSERT_EVALUATION_SQL = """
INSERT INTO response_evaluations (
    telemetry_id, provider, model, judge_provider, judge_model,
    score, reason, judge_telemetry_id
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
"""

# Upper bound on one analytics query. Unlike writes, a failed read fails
# the analytics request (there is nothing to fall back to).
READ_TIMEOUT_SECONDS = 5.0

# Aggregates shared by every analytics query. Any status other than
# 'success' counts as a failure, so success + failure == attempts. SUMs
# skip NULL usage/cost (failed attempts, unregistered models) and are
# COALESCEd so an empty set reads as 0; AVG and the rate stay NULL there.
# Values are returned unrounded; the API schemas own presentation. Only
# 'user' traffic is aggregated: LLM-judge calls are excluded.
_ATTEMPT_AGGREGATES = """
    COUNT(*) AS attempt_count,
    COUNT(*) FILTER (WHERE status = 'success') AS success_count,
    COUNT(*) FILTER (WHERE status <> 'success') AS failure_count,
    CAST(COUNT(*) FILTER (WHERE status = 'success') AS DOUBLE PRECISION)
        / NULLIF(COUNT(*), 0) AS success_rate,
    AVG(latency_ms) AS average_latency_ms,
    COALESCE(SUM(total_tokens), 0) AS total_tokens,
    CAST(COALESCE(SUM(estimated_cost_usd), 0) AS DOUBLE PRECISION) AS total_estimated_cost_usd
"""

SUMMARY_SQL = f"""
SELECT
    {_ATTEMPT_AGGREGATES},
    COALESCE(SUM(prompt_tokens), 0) AS total_prompt_tokens,
    COALESCE(SUM(completion_tokens), 0) AS total_completion_tokens
FROM request_telemetry
WHERE traffic = 'user'
"""

MODEL_STATS_SQL = f"""
SELECT model, provider, {_ATTEMPT_AGGREGATES}
FROM request_telemetry
WHERE traffic = 'user'
GROUP BY model, provider
ORDER BY attempt_count DESC, model ASC, provider ASC
"""

PROVIDER_STATS_SQL = f"""
SELECT provider, {_ATTEMPT_AGGREGATES}
FROM request_telemetry
WHERE traffic = 'user'
GROUP BY provider
ORDER BY attempt_count DESC, provider ASC
"""

QUALITY_STATS_SQL = """
SELECT
    model,
    provider,
    COUNT(*) AS evaluation_count,
    AVG(score) AS average_quality_score
FROM response_evaluations
GROUP BY model, provider
ORDER BY evaluation_count DESC, model ASC, provider ASC
"""


class TelemetryStore:
    """Persists RequestTelemetry rows to Postgres, and reads aggregates
    back, through one shared connection pool. Methods raise on database
    errors; callers that must not fail (see `telemetry.publish`) handle
    that."""

    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    async def init_schema(self) -> None:
        await self._pool.execute(SCHEMA_PATH.read_text())

    async def save(self, record: RequestTelemetry) -> int:
        """Insert one row and return its id."""
        return await asyncio.wait_for(
            self._pool.fetchval(
                INSERT_SQL,
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
            timeout=WRITE_TIMEOUT_SECONDS,
        )

    async def save_evaluation(self, evaluation: ResponseEvaluation) -> None:
        await asyncio.wait_for(
            self._pool.execute(
                INSERT_EVALUATION_SQL,
                evaluation.telemetry_id,
                evaluation.provider,
                evaluation.model,
                evaluation.judge_provider,
                evaluation.judge_model,
                evaluation.score,
                evaluation.reason,
                evaluation.judge_telemetry_id,
            ),
            timeout=WRITE_TIMEOUT_SECONDS,
        )

    # --- Read-only analytics (aggregated in Postgres, one row per group) ---

    async def fetch_summary(self) -> dict:
        row = await asyncio.wait_for(self._pool.fetchrow(SUMMARY_SQL), timeout=READ_TIMEOUT_SECONDS)
        return dict(row)

    async def fetch_model_stats(self) -> list[dict]:
        rows = await asyncio.wait_for(self._pool.fetch(MODEL_STATS_SQL), timeout=READ_TIMEOUT_SECONDS)
        return [dict(row) for row in rows]

    async def fetch_provider_stats(self) -> list[dict]:
        rows = await asyncio.wait_for(self._pool.fetch(PROVIDER_STATS_SQL), timeout=READ_TIMEOUT_SECONDS)
        return [dict(row) for row in rows]

    async def fetch_quality_stats(self) -> list[dict]:
        rows = await asyncio.wait_for(self._pool.fetch(QUALITY_STATS_SQL), timeout=READ_TIMEOUT_SECONDS)
        return [dict(row) for row in rows]

    async def close(self) -> None:
        await self._pool.close()


async def open_telemetry_store(database_url: str | None) -> TelemetryStore | None:
    """Create the shared pool and apply `schema.sql`, once at startup.

    Returns None (persistence disabled, logging only) if no URL is
    configured or the pool can't be created. If the database is merely
    unreachable right now, the store is still returned: schema setup is
    logged as failed, and writes succeed once the database is back.
    """
    if not database_url:
        logger.info("DATABASE_URL not set; telemetry is logged but not persisted.")
        return None

    try:
        pool = await asyncpg.create_pool(database_url, min_size=0, max_size=POOL_MAX_SIZE)
    except Exception:
        logger.exception("Could not create telemetry database pool; persistence disabled.")
        return None

    store = TelemetryStore(pool)

    try:
        await asyncio.wait_for(store.init_schema(), timeout=WRITE_TIMEOUT_SECONDS)
    except Exception:
        logger.exception("Could not apply telemetry schema; writes will be retried per request.")

    return store
