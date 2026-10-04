-- Azir telemetry storage. Idempotent: Azir runs this file at startup when
-- DATABASE_URL is set, and it is safe to run by hand:
--
--   psql "$DATABASE_URL" -f schema.sql
--
-- One row per non-streaming provider attempt (success or failure). Token
-- and cost columns are NULL when the attempt failed and no usage exists.
-- `traffic` is 'user' for client requests and 'judge' for LLM-judge
-- evaluation calls.

CREATE TABLE IF NOT EXISTS request_telemetry (
    id                 BIGSERIAL PRIMARY KEY,
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
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Tables created before `traffic` existed: existing rows become 'user'.
ALTER TABLE request_telemetry ADD COLUMN IF NOT EXISTS traffic TEXT NOT NULL DEFAULT 'user';

CREATE INDEX IF NOT EXISTS request_telemetry_model_created_at_idx
    ON request_telemetry (model, created_at);

-- One row per LLM-judge verdict on a successful non-streaming response.
-- telemetry_id is the evaluated attempt and judge_telemetry_id the judge
-- call (its cost and latency); either is NULL if that telemetry write
-- failed, since telemetry persistence is best-effort.
CREATE TABLE IF NOT EXISTS response_evaluations (
    id                 BIGSERIAL PRIMARY KEY,
    telemetry_id       BIGINT REFERENCES request_telemetry (id) ON DELETE SET NULL,
    provider           TEXT NOT NULL,
    model              TEXT NOT NULL,
    judge_provider     TEXT NOT NULL,
    judge_model        TEXT NOT NULL,
    score              DOUBLE PRECISION NOT NULL CHECK (score >= 0 AND score <= 1),
    reason             TEXT NOT NULL,
    judge_telemetry_id BIGINT REFERENCES request_telemetry (id) ON DELETE SET NULL,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS response_evaluations_model_created_at_idx
    ON response_evaluations (model, created_at);
