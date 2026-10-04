import json
import logging
from dataclasses import asdict, dataclass
from typing import Protocol

from model_registry import get_model

logger = logging.getLogger("azir.telemetry")

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def estimate_cost_usd(
    provider: str,
    model: str,
    prompt_tokens: int | None,
    completion_tokens: int | None,
) -> float | None:
    """Rough cost estimate from the registry's static per-1K-token rates.

    Models that aren't registered (or are registered under a different
    provider) have no estimate rather than being priced off a guess.
    """
    config = get_model(model)

    if (
        config is None
        or config.provider != provider
        or prompt_tokens is None
        or completion_tokens is None
    ):
        return None

    return config.estimate_cost_usd(prompt_tokens, completion_tokens)


@dataclass
class RequestTelemetry:
    provider: str
    model: str
    status: str
    status_code: int
    latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    estimated_cost_usd: float | None = None
    # "user" for attempts serving a client request, "judge" for LLM-judge
    # evaluation calls (see judge.py), so the two never mix in analytics.
    traffic: str = "user"

    def emit(self) -> None:
        logger.info(json.dumps(asdict(self)))


class TelemetrySink(Protocol):
    async def save(self, record: RequestTelemetry) -> int | None: ...


async def publish(record: RequestTelemetry, sink: TelemetrySink | None) -> int | None:
    """Log the record, then persist it to `sink` if one is configured.
    Returns the stored row's id, or None if it wasn't stored.

    Persistence is best-effort: any sink error (database down, timeout,
    ...) is logged and swallowed, so storing analytics can never fail the
    request that produced them.
    """
    record.emit()

    if sink is None:
        return None

    try:
        return await sink.save(record)
    except Exception:
        logger.exception("Failed to persist telemetry record for %s/%s", record.provider, record.model)
        return None
