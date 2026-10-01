import json
import logging
from dataclasses import asdict, dataclass

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

    return (
        (prompt_tokens / 1000) * config.input_cost_per_1k
        + (completion_tokens / 1000) * config.output_cost_per_1k
    )


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

    def emit(self) -> None:
        logger.info(json.dumps(asdict(self)))
