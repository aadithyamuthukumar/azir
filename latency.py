import threading
from dataclasses import dataclass

# Weight of the newest sample in each model's exponentially weighted moving
# average (EWMA) of observed latency:
#
#   first sample:  estimate = sample
#   afterwards:    estimate = ALPHA * sample + (1 - ALPHA) * previous_estimate
#
# 0.3 lets the estimate follow a sustained change within a handful of
# requests without one outlier dominating it.
LATENCY_EWMA_ALPHA = 0.3

# Latency `azir-auto` assumes for a model with no samples yet. Neither zero
# (an untried model would always win) nor infinite (it would never be tried,
# so never sampled): a model observed slower than this loses to an untried
# one, which then gets sampled.
DEFAULT_LATENCY_ESTIMATE_MS = 1000.0


@dataclass(frozen=True)
class LatencyStats:
    estimate_ms: float
    samples: int


# In-memory only, keyed by concrete model name; resets when the process
# restarts. The lock keeps updates atomic even if called from worker threads.
_stats: dict[str, LatencyStats] = {}
_lock = threading.Lock()


def record_latency(model: str, latency_ms: float) -> None:
    """Fold one observed latency for `model` into its EWMA estimate."""
    with _lock:
        previous = _stats.get(model)

        if previous is None:
            _stats[model] = LatencyStats(estimate_ms=latency_ms, samples=1)
            return

        estimate = LATENCY_EWMA_ALPHA * latency_ms + (1 - LATENCY_EWMA_ALPHA) * previous.estimate_ms
        _stats[model] = LatencyStats(estimate_ms=estimate, samples=previous.samples + 1)


def get_latency_stats(model: str) -> LatencyStats | None:
    with _lock:
        return _stats.get(model)


def get_latency_estimate(model: str) -> float | None:
    """The model's current EWMA latency in ms, or None if never observed."""
    stats = get_latency_stats(model)
    return None if stats is None else stats.estimate_ms


def reset_latency() -> None:
    with _lock:
        _stats.clear()
