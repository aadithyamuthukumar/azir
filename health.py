import threading
from collections import deque

# How many of each model's most recent provider outcomes are kept.
HEALTH_WINDOW_SIZE = 20

# Below this many recorded outcomes a model is never considered unhealthy:
# there isn't enough evidence yet (cold start).
MIN_HEALTH_SAMPLES = 5

# With enough samples, a model whose recent success rate is below this is
# unhealthy:
#
#   success_rate = successes in window / outcomes in window
#   unhealthy    = outcomes >= MIN_HEALTH_SAMPLES and success_rate < THRESHOLD
HEALTH_SUCCESS_THRESHOLD = 0.6


# In-memory only, keyed by concrete model name; resets when the process
# restarts. The lock keeps updates atomic even if called from worker threads.
_outcomes: dict[str, deque[bool]] = {}
_lock = threading.Lock()


def _record(model: str, success: bool) -> None:
    with _lock:
        _outcomes.setdefault(model, deque(maxlen=HEALTH_WINDOW_SIZE)).append(success)


def record_success(model: str) -> None:
    _record(model, True)


def record_failure(model: str) -> None:
    _record(model, False)


def get_sample_count(model: str) -> int:
    with _lock:
        return len(_outcomes.get(model, ()))


def get_success_rate(model: str) -> float | None:
    """Successes / outcomes over the model's recent window, or None if it
    has no recorded outcomes."""
    with _lock:
        window = _outcomes.get(model)

        if not window:
            return None

        return sum(window) / len(window)


def is_healthy(model: str) -> bool:
    if get_sample_count(model) < MIN_HEALTH_SAMPLES:
        return True

    return get_success_rate(model) >= HEALTH_SUCCESS_THRESHOLD


def reset_health() -> None:
    with _lock:
        _outcomes.clear()
