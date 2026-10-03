import pytest

from health import (
    HEALTH_SUCCESS_THRESHOLD,
    HEALTH_WINDOW_SIZE,
    MIN_HEALTH_SAMPLES,
    get_sample_count,
    get_success_rate,
    is_healthy,
    record_failure,
    record_success,
    reset_health,
)


def record(model: str, successes: int, failures: int) -> None:
    for _ in range(successes):
        record_success(model)
    for _ in range(failures):
        record_failure(model)


def test_constants():
    assert (HEALTH_WINDOW_SIZE, MIN_HEALTH_SAMPLES, HEALTH_SUCCESS_THRESHOLD) == (20, 5, 0.6)


def test_unknown_model_has_no_rate_and_is_healthy():
    assert get_success_rate("never-seen") is None
    assert get_sample_count("never-seen") == 0
    assert is_healthy("never-seen")


def test_success_rate_is_successes_over_outcomes():
    record("m", successes=3, failures=1)

    assert get_success_rate("m") == 0.75
    assert get_sample_count("m") == 4


def test_insufficient_history_is_never_unhealthy():
    # 4 straight failures: rate 0.0, but below MIN_HEALTH_SAMPLES
    record("m", successes=0, failures=MIN_HEALTH_SAMPLES - 1)

    assert get_success_rate("m") == 0.0
    assert is_healthy("m")


@pytest.mark.parametrize(
    "successes, failures, healthy",
    [
        (2, 3, False),  # 0.4
        (3, 2, True),  # 0.6 -- exactly at the threshold is healthy
        (5, 0, True),
        (0, 5, False),
    ],
)
def test_threshold_applies_from_min_samples(successes, failures, healthy):
    record("m", successes, failures)

    assert is_healthy("m") is healthy


def test_window_keeps_only_recent_outcomes():
    record("m", successes=0, failures=HEALTH_WINDOW_SIZE)
    assert not is_healthy("m")

    # 20 newer successes push every old failure out of the window
    record("m", successes=HEALTH_WINDOW_SIZE, failures=0)
    assert get_sample_count("m") == HEALTH_WINDOW_SIZE
    assert get_success_rate("m") == 1.0
    assert is_healthy("m")


def test_recovery_is_gradual_and_deterministic():
    record("m", successes=0, failures=10)
    record("m", successes=10, failures=0)
    # window: 10 failures + 10 successes -> 0.5 < 0.6
    assert get_success_rate("m") == 0.5
    assert not is_healthy("m")

    record("m", successes=2, failures=0)
    # oldest 2 failures dropped: 12 / 20 = 0.6
    assert get_success_rate("m") == 0.6
    assert is_healthy("m")


def test_models_are_tracked_independently_and_reset_clears():
    record("bad", successes=0, failures=5)
    record("good", successes=5, failures=0)

    assert not is_healthy("bad")
    assert is_healthy("good")

    reset_health()
    assert get_sample_count("bad") == 0
    assert is_healthy("bad")
