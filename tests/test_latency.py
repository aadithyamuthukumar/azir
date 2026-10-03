import pytest

from latency import (
    LATENCY_EWMA_ALPHA,
    get_latency_estimate,
    get_latency_stats,
    record_latency,
    reset_latency,
)


def test_unknown_model_has_no_estimate():
    assert get_latency_estimate("never-seen") is None
    assert get_latency_stats("never-seen") is None


def test_first_sample_becomes_the_estimate():
    record_latency("m", 250.0)

    assert get_latency_estimate("m") == 250.0
    assert get_latency_stats("m").samples == 1


def test_later_samples_follow_the_ewma_formula():
    assert LATENCY_EWMA_ALPHA == 0.3

    record_latency("m", 100.0)
    record_latency("m", 200.0)
    # 0.3 * 200 + 0.7 * 100
    assert get_latency_estimate("m") == pytest.approx(130.0)

    record_latency("m", 400.0)
    # 0.3 * 400 + 0.7 * 130
    assert get_latency_estimate("m") == pytest.approx(211.0)
    assert get_latency_stats("m").samples == 3


def test_estimates_are_deterministic_and_per_model():
    for latency_ms in (120.0, 80.0, 300.0):
        record_latency("a", latency_ms)
        record_latency("b", latency_ms)
    record_latency("c", 5.0)

    assert get_latency_estimate("a") == get_latency_estimate("b")
    assert get_latency_estimate("c") == 5.0


def test_reset_clears_all_models():
    record_latency("a", 1.0)
    reset_latency()

    assert get_latency_estimate("a") is None
