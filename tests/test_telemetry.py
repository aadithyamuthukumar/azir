import json
import logging

from telemetry import RequestTelemetry, estimate_cost_usd


def test_estimate_cost_usd_known_provider_and_model():
    cost = estimate_cost_usd("openai", "gpt-4o-mini", prompt_tokens=1000, completion_tokens=1000)

    assert cost == 0.00015 + 0.0006


def test_estimate_cost_usd_unknown_provider_returns_none():
    assert estimate_cost_usd("mystery", "gpt-4o-mini", prompt_tokens=10, completion_tokens=10) is None


def test_estimate_cost_usd_unknown_model_returns_none():
    assert estimate_cost_usd("openai", "gpt-5-turbo-ultra", prompt_tokens=10, completion_tokens=10) is None


def test_estimate_cost_usd_missing_usage_returns_none():
    assert estimate_cost_usd("openai", "gpt-4o-mini", prompt_tokens=None, completion_tokens=None) is None


def test_emit_logs_one_json_line_with_expected_fields(caplog):
    record = RequestTelemetry(
        provider="anthropic",
        model="claude-sonnet-4-6",
        status="success",
        status_code=200,
        latency_ms=123.4,
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        estimated_cost_usd=0.0001,
    )

    with caplog.at_level(logging.INFO, logger="azir.telemetry"):
        record.emit()

    assert len(caplog.records) == 1

    payload = json.loads(caplog.records[0].message)
    assert payload == {
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
        "status": "success",
        "status_code": 200,
        "latency_ms": 123.4,
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "estimated_cost_usd": 0.0001,
    }
