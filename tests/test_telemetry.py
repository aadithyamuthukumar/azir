import json
import logging

import pytest

from telemetry import RequestTelemetry, estimate_cost_usd, publish


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
        "traffic": "user",
    }


class RecordingSink:
    def __init__(self, error: Exception | None = None):
        self.error = error
        self.saved = []

    async def save(self, record):
        if self.error is not None:
            raise self.error
        self.saved.append(record)


def make_record() -> RequestTelemetry:
    return RequestTelemetry(provider="openai", model="gpt-4o-mini", status="success", status_code=200, latency_ms=1.0)


@pytest.mark.anyio
async def test_publish_logs_and_saves_to_sink(caplog):
    sink = RecordingSink()
    record = make_record()

    with caplog.at_level(logging.INFO, logger="azir.telemetry"):
        await publish(record, sink)

    assert len(caplog.records) == 1
    assert sink.saved == [record]


@pytest.mark.anyio
async def test_publish_without_sink_only_logs(caplog):
    with caplog.at_level(logging.INFO, logger="azir.telemetry"):
        await publish(make_record(), None)

    assert len(caplog.records) == 1


@pytest.mark.anyio
@pytest.mark.parametrize("error", [ConnectionRefusedError(), TimeoutError(), RuntimeError("boom")])
async def test_publish_logs_and_swallows_sink_errors(caplog, error):
    with caplog.at_level(logging.INFO, logger="azir.telemetry"):
        await publish(make_record(), RecordingSink(error=error))

    # the JSON record is still logged, followed by the persistence error
    assert json.loads(caplog.records[0].message)["model"] == "gpt-4o-mini"
    assert caplog.records[1].levelno == logging.ERROR
    assert "Failed to persist telemetry" in caplog.records[1].message
