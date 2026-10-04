import pytest
from fastapi.testclient import TestClient

import main
import telemetry_store
from main import app
from telemetry_store import INSERT_SQL, SCHEMA_PATH
from tests.test_router import StubProvider, make_response
from tests.test_telemetry_store import FakePool


async def fake_stream():
    yield 'data: {"choices":[{"delta":{"content":"Hi"},"index":0,"finish_reason":null}]}\n\n'
    yield "data: [DONE]\n\n"


@pytest.fixture
def client():
    # Entering the TestClient runs the real lifespan; the providers it
    # creates are then swapped for stubs so no upstream calls are made.
    with TestClient(app) as test_client:
        app.state.anthropic_provider = StubProvider(
            result=make_response("claude-sonnet-4-6"), stream_result=fake_stream()
        )
        app.state.openai_provider = StubProvider(
            result=make_response("gpt-4o-mini"), stream_result=fake_stream()
        )
        yield test_client


def body(model: str, **extra) -> dict:
    return {"model": model, "messages": [{"role": "user", "content": "hi"}], **extra}


def test_non_streaming_explicit_model(client):
    response = client.post("/v1/chat/completions", json=body("gpt-4o-mini"))

    assert response.status_code == 200
    assert response.json() == {
        "model": "gpt-4o-mini",
        "choices": [{"message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def test_non_streaming_azir_auto(client):
    response = client.post("/v1/chat/completions", json=body("azir-auto", task="coding"))

    assert response.status_code == 200
    assert app.state.anthropic_provider.requests[0].model == "claude-sonnet-4-6"


@pytest.mark.parametrize(
    "payload",
    [
        body("mystery-model"),
        body("azir-auto"),
        body("azir-auto", task="image-generation"),
        body("azir-auto", task="coding", max_cost_usd=0.0001),
    ],
)
def test_routing_errors_are_clean_400s(client, payload):
    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 400
    assert "detail" in response.json()


@pytest.mark.parametrize(
    "policy, provider_attr, expected",
    [
        # cold start: every model sits at the default latency
        ("cheap", "openai_provider", "gpt-4o-mini"),
        ("fast", "anthropic_provider", "claude-sonnet-4-6"),
        ("balanced", "openai_provider", "gpt-4o-mini"),
    ],
)
def test_non_streaming_azir_auto_routing_policy(client, policy, provider_attr, expected):
    response = client.post(
        "/v1/chat/completions", json=body("azir-auto", task="chat", routing_policy=policy)
    )

    assert response.status_code == 200
    assert getattr(app.state, provider_attr).requests[0].model == expected


def test_invalid_routing_policy_is_rejected_before_routing(client):
    response = client.post(
        "/v1/chat/completions", json=body("azir-auto", task="chat", routing_policy="fastest")
    )

    # FastAPI's standard request-validation status, like any other invalid field
    assert response.status_code == 422
    assert not app.state.anthropic_provider.requests
    assert not app.state.openai_provider.requests


def test_streaming_azir_auto_returns_event_stream(client):
    response = client.post(
        "/v1/chat/completions", json=body("azir-auto", task="classification", stream=True)
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.text.endswith("data: [DONE]\n\n")
    assert app.state.openai_provider.stream_requests[0].model == "gpt-4o-mini"


def test_lifespan_without_database_url_only_logs_telemetry(client):
    assert app.state.telemetry_store is None
    assert client.post("/v1/chat/completions", json=body("gpt-4o-mini")).status_code == 200


def test_lifespan_creates_one_pool_reuses_it_and_closes_it(monkeypatch):
    pool = FakePool()
    create_pool_calls = []

    async def fake_create_pool(dsn, **kwargs):
        create_pool_calls.append(dsn)
        return pool

    monkeypatch.setattr(main.settings, "database_url", "postgresql://user:pw@db:5432/azir")
    monkeypatch.setattr(telemetry_store.asyncpg, "create_pool", fake_create_pool)

    with TestClient(app) as test_client:
        app.state.anthropic_provider = StubProvider(error=_upstream_503())
        app.state.openai_provider = StubProvider(result=make_response("gpt-4o-mini"))

        assert test_client.post("/v1/chat/completions", json=body("gpt-4o-mini")).status_code == 200
        # explicit claude fails transiently, falls back to gpt-4o-mini: two rows
        assert test_client.post("/v1/chat/completions", json=body("claude-sonnet-4-6")).status_code == 200
        assert not pool.closed

    assert create_pool_calls == ["postgresql://user:pw@db:5432/azir"]
    assert pool.executed[0] == (SCHEMA_PATH.read_text(), ())
    assert [query for query, _ in pool.executed[1:]] == [INSERT_SQL] * 3
    assert [args[:4] for args in pool.inserts] == [
        ("openai", "gpt-4o-mini", "success", 200),
        ("anthropic", "claude-sonnet-4-6", "error", 502),
        ("openai", "gpt-4o-mini", "success", 200),
    ]
    assert pool.closed


def test_database_down_still_returns_successful_response(monkeypatch):
    async def fake_create_pool(dsn, **kwargs):
        return FakePool(error=ConnectionRefusedError())

    monkeypatch.setattr(main.settings, "database_url", "postgresql://user:pw@db:5432/azir")
    monkeypatch.setattr(telemetry_store.asyncpg, "create_pool", fake_create_pool)

    with TestClient(app) as test_client:
        app.state.openai_provider = StubProvider(result=make_response("gpt-4o-mini"))
        response = test_client.post("/v1/chat/completions", json=body("gpt-4o-mini"))

    assert response.status_code == 200
    assert response.json()["model"] == "gpt-4o-mini"


def _upstream_503():
    from tests.test_router import upstream_error

    return upstream_error(503, 502)


def test_streaming_routing_error_is_400_before_stream_starts(client):
    response = client.post("/v1/chat/completions", json=body("azir-auto", stream=True))

    assert response.status_code == 400
