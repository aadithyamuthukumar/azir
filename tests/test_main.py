import pytest
from fastapi.testclient import TestClient

from main import app
from tests.test_router import StubProvider, make_response


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
    [body("mystery-model"), body("azir-auto"), body("azir-auto", task="image-generation")],
)
def test_routing_errors_are_clean_400s(client, payload):
    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 400
    assert "detail" in response.json()


def test_streaming_azir_auto_returns_event_stream(client):
    response = client.post(
        "/v1/chat/completions", json=body("azir-auto", task="classification", stream=True)
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.text.endswith("data: [DONE]\n\n")
    assert app.state.openai_provider.stream_requests[0].model == "gpt-4o-mini"


def test_streaming_routing_error_is_400_before_stream_starts(client):
    response = client.post("/v1/chat/completions", json=body("azir-auto", stream=True))

    assert response.status_code == 400
