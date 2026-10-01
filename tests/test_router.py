import json
import logging
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException

from model_registry import MODEL_REGISTRY, ModelConfig
from router import plan_attempts, resolve_model, route_request, stream_chat_completion
from schemas import ChatRequest, ChatResponse, Choice, Message, Usage

ANTHROPIC_MODEL = "claude-sonnet-4-6"
OPENAI_MODEL = "gpt-4o-mini"


class StubProvider:
    def __init__(self, result=None, error=None, stream_result=None, stream_error=None):
        self.result = result
        self.error = error
        self.requests = []
        self.stream_result = stream_result
        self.stream_error = stream_error
        self.stream_requests = []

    async def complete(self, request):
        self.requests.append(request)

        if self.error is not None:
            raise self.error

        return self.result

    async def stream(self, request):
        self.stream_requests.append(request)

        if self.stream_error is not None:
            raise self.stream_error

        return self.stream_result


def make_app(anthropic=None, openai=None):
    return SimpleNamespace(
        state=SimpleNamespace(
            anthropic_provider=anthropic or StubProvider(),
            openai_provider=openai or StubProvider(),
        )
    )


def make_request(model: str, task: str | None = None) -> ChatRequest:
    return ChatRequest(model=model, task=task, messages=[Message(role="user", content="hi")])


def make_response(model: str) -> ChatResponse:
    return ChatResponse(
        model=model,
        choices=[Choice(message=Message(role="assistant", content="hello"), finish_reason="stop")],
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


def upstream_error(status_code: int, mapped_status: int) -> HTTPException:
    """An HTTPException shaped like raise_provider_error()'s output: the
    client-facing status, chained from the original upstream httpx error."""
    request = httpx.Request("POST", "https://example.invalid")
    response = httpx.Response(status_code, request=request)
    exc = HTTPException(status_code=mapped_status, detail="upstream failure")
    exc.__cause__ = httpx.HTTPStatusError("error", request=request, response=response)
    return exc


@pytest.fixture
def disable_model(monkeypatch):
    def _disable(name: str):
        monkeypatch.setattr(MODEL_REGISTRY[name], "enabled", False)

    return _disable


# --- Model resolution ---


def test_resolve_explicit_anthropic_model():
    config = resolve_model(make_request(ANTHROPIC_MODEL))

    assert (config.provider, config.name) == ("anthropic", ANTHROPIC_MODEL)


def test_resolve_explicit_openai_model():
    config = resolve_model(make_request(OPENAI_MODEL))

    assert (config.provider, config.name) == ("openai", OPENAI_MODEL)


@pytest.mark.parametrize("model", ["mystery-model", "claude-unregistered", "gpt-unregistered"])
def test_resolve_rejects_unknown_model_even_with_known_prefix(model):
    with pytest.raises(HTTPException) as exc_info:
        resolve_model(make_request(model))

    assert exc_info.value.status_code == 400
    assert "Unknown model" in exc_info.value.detail


def test_resolve_rejects_disabled_explicit_model(disable_model):
    disable_model(ANTHROPIC_MODEL)

    with pytest.raises(HTTPException) as exc_info:
        resolve_model(make_request(ANTHROPIC_MODEL))

    assert exc_info.value.status_code == 400


@pytest.mark.parametrize(
    "task, expected",
    [
        ("coding", ANTHROPIC_MODEL),
        ("reasoning", ANTHROPIC_MODEL),
        ("classification", OPENAI_MODEL),
        # several models can chat; registry order decides, deterministically
        ("chat", ANTHROPIC_MODEL),
    ],
)
def test_resolve_auto_picks_first_capable_model(task, expected):
    assert resolve_model(make_request("azir-auto", task)).name == expected


def test_resolve_auto_requires_task():
    with pytest.raises(HTTPException) as exc_info:
        resolve_model(make_request("azir-auto"))

    assert exc_info.value.status_code == 400
    assert "task" in exc_info.value.detail


def test_resolve_auto_rejects_unsupported_task():
    with pytest.raises(HTTPException) as exc_info:
        resolve_model(make_request("azir-auto", "image-generation"))

    assert exc_info.value.status_code == 400


def test_resolve_auto_skips_disabled_models(disable_model):
    disable_model(ANTHROPIC_MODEL)

    assert resolve_model(make_request("azir-auto", "chat")).name == OPENAI_MODEL


def test_resolve_auto_rejects_task_when_only_capable_model_is_disabled(disable_model):
    disable_model(ANTHROPIC_MODEL)

    with pytest.raises(HTTPException) as exc_info:
        resolve_model(make_request("azir-auto", "coding"))

    assert exc_info.value.status_code == 400


def test_resolve_surfaces_misconfigured_provider(monkeypatch):
    monkeypatch.setitem(
        MODEL_REGISTRY,
        "mystery-model",
        ModelConfig("mystery-model", "mystery", 0.0, 0.0, {"chat"}),
    )

    with pytest.raises(HTTPException) as exc_info:
        resolve_model(make_request("mystery-model"))

    assert exc_info.value.status_code == 500


# --- Fallback planning ---


def test_plan_explicit_model_then_one_fallback_per_other_provider():
    assert [c.name for c in plan_attempts(make_request(ANTHROPIC_MODEL))] == [ANTHROPIC_MODEL, OPENAI_MODEL]
    assert [c.name for c in plan_attempts(make_request(OPENAI_MODEL))] == [OPENAI_MODEL, ANTHROPIC_MODEL]


def test_plan_fallbacks_must_support_task():
    # gpt-4o-mini has no "coding" capability, so there is nothing to fall back to
    assert [c.name for c in plan_attempts(make_request("azir-auto", "coding"))] == [ANTHROPIC_MODEL]


def test_plan_skips_disabled_fallback(disable_model):
    disable_model(OPENAI_MODEL)

    assert [c.name for c in plan_attempts(make_request(ANTHROPIC_MODEL))] == [ANTHROPIC_MODEL]


# --- Non-streaming routing ---


@pytest.mark.anyio
async def test_route_request_uses_primary_provider_on_success():
    anthropic = StubProvider(result=make_response(ANTHROPIC_MODEL))
    openai = StubProvider()
    app = make_app(anthropic, openai)

    response = await route_request(app, make_request(ANTHROPIC_MODEL))

    assert response.model == ANTHROPIC_MODEL
    assert len(anthropic.requests) == 1
    assert not openai.requests


@pytest.mark.anyio
async def test_route_request_sends_concrete_model_for_azir_auto():
    openai = StubProvider(result=make_response(OPENAI_MODEL))
    app = make_app(StubProvider(), openai)

    await route_request(app, make_request("azir-auto", "classification"))

    assert openai.requests[0].model == OPENAI_MODEL


@pytest.mark.anyio
@pytest.mark.parametrize(
    "error",
    [
        upstream_error(429, 429),
        upstream_error(503, 502),
        HTTPException(status_code=504, detail="Timed out."),
        HTTPException(status_code=502, detail="Could not reach Anthropic."),
    ],
)
async def test_route_request_falls_back_on_transient_failure(error):
    anthropic = StubProvider(error=error)
    openai = StubProvider(result=make_response(OPENAI_MODEL))
    app = make_app(anthropic, openai)

    response = await route_request(app, make_request(ANTHROPIC_MODEL))

    assert response.model == OPENAI_MODEL
    assert len(anthropic.requests) == 1
    assert openai.requests[0].model == OPENAI_MODEL


@pytest.mark.anyio
async def test_route_request_falls_back_to_anthropic_when_openai_fails():
    anthropic = StubProvider(result=make_response(ANTHROPIC_MODEL))
    openai = StubProvider(error=upstream_error(429, 429))
    app = make_app(anthropic, openai)

    response = await route_request(app, make_request(OPENAI_MODEL))

    assert response.model == ANTHROPIC_MODEL
    assert anthropic.requests[0].model == ANTHROPIC_MODEL


@pytest.mark.anyio
async def test_route_request_azir_auto_falls_back_with_concrete_models():
    anthropic = StubProvider(error=upstream_error(503, 502))
    openai = StubProvider(result=make_response(OPENAI_MODEL))
    app = make_app(anthropic, openai)

    await route_request(app, make_request("azir-auto", "chat"))

    assert anthropic.requests[0].model == ANTHROPIC_MODEL
    assert openai.requests[0].model == OPENAI_MODEL


@pytest.mark.anyio
@pytest.mark.parametrize(
    "error",
    [
        upstream_error(400, 400),
        upstream_error(401, 401),
        upstream_error(403, 403),
        upstream_error(404, 404),
        # unrecognized upstream 4xx is surfaced as 502 but is still a request error
        upstream_error(422, 502),
    ],
)
async def test_route_request_does_not_fall_back_on_non_recoverable_error(error):
    anthropic = StubProvider(error=error)
    openai = StubProvider(result=make_response(OPENAI_MODEL))
    app = make_app(anthropic, openai)

    with pytest.raises(HTTPException) as exc_info:
        await route_request(app, make_request(ANTHROPIC_MODEL))

    assert exc_info.value is error
    assert not openai.requests


@pytest.mark.anyio
async def test_route_request_raises_last_error_when_all_providers_fail():
    anthropic = StubProvider(error=upstream_error(503, 502))
    openai = StubProvider(error=upstream_error(429, 429))
    app = make_app(anthropic, openai)

    with pytest.raises(HTTPException) as exc_info:
        await route_request(app, make_request(ANTHROPIC_MODEL))

    assert exc_info.value.status_code == 429
    assert len(anthropic.requests) == 1
    assert len(openai.requests) == 1


@pytest.mark.anyio
@pytest.mark.parametrize(
    "request_",
    [make_request("mystery-model"), make_request("azir-auto"), make_request("azir-auto", "image-generation")],
)
async def test_route_request_rejects_bad_routing_without_calling_providers(request_):
    anthropic, openai = StubProvider(), StubProvider()
    app = make_app(anthropic, openai)

    with pytest.raises(HTTPException) as exc_info:
        await route_request(app, request_)

    assert exc_info.value.status_code == 400
    assert not anthropic.requests
    assert not openai.requests


@pytest.mark.anyio
async def test_route_request_emits_telemetry_for_each_attempt(caplog):
    anthropic = StubProvider(error=upstream_error(503, 502))
    # the provider may report a more specific model name than was requested
    openai = StubProvider(result=make_response("gpt-4o-mini-2024-07-18"))
    app = make_app(anthropic, openai)

    with caplog.at_level(logging.INFO, logger="azir.telemetry"):
        await route_request(app, make_request("azir-auto", "chat"))

    records = [json.loads(r.message) for r in caplog.records]
    assert len(records) == 2

    assert records[0]["provider"] == "anthropic"
    assert records[0]["model"] == ANTHROPIC_MODEL
    assert records[0]["status"] == "error"
    assert records[0]["status_code"] == 502

    # telemetry records the concrete model Azir attempted, never "azir-auto"
    assert records[1]["provider"] == "openai"
    assert records[1]["model"] == OPENAI_MODEL
    assert records[1]["status"] == "success"
    assert records[1]["status_code"] == 200
    assert records[1]["total_tokens"] == 2
    assert records[1]["estimated_cost_usd"] is not None


# --- Streaming routing ---


@pytest.mark.anyio
@pytest.mark.parametrize("model, provider_name", [(ANTHROPIC_MODEL, "anthropic"), (OPENAI_MODEL, "openai")])
async def test_stream_chat_completion_uses_provider_for_explicit_model(model, provider_name):
    sentinel = object()
    providers = {"anthropic": StubProvider(), "openai": StubProvider()}
    providers[provider_name].stream_result = sentinel
    app = make_app(providers["anthropic"], providers["openai"])

    result = await stream_chat_completion(app, make_request(model))

    assert result is sentinel
    assert providers[provider_name].stream_requests[0].model == model


@pytest.mark.anyio
async def test_stream_chat_completion_resolves_azir_auto_before_streaming():
    openai = StubProvider(stream_result=object())
    app = make_app(StubProvider(), openai)

    await stream_chat_completion(app, make_request("azir-auto", "classification"))

    assert openai.stream_requests[0].model == OPENAI_MODEL


@pytest.mark.anyio
async def test_stream_chat_completion_propagates_pre_stream_provider_error():
    openai = StubProvider(stream_error=upstream_error(401, 401))
    app = make_app(StubProvider(), openai)

    with pytest.raises(HTTPException) as exc_info:
        await stream_chat_completion(app, make_request(OPENAI_MODEL))

    assert exc_info.value.status_code == 401


@pytest.mark.anyio
async def test_stream_chat_completion_does_not_fall_back_on_failure():
    anthropic = StubProvider(stream_error=upstream_error(503, 502))
    openai = StubProvider(stream_result=object())
    app = make_app(anthropic, openai)

    with pytest.raises(HTTPException) as exc_info:
        await stream_chat_completion(app, make_request(ANTHROPIC_MODEL))

    assert exc_info.value.status_code == 502
    assert not openai.stream_requests


@pytest.mark.anyio
@pytest.mark.parametrize("request_", [make_request("mystery-model"), make_request("azir-auto")])
async def test_stream_chat_completion_rejects_bad_routing(request_):
    anthropic, openai = StubProvider(), StubProvider()
    app = make_app(anthropic, openai)

    with pytest.raises(HTTPException) as exc_info:
        await stream_chat_completion(app, request_)

    assert exc_info.value.status_code == 400
    assert not anthropic.stream_requests
    assert not openai.stream_requests


@pytest.mark.anyio
async def test_stream_chat_completion_rejects_disabled_model(disable_model):
    disable_model(OPENAI_MODEL)
    openai = StubProvider(stream_result=object())
    app = make_app(StubProvider(), openai)

    with pytest.raises(HTTPException) as exc_info:
        await stream_chat_completion(app, make_request(OPENAI_MODEL))

    assert exc_info.value.status_code == 400
    assert not openai.stream_requests
