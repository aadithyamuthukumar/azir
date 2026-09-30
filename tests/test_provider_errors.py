from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import HTTPException

from providers.anthropic import AnthropicProvider
from providers.errors import raise_provider_error
from providers.openai import OpenAIProvider
from schemas import ChatRequest, Message

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"


def make_status_error(status_code: int, url: str) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", url)
    response = httpx.Response(status_code, json={"error": "irrelevant"}, request=request)
    return httpx.HTTPStatusError("error", request=request, response=response)


# --- Direct tests of the shared translation helper ---


@pytest.mark.parametrize(
    "status_code, expected_status",
    [
        (400, 400),
        (401, 401),
        (403, 403),
        (404, 404),
        (429, 429),
        (500, 502),
        (503, 502),
        (418, 502),
    ],
)
def test_raise_provider_error_maps_http_status_errors(status_code, expected_status):
    exc = make_status_error(status_code, ANTHROPIC_URL)

    with pytest.raises(HTTPException) as exc_info:
        raise_provider_error(exc, provider="Anthropic")

    assert exc_info.value.status_code == expected_status
    assert "Anthropic" in exc_info.value.detail
    # never leak the raw provider response body
    assert "irrelevant" not in exc_info.value.detail


def test_raise_provider_error_maps_timeout():
    exc = httpx.ConnectTimeout("timed out")

    with pytest.raises(HTTPException) as exc_info:
        raise_provider_error(exc, provider="OpenAI")

    assert exc_info.value.status_code == 504


def test_raise_provider_error_maps_connection_error():
    exc = httpx.ConnectError("connection refused")

    with pytest.raises(HTTPException) as exc_info:
        raise_provider_error(exc, provider="OpenAI")

    assert exc_info.value.status_code == 502


# --- End-to-end wiring through each provider's complete() ---


def make_request(model: str) -> ChatRequest:
    return ChatRequest(model=model, messages=[Message(role="user", content="Hi")])


@pytest.mark.anyio
async def test_anthropic_complete_translates_401():
    client = AsyncMock()
    client.post.return_value = httpx.Response(
        401,
        json={"error": {"message": "invalid x-api-key"}},
        request=httpx.Request("POST", ANTHROPIC_URL),
    )

    provider = AnthropicProvider(client=client)

    with pytest.raises(HTTPException) as exc_info:
        await provider.complete(make_request("claude-3-5-sonnet-20241022"))

    assert exc_info.value.status_code == 401
    assert "x-api-key" not in exc_info.value.detail


@pytest.mark.anyio
async def test_openai_complete_translates_429():
    client = AsyncMock()
    client.post.return_value = httpx.Response(
        429,
        json={"error": {"message": "rate limited"}},
        request=httpx.Request("POST", OPENAI_URL),
    )

    provider = OpenAIProvider(client=client)

    with pytest.raises(HTTPException) as exc_info:
        await provider.complete(make_request("gpt-4o-mini"))

    assert exc_info.value.status_code == 429


@pytest.mark.anyio
async def test_openai_complete_translates_timeout():
    client = AsyncMock()
    client.post.side_effect = httpx.ReadTimeout("timed out")

    provider = OpenAIProvider(client=client)

    with pytest.raises(HTTPException) as exc_info:
        await provider.complete(make_request("gpt-4o-mini"))

    assert exc_info.value.status_code == 504
