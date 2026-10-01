import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import HTTPException

from providers.openai import OpenAIProvider
from schemas import ChatRequest, Message

OPENAI_URL = "https://api.openai.com/v1/chat/completions"


def make_response(payload: dict) -> httpx.Response:
    request = httpx.Request("POST", OPENAI_URL)
    return httpx.Response(200, json=payload, request=request)


def make_streaming_client(response: httpx.Response) -> AsyncMock:
    """An AsyncMock client wired for OpenAIProvider.stream()'s
    build_request()+send(..., stream=True) pattern -- same shape as the
    Anthropic streaming test helper. build_request() is a plain (sync)
    method on httpx.AsyncClient, so it's overridden with a MagicMock.
    """
    client = AsyncMock()
    client.build_request = MagicMock(return_value=httpx.Request("POST", OPENAI_URL))
    client.send.return_value = response
    return client


def make_stream_request(model: str = "gpt-4o-mini") -> ChatRequest:
    return ChatRequest(model=model, messages=[Message(role="user", content="Hi")])


def parse_sse(chunk: str) -> dict:
    assert chunk.startswith("data: ")
    assert chunk.endswith("\n\n")
    return json.loads(chunk[len("data: "):])


OPENAI_STREAM_EVENTS = (
    b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini",'
    b'"choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}\n\n'
    b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini",'
    b'"choices":[{"index":0,"delta":{"content":"Hel"},"finish_reason":null}]}\n\n'
    b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini",'
    b'"choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":null}]}\n\n'
    b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini",'
    b'"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
    b'data: [DONE]\n\n'
)


@pytest.mark.anyio
async def test_complete_normalizes_openai_response():
    payload = {
        "model": "gpt-4o-mini",
        "choices": [
            {
                "message": {"role": "assistant", "content": "Hi!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 7,
            "completion_tokens": 3,
            "total_tokens": 10,
        },
    }

    client = AsyncMock()
    client.post.return_value = make_response(payload)

    provider = OpenAIProvider(client=client)
    request = ChatRequest(
        model="gpt-4o-mini",
        messages=[Message(role="user", content="Hi")],
    )

    response = await provider.complete(request)

    assert response.model == "gpt-4o-mini"
    assert response.choices[0].message.role == "assistant"
    assert response.choices[0].message.content == "Hi!"
    assert response.choices[0].finish_reason == "stop"
    assert response.usage.total_tokens == 10

    client.post.assert_awaited_once()


@pytest.mark.anyio
async def test_stream_emits_role_then_incremental_text_then_finish_then_done():
    response = httpx.Response(
        200, content=OPENAI_STREAM_EVENTS, request=httpx.Request("POST", OPENAI_URL)
    )
    client = make_streaming_client(response)
    provider = OpenAIProvider(client=client)

    chunks = [chunk async for chunk in await provider.stream(make_stream_request())]

    assert len(chunks) == 5

    assert parse_sse(chunks[0]) == {
        "choices": [{"delta": {"role": "assistant", "content": ""}, "index": 0, "finish_reason": None}]
    }
    assert parse_sse(chunks[1]) == {
        "choices": [{"delta": {"content": "Hel"}, "index": 0, "finish_reason": None}]
    }
    assert parse_sse(chunks[2]) == {
        "choices": [{"delta": {"content": "lo"}, "index": 0, "finish_reason": None}]
    }
    assert parse_sse(chunks[3]) == {
        "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}]
    }
    assert chunks[4] == "data: [DONE]\n\n"

    sent_payload = client.build_request.call_args.kwargs["json"]
    assert sent_payload["stream"] is True


@pytest.mark.anyio
async def test_stream_raises_clean_error_for_upstream_failure_before_streaming():
    response = httpx.Response(
        401,
        json={"error": {"message": "Incorrect API key provided"}},
        request=httpx.Request("POST", OPENAI_URL),
    )
    client = make_streaming_client(response)
    provider = OpenAIProvider(client=client)

    with pytest.raises(HTTPException) as exc_info:
        await provider.stream(make_stream_request())

    assert exc_info.value.status_code == 401
    assert "Incorrect API key" not in exc_info.value.detail


@pytest.mark.anyio
async def test_stream_emits_error_finish_reason_on_mid_stream_connection_failure():
    async def broken_lines():
        yield (
            'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},'
            '"finish_reason":null}]}'
        )
        raise httpx.ReadError("connection reset")

    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.aiter_lines = broken_lines
    response.aclose = AsyncMock()

    client = make_streaming_client(response)
    provider = OpenAIProvider(client=client)

    chunks = [chunk async for chunk in await provider.stream(make_stream_request())]

    assert parse_sse(chunks[0]) == {
        "choices": [{"delta": {"role": "assistant", "content": "Hi"}, "index": 0, "finish_reason": None}]
    }
    assert parse_sse(chunks[1]) == {
        "choices": [{"delta": {}, "index": 0, "finish_reason": "error"}]
    }
    assert chunks[2] == "data: [DONE]\n\n"
    response.aclose.assert_awaited_once()


@pytest.mark.anyio
async def test_stream_closes_cleanly_when_client_disconnects_early():
    async def endless_lines():
        while True:
            yield 'data: {"choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}'

    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.aiter_lines = endless_lines
    response.aclose = AsyncMock()

    provider = OpenAIProvider(client=make_streaming_client(response))
    stream = await provider.stream(make_stream_request())

    await stream.__anext__()
    # closing mid-stream must not raise "async generator ignored GeneratorExit"
    await stream.aclose()

    response.aclose.assert_awaited_once()
