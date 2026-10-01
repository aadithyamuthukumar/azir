import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import HTTPException

from providers.anthropic import AnthropicProvider
from schemas import ChatRequest, Message

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"


def make_response(payload: dict) -> httpx.Response:
    request = httpx.Request("POST", ANTHROPIC_URL)
    return httpx.Response(200, json=payload, request=request)


def make_streaming_client(response: httpx.Response) -> AsyncMock:
    """An AsyncMock client wired for AnthropicProvider.stream()'s
    build_request()+send(..., stream=True) pattern. build_request() is a
    plain (sync) method on httpx.AsyncClient, so it's overridden with a
    MagicMock -- left as an auto-created AsyncMock attribute, calling it
    would return an unawaited coroutine instead of a real Request.
    """
    client = AsyncMock()
    client.build_request = MagicMock(return_value=httpx.Request("POST", ANTHROPIC_URL))
    client.send.return_value = response
    return client


def make_stream_request(model: str = "claude-3-5-sonnet-20241022") -> ChatRequest:
    return ChatRequest(model=model, messages=[Message(role="user", content="Hi")])


def parse_sse(chunk: str) -> dict:
    assert chunk.startswith("data: ")
    assert chunk.endswith("\n\n")
    return json.loads(chunk[len("data: "):])


ANTHROPIC_STREAM_EVENTS = (
    b'event: message_start\n'
    b'data: {"type":"message_start","message":{"id":"msg_1","model":"claude-3-5-sonnet-20241022","usage":{"input_tokens":10,"output_tokens":1}}}\n\n'
    b'event: content_block_start\n'
    b'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n'
    b'event: ping\n'
    b'data: {"type":"ping"}\n\n'
    b'event: content_block_delta\n'
    b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hel"}}\n\n'
    b'event: content_block_delta\n'
    b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"lo"}}\n\n'
    b'event: content_block_stop\n'
    b'data: {"type":"content_block_stop","index":0}\n\n'
    b'event: message_delta\n'
    b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":2}}\n\n'
    b'event: message_stop\n'
    b'data: {"type":"message_stop"}\n\n'
)


@pytest.mark.parametrize(
    "stop_reason, expected",
    [
        ("end_turn", "stop"),
        ("stop_sequence", "stop"),
        ("max_tokens", "length"),
        ("tool_use", "tool_use"),
        (None, None),
    ],
)
def test_map_finish_reason(stop_reason, expected):
    provider = AnthropicProvider(client=AsyncMock())

    assert provider.map_finish_reason(stop_reason) == expected


@pytest.mark.anyio
async def test_complete_normalizes_anthropic_response():
    payload = {
        "model": "claude-3-5-sonnet-20241022",
        "content": [{"type": "text", "text": "Hello there!"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }

    client = AsyncMock()
    client.post.return_value = make_response(payload)

    provider = AnthropicProvider(client=client)
    request = ChatRequest(
        model="claude-3-5-sonnet-20241022",
        messages=[
            Message(role="system", content="Be nice."),
            Message(role="user", content="Hi"),
        ],
    )

    response = await provider.complete(request)

    assert response.model == "claude-3-5-sonnet-20241022"
    assert response.choices[0].message.role == "assistant"
    assert response.choices[0].message.content == "Hello there!"
    assert response.choices[0].finish_reason == "stop"
    assert response.usage.prompt_tokens == 10
    assert response.usage.completion_tokens == 5
    assert response.usage.total_tokens == 15

    sent_payload = client.post.call_args.kwargs["json"]
    assert sent_payload["system"] == "Be nice."
    assert all(message["role"] != "system" for message in sent_payload["messages"])


@pytest.mark.anyio
async def test_stream_emits_role_then_incremental_text_then_finish_then_done():
    response = httpx.Response(
        200, content=ANTHROPIC_STREAM_EVENTS, request=httpx.Request("POST", ANTHROPIC_URL)
    )
    client = make_streaming_client(response)
    provider = AnthropicProvider(client=client)

    chunks = [chunk async for chunk in await provider.stream(make_stream_request())]

    assert len(chunks) == 5

    assert parse_sse(chunks[0]) == {
        "choices": [{"delta": {"role": "assistant"}, "index": 0, "finish_reason": None}]
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
        json={"error": {"message": "invalid x-api-key"}},
        request=httpx.Request("POST", ANTHROPIC_URL),
    )
    client = make_streaming_client(response)
    provider = AnthropicProvider(client=client)

    with pytest.raises(HTTPException) as exc_info:
        await provider.stream(make_stream_request())

    assert exc_info.value.status_code == 401
    assert "x-api-key" not in exc_info.value.detail


@pytest.mark.anyio
async def test_stream_emits_error_finish_reason_on_mid_stream_anthropic_error_event():
    content = (
        b'event: message_start\n'
        b'data: {"type":"message_start","message":{"id":"msg_1"}}\n\n'
        b'event: content_block_delta\n'
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}\n\n'
        b'event: error\n'
        b'data: {"type":"error","error":{"type":"overloaded_error","message":"upstream is overloaded"}}\n\n'
    )
    response = httpx.Response(200, content=content, request=httpx.Request("POST", ANTHROPIC_URL))
    client = make_streaming_client(response)
    provider = AnthropicProvider(client=client)

    chunks = [chunk async for chunk in await provider.stream(make_stream_request())]

    assert parse_sse(chunks[1]) == {
        "choices": [{"delta": {"content": "Hi"}, "index": 0, "finish_reason": None}]
    }
    assert parse_sse(chunks[2]) == {
        "choices": [{"delta": {}, "index": 0, "finish_reason": "error"}]
    }
    assert chunks[3] == "data: [DONE]\n\n"
    assert len(chunks) == 4
    # the raw upstream error message must never reach the client
    assert "overloaded" not in "".join(chunks)


@pytest.mark.anyio
async def test_stream_emits_error_finish_reason_on_mid_stream_connection_failure():
    async def broken_lines():
        yield 'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}'
        raise httpx.ReadError("connection reset")

    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.aiter_lines = broken_lines
    response.aclose = AsyncMock()

    client = make_streaming_client(response)
    provider = AnthropicProvider(client=client)

    chunks = [chunk async for chunk in await provider.stream(make_stream_request())]

    assert parse_sse(chunks[1]) == {
        "choices": [{"delta": {"content": "Hi"}, "index": 0, "finish_reason": None}]
    }
    assert parse_sse(chunks[2]) == {
        "choices": [{"delta": {}, "index": 0, "finish_reason": "error"}]
    }
    assert chunks[3] == "data: [DONE]\n\n"
    response.aclose.assert_awaited_once()


@pytest.mark.anyio
async def test_stream_closes_cleanly_when_client_disconnects_early():
    async def endless_lines():
        while True:
            yield 'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}'

    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.aiter_lines = endless_lines
    response.aclose = AsyncMock()

    provider = AnthropicProvider(client=make_streaming_client(response))
    stream = await provider.stream(make_stream_request())

    await stream.__anext__()
    # closing mid-stream must not raise "async generator ignored GeneratorExit"
    await stream.aclose()

    response.aclose.assert_awaited_once()
