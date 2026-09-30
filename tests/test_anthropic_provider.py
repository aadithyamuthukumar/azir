from unittest.mock import AsyncMock

import httpx
import pytest

from providers.anthropic import AnthropicProvider
from schemas import ChatRequest, Message


def make_response(payload: dict) -> httpx.Response:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return httpx.Response(200, json=payload, request=request)


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
