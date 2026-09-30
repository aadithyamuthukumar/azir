from unittest.mock import AsyncMock

import httpx
import pytest

from providers.openai import OpenAIProvider
from schemas import ChatRequest, Message


def make_response(payload: dict) -> httpx.Response:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return httpx.Response(200, json=payload, request=request)


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
