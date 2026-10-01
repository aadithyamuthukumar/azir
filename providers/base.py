import json
from abc import ABC, abstractmethod
from typing import AsyncIterator

from schemas import ChatRequest, ChatResponse


class Provider(ABC):
    """Provider-specific HTTP behavior behind one contract.

    `request.model` is always a concrete, registry-resolved model name by
    the time a provider sees it -- never `azir-auto`. Upstream failures
    are raised as HTTPException via `providers.errors.raise_provider_error`.
    """

    @abstractmethod
    async def complete(self, request: ChatRequest) -> ChatResponse:
        pass

    @abstractmethod
    async def stream(self, request: ChatRequest) -> AsyncIterator[str]:
        """Open the upstream stream (raising HTTPException on pre-stream
        failure), then return an async iterator of Azir SSE chunks ending
        in `data: [DONE]`.
        """


def sse_chunk(delta: dict, *, finish_reason: str | None) -> str:
    """Format one Azir-style, OpenAI-compatible streaming chunk as an SSE
    `data:` line. Shared by every provider's stream() implementation so
    the outgoing wire format is identical regardless of which provider
    produced it -- this is pure wire formatting, not provider-specific
    parsing, which is why it lives here rather than in one provider file.
    """
    payload = {
        "choices": [
            {
                "delta": delta,
                "index": 0,
                "finish_reason": finish_reason,
            }
        ]
    }

    return f"data: {json.dumps(payload)}\n\n"