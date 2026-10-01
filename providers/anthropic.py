import json
import logging
from typing import AsyncIterator

import httpx

from config import settings
from schemas import ChatRequest, ChatResponse, Choice, Usage, Message
from providers.base import Provider, sse_chunk
from providers.errors import raise_provider_error

logger = logging.getLogger(__name__)

ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"


class AnthropicProvider(Provider):

    def __init__(self, client: httpx.AsyncClient):
        self.client = client


    def map_finish_reason(self, stop_reason: str | None) -> str | None:
        mapping = {
            "end_turn": "stop",
            "stop_sequence": "stop",
            "max_tokens": "length",
        }

        return mapping.get(stop_reason, stop_reason)

    def _headers(self) -> dict:
        return {
            "x-api-key": settings.anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    def _build_payload(self, request: ChatRequest, *, stream: bool) -> dict:
        system_messages = [
            message.content
            for message in request.messages
            if message.role == "system"
        ]

        payload = {
            "model": request.model,
            "max_tokens": request.max_tokens or 200,
            "stream": stream,
            "messages": [
                {
                    "role": message.role,
                    "content": message.content,
                }
                for message in request.messages
                if message.role != "system"
            ],
        }

        if system_messages:
            payload["system"] = "\n".join(system_messages)

        return payload


    async def complete(self, request: ChatRequest) -> ChatResponse:
        payload = self._build_payload(request, stream=False)

        try:
            response = await self.client.post(
                    ANTHROPIC_MESSAGES_URL,
                    headers=self._headers(),
                    json=payload,
                )

            response.raise_for_status()
        except (httpx.HTTPStatusError, httpx.RequestError) as exc:
            raise_provider_error(exc, provider="Anthropic")

        data = response.json()

        text = data["content"][0]["text"]

        return ChatResponse(
            model=data["model"],
            choices=[
                Choice(
                    message=Message(
                        role="assistant",
                        content=text,
                    ),
                    finish_reason = self.map_finish_reason(data.get("stop_reason"))
                )
            ],
            usage=Usage(
                prompt_tokens=data["usage"]["input_tokens"],
                completion_tokens=data["usage"]["output_tokens"],
                total_tokens=(
                    data["usage"]["input_tokens"]
                    + data["usage"]["output_tokens"]
                ),
            ),
        )

    async def stream(self, request: ChatRequest) -> AsyncIterator[str]:
        """Open a streaming connection to Anthropic and return an async
        iterator of ready-to-send, OpenAI-style SSE chunks.

        This is deliberately a plain coroutine, not an async generator: it
        eagerly sends the request and reads only the response headers
        before returning, so a failure that happens before any model
        output exists (bad credentials, unknown model, network failure)
        surfaces here as a normal HTTPException. That distinction matters
        because once the caller wraps the returned iterator in a
        StreamingResponse, the HTTP status line is sent to the client
        before the iterator's first chunk is even requested -- so it is
        too late at that point to change the status code. See
        `_translate_stream` for how failures that happen *after* this
        point (mid-stream) are handled instead.
        """
        payload = self._build_payload(request, stream=True)

        http_request = self.client.build_request(
            "POST", ANTHROPIC_MESSAGES_URL, headers=self._headers(), json=payload
        )

        try:
            response = await self.client.send(http_request, stream=True)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            await response.aclose()
            raise_provider_error(exc, provider="Anthropic")
        except httpx.RequestError as exc:
            raise_provider_error(exc, provider="Anthropic")

        return self._translate_stream(response)

    async def _translate_stream(self, response: httpx.Response) -> AsyncIterator[str]:
        """Translate Anthropic's SSE events into Azir's OpenAI-style SSE
        chunks, one Anthropic event at a time, without buffering the full
        response.

        Only event types that carry user-visible text or are needed to
        determine how the response ended are handled:

        - `content_block_delta` (`text_delta`) -> a content chunk
        - `message_delta` (`stop_reason`)       -> the finish-reason chunk

        Everything else (`message_start`, `content_block_start/stop`,
        `ping`, `message_stop`) carries no visible text and isn't needed
        for termination -- the stream ends naturally when Anthropic closes
        the connection, at which point this always emits `[DONE]`.

        Mid-stream failures (an Anthropic `error` event, or the connection
        dropping partway through) cannot be turned into a different HTTP
        status code, because the 200 status and headers were already sent
        to the client before this generator produced its first chunk. So
        instead of leaving the client hanging, both cases are handled the
        same way: stop producing content, emit one chunk with
        `finish_reason: "error"` (a clean, Azir-defined signal -- no raw
        upstream error text is ever forwarded to the client), then still
        emit `[DONE]`. Clients should treat `finish_reason: "error"` as
        "this response is incomplete."

        `[DONE]` is emitted after the `try`, not inside `finally`: if the
        client disconnects, the generator is closed and must not yield
        again -- `finally` only releases the upstream connection.
        """
        try:
            yield sse_chunk({"role": "assistant"}, finish_reason=None)

            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue

                raw = line[len("data:"):].strip()
                if not raw:
                    continue

                event = json.loads(raw)
                event_type = event.get("type")

                if event_type == "content_block_delta":
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta" and delta.get("text"):
                        yield sse_chunk({"content": delta["text"]}, finish_reason=None)

                elif event_type == "message_delta":
                    stop_reason = event.get("delta", {}).get("stop_reason")
                    yield sse_chunk({}, finish_reason=self.map_finish_reason(stop_reason))

                elif event_type == "error":
                    logger.warning(
                        "Anthropic sent a mid-stream error event: %s",
                        event.get("error", {}).get("type", "unknown"),
                    )
                    yield sse_chunk({}, finish_reason="error")
                    break
        except httpx.HTTPError:
            logger.warning("Anthropic stream connection failed mid-stream", exc_info=True)
            yield sse_chunk({}, finish_reason="error")
        finally:
            await response.aclose()

        yield "data: [DONE]\n\n"
