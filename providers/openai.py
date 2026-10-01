import json
import logging
from typing import AsyncIterator

import httpx

from config import settings
from providers.base import Provider, sse_chunk
from providers.errors import raise_provider_error
from schemas import ChatRequest, ChatResponse, Choice, Message, Usage

logger = logging.getLogger(__name__)

OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"


class OpenAIProvider(Provider):

    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {settings.openai_api_key}",
            "Content-Type": "application/json",
        }

    def _build_payload(self, request: ChatRequest, *, stream: bool) -> dict:
        payload = {
            "model": request.model,
            "messages": [
                {
                    "role": message.role,
                    "content": message.content,
                }
                for message in request.messages
            ],
            "stream": stream,
        }

        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens

        if request.temperature is not None:
            payload["temperature"] = request.temperature

        return payload

    async def complete(self, request: ChatRequest) -> ChatResponse:
        payload = self._build_payload(request, stream=False)

        try:
            response = await self.client.post(
                OPENAI_CHAT_COMPLETIONS_URL,
                headers=self._headers(),
                json=payload,
            )

            response.raise_for_status()
        except (httpx.HTTPStatusError, httpx.RequestError) as exc:
            raise_provider_error(exc, provider="OpenAI")

        data = response.json()

        return ChatResponse(
            model=data["model"],
            choices=[
                Choice(
                    message=Message(
                        role=data["choices"][0]["message"]["role"],
                        content=data["choices"][0]["message"]["content"],
                    ),
                    finish_reason=data["choices"][0]["finish_reason"],
                )
            ],
            usage=Usage(
                prompt_tokens=data["usage"]["prompt_tokens"],
                completion_tokens=data["usage"]["completion_tokens"],
                total_tokens=data["usage"]["total_tokens"],
            ),
        )

    async def stream(self, request: ChatRequest) -> AsyncIterator[str]:
        """Open a streaming connection to OpenAI and return an async
        iterator of ready-to-send SSE chunks.

        Same two-phase design as AnthropicProvider.stream(): this is a
        plain coroutine that eagerly sends the request and validates the
        response status before returning, so a pre-stream failure (bad
        credentials, unknown model, network failure) surfaces here as a
        normal HTTPException rather than after a 200 status has already
        been sent to the client. See `_translate_stream` for failures
        that happen after that point.
        """
        payload = self._build_payload(request, stream=True)

        http_request = self.client.build_request(
            "POST", OPENAI_CHAT_COMPLETIONS_URL, headers=self._headers(), json=payload
        )

        try:
            response = await self.client.send(http_request, stream=True)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            await response.aclose()
            raise_provider_error(exc, provider="OpenAI")
        except httpx.RequestError as exc:
            raise_provider_error(exc, provider="OpenAI")

        return self._translate_stream(response)

    async def _translate_stream(self, response: httpx.Response) -> AsyncIterator[str]:
        """Forward OpenAI's own chat-completion-chunk SSE stream, only
        normalizing the outer wrapper to match Azir's existing minimal
        per-chunk shape (dropping OpenAI's id/object/created/model fields
        -- the same normalization already applied to Anthropic's stream
        and to Azir's non-streaming ChatResponse). OpenAI's delta and
        finish_reason per chunk need no further translation: Azir's
        public stream contract already *is* this shape, unlike Anthropic,
        which requires dispatching on several distinct event types to
        assemble it.

        OpenAI's own raw stream already ends with a literal `data: [DONE]`
        line. That's recognized as "stop reading" rather than forwarded
        directly, so that Azir's own `[DONE]` -- emitted after the `try`
        below -- is always the single source of truth for ending the
        stream, including on a connection failure that never reaches
        OpenAI's own [DONE]. It is deliberately not yielded from `finally`:
        if the client disconnects, the generator is closed and must not
        yield again -- `finally` only releases the upstream connection.

        Mid-stream connection failures can't change the already-sent 200
        status (see providers/anthropic.py for the same reasoning), so
        they get the same treatment: stop forwarding content, emit one
        `finish_reason: "error"` chunk (never raw upstream text), then
        still emit [DONE].
        """
        try:
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue

                raw = line[len("data:"):].strip()
                if not raw:
                    continue

                if raw == "[DONE]":
                    break

                event = json.loads(raw)
                choices = event.get("choices")
                if not choices:
                    continue

                choice = choices[0]
                yield sse_chunk(choice.get("delta", {}), finish_reason=choice.get("finish_reason"))
        except httpx.HTTPError:
            logger.warning("OpenAI stream connection failed mid-stream", exc_info=True)
            yield sse_chunk({}, finish_reason="error")
        finally:
            await response.aclose()

        yield "data: [DONE]\n\n"
