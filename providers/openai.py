import httpx

from config import settings
from providers.base import Provider
from schemas import ChatRequest, ChatResponse, Choice, Message, Usage


class OpenAIProvider(Provider):

    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    async def complete(self, request: ChatRequest) -> ChatResponse:
        payload = {
            "model": request.model,
            "messages": [
                {
                    "role": message.role,
                    "content": message.content,
                }
                for message in request.messages
            ],
        }

        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens

        if request.temperature is not None:
            payload["temperature"] = request.temperature

        headers = {
            "Authorization": f"Bearer {settings.openai_api_key}",
            "Content-Type": "application/json",
        }

        response = await self.client.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
        )

        response.raise_for_status()

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