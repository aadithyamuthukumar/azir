import httpx

from config import settings
from schemas import ChatRequest, ChatResponse, Choice, Usage, Message
from providers.base import Provider

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


    async def complete(self, request: ChatRequest):
        
        system_messages = [
            message.content
            for message in request.messages
            if message.role == "system"
        ]

    
        payload = {
            "model": request.model,
            "max_tokens": request.max_tokens or 200,
            "messages":[
                {
                    "role": message.role,
                    "content": message.content,
                }
                for message in request.messages
                if message.role != "system"
            ]   
        }

        headers = {
            "x-api-key": settings.anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }


        if system_messages:
            payload["system"] = "\n".join(system_messages)

        response = await self.client.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
            )

        response.raise_for_status()

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

