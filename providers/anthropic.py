import httpx

from config import settings
from schemas import ChatRequest

async def complete(request: ChatRequest):
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

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
        )

    response.raise_for_status()

    return response.json()