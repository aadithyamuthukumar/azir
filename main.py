from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from providers.anthropic import AnthropicProvider
from providers.openai import OpenAIProvider
from router import route_request, stream_chat_completion
from schemas import ChatRequest, ChatResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    client = httpx.AsyncClient()

    app.state.anthropic_provider = AnthropicProvider(client)
    app.state.openai_provider = OpenAIProvider(client)

    yield

    await client.aclose()


app = FastAPI(lifespan=lifespan)


@app.get("/")
def root():
    return {"message": "Azir is running"}


@app.post(
    "/v1/chat/completions",
    response_model=ChatResponse,
)
async def chat(request: ChatRequest):
    if request.stream:
        event_stream = await stream_chat_completion(app, request)
        return StreamingResponse(event_stream, media_type="text/event-stream")

    return await route_request(app, request)