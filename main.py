from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException

from providers.anthropic import AnthropicProvider
from providers.openai import OpenAIProvider
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
        raise HTTPException(
            status_code=400,
            detail="Streaming is not supported yet",
        )

    if request.model.startswith("claude"):
        provider = app.state.anthropic_provider
    else:
        provider = app.state.openai_provider

    return await provider.complete(request)