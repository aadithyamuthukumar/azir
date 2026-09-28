from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException

from providers.anthropic import AnthropicProvider
from schemas import ChatRequest, ChatResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    client = httpx.AsyncClient()

    app.state.provider = AnthropicProvider(client)

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

    return await app.state.provider.complete(request)