import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import StreamingResponse

import telemetry_store
from config import settings
from providers.anthropic import AnthropicProvider
from providers.openai import OpenAIProvider
from router import route_request, stream_chat_completion
from schemas import (
    AnalyticsSummary,
    ChatRequest,
    ChatResponse,
    ModelAnalytics,
    ProviderAnalytics,
    QualityAnalytics,
)

logger = logging.getLogger("azir.analytics")


@asynccontextmanager
async def lifespan(app: FastAPI):
    client = httpx.AsyncClient()
    store = await telemetry_store.open_telemetry_store(settings.database_url)

    app.state.anthropic_provider = AnthropicProvider(client)
    app.state.openai_provider = OpenAIProvider(client)
    app.state.telemetry_store = store

    yield

    await client.aclose()

    if store is not None:
        await store.close()


app = FastAPI(lifespan=lifespan)


@app.get("/")
def root():
    return {"message": "Azir is running"}


@app.post(
    "/v1/chat/completions",
    response_model=ChatResponse,
)
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    if request.stream:
        # streams are never judged: no BackgroundTasks are passed on
        event_stream = await stream_chat_completion(app, request)
        return StreamingResponse(event_stream, media_type="text/event-stream")

    return await route_request(app, request, background_tasks)


async def _query_analytics(query):
    """Run `query(store)` against the shared telemetry store.

    Analytics read the database directly, so a missing store or a failed
    read is a 503. The internal error is logged; the client only gets a
    fixed message (never the DSN or driver error text).
    """
    store = app.state.telemetry_store

    if store is None:
        raise HTTPException(
            status_code=503,
            detail="Telemetry persistence is not configured; analytics are unavailable.",
        )

    try:
        return await query(store)
    except Exception:
        logger.exception("Telemetry analytics query failed")
        raise HTTPException(
            status_code=503,
            detail="Telemetry analytics are temporarily unavailable.",
        )


@app.get("/v1/analytics/summary", response_model=AnalyticsSummary)
async def analytics_summary():
    row = await _query_analytics(lambda store: store.fetch_summary())
    return AnalyticsSummary(
        total_attempts=row["attempt_count"],
        successful_attempts=row["success_count"],
        failed_attempts=row["failure_count"],
        success_rate=row["success_rate"],
        average_latency_ms=row["average_latency_ms"],
        total_prompt_tokens=row["total_prompt_tokens"],
        total_completion_tokens=row["total_completion_tokens"],
        total_tokens=row["total_tokens"],
        total_estimated_cost_usd=row["total_estimated_cost_usd"],
    )


@app.get("/v1/analytics/models", response_model=list[ModelAnalytics])
async def analytics_models():
    rows = await _query_analytics(lambda store: store.fetch_model_stats())
    return [ModelAnalytics(**row) for row in rows]


@app.get("/v1/analytics/providers", response_model=list[ProviderAnalytics])
async def analytics_providers():
    rows = await _query_analytics(lambda store: store.fetch_provider_stats())
    return [ProviderAnalytics(**row) for row in rows]


@app.get("/v1/analytics/quality", response_model=list[QualityAnalytics])
async def analytics_quality():
    rows = await _query_analytics(lambda store: store.fetch_quality_stats())
    return [QualityAnalytics(**row) for row in rows]
