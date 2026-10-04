from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, Field


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


RoutingPolicy = Literal["cheap", "fast", "balanced"]


class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    max_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = None
    stream: bool = False
    # Required capability when model is "azir-auto" (e.g. "coding"). When
    # given, it also restricts which models may serve as fallbacks.
    task: str | None = None
    # Upper bound on the *estimated* request cost for any model Azir picks
    # itself (the `azir-auto` choice and its fallbacks). An explicitly
    # requested model is always honored regardless of this value.
    max_cost_usd: float | None = Field(default=None, ge=0)
    # How `azir-auto` ranks the eligible models: lowest estimated cost
    # ("cheap"), lowest latency estimate ("fast"), or both ("balanced", the
    # default). Ignored for explicitly requested models.
    routing_policy: RoutingPolicy | None = None

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class Choice(BaseModel):
    message: Message
    finish_reason: str | None = None

class ChatResponse(BaseModel):
    model: str
    choices: list[Choice]
    usage: Usage


# --- Telemetry analytics (GET /v1/analytics/*) ---
#
# Presentation rounding lives here, not in the SQL: latency in ms to 2
# decimals, success rate and quality score as 0-1 fractions to 4 decimals.
# Costs are left unrounded because per-request estimates can be fractions
# of a cent.

LatencyMs = Annotated[float, AfterValidator(lambda v: round(v, 2))]
SuccessRate = Annotated[float, AfterValidator(lambda v: round(v, 4))]
QualityScore = SuccessRate


class AnalyticsSummary(BaseModel):
    total_attempts: int
    successful_attempts: int
    failed_attempts: int
    # null when there are no attempts
    success_rate: SuccessRate | None
    average_latency_ms: LatencyMs | None
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    total_estimated_cost_usd: float


class ModelAnalytics(BaseModel):
    model: str
    provider: str
    attempt_count: int
    success_count: int
    failure_count: int
    success_rate: SuccessRate
    average_latency_ms: LatencyMs
    total_tokens: int
    total_estimated_cost_usd: float


class ProviderAnalytics(BaseModel):
    provider: str
    attempt_count: int
    success_count: int
    failure_count: int
    success_rate: SuccessRate
    average_latency_ms: LatencyMs
    total_tokens: int
    total_estimated_cost_usd: float


class QualityAnalytics(BaseModel):
    # the evaluated model, not the judge
    model: str
    provider: str
    evaluation_count: int
    average_quality_score: QualityScore

