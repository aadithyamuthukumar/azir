from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


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

