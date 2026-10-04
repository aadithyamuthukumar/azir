import json
import logging

import pytest
from fastapi import BackgroundTasks, HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

import judge
import router
from config import settings
from health import get_sample_count
from judge import JUDGE_SYSTEM_PROMPT, MAX_REASON_CHARS, JudgeVerdict, evaluate_response, parse_verdict
from latency import get_latency_stats
from main import app
from model_registry import MODEL_REGISTRY
from router import route_request
from schemas import ChatRequest, ChatResponse, Choice, Message, Usage
from telemetry_store import INSERT_EVALUATION_SQL
from tests.test_router import (
    ANTHROPIC_MODEL,
    OPENAI_MODEL,
    StubProvider,
    make_app,
    make_request,
    make_response,
    store_with_pool,
    upstream_error,
)

VALID_VERDICT = '{"score": 0.87, "reason": "Correct and relevant, but omitted one requested edge case."}'


class SequenceProvider:
    """Returns (or raises) the scripted results in order, one per complete()."""

    def __init__(self, *results):
        self.results = list(results)
        self.requests: list[ChatRequest] = []
        self.stream_requests: list[ChatRequest] = []

    async def complete(self, request):
        self.requests.append(request)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    async def stream(self, request):
        self.stream_requests.append(request)
        raise AssertionError("not scripted")


def judge_reply(text: str) -> ChatResponse:
    return ChatResponse(
        model="judge-reported-name",
        choices=[Choice(message=Message(role="assistant", content=text), finish_reason="stop")],
        usage=Usage(prompt_tokens=50, completion_tokens=10, total_tokens=60),
    )


@pytest.fixture
def enable_judge(monkeypatch):
    def _enable(model: str = OPENAI_MODEL):
        monkeypatch.setattr(settings, "llm_judge_enabled", True)
        monkeypatch.setattr(settings, "llm_judge_model", model)

    _enable()
    return _enable


async def route_and_run_background(app, request):
    tasks = BackgroundTasks()
    response = await route_request(app, request, tasks)
    await tasks()
    return response, tasks


def evaluation_rows(pool):
    return [args for query, args in pool.executed if query == INSERT_EVALUATION_SQL]


# --- Parsing the verdict ---


def test_valid_verdict_parses_score_and_reason():
    verdict = parse_verdict(VALID_VERDICT)

    assert verdict.score == 0.87
    assert verdict.reason == "Correct and relevant, but omitted one requested edge case."


@pytest.mark.parametrize(
    "text",
    [
        '```json\n{"score": 1, "reason": "Perfect."}\n```',
        '```\n{"score": 1, "reason": "Perfect."}\n```',
        '  {"score": 1, "reason": "  Perfect.  "}  ',
    ],
)
def test_verdict_tolerates_one_code_fence_whitespace_and_integer_scores(text):
    assert parse_verdict(text) == JudgeVerdict(score=1.0, reason="Perfect.")


@pytest.mark.parametrize("score", [1.5, -0.1, 87])
def test_verdict_rejects_scores_outside_zero_to_one(score):
    with pytest.raises(ValidationError):
        parse_verdict(json.dumps({"score": score, "reason": "x"}))


@pytest.mark.parametrize(
    "text",
    [
        "The response is great. Score: 0.9",
        'Here you go: {"score": 0.9, "reason": "Good."}',
        '{"score": "0.9", "reason": "Good."}',
        '{"score": true, "reason": "Good."}',
        '{"score": 0.9}',
        '{"score": 0.9, "reason": "   "}',
        '[0.9, "Good."]',
        "",
    ],
)
def test_verdict_rejects_malformed_output(text):
    with pytest.raises(ValueError):
        parse_verdict(text)


def test_verdict_reason_is_truncated():
    verdict = parse_verdict(json.dumps({"score": 0.5, "reason": "x" * 5000}))

    assert len(verdict.reason) == MAX_REASON_CHARS


# --- When the judge runs ---


@pytest.mark.anyio
@pytest.mark.parametrize("request_", [make_request(ANTHROPIC_MODEL), make_request("azir-auto", "coding")])
async def test_successful_response_is_judged_when_enabled(enable_judge, request_):
    anthropic = StubProvider(result=make_response(ANTHROPIC_MODEL))
    openai = SequenceProvider(judge_reply(VALID_VERDICT))

    response, tasks = await route_and_run_background(make_app(anthropic, openai), request_)

    # the user's response is returned untouched
    assert response == make_response(ANTHROPIC_MODEL)
    assert len(tasks.tasks) == 1
    [judge_request] = openai.requests
    assert judge_request.model == OPENAI_MODEL


@pytest.mark.anyio
async def test_judging_disabled_makes_no_judge_call():
    openai = SequenceProvider()
    app = make_app(StubProvider(result=make_response(ANTHROPIC_MODEL)), openai)

    _, tasks = await route_and_run_background(app, make_request(ANTHROPIC_MODEL))

    assert tasks.tasks == []
    assert openai.requests == []


@pytest.mark.anyio
async def test_route_request_without_background_tasks_never_judges(enable_judge):
    openai = SequenceProvider()

    await route_request(make_app(StubProvider(result=make_response(ANTHROPIC_MODEL)), openai), make_request(ANTHROPIC_MODEL))

    assert openai.requests == []


@pytest.mark.anyio
async def test_failed_request_is_not_judged(enable_judge):
    tasks = BackgroundTasks()

    with pytest.raises(HTTPException):
        await route_request(make_app(StubProvider(error=upstream_error(401, 401))), make_request(ANTHROPIC_MODEL), tasks)

    assert tasks.tasks == []


@pytest.mark.anyio
@pytest.mark.parametrize("judge_model", ["azir-auto", "mystery-model"])
async def test_judge_must_be_a_registered_concrete_model(enable_judge, caplog, judge_model):
    enable_judge(judge_model)
    openai = SequenceProvider()
    app = make_app(StubProvider(result=make_response(ANTHROPIC_MODEL)), openai)

    with caplog.at_level(logging.INFO, logger="azir.telemetry"):
        response, _ = await route_and_run_background(app, make_request(ANTHROPIC_MODEL))

    assert response.model == ANTHROPIC_MODEL
    assert openai.requests == []
    assert "not an enabled registry model" in caplog.text


@pytest.mark.anyio
async def test_disabled_judge_model_is_not_called(enable_judge, monkeypatch):
    monkeypatch.setattr(MODEL_REGISTRY[OPENAI_MODEL], "enabled", False)
    openai = SequenceProvider()

    await route_and_run_background(
        make_app(StubProvider(result=make_response(ANTHROPIC_MODEL)), openai), make_request(ANTHROPIC_MODEL)
    )

    assert openai.requests == []


# --- What the judge sees ---


@pytest.mark.anyio
async def test_judge_receives_original_request_and_generated_answer(enable_judge):
    openai = SequenceProvider(judge_reply(VALID_VERDICT))
    request_ = ChatRequest(
        model="azir-auto",
        task="coding",
        messages=[
            Message(role="system", content="Answer in Python."),
            Message(role="user", content="Reverse a list."),
        ],
    )

    await route_and_run_background(make_app(StubProvider(result=make_response(ANTHROPIC_MODEL)), openai), request_)

    [judge_request] = openai.requests
    system, user = judge_request.messages
    assert (system.role, system.content) == ("system", JUDGE_SYSTEM_PROMPT)
    assert user.role == "user"
    assert json.loads(user.content) == {
        "task": "coding",
        "conversation": [
            {"role": "system", "content": "Answer in Python."},
            {"role": "user", "content": "Reverse a list."},
        ],
        "candidate_response": "hello",
    }
    assert judge_request.temperature == 0.0
    assert judge_request.max_tokens == judge.JUDGE_MAX_TOKENS
    # routing fields never reach the judge, and nor do secrets
    assert judge_request.task is None and judge_request.routing_policy is None
    assert settings.openai_api_key not in user.content + system.content
    assert settings.anthropic_api_key not in user.content + system.content


def test_judge_prompt_says_it_grades_rather_than_answers():
    assert "grading another AI model's response" in JUDGE_SYSTEM_PROMPT
    assert "Do not answer" in JUDGE_SYSTEM_PROMPT
    for criterion in ["correctness", "relevance", "completeness", "instruction following"]:
        assert criterion in JUDGE_SYSTEM_PROMPT


# --- Persistence ---


@pytest.mark.anyio
async def test_evaluation_is_persisted_with_concrete_models_and_linked_telemetry(enable_judge):
    store, pool = store_with_pool()
    anthropic = StubProvider(result=make_response(ANTHROPIC_MODEL))
    openai = SequenceProvider(judge_reply(VALID_VERDICT))

    await route_and_run_background(make_app(anthropic, openai, store), make_request("azir-auto", "coding"))

    # row 1: the user's attempt; row 2: the judge call, marked as judge traffic
    user_row, judge_row = pool.inserts
    assert (user_row[:4], user_row[-1]) == (("anthropic", ANTHROPIC_MODEL, "success", 200), "user")
    assert (judge_row[:4], judge_row[-1]) == (("openai", OPENAI_MODEL, "success", 200), "judge")
    assert judge_row[5:8] == (50, 10, 60)
    assert judge_row[8] == pytest.approx(0.05 * 0.00015 + 0.01 * 0.0006)  # judge cost is recorded

    assert evaluation_rows(pool) == [
        (
            1,  # telemetry_id of the evaluated attempt
            "anthropic",
            ANTHROPIC_MODEL,  # concrete model, never "azir-auto"
            "openai",
            OPENAI_MODEL,
            0.87,
            "Correct and relevant, but omitted one requested edge case.",
            2,  # judge_telemetry_id
            "coding",  # the request's task, for task-specific routing quality
        )
    ]


@pytest.mark.anyio
async def test_fallback_response_is_judged_as_the_model_that_served_it(enable_judge):
    store, pool = store_with_pool()
    anthropic = StubProvider(error=upstream_error(503, 502))
    openai = SequenceProvider(make_response(OPENAI_MODEL), judge_reply(VALID_VERDICT))

    await route_and_run_background(make_app(anthropic, openai, store), make_request(ANTHROPIC_MODEL))

    [row] = evaluation_rows(pool)
    assert row[:3] == (2, "openai", OPENAI_MODEL)


@pytest.mark.anyio
async def test_evaluation_without_database_is_only_logged(enable_judge, caplog):
    openai = SequenceProvider(judge_reply(VALID_VERDICT))
    app = make_app(StubProvider(result=make_response(ANTHROPIC_MODEL)), openai)

    with caplog.at_level(logging.INFO, logger="azir.telemetry"):
        await route_and_run_background(app, make_request(ANTHROPIC_MODEL))

    [logged] = [json.loads(r.message)["evaluation"] for r in caplog.records if '"evaluation"' in r.message]
    assert logged["score"] == 0.87
    assert logged["telemetry_id"] is None


# --- Failure isolation ---


@pytest.mark.anyio
@pytest.mark.parametrize("reply", ["Looks good to me!", '{"score": 1.7, "reason": "Great"}', '{"reason": "no score"}'])
async def test_malformed_judge_output_does_not_break_user_response(enable_judge, caplog, reply):
    store, pool = store_with_pool()
    openai = SequenceProvider(judge_reply(reply))
    app = make_app(StubProvider(result=make_response(ANTHROPIC_MODEL)), openai, store)

    with caplog.at_level(logging.INFO, logger="azir.telemetry"):
        response, _ = await route_and_run_background(app, make_request(ANTHROPIC_MODEL))

    assert response == make_response(ANTHROPIC_MODEL)
    assert evaluation_rows(pool) == []
    assert "malformed output" in caplog.text
    assert reply not in caplog.text  # raw judge text is never logged
    # the judge call still cost money, so it is still recorded
    assert [row[-1] for row in pool.inserts] == ["user", "judge"]


@pytest.mark.anyio
@pytest.mark.parametrize(
    "error, status_code",
    [(upstream_error(503, 502), 502), (upstream_error(401, 401), 401), (KeyError("content"), 500)],
)
async def test_judge_provider_failure_does_not_break_user_response(enable_judge, caplog, error, status_code):
    store, pool = store_with_pool()
    openai = SequenceProvider(error)
    app = make_app(StubProvider(result=make_response(ANTHROPIC_MODEL)), openai, store)

    with caplog.at_level(logging.INFO, logger="azir.telemetry"):
        response, _ = await route_and_run_background(app, make_request(ANTHROPIC_MODEL))

    assert response == make_response(ANTHROPIC_MODEL)
    assert evaluation_rows(pool) == []
    judge_row = pool.inserts[-1]
    assert (judge_row[:4], judge_row[-1]) == (("openai", OPENAI_MODEL, "error", status_code), "judge")
    assert "LLM judge call to openai/gpt-4o-mini failed" in caplog.text


@pytest.mark.anyio
async def test_judge_persistence_failure_does_not_break_user_response(enable_judge, caplog):
    store, pool = store_with_pool(error=ConnectionRefusedError())
    openai = SequenceProvider(judge_reply(VALID_VERDICT))
    app = make_app(StubProvider(result=make_response(ANTHROPIC_MODEL)), openai, store)

    with caplog.at_level(logging.INFO, logger="azir.telemetry"):
        response, _ = await route_and_run_background(app, make_request(ANTHROPIC_MODEL))

    assert response == make_response(ANTHROPIC_MODEL)
    assert "Failed to persist response evaluation" in caplog.text


@pytest.mark.anyio
async def test_evaluate_response_never_raises(enable_judge, caplog):
    # even an unexpected failure (here: no provider on app.state) is logged
    broken_app = make_app()
    del broken_app.state.openai_provider

    with caplog.at_level(logging.INFO, logger="azir.telemetry"):
        result = await evaluate_response(
            broken_app, make_request(ANTHROPIC_MODEL), MODEL_REGISTRY[ANTHROPIC_MODEL], make_response(ANTHROPIC_MODEL), None
        )

    assert result is None
    assert "Response evaluation failed" in caplog.text


# --- No recursive judging ---


@pytest.mark.anyio
async def test_judge_call_is_never_itself_judged(enable_judge, monkeypatch):
    # The judge model is the same model (and provider) that served the
    # user, the setup most likely to loop if judge calls were routed.
    store, pool = store_with_pool()
    openai = SequenceProvider(make_response(OPENAI_MODEL), judge_reply(VALID_VERDICT))
    app = make_app(openai=openai, telemetry_store=store)

    tasks = BackgroundTasks()
    await route_request(app, make_request(OPENAI_MODEL), tasks)

    # while the evaluation runs, nothing may route a request
    async def no_routing(*args, **kwargs):
        raise AssertionError("judge call went through route_request")

    monkeypatch.setattr(router, "route_request", no_routing)
    await tasks()

    assert len(openai.requests) == 2  # user call + exactly one judge call
    assert len(tasks.tasks) == 1  # the evaluation queued nothing further
    assert len(evaluation_rows(pool)) == 1
    # judge calls don't feed the routing state either
    assert get_sample_count(OPENAI_MODEL) == 1
    assert get_latency_stats(OPENAI_MODEL).samples == 1


# --- Through the HTTP endpoint ---


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def chat_body(**extra) -> dict:
    return {"model": ANTHROPIC_MODEL, "messages": [{"role": "user", "content": "hi"}], **extra}


def test_endpoint_judges_after_returning_unchanged_response(enable_judge, client):
    app.state.anthropic_provider = StubProvider(result=make_response(ANTHROPIC_MODEL))
    app.state.openai_provider = judge_provider = SequenceProvider(judge_reply("not json"))

    response = client.post("/v1/chat/completions", json=chat_body())

    assert response.status_code == 200
    assert response.json() == make_response(ANTHROPIC_MODEL).model_dump()
    assert len(judge_provider.requests) == 1  # judged, malformed verdict, still a 200


def test_streaming_responses_are_not_judged(enable_judge, client):
    async def fake_stream():
        yield "data: [DONE]\n\n"

    app.state.anthropic_provider = StubProvider(stream_result=fake_stream())
    app.state.openai_provider = judge_provider = SequenceProvider()

    response = client.post("/v1/chat/completions", json=chat_body(stream=True))

    assert response.status_code == 200
    assert response.text == "data: [DONE]\n\n"
    assert judge_provider.requests == []
