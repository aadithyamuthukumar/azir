import os

import pytest

# config.py builds Settings() at import time and requires both keys. Set
# dummy values before any test module imports it, so the suite runs on a
# fresh clone / CI without a .env file. Environment variables take priority
# over .env, so tests also never see a developer's real keys. No test makes
# a real provider call.
os.environ["ANTHROPIC_API_KEY"] = "test-anthropic-key"
os.environ["OPENAI_API_KEY"] = "test-openai-key"
# Overrides any DATABASE_URL in a developer's .env: tests never touch a real
# database (persistence is exercised with fakes).
os.environ["DATABASE_URL"] = ""
# LLM judging stays off unless a test enables it explicitly.
os.environ["LLM_JUDGE_ENABLED"] = "false"
os.environ["LLM_JUDGE_MODEL"] = "gpt-4o-mini"

from health import reset_health  # noqa: E402
from latency import reset_latency  # noqa: E402


@pytest.fixture(autouse=True)
def clean_routing_state():
    # Latency estimates and health windows are process-global; outcomes
    # from one test must not steer azir-auto selection in another.
    reset_latency()
    reset_health()
    yield
    reset_latency()
    reset_health()


@pytest.fixture
def anyio_backend():
    return "asyncio"
