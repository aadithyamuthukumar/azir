import os

import pytest

# config.py builds Settings() at import time and requires both keys. Set
# dummy values before any test module imports it, so the suite runs on a
# fresh clone / CI without a .env file. Environment variables take priority
# over .env, so tests also never see a developer's real keys. No test makes
# a real provider call.
os.environ["ANTHROPIC_API_KEY"] = "test-anthropic-key"
os.environ["OPENAI_API_KEY"] = "test-openai-key"


@pytest.fixture
def anyio_backend():
    return "asyncio"
