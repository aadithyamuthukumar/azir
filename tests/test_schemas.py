import pytest
from pydantic import ValidationError

from schemas import ChatRequest, Message


def test_chat_request_accepts_valid_messages():
    request = ChatRequest(
        model="gpt-4o",
        messages=[Message(role="user", content="hello")],
    )

    assert request.model == "gpt-4o"
    assert request.messages[0].role == "user"
    assert request.stream is False


def test_chat_request_optional_fields_default_to_none():
    request = ChatRequest(model="gpt-4o", messages=[])

    assert request.max_tokens is None
    assert request.temperature is None


def test_message_rejects_invalid_role():
    with pytest.raises(ValidationError):
        Message(role="bot", content="hello")


def test_chat_request_rejects_invalid_role_in_messages():
    with pytest.raises(ValidationError):
        ChatRequest(
            model="gpt-4o",
            messages=[{"role": "bot", "content": "hello"}],
        )
