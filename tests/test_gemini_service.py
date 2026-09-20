"""Unit tests for the Gemini service module (mocked client)."""

import pytest

from app.errors import AINotConfiguredError, AIUnavailableError
from app.services import gemini


class _Msg:
    def __init__(self, role: str, content: str) -> None:
        self.role = role
        self.content = content


def test_get_client_without_key_raises(monkeypatch):
    # Empty string (not delenv) so a developer's real .env is also overridden.
    monkeypatch.setenv("GEMINI_API_KEY", "")
    from app.config import get_settings

    get_settings.cache_clear()
    gemini.reset_client()
    try:
        with pytest.raises(AINotConfiguredError):
            gemini.get_client()
    finally:
        get_settings.cache_clear()
        gemini.reset_client()


def test_generate_reply_returns_text_and_model(fake_gemini):
    from app.config import get_settings

    reply, model = gemini.generate_reply([], "Hi")
    assert reply == "Hello! How can I help you?"
    assert model == get_settings().gemini_model


def test_generate_reply_maps_assistant_role_to_model(fake_gemini):
    gemini.generate_reply([_Msg("assistant", "earlier answer"), _Msg("user", "hi")], "next")
    contents = fake_gemini.calls[0]["contents"]
    assert contents[0].role == "model"
    assert contents[0].parts[0].text == "earlier answer"


def test_generate_reply_upstream_error_maps_to_ai_unavailable(failing_gemini):
    with pytest.raises(AIUnavailableError):
        gemini.generate_reply([], "Hi")


def test_stream_reply_yields_chunks(fake_gemini):
    chunks = list(gemini.stream_reply([], "Hi"))
    assert "".join(chunks) == "Hello! How can I help?"


def test_stream_reply_upstream_error_maps_to_ai_unavailable(failing_gemini):
    with pytest.raises(AIUnavailableError):
        list(gemini.stream_reply([], "Hi"))
