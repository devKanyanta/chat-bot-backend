"""SSE streaming endpoint tests (spec 15.5)."""

from .conftest import FakeChunk
from tests.conftest import FakeModels, GeminiFactory


def test_stream_returns_raw_text_chunks(api, fake_gemini):
    response = api.post("/api/v1/chat/stream", json={"message": "Hello"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "X-Conversation-Id" in response.headers
    assert "X-Conversation-Token" in response.headers  # new anon conversation

    events = [line[6:] for line in response.text.split("\n") if line.startswith("data: ")]
    assert "".join(events) == "Hello! How can I help?"


def test_stream_existing_conversation_no_token_redisclosure(api, fake_gemini):
    first = api.post("/api/v1/chat/stream", json={"message": "Hello"})
    assert "X-Conversation-Token" in first.headers
    conversation_id = first.headers["X-Conversation-Id"]

    second = api.post(
        "/api/v1/chat/stream",
        json={"message": "Continue", "conversation_id": conversation_id},
    )
    assert second.status_code == 200
    assert "X-Conversation-Token" not in second.headers  # no re-disclosure


def test_stream_stores_assistant_reply(api, fake_gemini):
    response = api.post("/api/v1/chat/stream", json={"message": "Hello"})
    conversation_id = response.headers["X-Conversation-Id"]

    history = api.get(f"/api/v1/conversations/{conversation_id}").json()
    roles = [m["role"] for m in history["messages"]]
    assert roles == ["user", "assistant"]
    assert history["messages"][1]["content"] == "Hello! How can I help?"


def test_stream_validation_error_is_json(api):
    response = api.post("/api/v1/chat/stream", json={"message": ""})
    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"]["code"] == "validation_error"


def test_stream_upstream_failure_before_output(api, fake_gemini):
    api.post("/api/v1/chat", json={"message": "Hello"})
    fake_gemini.fail = True  # upstream dies before any token is produced
    response = api.post("/api/v1/chat/stream", json={"message": "Continue"})
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "ai_unavailable"


def test_stream_splits_multiline_chunks_into_data_fields(api, monkeypatch):
    """Chunks containing newlines must be split across data: lines (SSE spec)."""
    from app.services import gemini as gemini_service

    def multiline_get_client():
        class MultiModels:
            @staticmethod
            def generate_content_stream(*, model, contents, config):
                yield FakeChunk("line one\nline two\n\nline four")

        class MultiClient:
            models = MultiModels()

        return MultiClient()

    monkeypatch.setattr(gemini_service, "get_client", multiline_get_client)
    response = api.post("/api/v1/chat/stream", json={"message": "Hello"})
    assert response.status_code == 200
    assert "data: line one\ndata: line two\ndata: \ndata: line four\n\n" in response.text


def test_stream_mid_stream_failure_keeps_partial_reply(api, fake_gemini, monkeypatch):
    from app.services import gemini as gemini_service

    original = gemini_service.get_client

    def flaky_get_client():
        class FlakyModels:
            @staticmethod
            def generate_content_stream(*, model, contents, config):
                yield FakeChunk("partial ")
                yield FakeChunk("reply")
                raise RuntimeError("mid-stream network drop")

        class FlakyClient:
            models = FlakyModels()

        return FlakyClient()

    monkeypatch.setattr(gemini_service, "get_client", flaky_get_client)
    try:
        response = api.post("/api/v1/chat/stream", json={"message": "Hello"})
        assert response.status_code == 200
        events = [
            line[6:] for line in response.text.split("\n") if line.startswith("data: ")
        ]
        assert events[-1].startswith("[error]")

        # Partial reply was persisted.
        history = api.get(
            f"/api/v1/conversations/{response.headers['X-Conversation-Id']}"
        ).json()
        assistant = [m for m in history["messages"] if m["role"] == "assistant"]
        assert assistant and assistant[0]["content"] == "partial reply"
    finally:
        monkeypatch.setattr(gemini_service, "get_client", original)
