"""Auto-continuation and transient-retry behavior (long-reply completeness)."""

from tests.conftest import FakeModels, GeminiFactory


def _chat(client):
    return client.post("/api/v1/chat", json={"message": "Teach me OpenGL"})


def test_nonstream_continues_on_max_tokens(auth_api, monkeypatch):
    from app.services import gemini as gs

    models_fake = FakeModels()
    models_fake.reply_queue = ["Part one that got cut"]
    models_fake.reply_reason_queue = ["MAX_TOKENS"]
    models_fake.replies = ["...and the continuation completes the answer."]
    monkeypatch.setattr(gs, "get_client", GeminiFactory(models_fake))

    body = _chat(auth_api).json()
    assert body["reply"] == (
        "Part one that got cut\n...and the continuation completes the answer."
    )
    assert len(models_fake.calls) == 2  # original + one continuation
    # Continuation request must contain the partial text as model context.
    continuation_contents = models_fake.calls[1]["contents"]
    assert continuation_contents[-2].role == "model"
    assert "Part one" in continuation_contents[-2].parts[0].text


def test_nonstream_no_continuation_on_stop(auth_api, fake_gemini):
    _chat(auth_api)
    assert len(fake_gemini.calls) == 1  # clean STOP: exactly one call


def test_nonstream_respects_continuation_limit(auth_api, fake_gemini, monkeypatch):
    monkeypatch.setenv("MAX_REPLY_CONTINUATIONS", "2")
    from app.config import get_settings

    get_settings.cache_clear()
    try:
        fake_gemini.reply_queue = ["seg1", "seg2", "seg3"]
        fake_gemini.reply_reason_queue = ["MAX_TOKENS", "MAX_TOKENS", "MAX_TOKENS"]
        body = _chat(auth_api).json()
        assert body["reply"] == "seg1\nseg2\nseg3"
        assert len(fake_gemini.calls) == 3  # initial + 2 continuations (limit)
    finally:
        get_settings.cache_clear()


def test_stream_continues_past_output_cap(auth_api, monkeypatch):
    from app.services import gemini as gs

    fake = FakeModels(
        stream_segments=[
            (["Think of the GPU as ", "a factory."], "MAX_TOKENS"),
            ([" It has many ", "cores."], "STOP"),
        ]
    )
    monkeypatch.setattr(gs, "get_client", GeminiFactory(fake))

    response = auth_api.post("/api/v1/chat/stream", json={"message": "Explain GPUs"})
    assert response.status_code == 200
    text = "".join(
        line[6:] for line in response.text.split("\n") if line.startswith("data: ")
    )
    assert text == "Think of the GPU as a factory. It has many cores."
    assert "[error]" not in text
    assert len(fake.calls) == 2

    # The stored assistant message is the full, joined reply.
    history = auth_api.get(
        f"/api/v1/conversations/{response.headers['X-Conversation-Id']}"
    ).json()
    assistant = [m for m in history["messages"] if m["role"] == "assistant"]
    assert assistant[0]["content"] == "Think of the GPU as a factory. It has many cores."


def test_stream_no_extra_call_on_stop(auth_api, fake_gemini):
    response = auth_api.post("/api/v1/chat/stream", json={"message": "Hi"})
    assert response.status_code == 200
    assert len(fake_gemini.calls) == 1


def test_stream_continuation_limit_enforced(auth_api, fake_gemini, monkeypatch):
    monkeypatch.setenv("MAX_REPLY_CONTINUATIONS", "1")
    from app.config import get_settings

    get_settings.cache_clear()
    try:
        fake_gemini.stream_segments = [
            (["part A"], "MAX_TOKENS"),
            (["part B"], "MAX_TOKENS"),  # cap reached again, limit=1 -> stop
        ]
        response = auth_api.post("/api/v1/chat/stream", json={"message": "Hi"})
        text = "".join(
            line[6:] for line in response.text.split("\n") if line.startswith("data: ")
        )
        assert text == "part Apart B"
        assert len(fake_gemini.calls) == 2
        assert "[error]" not in text
    finally:
        get_settings.cache_clear()


def test_nonstream_retries_transient_503(auth_api, fake_gemini, monkeypatch):
    from app.services import gemini as gemini_module

    monkeypatch.setattr(gemini_module, "_RETRY_BACKOFF_SECONDS", 0.0)
    fake_gemini.transient_failures = 1  # first call 503s, second succeeds
    body = _chat(auth_api).json()
    assert body["reply"] == "Hello! How can I help you?"
    assert len(fake_gemini.calls) == 2


def test_nonstream_503_exhausted_maps_to_502(auth_api, failing_gemini):
    response = _chat(auth_api)
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "ai_unavailable"


def test_stream_retries_transient_503(auth_api, fake_gemini, monkeypatch):
    from app.services import gemini as gemini_module

    monkeypatch.setattr(gemini_module, "_RETRY_BACKOFF_SECONDS", 0.0)
    fake_gemini.transient_failures = 1
    response = auth_api.post("/api/v1/chat/stream", json={"message": "Hi"})
    assert response.status_code == 200
    assert len(fake_gemini.calls) == 2


def test_continuation_failure_keeps_partial_reply(auth_api, fake_gemini):
    fake_gemini.reply_queue = ["partial answer that is already useful"]
    fake_gemini.reply_reason_queue = ["MAX_TOKENS"]
    fake_gemini.fail_after_calls = 0  # the continuation call fails permanently
    body = _chat(auth_api).json()
    assert "partial answer" in body["reply"]  # partial text preserved, no error
    assert len(fake_gemini.calls) == 2
