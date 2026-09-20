"""End-to-end chat flow (user + anonymous), streaming, and rate limiting."""


def _sse_text(response) -> str:
    return "".join(
        line[6:] for line in response.text.split("\n") if line.startswith("data: ")
    )


def test_user_chat_persists_and_titles(auth_api, fake_gemini):
    body = auth_api.post("/api/v1/chat", json={"message": "Explain the ORM"}).json()
    assert body["reply"] == "Hello! How can I help you?"
    assert body["model"]
    assert body["conversation_id"]
    assert body["title"] == "Explain the ORM"
    assert body["anon_token"] is None
    assert body["created_at"].endswith("Z")


def test_streaming_chat_for_user(auth_api, fake_gemini):
    response = auth_api.post("/api/v1/chat/stream", json={"message": "Hello streamed"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert _sse_text(response) == "Hello! How can I help?"
    assert "X-Conversation-Token" not in response.headers


def test_gemini_receives_context_from_history(auth_api, fake_gemini):
    first = auth_api.post("/api/v1/chat", json={"message": "What is HTTP?"}).json()
    auth_api.post(
        "/api/v1/chat",
        json={"message": "And HTTPS?", "conversation_id": first["conversation_id"]},
    )

    contents = fake_gemini.calls[1]["contents"]
    # user("What is HTTP?") + assistant(reply) + user("And HTTPS?")
    assert len(contents) == 3
    assert contents[0].parts[0].text == "What is HTTP?"
    assert contents[1].role == "model"
    assert contents[2].parts[0].text == "And HTTPS?"


def test_gemini_receives_system_prompt_and_params(auth_api, fake_gemini):
    from app.config import get_settings

    auth_api.post("/api/v1/chat", json={"message": "Hello"})
    config = fake_gemini.calls[0]["config"]
    assert "ZedChatBot" in config.system_instruction
    assert config.temperature == 0.7
    assert config.max_output_tokens == get_settings().max_output_tokens
    assert config.thinking_config.thinking_budget == get_settings().thinking_budget
    assert fake_gemini.calls[0]["model"] == get_settings().gemini_model


def test_gemini_error_maps_to_502(auth_api, failing_gemini):
    response = auth_api.post("/api/v1/chat", json={"message": "Hello"})
    assert response.status_code == 502
    body = response.json()
    assert body["error"]["code"] == "ai_unavailable"
    assert "outage" not in body["error"]["message"]


def test_no_api_key_maps_to_503(auth_api, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    from app.config import get_settings

    get_settings.cache_clear()
    try:
        from app.services import gemini as gemini_service

        gemini_service.reset_client()
        response = auth_api.post("/api/v1/chat", json={"message": "Hello"})
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "service_not_configured"
    finally:
        get_settings.cache_clear()
        gemini_service.reset_client()


def test_rate_limit_returns_429(api, fake_gemini, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "2")
    from app.config import get_settings

    get_settings.cache_clear()
    from app.ratelimit import limiter

    limiter._hits.clear()
    try:
        assert api.post("/api/v1/chat", json={"message": "one"}).status_code == 200
        assert api.post("/api/v1/chat", json={"message": "two"}).status_code == 200

        third = api.post("/api/v1/chat", json={"message": "three"})
        assert third.status_code == 429
        assert third.json()["error"]["code"] == "rate_limited"
        assert third.headers.get("retry-after") == "60"
    finally:
        get_settings.cache_clear()
        limiter._hits.clear()


def test_history_trimming_to_context_turns(auth_api, fake_gemini, monkeypatch):
    monkeypatch.setenv("HISTORY_CONTEXT_TURNS", "2")
    from app.config import get_settings

    get_settings.cache_clear()
    try:
        first = auth_api.post("/api/v1/chat", json={"message": "q1"}).json()
        for q in ("q2", "q3", "q4"):
            auth_api.post(
                "/api/v1/chat",
                json={"message": q, "conversation_id": first["conversation_id"]},
            )

        contents = fake_gemini.calls[-1]["contents"]
        # Last 2 stored turns before q4: assistant(q3 reply), user(q3) + new q4
        assert len(contents) == 3
        assert contents[-1].parts[0].text == "q4"
    finally:
        get_settings.cache_clear()


def test_failed_chat_keeps_user_message_but_no_assistant(auth_api, failing_gemini):
    """The user's message is stored (retryable); no assistant reply is faked."""
    response = auth_api.post("/api/v1/chat", json={"message": "Hello"})
    assert response.status_code == 502
    conversation_id = response.headers["X-Conversation-Id"]

    listing = auth_api.get("/api/v1/conversations").json()["conversations"]
    assert len(listing) == 1 and listing[0]["message_count"] == 1

    detail = auth_api.get(f"/api/v1/conversations/{conversation_id}").json()
    assert [m["role"] for m in detail["messages"]] == ["user"]
