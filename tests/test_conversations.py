"""Conversation lifecycle tests: ownership, list, detail, delete."""

import time


def test_first_chat_creates_conversation_for_user(auth_api, fake_gemini):
    response = auth_api.post("/api/v1/chat/stream", json={"message": "What is HTTP?"})
    assert response.status_code == 200
    conversation_id = response.headers["X-Conversation-Id"]
    assert conversation_id
    assert "X-Conversation-Token" not in response.headers  # users need no anon token

    history = auth_api.get(f"/api/v1/conversations/{conversation_id}").json()
    assert history["title"].startswith("What is HTTP?")
    assert [m["role"] for m in history["messages"]] == ["user", "assistant"]


def test_conversations_listed_newest_first(auth_api, fake_gemini):
    for i in range(3):
        response = auth_api.post(
            "/api/v1/chat", json={"message": f"Conversation number {i} question?"}
        )
        assert response.status_code == 200

    listing = auth_api.get("/api/v1/conversations").json()["conversations"]
    assert len(listing) == 3
    assert listing[0]["title"].startswith("Conversation number 2")  # most recent first
    assert listing[0]["message_count"] == 2  # user + assistant


def test_conversation_detail_auth_scoped(api, registered_user, fake_gemini):
    """Another user cannot read or delete someone else's conversation."""
    # Owner creates a conversation.
    owner_api = api
    owner_api.token = registered_user["token"]
    created = owner_api.post("/api/v1/chat", json={"message": "Private question"}).json()
    conversation_id = created["conversation_id"]

    # Intruder registers and tries to read it.
    intruder = api.post(
        "/api/v1/auth/register", json={"email": "intruder@example.com", "password": "securePass123"}
    ).json()
    api.token = intruder["access_token"]

    read = api.get(f"/api/v1/conversations/{conversation_id}")
    assert read.status_code == 404  # not revealed to non-owners
    delete_attempt = api.delete(f"/api/v1/conversations/{conversation_id}")
    assert delete_attempt.status_code == 404


def test_anon_conversation_token_required(api, fake_gemini):
    """An anonymous conversation without its token becomes inaccessible."""
    created = api.post("/api/v1/chat", json={"message": "Anon question"}).json()
    conversation_id = created["conversation_id"]

    # Same client (token captured) can read it.
    ok = api.get(f"/api/v1/conversations/{conversation_id}")
    assert ok.status_code == 200

    # Without the token: 403.
    saved_token = api.anon_token
    api.anon_token = None
    denied = api.get(f"/api/v1/conversations/{conversation_id}")
    assert denied.status_code == 403

    # Wrong token: 403.
    api.anon_token = "totally-wrong-token"
    wrong = api.get(f"/api/v1/conversations/{conversation_id}")
    assert wrong.status_code == 403
    api.anon_token = saved_token


def test_user_takes_over_no_longer_anonymous(api, registered_user, fake_gemini):
    """A logged-in user's chats are never anonymous (no token needed)."""
    api.token = registered_user["token"]
    created = api.post("/api/v1/chat", json={"message": "Hello"}).json()
    assert created["anon_token"] is None


def test_delete_conversation(auth_api, fake_gemini, started_conversation):
    deleted = auth_api.delete(f"/api/v1/conversations/{started_conversation}")
    assert deleted.status_code == 200
    assert deleted.json() == {"deleted": True, "conversation_id": started_conversation}

    detail = auth_api.get(f"/api/v1/conversations/{started_conversation}")
    assert detail.status_code == 404


def test_anon_conversation_auto_expires(api, fake_gemini, monkeypatch):
    """Anonymous conversations age out; user conversations never do."""
    created = api.post("/api/v1/chat", json={"message": "Ephemeral question"}).json()
    conversation_id = created["conversation_id"]

    from datetime import timedelta

    from app.db import SessionLocal
    from app.models import Conversation, utcnow
    from app.retention import purge_expired_anonymous_conversations

    monkeypatch.setenv("SESSION_EXPIRY_HOURS", "1")
    from app.config import get_settings

    get_settings.cache_clear()
    try:
        db = SessionLocal()
        try:
            record = db.get(Conversation, conversation_id)
            record.last_activity_at = utcnow() - timedelta(minutes=90)
            db.commit()
        finally:
            db.close()

        assert purge_expired_anonymous_conversations() == 1
        gone = api.get(f"/api/v1/conversations/{conversation_id}")
        assert gone.status_code == 404
    finally:
        get_settings.cache_clear()


def test_user_conversations_never_purged(auth_api, fake_gemini, started_conversation):
    from datetime import timedelta

    from app.db import SessionLocal
    from app.models import Conversation, utcnow
    from app.retention import purge_expired_anonymous_conversations

    db = SessionLocal()
    try:
        record = db.get(Conversation, started_conversation)
        record.last_activity_at = utcnow() - timedelta(days=365)
        db.commit()
    finally:
        db.close()

    assert purge_expired_anonymous_conversations() == 0
    ok = auth_api.get(f"/api/v1/conversations/{started_conversation}")
    assert ok.status_code == 200
