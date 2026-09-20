"""Retention sweeper tests: anonymous conversations expire; user chats persist."""

from datetime import timedelta

import app.retention
from app.models import Conversation, utcnow


def test_purge_removes_only_stale_anon_conversations(api, fake_gemini):
    created = api.post("/api/v1/chat", json={"message": "Ephemeral"}).json()
    conversation_id = created["conversation_id"]

    # A fresh second conversation must survive.
    fresh = api.post("/api/v1/chat", json={"message": "Fresh chat"}).json()

    from app.db import SessionLocal

    db = SessionLocal()
    try:
        record = db.get(Conversation, conversation_id)
        record.last_activity_at = utcnow() - timedelta(hours=25)
        db.commit()
    finally:
        db.close()

    removed = app.retention.purge_expired_anonymous_conversations()
    assert removed == 1

    assert api.get(f"/api/v1/conversations/{conversation_id}").status_code == 404
    ok = api.get(f"/api/v1/conversations/{fresh['conversation_id']}")
    assert ok.status_code == 200


def test_purge_respects_expiry_window(api, fake_gemini, monkeypatch):
    created = api.post("/api/v1/chat", json={"message": "Window test"}).json()
    conversation_id = created["conversation_id"]

    monkeypatch.setenv("SESSION_EXPIRY_HOURS", "1")
    from app.config import get_settings

    get_settings.cache_clear()
    try:
        from app.db import SessionLocal

        db = SessionLocal()
        try:
            record = db.get(Conversation, conversation_id)
            record.last_activity_at = utcnow() - timedelta(minutes=90)
            db.commit()
        finally:
            db.close()

        assert app.retention.purge_expired_anonymous_conversations() == 1
        assert api.get(f"/api/v1/conversations/{conversation_id}").status_code == 404
    finally:
        get_settings.cache_clear()


def test_purge_noop_when_nothing_stale(api, fake_gemini):
    api.post("/api/v1/chat", json={"message": "Recent"})
    assert app.retention.purge_expired_anonymous_conversations() == 0


def test_user_conversations_never_purged(auth_api, fake_gemini, monkeypatch):
    response = auth_api.post("/api/v1/chat", json={"message": "User owned chat"})
    conversation_id = response.headers["X-Conversation-Id"]

    from app.db import SessionLocal

    db = SessionLocal()
    try:
        record = db.get(Conversation, conversation_id)
        record.last_activity_at = utcnow() - timedelta(days=365)
        db.commit()
    finally:
        db.close()

    assert app.retention.purge_expired_anonymous_conversations() == 0
    ok = auth_api.get(f"/api/v1/conversations/{conversation_id}")
    assert ok.status_code == 200
