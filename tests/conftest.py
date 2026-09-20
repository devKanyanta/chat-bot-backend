"""Shared fixtures: per-test SQLite DB, scriptable fake Gemini, API helpers."""

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.services import gemini as gemini_service  # noqa: E402


# --- Gemini test doubles ------------------------------------------------------


class Fake503(Exception):
    """Duck-typed transient upstream error (google.genai ServerError shape)."""

    code = 503


class FakeCandidate:
    def __init__(self, finish_reason=None):
        self.finish_reason = finish_reason


class FakeFinishReason:
    def __init__(self, name):
        self.name = name


class FakeChunk:
    def __init__(self, text, candidates=None):
        self.text = text
        self.candidates = candidates or []


class FakeResponse:
    def __init__(self, text, finish_reason="STOP"):
        self.text = text
        self.candidates = [FakeCandidate(FakeFinishReason(finish_reason))]


class FakeModels:
    """Scriptable stand-in for google-genai client.models.

    Attributes:
        replies: non-streaming reply texts. `reply_queue` (popped per call)
            takes precedence; otherwise `replies[0]` is returned every time.
        reply_queue / reply_reason_queue: scripted replies + finish reasons.
        stream_segments: streaming script — list of (chunks, finish_reason),
            popped per call. None = legacy single canned reply.
        fail: raise a permanent (non-transient) error on every call.
        fail_after_calls: fail permanently once more than N successful calls
            have happened.
        transient_failures: raise a 503-shaped error on the next N calls.
        stream_fail_after: per stream call, yield N chunks then raise Fake503
            (simulates a mid-stream dropout). Entries align with stream_segments.
    """

    def __init__(
        self,
        replies=None,
        fail: bool = False,
        stream_segments=None,
        transient_failures: int = 0,
        fail_after_calls=None,
    ) -> None:
        self.replies = replies if replies is not None else ["Hello! How can I help you?"]
        self.reply_queue: list[str] = []
        self.reply_reason_queue: list[str] = []
        self.fail = fail
        self.stream_segments = stream_segments
        self.transient_failures = transient_failures
        self.fail_after_calls = fail_after_calls
        self.stream_fail_after = None
        self.calls = []

    def _maybe_raise(self):
        # self.calls already includes the current call, so "more than N
        # successful prior calls" means len(self.calls) - 1 > N.
        if self.fail:
            raise RuntimeError("simulated gemini outage")
        if self.fail_after_calls is not None and (len(self.calls) - 1) > self.fail_after_calls:
            raise RuntimeError("simulated gemini outage after N calls")
        if self.transient_failures > 0:
            self.transient_failures -= 1
            raise Fake503("simulated 503 high demand")

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        self._maybe_raise()
        if self.reply_queue:
            reply = self.reply_queue.pop(0)
            reason = self.reply_reason_queue.pop(0) if self.reply_reason_queue else "STOP"
        else:
            reply = self.replies[0] if self.replies else ""
            reason = "STOP"
        return FakeResponse(reply, reason)

    def generate_content_stream(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        self._maybe_raise()
        if self.stream_segments is not None:
            chunks, reason = (
                self.stream_segments.pop(0) if self.stream_segments else ([], "STOP")
            )
            fail_after = self.stream_fail_after.pop(0) if self.stream_fail_after else None
            produced = 0
            for part in chunks:
                yield FakeChunk(part)
                produced += 1
                if fail_after is not None and produced >= fail_after:
                    raise Fake503("simulated mid-stream dropout")
            # Finish reasons arrive on a final (possibly empty) chunk.
            if reason:
                yield FakeChunk("", candidates=[FakeCandidate(FakeFinishReason(reason))])
            return
        for part in ("Hel", "lo!", " How can I help?"):
            yield FakeChunk(part)


class FakeClient:
    def __init__(self, models: FakeModels) -> None:
        self.models = models


class GeminiFactory:
    """Callable used as get_client(); swap behavior between requests via .models."""

    def __init__(self, models: FakeModels) -> None:
        self.models = models

    def __call__(self) -> FakeClient:
        return FakeClient(self.models)


@pytest.fixture()
def fake_gemini(monkeypatch):
    """Replace the Gemini client getter with a controllable fake; returns FakeModels."""
    models = FakeModels()
    monkeypatch.setattr(gemini_service, "get_client", GeminiFactory(models))
    return models


@pytest.fixture()
def failing_gemini(monkeypatch):
    """Gemini fake whose calls raise permanently (simulates upstream outage)."""
    models = FakeModels(fail=True)
    monkeypatch.setattr(gemini_service, "get_client", GeminiFactory(models))
    return models


# --- API client with auth handling --------------------------------------------


class ApiClient:
    """Thin wrapper over TestClient that injects auth/anon headers and
    auto-captures anonymous conversation tokens from response headers."""

    def __init__(self, test_client: TestClient) -> None:
        self.tc = test_client
        self.token: str | None = None
        self.anon_token: str | None = None

    def _headers(self, headers):
        merged = {}
        if self.token:
            merged["Authorization"] = f"Bearer {self.token}"
        if self.anon_token:
            merged["X-Conversation-Token"] = self.anon_token
        merged.update(headers or {})
        return merged

    def _capture(self, response):
        token = response.headers.get("X-Conversation-Token")
        if token:
            self.anon_token = token
        return response

    def post(self, path, headers=None, **kwargs):
        return self._capture(self.tc.post(path, headers=self._headers(headers), **kwargs))

    def get(self, path, headers=None, **kwargs):
        return self._capture(self.tc.get(path, headers=self._headers(headers), **kwargs))

    def delete(self, path, headers=None, **kwargs):
        return self._capture(self.tc.delete(path, headers=self._headers(headers), **kwargs))


# --- core fixtures -------------------------------------------------------------


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """TestClient bound to a per-test SQLite DB with overridden settings.

    app.db builds its engine at import time, so the db/retention/main modules
    are reloaded here to bind every test to its own fresh database file.
    """
    db_file = tmp_path / "test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_file}")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "1000")
    monkeypatch.setenv("JWT_SECRET", "test-secret-key-for-tests-only")
    get_settings.cache_clear()

    import app.db as app_db
    import app.retention as app_retention
    import app.main as app_main

    # Reload in dependency order so every module-level `from .db import ...`
    # binding is refreshed against the new per-test engine.
    importlib.reload(app_db)
    importlib.reload(app_retention)
    importlib.reload(app_main)
    app = app_main.app

    from app.ratelimit import limiter

    limiter._hits.clear()

    with TestClient(app) as test_client:
        yield test_client

    get_settings.cache_clear()


@pytest.fixture()
def api(client):
    """ApiClient wrapper without credentials (anonymous)."""
    return ApiClient(client)


@pytest.fixture()
def registered_user(api):
    """Register a user via the API; returns dict with user info + token."""
    response = api.post(
        "/api/v1/auth/register",
        json={
            "email": "annie@example.com",
            "password": "securePass123",
            "display_name": "Annie",
        },
    )
    assert response.status_code == 201, response.text
    body = response.json()
    return {
        "user_id": body["user"]["user_id"],
        "email": body["user"]["email"],
        "display_name": body["user"]["display_name"],
        "token": body["access_token"],
        "password": "securePass123",
    }


@pytest.fixture()
def auth_api(api, registered_user):
    """ApiClient authenticated as the registered user."""
    api.token = registered_user["token"]
    return api


@pytest.fixture()
def started_conversation(auth_api, fake_gemini):
    """Send one streamed chat message as the registered user; returns conv id."""
    response = auth_api.post("/api/v1/chat/stream", json={"message": "Hi there"})
    assert response.status_code == 200, response.text
    return response.headers["X-Conversation-Id"]
