"""Unit tests for the Gemini service module (mocked client)."""

import httpx
import pytest

from app.errors import AINotConfiguredError, AIUnavailableError
from app.services import gemini
from tests.conftest import (
    Fake503,
    FakeCandidate,
    FakeChunk,
    FakeFinishReason,
    FakeResponse,
    GeminiFactory,
)


class _SelectiveModels:
    """Fake models where only some model names can serve a request.

    ``broken`` models answer with an upstream 503 (overloaded); ``hard_fail``
    models raise an error no other model could fix.
    """

    def __init__(self, broken=(), hard_fail=()) -> None:
        self.broken = set(broken)
        self.hard_fail = set(hard_fail)
        self.calls: list[str] = []

    def _maybe_raise(self, model: str) -> None:
        if model in self.hard_fail:
            raise RuntimeError("simulated malformed request")
        if model in self.broken:
            raise Fake503("simulated 503 high demand")

    def generate_content(self, *, model, contents, config):
        self.calls.append(model)
        self._maybe_raise(model)
        return FakeResponse("Served by the backup model.")

    def generate_content_stream(self, *, model, contents, config):
        self.calls.append(model)
        self._maybe_raise(model)
        for part in ("Back", "up", " model"):
            yield FakeChunk(part)


def _install_selective_gemini(monkeypatch, *, broken=(), hard_fail=()):
    """Point the service at a primary/fallback pair of scripted models."""
    from app.config import get_settings

    monkeypatch.setenv("GEMINI_MODEL", "primary-model")
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "backup-model")
    monkeypatch.setattr(gemini, "_RETRY_BACKOFF_SECONDS", 0.0)
    get_settings.cache_clear()
    gemini.reset_client()
    models = _SelectiveModels(broken=broken, hard_fail=hard_fail)
    monkeypatch.setattr(gemini, "get_client", GeminiFactory(models))
    return models


@pytest.fixture()
def overloaded_primary(monkeypatch):
    """The configured primary model 503s; the fallback model serves instead."""
    from app.config import get_settings

    models = _install_selective_gemini(monkeypatch, broken={"primary-model"})
    try:
        yield models
    finally:
        get_settings.cache_clear()
        gemini.reset_client()


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


def test_generate_reply_falls_back_when_primary_is_overloaded(overloaded_primary):
    reply, model = gemini.generate_reply([], "Hi")
    assert reply == "Served by the backup model."
    # The reply reports the model that actually served it, not the primary.
    assert model == "backup-model"
    # Primary retried before being skipped, then the fallback was used.
    assert overloaded_primary.calls.count("primary-model") == 3
    assert overloaded_primary.calls[-1] == "backup-model"


def test_stream_reply_falls_back_when_primary_is_overloaded(overloaded_primary):
    chunks = list(gemini.stream_reply([], "Hi"))
    assert "".join(chunks) == "Backup model"
    assert overloaded_primary.calls[-1] == "backup-model"


def test_model_chain_keeps_primary_first_and_drops_duplicates(monkeypatch):
    from app.config import get_settings

    monkeypatch.setenv("GEMINI_MODEL", "primary-model")
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "backup-model,primary-model,,backup-model")
    get_settings.cache_clear()
    try:
        assert get_settings().model_chain == ["primary-model", "backup-model"]
    finally:
        get_settings.cache_clear()


# --- Transport failures (DNS/connection) -----------------------------------


class _FlakyTransportModels:
    """Fake models whose next ``failures`` calls die before reaching Google.

    Mirrors the real ``ConnectError: [Errno -3] Temporary failure in name
    resolution`` seen when the LAN's DNS resolver blinks.
    """

    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls: list[str] = []

    def _maybe_raise(self) -> None:
        if self.failures > 0:
            self.failures -= 1
            raise httpx.ConnectError("[Errno -3] Temporary failure in name resolution")

    def generate_content(self, *, model, contents, config):
        self.calls.append(model)
        self._maybe_raise()
        return FakeResponse("Recovered reply.")

    def generate_content_stream(self, *, model, contents, config):
        self.calls.append(model)
        self._maybe_raise()
        for part in ("Rec", "overed"):
            yield FakeChunk(part)


def _install_flaky_transport(monkeypatch, *, failures: int):
    """Primary/fallback pair where the network itself is failing."""
    from app.config import get_settings

    monkeypatch.setenv("GEMINI_MODEL", "primary-model")
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "backup-model")
    monkeypatch.setattr(gemini, "_RETRY_BACKOFF_SECONDS", 0.0)
    get_settings.cache_clear()
    gemini.reset_client()
    models = _FlakyTransportModels(failures)
    monkeypatch.setattr(gemini, "get_client", GeminiFactory(models))
    return models


def test_transport_error_is_retried_on_the_same_model(monkeypatch):
    from app.config import get_settings

    # A DNS blip clears within one retry, so the user still gets a reply.
    models = _install_flaky_transport(monkeypatch, failures=1)
    try:
        reply, model = gemini.generate_reply([], "Hi")
        assert reply == "Recovered reply."
        assert model == "primary-model"
        assert models.calls == ["primary-model", "primary-model"]
    finally:
        get_settings.cache_clear()
        gemini.reset_client()


def test_stream_retries_transport_error_before_the_first_chunk(monkeypatch):
    from app.config import get_settings

    models = _install_flaky_transport(monkeypatch, failures=1)
    try:
        assert "".join(gemini.stream_reply([], "Hi")) == "Recovered"
        assert models.calls == ["primary-model", "primary-model"]
    finally:
        get_settings.cache_clear()
        gemini.reset_client()


def test_transport_error_does_not_walk_the_model_chain(monkeypatch):
    from app.config import get_settings

    # Every model shares one endpoint, so a dead network is not a model problem:
    # retry the primary, then fail rather than blame each model in turn.
    models = _install_flaky_transport(monkeypatch, failures=99)
    try:
        with pytest.raises(AIUnavailableError):
            gemini.generate_reply([], "Hi")
        assert models.calls == ["primary-model"] * gemini._INITIAL_ATTEMPTS
    finally:
        get_settings.cache_clear()
        gemini.reset_client()


class _MidStreamDropModels:
    """Fake models whose first stream drops mid-generation on a network error."""

    def __init__(self) -> None:
        self.calls: list[list] = []

    def generate_content_stream(self, *, model, contents, config):
        self.calls.append(contents)
        if len(self.calls) == 1:
            yield FakeChunk("half an answer")
            raise httpx.ConnectError("[Errno -3] Temporary failure in name resolution")
        yield FakeChunk(" that resumes")
        yield FakeChunk("", candidates=[FakeCandidate(FakeFinishReason("STOP"))])


def test_stream_resumes_after_a_midstream_transport_error(monkeypatch):
    from app.config import get_settings

    monkeypatch.setattr(gemini, "_RETRY_BACKOFF_SECONDS", 0.0)
    get_settings.cache_clear()
    gemini.reset_client()
    models = _MidStreamDropModels()
    monkeypatch.setattr(gemini, "get_client", GeminiFactory(models))
    try:
        text = "".join(gemini.stream_reply([], "Hi"))
        assert text == "half an answer that resumes"
        # The resume hands the model the partial text so it can carry on.
        resumed = models.calls[1]
        assert resumed[-2].role == "model"
        assert resumed[-2].parts[0].text.strip() == "half an answer"
        assert resumed[-1].role == "user"
    finally:
        get_settings.cache_clear()
        gemini.reset_client()


def test_generation_config_disables_automatic_function_calling():
    # This service declares no tools, so AFC has nothing to call; leaving it on
    # only wraps every request and logs a "not recommended" warning per process.
    from google.genai import _extra_utils

    assert _extra_utils.should_disable_afc(gemini._generation_config()) is True


def test_unfixable_error_does_not_try_a_fallback(monkeypatch):
    from app.config import get_settings

    # A malformed request fails the same way on every model, so there is no
    # point switching models: fail fast and only touch the primary.
    models = _install_selective_gemini(monkeypatch, hard_fail={"primary-model"})
    try:
        with pytest.raises(AIUnavailableError):
            gemini.generate_reply([], "Hi")
        assert models.calls == ["primary-model"]
    finally:
        get_settings.cache_clear()
        gemini.reset_client()
