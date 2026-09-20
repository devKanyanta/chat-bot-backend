"""FR-004 input validation tests (anonymous caller)."""

MAX = 2000


def test_empty_message_rejected(api):
    response = api.post("/api/v1/chat", json={"message": ""})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_whitespace_message_rejected(api):
    response = api.post("/api/v1/chat", json={"message": "   \n\t  "})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_too_long_message_rejected(api, fake_gemini, monkeypatch):
    monkeypatch.setenv("MAX_MESSAGE_CHARS", "50")
    from app.config import get_settings

    get_settings.cache_clear()
    try:
        response = api.post("/api/v1/chat", json={"message": "x" * 51})
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_error"

        ok = api.post("/api/v1/chat", json={"message": "x" * 50})
        assert ok.status_code == 200
    finally:
        get_settings.cache_clear()


def test_missing_body_rejected(api):
    response = api.post("/api/v1/chat", json={})
    assert response.status_code == 422


def test_error_contract_shape(api):
    response = api.post("/api/v1/chat", json={"message": ""})
    body = response.json()
    assert set(body.keys()) == {"error"}
    assert set(body["error"].keys()) == {"code", "message"}
    assert "traceback" not in body["error"]["message"].lower()
