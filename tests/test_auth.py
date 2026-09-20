"""Auth endpoint tests: register, login, me, JWT rejection cases."""

import time

from app.security import create_access_token


def test_register_returns_token_and_user(api):
    response = api.post(
        "/api/v1/auth/register",
        json={"email": "new@example.com", "password": "securePass123", "display_name": "New"},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["access_token"]
    assert body["token_type"] == "bearer"
    assert body["user"]["email"] == "new@example.com"
    assert body["user"]["display_name"] == "New"
    assert "password" not in body["user"]
    assert "password_hash" not in body["user"]


def test_register_duplicate_email_conflict(api):
    payload = {"email": "dupe@example.com", "password": "securePass123"}
    assert api.post("/api/v1/auth/register", json=payload).status_code == 201
    response = api.post("/api/v1/auth/register", json=payload)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "email_taken"


def test_register_rejects_short_password(api):
    response = api.post(
        "/api/v1/auth/register", json={"email": "short@example.com", "password": "short"}
    )
    assert response.status_code == 422


def test_register_rejects_invalid_email(api):
    response = api.post(
        "/api/v1/auth/register", json={"email": "not-an-email", "password": "securePass123"}
    )
    assert response.status_code == 422


def test_login_success(api, registered_user):
    response = api.post(
        "/api/v1/auth/login",
        json={"email": registered_user["email"], "password": registered_user["password"]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["access_token"]
    assert body["user"]["user_id"] == registered_user["user_id"]


def test_login_wrong_password(api, registered_user):
    response = api.post(
        "/api/v1/auth/login",
        json={"email": registered_user["email"], "password": "wrongPassword1"},
    )
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_credentials"


def test_login_unknown_email(api):
    response = api.post(
        "/api/v1/auth/login", json={"email": "ghost@example.com", "password": "whatever123"}
    )
    assert response.status_code == 401


def test_me_requires_token(api):
    response = api.get("/api/v1/auth/me")
    assert response.status_code == 401


def test_me_returns_user(api, auth_api, registered_user):
    response = auth_api.get("/api/v1/auth/me")
    assert response.status_code == 200
    body = response.json()
    assert body["email"] == registered_user["email"]
    assert body["user_id"] == registered_user["user_id"]


def test_me_rejects_garbage_token(api):
    response = api.get("/api/v1/auth/me", headers={"Authorization": "Bearer not.a.jwt"})
    assert response.status_code == 401


def test_me_rejects_expired_token(api, registered_user, monkeypatch):
    monkeypatch.setenv("ACCESS_TOKEN_EXPIRE_MINUTES", "-1")  # already expired
    from app.config import get_settings

    get_settings.cache_clear()
    try:
        expired = create_access_token(registered_user["user_id"])
    finally:
        get_settings.cache_clear()

    response = api.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {expired}"})
    assert response.status_code == 401


def test_me_rejects_token_signed_with_wrong_secret(api, registered_user):
    import jwt

    forged = jwt.encode(
        {"sub": registered_user["user_id"], "iat": int(time.time()), "exp": int(time.time()) + 600},
        "wrong-secret",
        algorithm="HS256",
    )
    response = api.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {forged}"})
    assert response.status_code == 401


def test_passwords_are_hashed_not_stored(api, registered_user):
    from app.db import SessionLocal
    from app.models import User

    db = SessionLocal()
    try:
        user = db.get(User, registered_user["user_id"])
        assert user.password_hash != registered_user["password"]
        assert user.password_hash.startswith("$2")  # bcrypt
    finally:
        db.close()
