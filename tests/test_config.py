"""Regression tests for CORS_ORIGINS parsing (pydantic-settings NoDecode).

Without NoDecode, pydantic-settings JSON-decodes list fields from .env / OS
env before our validator runs, crashing on values like "*" (SettingsError:
error parsing value for field "cors_origins").
"""

import pytest

from app.config import Settings


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("*", ["*"]),
        ("", ["*"]),
        ("http://a.com,http://b.com", ["http://a.com", "http://b.com"]),
        ('["http://a.com", "http://b.com"]', ["http://a.com", "http://b.com"]),
        ("  http://a.com  ,  ", ["http://a.com"]),
    ],
)
def test_cors_origins_from_env_string(raw, expected, monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", raw)
    assert Settings(_env_file=None).cors_origins == expected


def test_cors_origins_default(monkeypatch):
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    assert Settings(_env_file=None).cors_origins == ["*"]


def test_cors_origins_from_dotenv_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("CORS_ORIGINS=*\n")
    assert Settings(_env_file=str(env_file)).cors_origins == ["*"]
