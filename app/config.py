from functools import lru_cache
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

DEFAULT_SYSTEM_PROMPT = (
    "You are ZedChatBot, a friendly, general-purpose assistant for users in Zambia. "
    "Answer clearly, helpfully, and accurately in English. You help with everyday "
    "questions and academic study. If you are not sure about something, say so and "
    "suggest how the user can verify it. Keep responses concise and easy to understand. "
    "Whenever your answer includes mathematics, always write it in LaTeX: use $...$ for "
    "inline math and $$...$$ for displayed equations (for example $x^2 + 1$, "
    "$$\\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$$). Never describe math in plain text when "
    "proper notation is possible, and show worked steps one equation per line."
)


class Settings(BaseSettings):
    app_name: str = Field(default="ZedChatBot API", alias="APP_NAME")
    app_version: str = Field(default="1.0.0", alias="APP_VERSION")
    app_description: str = Field(
        default="Backend API for the ZedChatBot general-purpose chatbot.",
        alias="APP_DESCRIPTION",
    )

    # AI service
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.5-flash", alias="GEMINI_MODEL")
    # Backup models, tried in order when the primary cannot serve a request:
    # Google-side overload (503 "currently experiencing high demand"), a retired
    # or unavailable model name (404) or exhausted quota (429). Space for
    # resilience so one saturated model does not take chat down.
    gemini_fallback_models: Annotated[list[str], NoDecode] = Field(
        default=[], alias="GEMINI_FALLBACK_MODELS"
    )
    system_prompt: str = Field(default=DEFAULT_SYSTEM_PROMPT, alias="SYSTEM_PROMPT")
    temperature: float = Field(default=0.7, alias="TEMPERATURE")
    max_output_tokens: int = Field(default=8192, alias="MAX_OUTPUT_TOKENS")
    # Gemini 2.5 models "think" before answering and the thinking tokens count
    # against max_output_tokens. 0 disables thinking entirely (best for a
    # concise chatbot); -1 lets the model decide dynamically.
    thinking_budget: int = Field(default=0, alias="THINKING_BUDGET")
    # Times the backend may auto-continue a reply that hit the output cap,
    # so long answers stream to completion instead of stopping mid-sentence.
    max_reply_continuations: int = Field(default=4, alias="MAX_REPLY_CONTINUATIONS")

    # Database
    database_url: str = Field(default="sqlite:///./zedchatbot.db", alias="DATABASE_URL")
    db_echo: bool = Field(default=False, alias="DB_ECHO")

    # Auth / accounts
    jwt_secret: str = Field(
        default="change-me-in-production-9f2c1b7a4e6d8f3a1b5c7d9e2f4a6b8c",
        alias="JWT_SECRET",
    )
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(default=43200, alias="ACCESS_TOKEN_EXPIRE_MINUTES")
    # Min length for registration passwords.
    password_min_chars: int = Field(default=8, alias="PASSWORD_MIN_CHARS")

    # Sessions / retention
    history_context_turns: int = Field(default=20, alias="HISTORY_CONTEXT_TURNS")
    session_expiry_hours: int = Field(default=24, alias="SESSION_EXPIRY_HOURS")
    retention_sweep_minutes: int = Field(default=10, alias="RETENTION_SWEEP_MINUTES")

    # API behavior
    rate_limit_per_minute: int = Field(default=30, alias="RATE_LIMIT_PER_MINUTE")
    max_message_chars: int = Field(default=2000, alias="MAX_MESSAGE_CHARS")

    # NoDecode: the raw string arrives untouched (from .env or OS env) and is
    # parsed by the validator below. Without this, pydantic-settings tries
    # json.loads() first and crashes on values like "*" or "a,b".
    cors_origins: Annotated[list[str], NoDecode] = Field(default=["*"], alias="CORS_ORIGINS")  # type: ignore[assignment]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    @property
    def model_chain(self) -> list[str]:
        """The primary model followed by its fallbacks, order kept, de-duplicated."""
        chain: list[str] = []
        for model in (self.gemini_model, *self.gemini_fallback_models):
            name = (model or "").strip()
            if name and name not in chain:
                chain.append(name)
        return chain

    @field_validator("gemini_fallback_models", mode="before")
    @classmethod
    def parse_model_list(cls, value: object) -> list[str]:
        """Accept a comma-separated list, a JSON array, or an empty value."""
        if value is None:
            return []
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            if text.startswith("["):
                import json

                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, list):
                        return [str(item).strip() for item in parsed if str(item).strip()]
                except ValueError:
                    pass
            return [item.strip() for item in text.split(",") if item.strip()]
        if isinstance(value, (list, tuple)):
            return [str(item).strip() for item in value if str(item).strip()]
        return []

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: object) -> list[str]:
        """Accept "*", empty, comma-separated lists, and JSON arrays."""
        if value is None:
            return ["*"]
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return ["*"]
            if text.startswith("["):
                import json

                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, list):
                        return [str(item).strip() for item in parsed if str(item).strip()]
                except ValueError:
                    pass
            if text == "*":
                return ["*"]
            return [item.strip() for item in text.split(",") if item.strip()]
        if isinstance(value, (list, tuple)):
            return [str(item).strip() for item in value if str(item).strip()]
        return ["*"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
