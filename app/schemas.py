"""Pydantic request/response models (uniform error contract lives in errors.py)."""

from datetime import datetime, timezone

from pydantic import BaseModel, EmailStr, Field, field_serializer, field_validator


def _iso_utc(value: datetime) -> str:
    """Serialize a datetime as ISO-8601 with a Z suffix (naive values are UTC)."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


class UtcTimestampModel(BaseModel):
    @field_serializer("*", mode="plain")
    def _serialize_datetime(self, value, info):
        if isinstance(value, datetime):
            return _iso_utc(value)
        return value


# --- Auth ---------------------------------------------------------------------


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)
    display_name: str | None = Field(default=None, max_length=60)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        return value.strip().lower()

    @field_validator("password")
    @classmethod
    def password_strength(cls, value: str) -> str:
        from .config import get_settings

        minimum = get_settings().password_min_chars
        if len(value) < minimum:
            raise ValueError(f"Password must be at least {minimum} characters long.")
        return value


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        return value.strip().lower()


class UserOut(UtcTimestampModel):
    user_id: str
    email: str
    display_name: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class MeResponse(UserOut):
    pass


# --- Chat ---------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, description="Latest user prompt.")
    conversation_id: str | None = Field(
        default=None,
        description="Existing conversation ID. Omit to start a new conversation.",
    )

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        from .config import get_settings

        trimmed = value.strip()
        if not trimmed:
            raise ValueError("Message must not be empty.")
        max_chars = get_settings().max_message_chars
        if len(trimmed) > max_chars:
            raise ValueError(f"Message must be at most {max_chars} characters long.")
        return trimmed


class ChatResponse(UtcTimestampModel):
    reply: str
    model: str
    conversation_id: str
    title: str | None = None
    created_at: datetime
    anon_token: str | None = Field(
        default=None,
        description="Returned only when a new anonymous conversation was created.",
    )


class MessageOut(UtcTimestampModel):
    id: int = Field(validation_alias="message_id")
    role: str
    content: str
    created_at: datetime

    model_config = {"from_attributes": True}


class ConversationSummary(UtcTimestampModel):
    conversation_id: str
    title: str | None
    created_at: datetime
    last_activity_at: datetime
    message_count: int


class ConversationListResponse(BaseModel):
    conversations: list[ConversationSummary]


class ConversationDetailResponse(UtcTimestampModel):
    conversation_id: str
    title: str | None
    created_at: datetime
    messages: list[MessageOut]


class DeleteConversationResponse(BaseModel):
    deleted: bool
    conversation_id: str


class HealthResponse(BaseModel):
    status: str
    database_connected: bool
    gemini_configured: bool
    gemini_model: str
    max_output_tokens: int
    thinking_budget: int
    max_reply_continuations: int
    database_backend: str
