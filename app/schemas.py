from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(description="Message role, e.g. user or assistant.")
    content: str = Field(min_length=1, description="Message text.")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, description="Latest user prompt.")
    history: list[ChatMessage] = Field(
        default_factory=list,
        description="Previous conversation messages in order.",
    )
    system_prompt: str | None = Field(
        default=None,
        description="Optional instruction to steer the assistant.",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Sampling temperature for Gemini.",
    )
    max_output_tokens: int = Field(
        default=512,
        ge=1,
        le=50000,
        description="Maximum number of tokens to generate.",
    )


class ChatResponse(BaseModel):
    reply: str
    model: str


class HealthResponse(BaseModel):
    status: str
    database_connected: bool
    gemini_configured: bool
