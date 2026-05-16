from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .db import check_database_connection
from .schemas import ChatRequest, ChatResponse, HealthResponse
from .services.gemini import GeminiChatService

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=settings.app_description,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "ZedChatBot backend is running."}


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    database_connected = False
    if settings.database_url:
        try:
            database_connected = check_database_connection()
        except Exception:
            database_connected = False

    return HealthResponse(
        status="ok",
        database_connected=database_connected,
        gemini_configured=bool(settings.gemini_api_key),
    )


@app.post("/api/v1/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    service = GeminiChatService()
    reply, model = service.generate_reply(payload)
    return ChatResponse(reply=reply, model=model)
