"""ZedChatBot FastAPI application.

Implements (URD sections in parentheses):
- Accounts: register / login / me with bcrypt + JWT
- Anonymous chat: no account needed; anonymous conversations expire (8.3)
- User conversations: kept until deleted (FR-008/009/010/015)
- Chat: JSON + SSE streaming with auto-continuation and mid-stream resume
- Uniform, non-leaking error contract (FR-011, UC-04)
"""

import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from sqlalchemy import func
from sqlalchemy.orm import Session as DBSession

from . import errors as errors_module
from .config import get_settings
from .db import SessionLocal, check_database_connection, init_db
from .deps import (
    ANON_TOKEN_HEADER,
    CurrentAuth,
    check_rate_limit,
    db_dep,
    generate_anon_token,
    get_current_auth,
    get_current_user,
    load_history,
    resolve_conversation_access,
    store_message,
)
from .errors import AIUnavailableError, ApiError
from .models import Conversation, Message, User, utcnow
from .ratelimit import limiter
from .retention import retention_task
from .schemas import (
    AuthResponse,
    ChatRequest,
    ChatResponse,
    ConversationDetailResponse,
    ConversationListResponse,
    ConversationSummary,
    DeleteConversationResponse,
    HealthResponse,
    LoginRequest,
    MeResponse,
    MessageOut,
    RegisterRequest,
)
from .security import create_access_token, hash_password, verify_password
from .services import gemini

logger = logging.getLogger(__name__)

STREAM_ERROR_PREFIX = "[error] "
TITLE_MAX_CHARS = 80


def _sse_event(text: str) -> str:
    """Format one SSE event; multi-line text becomes multiple data: fields."""
    return "".join(f"data: {line}\n" for line in text.split("\n")) + "\n"


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    retention_task.start()
    logger.info("ZedChatBot backend started (model=%s).", get_settings().gemini_model)
    yield
    await retention_task.stop()


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=settings.app_description,
    lifespan=lifespan,
)

cors_headers = [
    "X-Conversation-Id",
    "X-Conversation-Token",
    "X-Session-Id",
    "X-Session-Token",
    "Retry-After",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=cors_headers,
)

errors_module.register_handlers(app)


@app.middleware("http")
async def db_session_middleware(request: Request, call_next):
    request.state.db = SessionLocal()
    try:
        return await call_next(request)
    finally:
        request.state.db.close()


# --- basic endpoints ---------------------------------------------------------


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "ZedChatBot backend is running."}


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    database_connected = False
    try:
        database_connected = check_database_connection()
    except Exception:
        logger.exception("Database health check failed.")
    s = get_settings()
    return HealthResponse(
        status="ok",
        database_connected=database_connected,
        gemini_configured=bool(s.gemini_api_key),
        gemini_model=s.gemini_model,
        max_output_tokens=s.max_output_tokens,
        thinking_budget=s.thinking_budget,
        max_reply_continuations=s.max_reply_continuations,
        database_backend=s.database_url.split("://", 1)[0],
    )


# --- auth endpoints ----------------------------------------------------------


def _auth_response(user: User) -> AuthResponse:
    return AuthResponse(
        access_token=create_access_token(user.user_id),
        token_type="bearer",
        user=user,
    )


@app.post("/api/v1/auth/register", response_model=AuthResponse, status_code=201)
def register(payload: RegisterRequest, request: Request, db: DBSession = Depends(db_dep)):
    existing = db.query(User).filter(User.email == payload.email).first()
    if existing is not None:
        raise ApiError(
            message="An account with this email already exists. Try signing in instead.",
            status_code=409,
            code="email_taken",
        )
    user = User(
        user_id=str(uuid.uuid4()),
        email=payload.email,
        password_hash=hash_password(payload.password),
        display_name=(payload.display_name or payload.email.split("@", 1)[0])[:60],
        created_at=utcnow(),
        last_login_at=utcnow(),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return _auth_response(user)


@app.post("/api/v1/auth/login", response_model=AuthResponse)
def login(payload: LoginRequest, db: DBSession = Depends(db_dep)):
    user = db.query(User).filter(User.email == payload.email).first()
    if user is None or not verify_password(payload.password, user.password_hash):
        raise ApiError(
            message="Incorrect email or password.",
            status_code=401,
            code="invalid_credentials",
        )
    user.last_login_at = utcnow()
    db.commit()
    return _auth_response(user)


@app.get("/api/v1/auth/me", response_model=MeResponse)
def me(user: User = Depends(get_current_user)) -> MeResponse:
    return MeResponse(user_id=user.user_id, email=user.email, display_name=user.display_name, created_at=user.created_at)


# --- chat core ---------------------------------------------------------------


def _prepare_chat(
    db: DBSession, auth: CurrentAuth, payload: ChatRequest
) -> tuple[Conversation, bool, str, list[Message]]:
    """Resolve/create the conversation, store the user message, load context."""
    conversation: Conversation | None = None
    if payload.conversation_id:
        conversation = resolve_conversation_access(db, auth, payload.conversation_id)

    is_new = conversation is None
    if is_new:
        if auth.is_authenticated:
            conversation = Conversation(
                conversation_id=str(uuid.uuid4()),
                user_id=auth.user.user_id,
                anon_token=None,
                created_at=utcnow(),
                last_activity_at=utcnow(),
            )
        else:
            conversation = Conversation(
                conversation_id=str(uuid.uuid4()),
                user_id=None,
                anon_token=generate_anon_token(),
                created_at=utcnow(),
                last_activity_at=utcnow(),
            )
        db.add(conversation)
        db.commit()
        db.refresh(conversation)

    assert conversation is not None
    history = load_history(db, conversation.conversation_id, get_settings().history_context_turns)
    store_message(db, conversation.conversation_id, "user", payload.message)

    if not conversation.title:
        trimmed = payload.message.strip()
        conversation.title = trimmed[:TITLE_MAX_CHARS] + ("…" if len(trimmed) > TITLE_MAX_CHARS else "")
        db.commit()

    return conversation, is_new, payload.message, history


def _persist_assistant(conversation_id: str, content: str) -> None:
    """Store an assistant message on its own connection (stream-safe)."""
    db = SessionLocal()
    try:
        store_message(db, conversation_id, "assistant", content)
    finally:
        db.close()


@app.post("/api/v1/chat", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    request: Request,
    db: DBSession = Depends(db_dep),
    auth: CurrentAuth = Depends(get_current_auth),
    _: None = Depends(check_rate_limit),
) -> ChatResponse:
    conversation, is_new, message, history = _prepare_chat(db, auth, payload)

    try:
        reply, model = gemini.generate_reply(history, message)
    except ApiError as exc:
        # Even on failure, let the client bind to the (created) conversation
        # so it can retry inside it.
        exc.headers = {"X-Conversation-Id": conversation.conversation_id}
        raise
    assistant_message = store_message(db, conversation.conversation_id, "assistant", reply)

    body = ChatResponse(
        reply=reply,
        model=model,
        conversation_id=conversation.conversation_id,
        title=conversation.title,
        created_at=assistant_message.created_at,
        anon_token=conversation.anon_token
        if (is_new and not auth.is_authenticated)
        else None,
    )
    return Response(
        status_code=200,
        headers={
            "X-Conversation-Id": conversation.conversation_id,
            **(
                {"X-Conversation-Token": conversation.anon_token or ""}
                if is_new and not auth.is_authenticated
                else {}
            ),
        },
        content=body.model_dump_json(),
        media_type="application/json",
    )


@app.post("/api/v1/chat/stream")
def chat_stream(
    payload: ChatRequest,
    request: Request,
    db: DBSession = Depends(db_dep),
    auth: CurrentAuth = Depends(get_current_auth),
    _: None = Depends(check_rate_limit),
) -> StreamingResponse:
    conversation, is_new, message, history = _prepare_chat(db, auth, payload)
    conversation_id = conversation.conversation_id

    headers = {
        "X-Conversation-Id": conversation_id,
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    if is_new and not auth.is_authenticated:
        headers["X-Conversation-Token"] = conversation.anon_token or ""

    # Pull the first chunk eagerly so upstream failures surface as a proper
    # JSON error response (the SSE status line is sent once streaming starts).
    stream_iterator = gemini.stream_reply(history, message)
    try:
        first_chunk = next(stream_iterator)
    except ApiError as exc:
        exc.headers = {"X-Conversation-Id": conversation_id}
        raise
    except StopIteration:  # pragma: no cover - stream_reply raises instead
        raise AIUnavailableError(headers={"X-Conversation-Id": conversation_id})

    def event_generator():
        accumulated: list[str] = []
        try:
            if first_chunk:
                accumulated.append(first_chunk)
                yield _sse_event(first_chunk)
            for chunk in stream_iterator:
                accumulated.append(chunk)
                yield _sse_event(chunk)
        except ApiError as exc:
            if accumulated:
                _persist_assistant(conversation_id, "".join(accumulated).strip())
            yield _sse_event(f"{STREAM_ERROR_PREFIX}{exc.message}")
            return
        except Exception:
            logger.exception("Unexpected failure terminated the stream.")
            if accumulated:
                _persist_assistant(conversation_id, "".join(accumulated).strip())
            yield _sse_event(
                f"{STREAM_ERROR_PREFIX}The response was interrupted. Please try again."
            )
            return

        full_reply = "".join(accumulated).strip()
        if full_reply:
            _persist_assistant(conversation_id, full_reply)
        else:
            yield _sse_event(
                f"{STREAM_ERROR_PREFIX}The response was interrupted. Please try again."
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers=headers,
    )


# --- conversation endpoints --------------------------------------------------


@app.get("/api/v1/conversations", response_model=ConversationListResponse)
def list_conversations(
    limit: int = Query(default=100, ge=1, le=500),
    user: User = Depends(get_current_user),
    db: DBSession = Depends(db_dep),
) -> ConversationListResponse:
    conversations = (
        db.query(Conversation)
        .filter(Conversation.user_id == user.user_id)
        .order_by(Conversation.last_activity_at.desc())
        .limit(limit)
        .all()
    )
    ids = [c.conversation_id for c in conversations]
    counts: dict[str, int] = {}
    if ids:
        counts = dict(
            db.query(Message.conversation_id, func.count(Message.message_id))
            .filter(Message.conversation_id.in_(ids))
            .group_by(Message.conversation_id)
            .all()
        )
    return ConversationListResponse(
        conversations=[
            ConversationSummary(
                conversation_id=c.conversation_id,
                title=c.title,
                created_at=c.created_at,
                last_activity_at=c.last_activity_at,
                message_count=counts.get(c.conversation_id, 0),
            )
            for c in conversations
        ]
    )


@app.get("/api/v1/conversations/{conversation_id}", response_model=ConversationDetailResponse)
def get_conversation_detail(
    conversation_id: str,
    limit: int = Query(default=1000, ge=1, le=1000),
    auth: CurrentAuth = Depends(get_current_auth),
    db: DBSession = Depends(db_dep),
) -> ConversationDetailResponse:
    conversation = resolve_conversation_access(db, auth, conversation_id)
    messages = load_history(db, conversation_id, limit)
    return ConversationDetailResponse(
        conversation_id=conversation.conversation_id,
        title=conversation.title,
        created_at=conversation.created_at,
        messages=[MessageOut.model_validate(m) for m in messages],
    )


@app.delete("/api/v1/conversations/{conversation_id}", response_model=DeleteConversationResponse)
def delete_conversation(
    conversation_id: str,
    auth: CurrentAuth = Depends(get_current_auth),
    db: DBSession = Depends(db_dep),
) -> DeleteConversationResponse:
    conversation = resolve_conversation_access(db, auth, conversation_id)
    db.query(Message).filter(Message.conversation_id == conversation.conversation_id).delete()
    db.query(Conversation).filter(Conversation.conversation_id == conversation.conversation_id).delete()
    db.commit()
    return DeleteConversationResponse(deleted=True, conversation_id=conversation.conversation_id)
