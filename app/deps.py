"""Shared dependencies: DB session, rate limiting, auth, conversation access."""

import secrets

from fastapi import Depends, Request
from sqlalchemy.orm import Session as DBSession

from .errors import ForbiddenError, NotFoundError, RateLimitedError, UnauthorizedError
from .models import Conversation, Message, User, utcnow
from .security import decode_access_token

ANON_TOKEN_HEADER = "X-Conversation-Token"


def db_dep(request: Request) -> DBSession:
    return request.state.db


def check_rate_limit(request: Request) -> None:
    from .ratelimit import limiter

    limiter.check(request)


# --- Auth ---------------------------------------------------------------------


class CurrentAuth:
    """Resolved caller identity: a registered user OR an anonymous visitor."""

    def __init__(self, user: User | None = None, anon_token: str | None = None) -> None:
        self.user = user
        self.anon_token = anon_token

    @property
    def is_authenticated(self) -> bool:
        return self.user is not None


def get_current_auth(request: Request, db: DBSession = Depends(db_dep)) -> CurrentAuth:
    """Resolve the caller from `Authorization: Bearer <JWT>` or the anon header.

    Missing credentials are fine (anonymous chat is allowed); *invalid*
    credentials raise 401.
    """
    authorization = request.headers.get("Authorization")
    x_conversation_token = request.headers.get(ANON_TOKEN_HEADER)

    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
        if not token:
            raise UnauthorizedError()
        user_id = decode_access_token(token)
        db_user = db.get(User, user_id)
        if db_user is None:
            raise UnauthorizedError()
        return CurrentAuth(user=db_user)

    if x_conversation_token:
        return CurrentAuth(anon_token=x_conversation_token)

    return CurrentAuth()


def get_current_user(auth: CurrentAuth = Depends(get_current_auth)) -> User:
    """Require a logged-in user (for /me, conversation list, etc.)."""
    if not auth.is_authenticated:
        raise UnauthorizedError()
    return auth.user


# --- Conversation access ------------------------------------------------------


def resolve_conversation_access(
    db: DBSession, auth: CurrentAuth, conversation_id: str
) -> Conversation:
    """Load a conversation and verify the caller may access it; 404/403 on failure."""
    conversation = db.get(Conversation, conversation_id)
    if conversation is None:
        raise NotFoundError("Conversation not found. It may have been deleted.")
    if conversation.user_id is not None:
        if not auth.is_authenticated or auth.user.user_id != conversation.user_id:
            # Do not reveal other users' conversations; 404 rather than 403.
            raise NotFoundError("Conversation not found. It may have been deleted.")
    else:
        if auth.anon_token != conversation.anon_token:
            raise ForbiddenError()
    return conversation


def load_history(db: DBSession, conversation_id: str, limit: int) -> list[Message]:
    """Load the most recent `limit` messages in chronological order."""
    rows = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc(), Message.message_id.desc())
        .limit(limit)
        .all()
    )
    rows.reverse()  # oldest -> newest for the model
    return rows


def store_message(db: DBSession, conversation_id: str, role: str, content: str) -> Message:
    message = Message(
        conversation_id=conversation_id, role=role, content=content, created_at=utcnow()
    )
    db.add(message)
    db.query(Conversation).filter(
        Conversation.conversation_id == conversation_id
    ).update({"last_activity_at": utcnow()})
    db.commit()
    db.refresh(message)
    return message


def generate_anon_token() -> str:
    return secrets.token_urlsafe(32)
