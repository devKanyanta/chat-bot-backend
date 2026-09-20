"""Retention sweeper: anonymous conversations expire; user chats are kept."""

import asyncio
import logging
from datetime import timedelta

from sqlalchemy import delete

from .config import get_settings
from .db import SessionLocal
from .models import Conversation, Message, utcnow

logger = logging.getLogger(__name__)


def purge_expired_anonymous_conversations() -> int:
    """Delete anonymous conversations inactive beyond the expiry window.

    Registered users' conversations are never swept here — they persist until
    the user deletes them.
    """
    settings = get_settings()
    cutoff = utcnow() - timedelta(hours=settings.session_expiry_hours)

    db = SessionLocal()
    try:
        stale = (
            db.query(Conversation)
            .filter(
                Conversation.user_id.is_(None),
                Conversation.last_activity_at < cutoff,
            )
            .all()
        )
        if not stale:
            return 0
        conversation_ids = [c.conversation_id for c in stale]
        db.execute(
            delete(Message).where(Message.conversation_id.in_(conversation_ids))
        )
        db.execute(
            delete(Conversation).where(Conversation.conversation_id.in_(conversation_ids))
        )
        db.commit()
        logger.info(
            "Retention sweep removed %d expired anonymous conversation(s).",
            len(conversation_ids),
        )
        return len(conversation_ids)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


async def _sweep_loop() -> None:
    settings = get_settings()
    interval = max(1, settings.retention_sweep_minutes) * 60
    while True:
        await asyncio.sleep(interval)
        try:
            await asyncio.to_thread(purge_expired_anonymous_conversations)
        except Exception:
            logger.exception("Retention sweep failed; will retry next cycle.")


class RetentionTask:
    """Background asyncio task running the expiry sweep every N minutes."""

    def __init__(self) -> None:
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(_sweep_loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None


retention_task = RetentionTask()
