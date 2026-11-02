"""Synthetic, database-backed history adapters for the recorded evaluation CLI."""

from __future__ import annotations

import json
import logging
import uuid
from contextlib import contextmanager
from datetime import datetime
from types import SimpleNamespace

from sqlalchemy import BigInteger, Column, DateTime, MetaData, String, Table, select
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import registry

from kai_memory import AgentIdentity, Chat, MemorySettings, ModelRoles, NoopTelemetry, ports
from kai_memory.configuration import NeutralHostPolicy
from memory_evaluation.eval.providers import TokenHashEmbedder
from kai_memory.sdk import Message
from kai_memory.host_calls import utc_now

logger = logging.getLogger(__name__)


def _warn_recovery_failure(operation: str, error: Exception) -> None:
    try:
        logger.warning("Recorded evaluation session %s failed: %s", operation, type(error).__name__)
    except Exception:
        pass


HISTORY = Table(
    "conversation_messages",
    MetaData(),
    Column("id", UUID(as_uuid=True), primary_key=True),
    Column("chat_id", BigInteger),
    Column("topic_id", BigInteger),
    Column("venue", String),
    Column("role", String),
    Column("content_blocks", JSONB),
    Column("message_metadata", JSONB),
    Column("sender_user_id", BigInteger),
    Column("sender_name", String),
    Column("created_at", DateTime(timezone=True)),
)


def _message(row):
    return Message(
        id=row.id,
        chat_id=row.chat_id,
        topic_id=row.topic_id,
        venue=row.venue,
        role=row.role,
        text="".join(b.get("text", "") for b in row.content_blocks),
        message_metadata=row.message_metadata or {},
        created_at=row.created_at,
        sender_user_id=row.sender_user_id,
        sender_name=row.sender_name,
    )


class Conversations:
    def __init__(self, session):
        self.session = session

    async def save_message(
        self, chat_id, topic_id, *, role, content, sender_user_id=None, sender_name=None, venue="telegram"
    ):
        # The runner stamps created_at before flush; retain that explicit clock.
        row = HistoryRow(
            id=uuid.uuid4(),
            chat_id=chat_id,
            topic_id=topic_id,
            role=role,
            content_blocks=[{"text": content}],
            message_metadata={},
            sender_user_id=sender_user_id,
            sender_name=sender_name,
            venue=venue,
            created_at=utc_now(),
        )
        self.session.add(row)
        return row

    async def get_messages_since(self, chat_id, topic_id, since, limit=200, venue="telegram"):
        chat = Chat(chat_id=chat_id, topic_id=topic_id, venue=venue)
        return await read_page(self.session, chat, SimpleNamespace(created_at=since, id=None), limit)


class HistoryRow:
    @property
    def text(self):
        return "".join(b.get("text", "") for b in self.content_blocks or [])


registry().map_imperatively(HistoryRow, HISTORY)


async def read_page(session, chat, after, limit):
    query = select(HISTORY).where(
        HISTORY.c.chat_id == chat.chat_id, HISTORY.c.topic_id == chat.topic_id, HISTORY.c.venue == chat.venue
    )
    if after is not None:
        if after.id is None:
            query = query.where(HISTORY.c.created_at > after.created_at)
        else:
            from sqlalchemy import tuple_

            query = query.where(tuple_(HISTORY.c.created_at, HISTORY.c.id) > (after.created_at, after.id))
    rows = (await session.execute(query.order_by(HISTORY.c.created_at, HISTORY.c.id).limit(limit))).all()
    return [_message(row) for row in rows]


async def senders_seen_before(session, senders, before):
    return set(
        (
            await session.scalars(
                select(HISTORY.c.sender_user_id).where(
                    HISTORY.c.sender_user_id.in_(senders), HISTORY.c.created_at < before
                )
            )
        ).all()
    )


@contextmanager
def recorded_host(now: datetime):
    """No network inference, paid provider, application registry or host resolver."""
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("recorded evaluation clock must be timezone-aware")
    settings = MemorySettings()
    policy = NeutralHostPolicy()

    async def no_inference(*args, **kwargs):
        raise RuntimeError("recorded evaluation cannot call a live model")

    async def recover_session(session):
        try:
            await session.rollback()
        except Exception as error:
            _warn_recovery_failure("rollback", error)
            try:
                await session.invalidate()
            except Exception as error:
                _warn_recovery_failure("invalidate", error)

    bindings = {
        "recover_session": recover_session,
        "get_settings": lambda: settings,
        "clock": lambda: now,
        "get_agent_identity": lambda: AgentIdentity("example-assistant", "Atlas"),
        "get_model_roles": ModelRoles,
        "get_host_policy": lambda: policy,
        "telemetry": NoopTelemetry(),
        "embedder": TokenHashEmbedder(),
        "ConversationManager": Conversations,
        "ChatContext": Chat,
        "read_conversation_page": read_page,
        "senders_seen_before": senders_seen_before,
        "parse_json_object_response": json.loads,
        "parse_json_array_response": json.loads,
        "get_provider_for_model": lambda *a: None,
        "call_llm_text": no_inference,
        "record_generation_provenance": lambda *a, **k: None,
        "scrub_secrets": lambda value: value,
        "validate_skill_content": lambda value: [],
        "strip_msg_meta_envelope": lambda value: value,
        "strip_echoed_metadata": lambda value: value or "",
        "redact_urls_for_trace": lambda value: value,
        "is_admin_user": lambda actor, *a: (
            actor is not None and str(actor) in settings.telegram_admin_user_ids.split(",")
        ),
        "invalidate_prompt_cache": lambda *a, **k: None,
        "MODEL_METADATA_KEY": "model",
    }
    with ports.scoped(bindings, inherit=False):
        yield bindings
