"""Owned application data. PostgreSQL is authoritative; the model is not."""
from __future__ import annotations

import hashlib
import secrets
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator

UserId = Literal["alice", "bob"]
USERS: tuple[UserId, ...] = ("alice", "bob")


class NotFound(Exception):
    pass


class Conflict(Exception):
    pass


class Busy(Conflict):
    pass


class TaskInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=1, max_length=240)

    @field_validator("title")
    @classmethod
    def clean_title(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("A task needs a title.")
        return value


class Preferences(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tone: Literal["concise", "friendly"] = "concise"
    notification_hour: int = Field(default=9, ge=0, le=23, strict=True)
    timezone: str = Field(default="Asia/Tokyo", min_length=1, max_length=100)

    @field_validator("timezone")
    @classmethod
    def known_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError):
            raise ValueError("Choose an IANA time zone, such as Asia/Tokyo.") from None
        return value


@dataclass(frozen=True)
class Session:
    id: str
    user_id: UserId
    csrf_token: str


@dataclass(frozen=True)
class UserContext:
    user_id: UserId
    thread_id: str


def token_digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def lock_id(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big", signed=True)


SCHEMA = (
    """CREATE TABLE IF NOT EXISTS tm_threads (
        user_id text NOT NULL CHECK (user_id IN ('alice', 'bob')),
        id uuid NOT NULL,
        created_at timestamptz NOT NULL DEFAULT now(),
        PRIMARY KEY (user_id, id)
    )""",
    """CREATE TABLE IF NOT EXISTS tm_tasks (
        user_id text NOT NULL CHECK (user_id IN ('alice', 'bob')),
        id uuid NOT NULL,
        title text NOT NULL CHECK (length(title) BETWEEN 1 AND 240),
        completed boolean NOT NULL DEFAULT false,
        created_at timestamptz NOT NULL DEFAULT now(),
        thread_id uuid,
        tool_call_id text,
        PRIMARY KEY (user_id, id),
        UNIQUE (user_id, thread_id, tool_call_id),
        FOREIGN KEY (user_id, thread_id) REFERENCES tm_threads(user_id, id),
        CHECK ((thread_id IS NULL) = (tool_call_id IS NULL))
    )""",
    """CREATE TABLE IF NOT EXISTS tm_sessions (
        token_hash text PRIMARY KEY,
        user_id text NOT NULL CHECK (user_id IN ('alice', 'bob')),
        csrf_token text NOT NULL,
        expires_at timestamptz NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS tm_chat_requests (
        sequence bigserial UNIQUE,
        user_id text NOT NULL,
        thread_id uuid NOT NULL,
        request_id uuid NOT NULL,
        message text NOT NULL,
        status text NOT NULL CHECK (status IN ('running', 'completed')),
        reply text,
        created_at timestamptz NOT NULL DEFAULT now(),
        PRIMARY KEY (user_id, thread_id, request_id),
        FOREIGN KEY (user_id, thread_id) REFERENCES tm_threads(user_id, id),
        CHECK ((status = 'running' AND reply IS NULL)
            OR (status = 'completed' AND reply IS NOT NULL))
    )""",
    """CREATE UNIQUE INDEX IF NOT EXISTS tm_one_running_chat
        ON tm_chat_requests(user_id, thread_id) WHERE status = 'running'""",
    """CREATE TABLE IF NOT EXISTS tm_notifications (
        user_id text NOT NULL CHECK (user_id IN ('alice', 'bob')),
        id uuid NOT NULL,
        dedupe_key text NOT NULL,
        kind text NOT NULL CHECK (kind IN ('daily', 'preview')),
        content text NOT NULL,
        created_at timestamptz NOT NULL DEFAULT now(),
        PRIMARY KEY (user_id, id),
        UNIQUE (user_id, dedupe_key)
    )""",
)


class Repository:
    def __init__(self, pool, store):
        self.pool = pool
        self.store = store

    async def setup(self) -> None:
        async with self.pool.connection() as conn:
            for statement in SCHEMA:
                await conn.execute(statement)

    @asynccontextmanager
    async def lock(self, key: str):
        async with self.pool.connection() as conn:
            async with conn.transaction():
                row = await (
                    await conn.execute(
                        "SELECT pg_try_advisory_xact_lock(%s) AS acquired", (lock_id(key),)
                    )
                ).fetchone()
                if not row["acquired"]:
                    raise Busy("This operation is already running. Retry with the same request ID.")
                yield

    async def new_session(
        self, user_id: UserId, previous_id: str | None = None
    ) -> tuple[str, Session]:
        token = secrets.token_urlsafe(32)
        session = Session(token_digest(token), user_id, secrets.token_urlsafe(32))
        async with self.pool.connection() as conn:
            async with conn.transaction():
                if previous_id:
                    await conn.execute(
                        "DELETE FROM tm_sessions WHERE token_hash = %s", (previous_id,)
                    )
                await conn.execute(
                    "DELETE FROM tm_sessions WHERE expires_at <= now()"
                )
                await conn.execute(
                    """INSERT INTO tm_sessions
                       (token_hash, user_id, csrf_token, expires_at) VALUES (%s, %s, %s, %s)""",
                    (session.id, user_id, session.csrf_token, datetime.now(UTC) + timedelta(days=7)),
                )
        return token, session

    async def session(self, token: str | None) -> Session | None:
        if not token or len(token) > 128:
            return None
        async with self.pool.connection() as conn:
            row = await (
                await conn.execute(
                    """SELECT token_hash, user_id, csrf_token FROM tm_sessions
                       WHERE token_hash = %s AND expires_at > now()""",
                    (token_digest(token),),
                )
            ).fetchone()
        return Session(row["token_hash"], row["user_id"], row["csrf_token"]) if row else None

    async def tasks(self, user_id: UserId) -> list[dict]:
        async with self.pool.connection() as conn:
            return await (
                await conn.execute(
                    """SELECT id, title, completed, created_at FROM tm_tasks
                       WHERE user_id = %s ORDER BY created_at, id""", (user_id,)
                )
            ).fetchall()

    async def create_task(
        self,
        user_id: UserId,
        task: TaskInput,
        *,
        thread_id: str | None = None,
        tool_call_id: str | None = None,
    ) -> dict:
        async with self.pool.connection() as conn:
            async with conn.transaction():
                row = await (
                    await conn.execute(
                        """INSERT INTO tm_tasks
                           (user_id, id, title, thread_id, tool_call_id)
                           VALUES (%s, %s, %s, %s, %s)
                           ON CONFLICT (user_id, thread_id, tool_call_id)
                           DO UPDATE SET title = tm_tasks.title
                           RETURNING id, title, completed, created_at""",
                        (user_id, uuid4(), task.title, thread_id, tool_call_id),
                    )
                ).fetchone()
                if row["title"] != task.title:
                    raise Conflict("That tool-call ID was already used for a different task.")
                return row

    async def complete_task(self, user_id: UserId, task_id: UUID) -> dict:
        async with self.pool.connection() as conn:
            row = await (
                await conn.execute(
                    """UPDATE tm_tasks SET completed = true
                       WHERE user_id = %s AND id = %s
                       RETURNING id, title, completed, created_at""", (user_id, task_id)
                )
            ).fetchone()
        if row is None:
            raise NotFound("Task not found.")
        return row

    async def preferences(self, user_id: UserId) -> Preferences:
        item = await self.store.aget(("users", user_id, "preferences"), "profile")
        return Preferences.model_validate(item.value) if item else Preferences()

    async def save_preferences(self, user_id: UserId, prefs: Preferences) -> Preferences:
        await self.store.aput(
            ("users", user_id, "preferences"), "profile", prefs.model_dump()
        )
        return prefs

    async def new_thread(self, user_id: UserId) -> str:
        thread_id = uuid4()
        async with self.pool.connection() as conn:
            await conn.execute(
                "INSERT INTO tm_threads (user_id, id) VALUES (%s, %s)", (user_id, thread_id)
            )
        return str(thread_id)

    async def require_thread(self, user_id: UserId, thread_id: str) -> None:
        async with self.pool.connection() as conn:
            row = await (
                await conn.execute(
                    "SELECT 1 FROM tm_threads WHERE user_id = %s AND id = %s",
                    (user_id, thread_id),
                )
            ).fetchone()
        if row is None:
            raise NotFound("Thread not found.")

    async def cached_reply(
        self, user_id: UserId, thread_id: str, request_id: str, message: str
    ) -> str | None:
        async with self.pool.connection() as conn:
            row = await (
                await conn.execute(
                    """SELECT request_id, message, reply FROM tm_chat_requests
                       WHERE user_id = %s AND thread_id = %s AND request_id = %s""",
                    (user_id, thread_id, request_id),
                )
            ).fetchone()
        if row and row["message"] != message:
            raise Conflict("A request ID cannot be reused with different text.")
        return row["reply"] if row else None

    async def begin_chat(
        self, user_id: UserId, thread_id: str, request_id: str, message: str
    ) -> dict:
        await self.require_thread(user_id, thread_id)
        async with self.pool.connection() as conn:
            async with conn.transaction():
                row = await (
                    await conn.execute(
                        """SELECT request_id, message, status, reply FROM tm_chat_requests
                           WHERE user_id = %s AND thread_id = %s AND request_id = %s""",
                        (user_id, thread_id, request_id),
                    )
                ).fetchone()
                if row:
                    if row["message"] != message:
                        raise Conflict("A request ID cannot be reused with different text.")
                    return row
                pending = await (
                    await conn.execute(
                        """SELECT 1 FROM tm_chat_requests
                           WHERE user_id = %s AND thread_id = %s AND status = 'running'""",
                        (user_id, thread_id),
                    )
                ).fetchone()
                if pending:
                    raise Conflict("Retry the unfinished message before sending a new one.")
                return await (
                    await conn.execute(
                        """INSERT INTO tm_chat_requests
                           (user_id, thread_id, request_id, message, status)
                           VALUES (%s, %s, %s, %s, 'running')
                           RETURNING request_id, message, status, reply""",
                        (user_id, thread_id, request_id, message),
                    )
                ).fetchone()

    async def finish_chat(
        self, user_id: UserId, thread_id: str, request_id: str, reply: str
    ) -> None:
        async with self.pool.connection() as conn:
            await conn.execute(
                """UPDATE tm_chat_requests SET status = 'completed', reply = %s
                   WHERE user_id = %s AND thread_id = %s AND request_id = %s""",
                (reply, user_id, thread_id, request_id),
            )

    async def messages(self, user_id: UserId, thread_id: str) -> list[dict]:
        await self.require_thread(user_id, thread_id)
        async with self.pool.connection() as conn:
            rows = await (
                await conn.execute(
                    """SELECT request_id, message, reply FROM tm_chat_requests
                       WHERE user_id = %s AND thread_id = %s ORDER BY sequence""",
                    (user_id, thread_id),
                )
            ).fetchall()
        result = []
        for row in rows:
            message = {"role": "user", "content": row["message"]}
            if row["reply"] is None:
                message.update(request_id=str(row["request_id"]), pending=True)
            result.append(message)
            if row["reply"] is not None:
                result.append({"role": "assistant", "content": row["reply"]})
        return result

    async def notification(self, user_id: UserId, dedupe_key: str) -> dict | None:
        async with self.pool.connection() as conn:
            return await (
                await conn.execute(
                    """SELECT id, kind, content, created_at FROM tm_notifications
                       WHERE user_id = %s AND dedupe_key = %s""", (user_id, dedupe_key)
                )
            ).fetchone()

    async def add_notification(
        self, user_id: UserId, dedupe_key: str, kind: str, content: str
    ) -> dict:
        async with self.pool.connection() as conn:
            return await (
                await conn.execute(
                    """INSERT INTO tm_notifications (user_id, id, dedupe_key, kind, content)
                       VALUES (%s, %s, %s, %s, %s)
                       ON CONFLICT (user_id, dedupe_key)
                       DO UPDATE SET content = tm_notifications.content
                       RETURNING id, kind, content, created_at""",
                    (user_id, uuid4(), dedupe_key, kind, content),
                )
            ).fetchone()

    async def notifications(self, user_id: UserId) -> list[dict]:
        async with self.pool.connection() as conn:
            return await (
                await conn.execute(
                    """SELECT id, kind, content, created_at FROM tm_notifications
                       WHERE user_id = %s ORDER BY created_at DESC, id DESC LIMIT 50""",
                    (user_id,),
                )
            ).fetchall()
