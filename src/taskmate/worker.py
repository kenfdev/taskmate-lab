"""Deterministic scheduling and in-app delivery. No external notification channel."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

from .domain import USERS, Busy, Preferences

logger = logging.getLogger(__name__)


def due_day(preferences: Preferences, now: datetime) -> date | None:
    if now.tzinfo is None:
        raise ValueError("An aware timestamp is required.")
    local = now.astimezone(ZoneInfo(preferences.timezone))
    return local.date() if local.hour >= preferences.notification_hour else None


def reminder_data(preferences: Preferences, tasks: list[dict]) -> dict:
    return {
        "tone": preferences.tone,
        "timezone": preferences.timezone,
        "open_tasks": [
            {"id": str(task["id"]), "title": task["title"]}
            for task in tasks if not task["completed"]
        ],
    }


def preview_text(data: dict) -> str:
    tasks = data["open_tasks"]
    if not tasks:
        return "No open tasks. Enjoy the breathing room!"
    titles = "; ".join(task["title"] for task in tasks[:3])
    extra = f" (+{len(tasks) - 3} more)" if len(tasks) > 3 else ""
    prefix = "A friendly nudge" if data["tone"] == "friendly" else "Task reminder"
    return f"{prefix}: {titles}{extra}."


async def preview(services, user_id: str) -> dict:
    repo = services.repo
    prefs = await repo.preferences(user_id)
    data = reminder_data(prefs, await repo.tasks(user_id))
    digest = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    key = f"preview:{digest}"
    async with repo.lock(f"notification-preview:{user_id}"):
        existing = await repo.notification(user_id, key)
        if existing:
            return existing
        content = await services.chat.reminder(user_id, data)
        return await repo.add_notification(user_id, key, "preview", content)


async def deliver_due(services, now: datetime) -> int:
    delivered = 0
    for user_id in USERS:
        prefs = await services.repo.preferences(user_id)
        day = due_day(prefs, now)
        if day is None:
            continue
        key = f"daily:{day.isoformat()}"
        try:
            async with services.repo.lock(f"notification:{user_id}:{key}"):
                if await services.repo.notification(user_id, key):
                    continue
                data = reminder_data(prefs, await services.repo.tasks(user_id))
                if not data["open_tasks"]:
                    continue
                if not (await services.auth.status(user_id))["connected"]:
                    continue
                content = await services.chat.reminder(user_id, data)
                await services.repo.add_notification(user_id, key, "daily", content)
                delivered += 1
        except Busy:
            continue
        except Exception as exc:
            logger.warning("Reminder not delivered for %s (%s)", user_id, type(exc).__name__)
    return delivered


async def run() -> None:
    from dotenv import load_dotenv

    from .app import build_services

    load_dotenv()
    async with build_services() as services:
        if not services.model:
            raise RuntimeError("Choose an ID from /api/models, set TASKMATE_MODEL, and restart.")
        while True:
            await deliver_due(services, datetime.now(UTC))
            await asyncio.sleep(60)


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
