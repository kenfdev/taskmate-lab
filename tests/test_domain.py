from __future__ import annotations

import os
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest
import pytest_asyncio
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore
from psycopg import AsyncConnection, sql
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from pydantic import ValidationError

from taskmate.app import AgentService, domain_tools
from taskmate.domain import Busy, Conflict, NotFound, Preferences, Repository, TaskInput
from taskmate.worker import deliver_due, due_day, preview, preview_text, reminder_data


@pytest.mark.parametrize("title", ["", "   ", "x" * 241])
def test_task_title_must_be_nonempty_and_bounded(title):
    with pytest.raises(ValidationError):
        TaskInput(title=title)


def test_task_title_is_normalized():
    assert TaskInput(title="  Ship the small version  ").title == "Ship the small version"


@pytest.mark.parametrize("data", [
    {"notification_hour": -1},
    {"notification_hour": 24},
    {"notification_hour": True},
    {"timezone": "not/a/timezone"},
    {"timezone": "../etc/passwd"},
    {"tone": "hostile"},
    {"user_id": "bob"},
])
def test_preferences_reject_invalid_boundary_input(data):
    with pytest.raises(ValidationError):
        Preferences(**data)


def test_schedule_uses_users_local_day_and_catches_up_after_scheduled_hour():
    prefs = Preferences(timezone="Asia/Tokyo", notification_hour=9)
    assert due_day(prefs, datetime(2026, 10, 18, 23, 59, tzinfo=UTC)) is None
    assert due_day(prefs, datetime(2026, 10, 19, 0, 0, tzinfo=UTC)).isoformat() == "2026-10-19"
    assert due_day(prefs, datetime(2026, 10, 19, 8, 0, tzinfo=UTC)).isoformat() == "2026-10-19"


def test_repeated_dst_hour_has_same_daily_key():
    prefs = Preferences(timezone="America/New_York", notification_hour=1)
    first = due_day(prefs, datetime(2026, 11, 1, 5, 30, tzinfo=UTC))
    repeated = due_day(prefs, datetime(2026, 11, 1, 6, 30, tzinfo=UTC))
    assert first == repeated
    assert first.isoformat() == "2026-11-01"


def test_schedule_rejects_ambiguous_naive_datetime():
    with pytest.raises(ValueError):
        due_day(Preferences(), datetime(2026, 10, 19, 12))


def test_preview_contains_only_open_tasks_and_honors_tone():
    data = reminder_data(Preferences(tone="friendly"), [
        {"id": uuid4(), "title": "Buy tea", "completed": False},
        {"id": uuid4(), "title": "Old task", "completed": True},
    ])
    assert preview_text(data) == "A friendly nudge: Buy tea."


def test_model_tools_have_no_user_identity_argument():
    for current in domain_tools(None):
        properties = current.tool_call_schema.model_json_schema().get("properties", {})
        assert "user_id" not in properties
        assert "thread_id" not in properties
        assert "runtime" not in properties


@pytest_asyncio.fixture
async def repository():
    dsn = os.getenv("TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("Set TEST_DATABASE_URL to run PostgreSQL integration tests.")
    schema = "taskmate_test_" + uuid4().hex
    async with await AsyncConnection.connect(dsn, autocommit=True) as admin:
        await admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        pool = AsyncConnectionPool(
            dsn, open=False, min_size=1, max_size=8,
            kwargs={
                "autocommit": True,
                "row_factory": dict_row,
                "prepare_threshold": 0,
                "options": f"-c search_path={schema}",
            },
        )
        try:
            await pool.open()
            await pool.wait()
            store = AsyncPostgresStore(pool)
            await store.setup()
            repo = Repository(pool, store)
            await repo.setup()
            yield repo
        finally:
            await pool.close()
            await admin.execute(
                sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema))
            )


@pytest.mark.asyncio
async def test_postgres_owns_tasks_threads_and_cross_thread_preferences(repository):
    repo = repository
    first = await repo.new_thread("alice")
    second = await repo.new_thread("alice")
    task = await repo.create_task(
        "alice", TaskInput(title="Write notes"), thread_id=first, tool_call_id="call-1"
    )
    repeated = await repo.create_task(
        "alice", TaskInput(title="Write notes"), thread_id=first, tool_call_id="call-1"
    )
    assert repeated["id"] == task["id"]
    assert len(await repo.tasks("alice")) == 1
    assert await repo.tasks("bob") == []
    with pytest.raises(NotFound):
        await repo.complete_task("bob", task["id"])
    with pytest.raises(NotFound):
        await repo.messages("bob", first)
    with pytest.raises(Conflict):
        await repo.create_task(
            "alice", TaskInput(title="Different"), thread_id=first, tool_call_id="call-1"
        )
    await repo.complete_task("alice", task["id"])
    await repo.complete_task("alice", task["id"])
    assert (await repo.tasks("alice"))[0]["completed"] is True
    await repo.save_preferences(
        "alice", Preferences(tone="friendly", notification_hour=18, timezone="Asia/Tokyo")
    )
    reopened = Repository(repo.pool, AsyncPostgresStore(repo.pool))
    await reopened.require_thread("alice", second)
    assert (await reopened.preferences("alice")).tone == "friendly"
    assert (await reopened.preferences("bob")).tone == "concise"


@pytest.mark.asyncio
async def test_postgres_session_rotation_revokes_old_token(repository):
    token, original = await repository.new_session("alice")
    new_token, updated = await repository.new_session("bob", original.id)
    assert await repository.session(token) is None
    assert (await repository.session(new_token)).user_id == "bob"
    assert original.csrf_token != updated.csrf_token


@pytest.mark.asyncio
async def test_postgres_lock_excludes_other_connections_and_releases_on_error(repository):
    with pytest.raises(RuntimeError):
        async with repository.lock("shared-operation"):
            with pytest.raises(Busy):
                async with repository.lock("shared-operation"):
                    pytest.fail("A second connection entered the protected operation.")
            raise RuntimeError("simulated interrupted request")
    async with repository.lock("shared-operation"):
        pass


class FakeAuth:
    connected = True

    async def status(self, user_id):
        return {"connected": self.connected}


class InterruptedGraph:
    def __init__(self, repo, thread_id):
        self.repo = repo
        self.thread_id = thread_id
        self.values = {}
        self.next = ()
        self.inputs = []

    async def aget_state(self, config):
        return SimpleNamespace(values=self.values, next=self.next)

    async def ainvoke(self, value, *, config, context, durability):
        self.inputs.append(value)
        if value is not None:
            self.values = value
        await self.repo.create_task(
            context.user_id, TaskInput(title="Durable task"),
            thread_id=self.thread_id, tool_call_id="stable-tool-call",
        )
        if len(self.inputs) == 1:
            self.next = ("tools",)
            raise RuntimeError("simulated crash after tool side effect")
        self.next = ()
        self.values = {
            **self.values, "messages": [AIMessage(content="Created the task.")]
        }
        return self.values


@pytest.mark.asyncio
async def test_interrupted_chat_resumes_without_duplicate_effects(repository):
    thread_id = await repository.new_thread("alice")
    request_id = str(uuid4())
    auth = FakeAuth()
    graph = InterruptedGraph(repository, thread_id)
    service = AgentService(repository, auth, None, None, None, "mock-model")
    service.agent = lambda user_id: graph
    with pytest.raises(RuntimeError):
        await service.reply("alice", thread_id, request_id, "Create a task")
    assert await repository.messages("alice", thread_id) == [{
        "role": "user", "content": "Create a task",
        "request_id": request_id, "pending": True,
    }]
    with pytest.raises(NotFound):
        await repository.messages("bob", thread_id)
    with pytest.raises(Conflict):
        await service.reply("alice", thread_id, str(uuid4()), "A different message")
    assert await service.reply(
        "alice", thread_id, request_id, "Create a task"
    ) == "Created the task."
    assert graph.inputs[1] is None
    assert len(await repository.tasks("alice")) == 1
    auth.connected = False
    assert await service.reply(
        "alice", thread_id, request_id, "Create a task"
    ) == "Created the task."
    assert len(graph.inputs) == 2
    with pytest.raises(Conflict):
        await service.reply("alice", thread_id, request_id, "Changed text")
    assert await repository.messages("alice", thread_id) == [
        {"role": "user", "content": "Create a task"},
        {"role": "assistant", "content": "Created the task."},
    ]


@pytest.mark.asyncio
async def test_finished_checkpoint_recovers_before_command_commit(repository):
    thread_id = await repository.new_thread("alice")
    request_id = str(uuid4())
    await repository.begin_chat("alice", thread_id, request_id, "Hello")
    snapshot = SimpleNamespace(
        values={"request_id": request_id, "messages": [AIMessage(content="Hello back.")]},
        next=(),
    )

    class CompletedGraph:
        async def aget_state(self, config):
            return snapshot

        async def ainvoke(self, *args, **kwargs):
            pytest.fail("A completed checkpoint must not invoke the model again.")

    service = AgentService(repository, FakeAuth(), None, None, None, "mock-model")
    service.agent = lambda user_id: CompletedGraph()
    assert await service.reply("alice", thread_id, request_id, "Hello") == "Hello back."


@pytest.mark.asyncio
async def test_notifications_are_owner_scoped_and_daily_idempotent(repository):
    await repository.create_task("alice", TaskInput(title="Alice task"))
    await repository.create_task("bob", TaskInput(title="Bob task"))
    await repository.save_preferences("alice", Preferences(tone="friendly"))
    calls = []

    class FakeChat:
        async def reminder(self, user_id, data):
            calls.append((user_id, data))
            return preview_text(data)

    services = SimpleNamespace(repo=repository, auth=FakeAuth(), chat=FakeChat())
    now = datetime(2026, 10, 19, 12, tzinfo=UTC)
    assert await deliver_due(services, now) == 2
    assert await deliver_due(services, now) == 0
    assert len(calls) == 2
    assert calls[0][1]["open_tasks"][0]["title"] == "Alice task"
    assert calls[1][1]["open_tasks"][0]["title"] == "Bob task"
    assert (await repository.notifications("alice"))[0]["content"] == (
        "A friendly nudge: Alice task."
    )
    first = await preview(services, "alice")
    repeated = await preview(services, "alice")
    assert first["id"] == repeated["id"]
    assert len(await repository.notifications("alice")) == 2
    assert len(calls) == 3


class ScriptedModel(BaseChatModel):
    replies: list[AIMessage]
    cursor: int = 0

    @property
    def _llm_type(self):
        return "taskmate-test"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise AssertionError("The app must use async model calls.")

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        message = self.replies[self.cursor]
        self.cursor += 1
        return ChatResult(generations=[ChatGeneration(message=message)])

    def get_num_tokens(self, text):
        return max(1, len(text) // 4)

    def get_num_tokens_from_messages(self, messages, tools=None):
        return sum(self.get_num_tokens(str(message.content)) for message in messages)


@pytest.mark.asyncio
async def test_real_deepagents_graph_injects_identity_and_persists_checkpoint(repository):
    saver = AsyncPostgresSaver(repository.pool)
    await saver.setup()
    model = ScriptedModel(replies=[
        AIMessage(content="", tool_calls=[{
            "name": "create_task", "args": {"title": "Graph-created task"},
            "id": "graph-call-1", "type": "tool_call",
        }]),
        AIMessage(content="Created the task."),
    ])
    service = AgentService(
        repository, FakeAuth(), saver, repository.store, None, "scripted"
    )
    service.model = lambda user_id: model
    thread_id = await repository.new_thread("alice")
    request_id = str(uuid4())
    assert await service.reply(
        "alice", thread_id, request_id, "Create a task"
    ) == "Created the task."
    assert (await repository.tasks("alice"))[0]["title"] == "Graph-created task"
    assert await repository.tasks("bob") == []
    snapshot = await service.agent("alice").aget_state({
        "configurable": {"thread_id": f"alice:{thread_id}"}
    })
    assert snapshot.values["request_id"] == request_id
    assert not snapshot.next
    assert await service.reply(
        "alice", thread_id, request_id, "Create a task"
    ) == "Created the task."
    assert model.cursor == 2
