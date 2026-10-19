from __future__ import annotations

import secrets
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from uuid import uuid4

import httpx
import pytest
import pytest_asyncio

from taskmate.app import COOKIE, ORIGIN, Services, create_app
from taskmate.domain import Conflict, NotFound, Preferences, Session, token_digest


class FakeRepository:
    def __init__(self):
        self.sessions = {}
        self.task_rows = {"alice": [], "bob": []}
        self.prefs = {}
        self.threads = {}
        self.transcripts = {}
        self.notices = {}

    @asynccontextmanager
    async def lock(self, key):
        yield

    async def notification(self, user_id, key):
        return self.notices.get((user_id, key))

    async def new_session(self, user_id, previous_id=None):
        self.sessions = {
            token: session for token, session in self.sessions.items()
            if session.id != previous_id
        }
        token = secrets.token_urlsafe(32)
        current = Session(token_digest(token), user_id, secrets.token_urlsafe(32))
        self.sessions[token] = current
        return token, current

    async def session(self, token):
        return self.sessions.get(token)

    async def tasks(self, user_id):
        return self.task_rows[user_id]

    async def create_task(self, user_id, task):
        row = {
            "id": uuid4(), "title": task.title, "completed": False,
            "created_at": datetime.now(UTC),
        }
        self.task_rows[user_id].append(row)
        return row

    async def complete_task(self, user_id, task_id):
        for row in self.task_rows[user_id]:
            if row["id"] == task_id:
                row["completed"] = True
                return row
        raise NotFound("Task not found.")

    async def preferences(self, user_id):
        return self.prefs.get(user_id, Preferences())

    async def save_preferences(self, user_id, prefs):
        self.prefs[user_id] = prefs
        return prefs

    async def new_thread(self, user_id):
        thread_id = str(uuid4())
        self.threads[thread_id] = user_id
        self.transcripts[thread_id] = []
        return thread_id

    async def require_thread(self, user_id, thread_id):
        if self.threads.get(thread_id) != user_id:
            raise NotFound("Thread not found.")

    async def messages(self, user_id, thread_id):
        await self.require_thread(user_id, thread_id)
        return self.transcripts[thread_id]

    async def add_notification(self, user_id, key, kind, content):
        self.notices.setdefault((user_id, key), {
            "id": uuid4(), "kind": kind, "content": content, "created_at": datetime.now(UTC),
        })
        return self.notices[(user_id, key)]

    async def notifications(self, user_id):
        return [row for (owner, _), row in self.notices.items() if owner == user_id]


class FakeAuth:
    def __init__(self):
        self.connected = {"alice": False, "bob": False}
        self.callback = None

    async def status(self, user_id):
        return {"connected": self.connected[user_id]}

    async def begin(self, user_id, browser_session_id):
        return "https://auth.openai.com/oauth/authorize?state=mock-state"

    async def complete(self, user_id, browser_session_id, **params):
        self.callback = (user_id, browser_session_id, params)
        self.connected[user_id] = True
        return {"connected": True}

    async def disconnect(self, user_id):
        self.connected[user_id] = False
        return True


class FakeChat:
    def __init__(self, repo):
        self.repo = repo
        self.commands = {}

    async def reminder(self, user_id, data):
        return f"Personalized preview for {user_id}."

    async def available_models(self, user_id):
        return ["account-supported-model"]

    async def reply(self, user_id, thread_id, request_id, message):
        await self.repo.require_thread(user_id, thread_id)
        key = (user_id, thread_id, request_id)
        if key in self.commands:
            previous, reply = self.commands[key]
            if previous != message:
                raise Conflict("A request ID cannot be reused with different text.")
            return reply
        reply = f"Reply for {user_id}."
        self.commands[key] = (message, reply)
        self.repo.transcripts[thread_id] += [
            {"role": "user", "content": message},
            {"role": "assistant", "content": reply},
        ]
        return reply


@pytest_asyncio.fixture
async def client():
    repo = FakeRepository()
    auth = FakeAuth()
    services = Services(repo, auth, FakeChat(repo), None)
    app = create_app(services)
    transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 45000))
    async with httpx.AsyncClient(transport=transport, base_url=ORIGIN) as current:
        current.services = services
        yield current


async def browser_session(client):
    response = await client.get("/api/session")
    assert response.status_code == 200
    return response.json()


def write_headers(session):
    return {"origin": ORIGIN, "x-csrf-token": session["csrf_token"]}


@pytest.mark.asyncio
async def test_bootstrap_does_not_require_a_configured_model(client):
    response = await client.get("/api/session")
    assert response.status_code == 200
    assert response.json()["user_id"] == "alice"
    assert response.json()["model"] is None
    assert response.json()["chatgpt_connected"] is False
    cookie = response.headers["set-cookie"].lower()
    assert "httponly" in cookie
    assert "samesite=lax" in cookie
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.asyncio
@pytest.mark.parametrize("method,path,body", [
    ("POST", "/api/session", {"user_id": "bob"}),
    ("POST", "/api/oauth/start", {}),
    ("POST", "/api/oauth/disconnect", {}),
    ("POST", "/api/tasks", {"title": "Do it"}),
    ("POST", "/api/tasks/00000000-0000-0000-0000-000000000001/complete", {}),
    ("PUT", "/api/preferences", {"tone": "friendly"}),
    ("POST", "/api/threads", {}),
    ("POST", "/api/threads/00000000-0000-0000-0000-000000000001/chat",
     {"message": "Hello", "request_id": str(uuid4())}),
    ("POST", "/api/notifications/preview", {}),
])
async def test_every_api_mutation_requires_csrf(client, method, path, body):
    await browser_session(client)
    response = await client.request(method, path, json=body, headers={"origin": ORIGIN})
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_cross_origin_wrong_host_and_remote_clients_are_rejected(client):
    session = await browser_session(client)
    assert (await client.get("/api/session", headers={"host": "evil.example"})).status_code == 403
    assert (await client.get(
        "/api/session", headers={"origin": "https://evil.example"}
    )).status_code == 403
    assert (await client.post(
        "/api/tasks", json={"title": "Bad"},
        headers={**write_headers(session), "origin": "https://evil.example"},
    )).status_code == 403
    app = create_app(client.services)
    remote = httpx.ASGITransport(app=app, client=("192.0.2.1", 40000))
    async with httpx.AsyncClient(transport=remote, base_url=ORIGIN) as external:
        assert (await external.get("/api/session")).status_code == 403


@pytest.mark.asyncio
async def test_session_switch_rotates_tokens_and_isolates_data(client):
    alice = await browser_session(client)
    old_cookie = client.cookies.get(COOKIE)
    task = (await client.post(
        "/api/tasks", json={"title": "Alice task"}, headers=write_headers(alice)
    )).json()
    thread_id = (await client.post("/api/threads", headers=write_headers(alice))).json()["id"]
    response = await client.post(
        "/api/session", json={"user_id": "bob"}, headers=write_headers(alice)
    )
    bob = response.json()
    assert bob["user_id"] == "bob"
    assert bob["csrf_token"] != alice["csrf_token"]
    assert client.cookies.get(COOKIE) != old_cookie
    assert (await client.get("/api/tasks")).json() == []
    assert (await client.get(f"/api/threads/{thread_id}/messages")).status_code == 404
    assert (await client.post(
        f"/api/tasks/{task['id']}/complete", headers=write_headers(bob)
    )).status_code == 404
    assert (await client.post(
        "/api/tasks", json={"title": "Stale session"},
        headers={**write_headers(alice), "cookie": f"{COOKIE}={old_cookie}"},
    )).status_code == 403
    assert (await client.post(
        "/api/session", json={"user_id": "charlie"}, headers=write_headers(bob)
    )).status_code == 422


@pytest.mark.asyncio
async def test_oauth_start_and_callback_are_bound_to_browser_session(client):
    session = await browser_session(client)
    response = await client.post("/api/oauth/start", headers=write_headers(session))
    assert response.status_code == 200
    assert response.json()["authorization_url"].startswith("https://auth.openai.com/")
    callback = await client.get(
        "/auth/callback?state=mock-state&code=mock-code&client_id=mock-client",
        headers={"sec-fetch-site": "cross-site"},
    )
    assert callback.status_code == 303
    user_id, binding, params = client.services.auth.callback
    assert user_id == "alice"
    assert binding == token_digest(client.cookies.get(COOKIE))
    assert params["code"] == "mock-code"
    assert (await client.get("/api/session")).json()["chatgpt_connected"] is True
    assert (await client.get("/api/models")).json() == {"models": ["account-supported-model"]}


@pytest.mark.asyncio
async def test_api_validation_and_chat_request_replay(client):
    session = await browser_session(client)
    headers = write_headers(session)
    assert (await client.post(
        "/api/tasks", json={"title": "  "}, headers=headers
    )).status_code == 422
    assert (await client.put(
        "/api/preferences", json={"timezone": "invalid/zone"}, headers=headers
    )).status_code == 422
    assert (await client.put(
        "/api/preferences",
        json={"tone": "friendly", "notification_hour": 8, "timezone": "Asia/Tokyo"},
        headers=headers,
    )).status_code == 200
    thread_id = (await client.post("/api/threads", headers=headers)).json()["id"]
    payload = {"message": "Hello", "request_id": str(uuid4())}
    url = f"/api/threads/{thread_id}/chat"
    assert (await client.post(url, json=payload, headers=headers)).json()["reply"] == "Reply for alice."
    assert (await client.post(url, json=payload, headers=headers)).status_code == 200
    assert len((await client.get(f"/api/threads/{thread_id}/messages")).json()) == 2
    assert (await client.post(
        url, json={**payload, "message": "Changed"}, headers=headers
    )).status_code == 409
    assert (await client.post("/api/notifications/preview", headers=headers)).status_code == 401
    client.services.auth.connected["alice"] = True
    first = (await client.post("/api/notifications/preview", headers=headers)).json()
    second = (await client.post("/api/notifications/preview", headers=headers)).json()
    assert first["id"] == second["id"]


@pytest.mark.asyncio
async def test_pending_command_can_be_restored_after_reload(client):
    session = await browser_session(client)
    thread_id = (await client.post(
        "/api/threads", headers=write_headers(session)
    )).json()["id"]
    assert (await client.get(f"/api/threads/{thread_id}/messages")).json() == []
    pending = {
        "role": "user", "content": "Create a task",
        "request_id": str(uuid4()), "pending": True,
    }
    client.services.repo.transcripts[thread_id] = [pending]
    assert (await client.get(f"/api/threads/{thread_id}/messages")).json() == [pending]


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["code=mock-code", "state=&code=mock-code"])
async def test_oauth_callback_rejects_missing_state_before_auth(client, query):
    await browser_session(client)
    response = await client.get(f"/auth/callback?{query}")
    assert response.status_code == 400
    assert client.services.auth.callback is None
