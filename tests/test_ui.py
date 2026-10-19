"""Browser checks against the real HTML with synthetic, intercepted API responses.

No OAuth exchange or model request occurs. The loopback HTTP server serves only
the repository's index.html. CI can retain videos with pytest-playwright's
--video=on --output=artifacts/browser options.
"""

from __future__ import annotations

import json
import threading
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
from uuid import UUID, uuid4

import pytest

playwright = pytest.importorskip("playwright.sync_api")
expect = playwright.expect

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "src" / "taskmate" / "static" / "index.html"


@pytest.fixture(scope="session")
def ui_origin():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if urlsplit(self.path).path not in {"/", "/index.html"}:
                self.send_error(404)
                return
            body = HTML.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class MockApi:
    def __init__(self):
        self.user_id = "alice"
        self.connected = {"alice": False, "bob": False}
        self.model = None
        self.models = ["mock-model-small", "mock-model-large"]
        self.tasks = {
            "alice": [{"id": "alice-1", "title": "Aliceの見積もり", "completed": False}],
            "bob": [{"id": "bob-1", "title": "Bobの買い物", "completed": False}],
        }
        self.preferences = {
            "alice": {"tone": "concise", "notification_hour": 9, "timezone": "Asia/Tokyo"},
            "bob": {"tone": "friendly", "notification_hour": 18, "timezone": "Asia/Tokyo"},
        }
        self.notifications = {"alice": [], "bob": []}
        self.threads = {}
        self.calls = []
        self.unhandled = []
        self.csrf_errors = []
        self.reply = "モック応答です。タスクをひとつずつ進めましょう。"
        self.fail_next_chat = False
        self.remote_revocation_confirmed = False
        self.preferences_after_chat = None

    @property
    def csrf(self):
        return f"synthetic-csrf-{self.user_id}"

    def session(self):
        return {
            "user_id": self.user_id,
            "csrf_token": self.csrf,
            "chatgpt_connected": self.connected[self.user_id],
            "model": self.model,
        }

    @staticmethod
    def respond(route, payload, status=200):
        route.fulfill(
            status=status,
            content_type="application/json",
            body=json.dumps(payload, ensure_ascii=False),
        )

    def handle(self, route):
        request = route.request
        path = urlsplit(request.url).path
        method = request.method
        data = request.post_data_json if request.post_data else None
        user = self.user_id
        self.calls.append({
            "method": method,
            "path": path,
            "user_id": user,
            "body": deepcopy(data),
            "csrf": request.headers.get("x-csrf-token"),
        })
        if method not in {"GET", "HEAD"} and request.headers.get("x-csrf-token") != self.csrf:
            self.csrf_errors.append(f"{method} {path} for {user}")
            self.respond(route, {"detail": "Synthetic CSRF rejection"}, status=403)
            return

        if path == "/api/session" and method == "GET":
            self.respond(route, self.session())
            return
        if path == "/api/session" and method == "POST":
            if data["user_id"] not in {"alice", "bob"}:
                self.respond(route, {"detail": "Unknown demo profile"}, status=422)
                return
            self.user_id = data["user_id"]
            self.respond(route, self.session())
            return
        if path == "/api/tasks" and method == "GET":
            self.respond(route, self.tasks[user])
            return
        if path == "/api/tasks" and method == "POST":
            task = {"id": str(uuid4()), "title": data["title"], "completed": False}
            self.tasks[user].append(task)
            self.respond(route, task, status=201)
            return
        if path == "/api/preferences" and method == "GET":
            self.respond(route, self.preferences[user])
            return
        if path == "/api/preferences" and method == "PUT":
            self.preferences[user] = deepcopy(data)
            self.respond(route, self.preferences[user])
            return
        if path == "/api/models" and method == "GET":
            if not self.connected[user]:
                self.respond(route, {"detail": "Not connected"}, status=401)
            else:
                self.respond(route, {"models": self.models})
            return
        if path == "/api/oauth/disconnect" and method == "POST":
            self.connected[user] = False
            self.respond(route, {
                "disconnected": True,
                "remote_revocation_confirmed": self.remote_revocation_confirmed,
            })
            return
        if path == "/api/notifications" and method == "GET":
            self.respond(route, self.notifications[user])
            return
        if path == "/api/notifications/preview" and method == "POST":
            preview = {
                "id": str(uuid4()),
                "content": "モック通知です。今日のタスクを確認しましょう。",
                "created_at": "2026-10-19T00:00:00+00:00",
            }
            self.notifications[user].append(preview)
            self.respond(route, {"content": preview["content"]})
            return
        if path == "/api/threads" and method == "POST":
            thread_id = str(uuid4())
            self.threads[thread_id] = {"user_id": user, "messages": [], "replies": {}}
            self.respond(route, {"id": thread_id}, status=201)
            return

        parts = path.strip("/").split("/")
        if len(parts) == 4 and parts[:2] == ["api", "tasks"] and parts[3] == "complete" and method == "POST":
            task = next((item for item in self.tasks[user] if item["id"] == parts[2]), None)
            if task is None:
                self.respond(route, {"detail": "Task not found"}, status=404)
            else:
                task["completed"] = True
                self.respond(route, task)
            return
        if len(parts) == 4 and parts[:2] == ["api", "threads"]:
            thread = self.threads.get(parts[2])
            if thread is None or thread["user_id"] != user:
                self.respond(route, {"detail": "Thread not found"}, status=404)
                return
            if parts[3] == "messages" and method == "GET":
                self.respond(route, thread["messages"])
                return
            if parts[3] == "chat" and method == "POST":
                if not self.connected[user] or self.model is None:
                    self.respond(route, {"detail": "Model unavailable"}, status=503)
                    return
                request_id = data["request_id"]
                UUID(request_id)
                pending = next((message for message in thread["messages"] if message.get("pending")), None)
                if pending and pending["request_id"] != request_id:
                    self.respond(route, {"detail": "Another request is pending"}, status=409)
                    return
                if self.fail_next_chat:
                    self.fail_next_chat = False
                    thread["messages"].append({
                        "role": "user",
                        "content": data["message"],
                        "request_id": request_id,
                        "pending": True,
                    })
                    self.respond(route, {"detail": "Synthetic temporary failure"}, status=503)
                    return
                if request_id not in thread["replies"]:
                    thread["messages"] = [message for message in thread["messages"] if not message.get("pending")]
                    thread["messages"].extend([
                        {"role": "user", "content": data["message"]},
                        {"role": "assistant", "content": self.reply},
                    ])
                    thread["replies"][request_id] = self.reply
                    if self.preferences_after_chat is not None:
                        self.preferences[user].update(self.preferences_after_chat)
                        self.preferences_after_chat = None
                self.respond(route, {"reply": thread["replies"][request_id]})
                return

        self.unhandled.append(f"{method} {path}")
        self.respond(route, {"detail": "Unexpected mock request"}, status=404)


@pytest.fixture
def mocked_app(page, ui_origin):
    api = MockApi()
    page.route("**/api/**", api.handle)

    def open_app():
        page.goto(ui_origin, wait_until="domcontentloaded")
        expect(page.locator("#runtime-status")).to_have_text("準備できました")
        expect(page.locator("#error-banner")).to_be_hidden()

    yield api, open_app
    assert not api.unhandled, api.unhandled
    assert not api.csrf_errors, api.csrf_errors


def test_initial_state_and_direct_tasks_without_model(page, mocked_app):
    api, open_app = mocked_app
    open_app()

    expect(page.get_by_role("button", name="ChatGPTにログイン", exact=True)).to_be_enabled()
    expect(page.locator("#chat-message")).to_be_disabled()
    expect(page.locator("#preview-notification")).to_be_disabled()
    expect(page.locator("#task-title")).to_be_enabled()
    expect(page.locator("#tone")).to_be_enabled()
    expect(page.locator("#model-setup")).to_be_hidden()
    expect(page.locator("#task-list")).to_contain_text("Aliceの見積もり")

    page.get_by_label("新しいタスク", exact=True).fill("資料をまとめる")
    page.get_by_role("button", name="追加", exact=True).click()
    expect(page.locator("#task-list").get_by_text("資料をまとめる", exact=True)).to_be_visible()
    expect(page.locator("#task-count")).to_have_text("残り 2 件")
    page.get_by_role("button", name="資料をまとめるを完了にする", exact=True).click()
    expect(page.locator(".task-row.completed")).to_contain_text("資料をまとめる")
    expect(page.locator("#task-count")).to_have_text("残り 1 件")

    page.locator("#tone").select_option("friendly")
    page.locator("#notification-hour").select_option("14")
    page.get_by_role("button", name="設定を保存", exact=True).click()
    expect(page.locator("#preference-status")).to_have_text("保存済み")
    assert api.preferences["alice"]["tone"] == "friendly"
    assert api.preferences["alice"]["notification_hour"] == 14
    assert api.model is None
    assert all(
        call["csrf"] == f"synthetic-csrf-{call['user_id']}"
        for call in api.calls if call["method"] not in {"GET", "HEAD"}
    )


def test_profile_switch_clears_previous_tasks_and_chat(page, mocked_app):
    api, open_app = mocked_app
    api.connected = {"alice": True, "bob": True}
    api.model = "mock-model-small"
    open_app()

    page.get_by_label("メッセージ", exact=True).fill("Aliceだけの相談")
    page.get_by_role("button", name="送信", exact=True).click()
    expect(page.locator("#chat-log")).to_contain_text(api.reply)
    expect(page.locator("#runtime-status")).to_have_text("準備できました")
    page.locator("#demo-user").select_option("bob")

    expect(page.locator("#runtime-status")).to_have_text("準備できました")
    expect(page.locator("#task-list")).to_contain_text("Bobの買い物")
    expect(page.locator("#task-list")).not_to_contain_text("Aliceの見積もり")
    expect(page.locator("#chat-log")).not_to_contain_text("Aliceだけの相談")
    expect(page.locator("#chat-log")).not_to_contain_text(api.reply)
    expect(page.locator("#chat-message")).to_have_value("")
    expect(page.locator("#tone")).to_have_value("friendly")
    expect(page.locator("#notification-hour")).to_have_value("18")
    assert len(api.threads) == 2

    page.get_by_label("メッセージ", exact=True).fill("Bobの相談")
    page.get_by_role("button", name="送信", exact=True).click()
    expect(page.locator("#chat-log")).to_contain_text(api.reply)
    chat_calls = [call for call in api.calls if call["path"].endswith("/chat")]
    assert [call["user_id"] for call in chat_calls] == ["alice", "bob"]
    assert chat_calls[0]["path"] != chat_calls[1]["path"]


def test_reload_and_profile_switch_restore_each_profiles_thread(page, mocked_app):
    api, open_app = mocked_app
    api.connected = {"alice": True, "bob": True}
    api.model = "mock-model-small"
    open_app()

    page.get_by_label("メッセージ", exact=True).fill("Aliceの保存される会話")
    page.get_by_role("button", name="送信", exact=True).click()
    expect(page.locator("#chat-log")).to_contain_text(api.reply)
    expect(page.locator("#runtime-status")).to_have_text("準備できました")
    alice_thread = next(iter(api.threads))
    page.reload(wait_until="domcontentloaded")
    expect(page.locator("#runtime-status")).to_have_text("準備できました")
    expect(page.locator("#chat-log")).to_contain_text("Aliceの保存される会話")
    assert len(api.threads) == 1
    assert page.evaluate("sessionStorage.getItem('taskmate.thread.alice')") == alice_thread

    page.locator("#demo-user").select_option("bob")
    expect(page.locator("#runtime-status")).to_have_text("準備できました")
    page.get_by_label("メッセージ", exact=True).fill("Bobの保存される会話")
    page.get_by_role("button", name="送信", exact=True).click()
    expect(page.locator("#chat-log")).to_contain_text(api.reply)
    expect(page.locator("#runtime-status")).to_have_text("準備できました")
    bob_thread = next(key for key, value in api.threads.items() if value["user_id"] == "bob")

    page.locator("#demo-user").select_option("alice")
    expect(page.locator("#runtime-status")).to_have_text("準備できました")
    expect(page.locator("#chat-log")).to_contain_text("Aliceの保存される会話")
    expect(page.locator("#chat-log")).not_to_contain_text("Bobの保存される会話")
    page.locator("#demo-user").select_option("bob")
    expect(page.locator("#runtime-status")).to_have_text("準備できました")
    expect(page.locator("#chat-log")).to_contain_text("Bobの保存される会話")
    expect(page.locator("#chat-log")).not_to_contain_text("Aliceの保存される会話")
    assert len(api.threads) == 2
    assert page.evaluate("sessionStorage.getItem('taskmate.thread.bob')") == bob_thread
    storage = page.evaluate("Object.fromEntries(Object.entries(sessionStorage))")
    assert storage == {"taskmate.thread.alice": alice_thread, "taskmate.thread.bob": bob_thread}


def test_model_catalog_is_explicit_and_does_not_select_a_default(page, mocked_app):
    api, open_app = mocked_app
    api.connected["alice"] = True
    open_app()

    expect(page.locator("#load-models")).to_be_enabled()
    expect(page.locator("#model-help")).to_be_visible()
    expect(page.locator("#model-help")).to_contain_text("TASKMATE_MODEL")
    expect(page.locator("#chat-message")).to_be_disabled()
    expect(page.locator("#preview-notification")).to_be_disabled()
    assert not any(call["path"] == "/api/models" for call in api.calls)

    page.get_by_role("button", name="一覧を取得", exact=True).click()
    expect(page.locator("#model-list li")).to_have_count(2)
    for model in api.models:
        expect(page.locator("#model-list").get_by_text(model, exact=True)).to_be_visible()
    expect(page.locator("#chat-message")).to_be_disabled()
    expect(page.locator("#task-title")).to_be_enabled()
    assert api.model is None
    assert len([call for call in api.calls if call["path"] == "/api/models"]) == 1


def test_model_content_and_task_titles_render_as_text(page, mocked_app):
    api, open_app = mocked_app
    payload = '<img src=x onerror="window.__taskmate_xss = true">'
    api.connected["alice"] = True
    api.model = "mock-model-small"
    api.tasks["alice"][0]["title"] = payload
    api.reply = payload
    api.notifications["alice"] = [{
        "id": "synthetic-notification",
        "content": payload,
        "created_at": "2026-10-19T00:00:00+00:00",
    }]
    open_app()

    expect(page.locator("#task-list .task-title")).to_have_text(payload)
    expect(page.locator("#notification-list .notification-content")).to_have_text(payload)
    page.get_by_label("メッセージ", exact=True).fill("安全な表示を確認")
    page.get_by_role("button", name="送信", exact=True).click()
    expect(page.locator("#chat-log .assistant .message-content")).to_have_text(payload)
    expect(page.locator("#task-list img, #chat-log img, #notification-list img")).to_have_count(0)
    assert page.evaluate("window.__taskmate_xss") is None


def test_failed_chat_retries_same_uuid_v4_with_fallback(page, mocked_app):
    api, open_app = mocked_app
    api.connected["alice"] = True
    api.model = "mock-model-small"
    api.fail_next_chat = True
    page.add_init_script(
        "Object.defineProperty(window.crypto, 'randomUUID', "
        "{value: undefined, configurable: true});"
    )
    open_app()

    page.get_by_label("メッセージ", exact=True).fill("もう一度確認する")
    page.get_by_role("button", name="送信", exact=True).click()
    expect(page.get_by_role("button", name="再試行", exact=True)).to_be_enabled()
    expect(page.locator("#chat-message")).to_have_attribute("readonly", "")
    page.get_by_role("button", name="再試行", exact=True).click()
    expect(page.locator("#chat-log")).to_contain_text(api.reply)
    chat_calls = [call for call in api.calls if call["path"].endswith("/chat")]
    assert len(chat_calls) == 2
    first_id = chat_calls[0]["body"]["request_id"]
    assert first_id == chat_calls[1]["body"]["request_id"]
    parsed = UUID(first_id)
    assert parsed.version == 4
    assert str(parsed) == first_id
    expect(page.locator("#chat-log .user")).to_have_count(1)


def test_reload_restores_pending_request_without_duplicate_user_message(page, mocked_app):
    api, open_app = mocked_app
    api.connected["alice"] = True
    api.model = "mock-model-small"
    api.fail_next_chat = True
    open_app()

    page.get_by_label("メッセージ", exact=True).fill("途中の相談を再開する")
    page.get_by_role("button", name="送信", exact=True).click()
    expect(page.get_by_role("button", name="再試行", exact=True)).to_be_enabled()
    first_call = next(call for call in api.calls if call["path"].endswith("/chat"))
    page.reload(wait_until="domcontentloaded")
    expect(page.locator("#runtime-status")).to_have_text("準備できました")
    expect(page.get_by_role("button", name="再試行", exact=True)).to_be_enabled()
    expect(page.locator("#chat-message")).to_have_value("途中の相談を再開する")
    expect(page.locator("#chat-log .user")).to_have_count(1)
    assert len(api.threads) == 1

    page.get_by_role("button", name="再試行", exact=True).click()
    expect(page.locator("#chat-log")).to_contain_text(api.reply)
    expect(page.locator("#chat-log .user")).to_have_count(1)
    chat_calls = [call for call in api.calls if call["path"].endswith("/chat")]
    assert len(chat_calls) == 2
    assert chat_calls[1]["body"] == first_call["body"]


def test_profile_timezone_can_load_utc_and_change_to_london_after_chat(page, mocked_app):
    api, open_app = mocked_app
    api.connected["alice"] = True
    api.model = "mock-model-small"
    api.preferences["alice"]["timezone"] = "UTC"
    api.notifications["alice"] = [{
        "id": "timezone-notification",
        "content": "タイムゾーンを確認するためのモック通知です。",
        "created_at": "2026-10-19T00:00:00+00:00",
    }]
    open_app()

    expect(page.locator("#timezone")).to_have_value("UTC")
    expect(page.locator("#timezone")).to_be_enabled()
    expect(page.locator("#tone")).to_be_enabled()
    expect(page.locator("#notification-list .notification-time")).to_contain_text("UTC")

    api.preferences_after_chat = {"timezone": "Europe/London"}
    page.get_by_label("メッセージ", exact=True).fill("通知のタイムゾーンをロンドンに変えて")
    page.get_by_role("button", name="送信", exact=True).click()
    expect(page.locator("#chat-log")).to_contain_text(api.reply)
    expect(page.locator("#runtime-status")).to_have_text("準備できました")
    expect(page.locator("#timezone")).to_have_value("Europe/London")
    expect(page.locator("#timezone")).to_be_enabled()
    expect(page.locator("#tone")).to_be_enabled()
    expect(page.locator("#timezone option[value='Europe/London']")).to_have_text("Europe/London")
    expect(page.locator("#notification-list .notification-time")).to_contain_text("Europe/London")
    expect(page.locator("#error-banner")).to_be_hidden()


def test_disconnect_warns_when_remote_revocation_is_unconfirmed(page, mocked_app):
    api, open_app = mocked_app
    api.connected["alice"] = True
    api.model = "mock-model-small"
    open_app()

    page.get_by_role("button", name="接続を解除", exact=True).click()
    expect(page.locator("#connection-state")).to_have_text("ChatGPT 未接続")
    expect(page.get_by_role("button", name="ChatGPTにログイン", exact=True)).to_be_visible()
    expect(page.locator("#error-banner")).to_be_visible()
    expect(page.locator("#error-banner")).to_contain_text("失効は確認できませんでした")
    expect(page.locator("#chat-message")).to_be_disabled()
    expect(page.locator("#task-title")).to_be_enabled()


def test_synthetic_demo_screenshot(page, mocked_app):
    api, open_app = mocked_app
    api.connected["alice"] = True
    api.model = "mock-model-small"
    api.tasks["alice"] = [
        {"id": "demo-1", "title": "見積もりを送る", "completed": False},
        {"id": "demo-2", "title": "打ち合わせのメモを整理する", "completed": False},
        {"id": "demo-3", "title": "机の上を片づける", "completed": True},
    ]
    api.notifications["alice"] = [{
        "id": "demo-notification",
        "content": "今日は2件のタスクが残っています。ひとつずつ進めましょう。",
        "created_at": "2026-10-19T00:00:00+00:00",
    }]
    api.reply = "まずは「見積もりを送る」から始めましょう。\n送信前に、金額と宛先を確認すると安心です。"
    page.set_viewport_size({"width": 1440, "height": 1040})
    open_app()
    page.get_by_role("button", name="一覧を取得", exact=True).click()
    expect(page.locator("#model-list li")).to_have_count(2)
    page.get_by_label("メッセージ", exact=True).fill("今日のタスク、どこから手をつけよう？")
    page.get_by_role("button", name="送信", exact=True).click()
    expect(page.locator("#chat-log")).to_contain_text(api.reply)
    expect(page.locator("#runtime-status")).to_have_text("準備できました")

    page.evaluate("""() => {
        const label = document.createElement("div");
        label.textContent = "UIテスト画像 · すべて合成データ · 実際のOAuth・モデルは未使用";
        label.style.cssText = "padding:8px 16px;text-align:center;background:#233a32;color:white;font:12px sans-serif";
        document.body.prepend(label);
    }""")
    artifact_dir = ROOT / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    page.screenshot(path=str(artifact_dir / "taskmate-demo.png"), full_page=True)
