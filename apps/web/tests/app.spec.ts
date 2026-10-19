import { test as base, expect, type Page, type Route } from "@playwright/test";
import { randomUUID } from "node:crypto";

type DemoUser = "alice" | "bob";
type Tone = "concise" | "friendly";
type Preferences = { tone: Tone; notification_hour: number; timezone: string };
type Task = { id: string; title: string; completed: boolean };
type StoredMessage = {
  role: "user" | "assistant";
  content: string;
  request_id?: string;
  pending?: boolean;
};
type Notice = { id: string; content: string; created_at: string };
type MockThread = { userId: DemoUser; messages: StoredMessage[]; replies: Map<string, string> };
type ApiCall = {
  method: string;
  path: string;
  userId: DemoUser;
  body: Record<string, unknown> | null;
  csrf: string | undefined;
};

class MockApi {
  userId: DemoUser = "alice";
  connected: Record<DemoUser, boolean> = { alice: false, bob: false };
  model: string | null = null;
  models = ["mock-model-small", "mock-model-large"];
  tasks: Record<DemoUser, Task[]> = {
    alice: [{ id: "alice-1", title: "Aliceの見積もり", completed: false }],
    bob: [{ id: "bob-1", title: "Bobの買い物", completed: false }]
  };
  preferences: Record<DemoUser, Preferences> = {
    alice: { tone: "concise", notification_hour: 9, timezone: "Asia/Tokyo" },
    bob: { tone: "friendly", notification_hour: 18, timezone: "Asia/Tokyo" }
  };
  notifications: Record<DemoUser, Notice[]> = { alice: [], bob: [] };
  threads = new Map<string, MockThread>();
  calls: ApiCall[] = [];
  unhandled: string[] = [];
  csrfErrors: string[] = [];
  reply = "モック応答です。タスクをひとつずつ進めましょう。";
  failNextChat = false;
  remoteRevocationConfirmed = false;
  preferencesAfterChat: Partial<Preferences> | null = null;
  chatGate: Promise<void> | null = null;
  chatStarted: (() => void) | null = null;
  completedChats: DemoUser[] = [];

  get csrf(): string {
    return "synthetic-csrf-" + this.userId;
  }

  session() {
    return {
      user_id: this.userId,
      csrf_token: this.csrf,
      chatgpt_connected: this.connected[this.userId],
      model: this.model
    };
  }

  async respond(route: Route, payload: unknown, status = 200): Promise<void> {
    await route.fulfill({ status, contentType: "application/json", body: JSON.stringify(payload) });
  }

  async handle(route: Route): Promise<void> {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    const method = request.method();
    const body = request.postData() ? request.postDataJSON() as Record<string, unknown> : null;
    const user = this.userId;
    const csrf = request.headers()["x-csrf-token"];
    this.calls.push({ method, path, userId: user, body: structuredClone(body), csrf });
    if (method !== "GET" && method !== "HEAD" && csrf !== this.csrf) {
      this.csrfErrors.push(method + " " + path);
      await this.respond(route, { detail: "Synthetic CSRF rejection" }, 403);
      return;
    }
    if (path === "/api/session" && method === "GET") {
      await this.respond(route, this.session());
      return;
    }
    if (path === "/api/session" && method === "POST") {
      const nextUser = body?.user_id;
      if (nextUser !== "alice" && nextUser !== "bob") {
        await this.respond(route, { detail: "Unknown profile" }, 422);
        return;
      }
      this.userId = nextUser;
      await this.respond(route, this.session());
      return;
    }
    if (path === "/api/tasks" && method === "GET") {
      await this.respond(route, this.tasks[user]);
      return;
    }
    if (path === "/api/tasks" && method === "POST") {
      const task = { id: randomUUID(), title: String(body?.title), completed: false };
      this.tasks[user].push(task);
      await this.respond(route, task, 201);
      return;
    }
    if (path === "/api/preferences" && method === "GET") {
      await this.respond(route, this.preferences[user]);
      return;
    }
    if (path === "/api/preferences" && method === "PUT") {
      this.preferences[user] = {
        tone: body?.tone as Tone,
        notification_hour: Number(body?.notification_hour),
        timezone: String(body?.timezone)
      };
      await this.respond(route, this.preferences[user]);
      return;
    }
    if (path === "/api/models" && method === "GET") {
      await this.respond(route, this.connected[user]
        ? { models: this.models } : { detail: "Not connected" }, this.connected[user] ? 200 : 401);
      return;
    }
    if (path === "/api/oauth/disconnect" && method === "POST") {
      this.connected[user] = false;
      await this.respond(route, {
        disconnected: true,
        remote_revocation_confirmed: this.remoteRevocationConfirmed
      });
      return;
    }
    if (path === "/api/notifications" && method === "GET") {
      await this.respond(route, this.notifications[user]);
      return;
    }
    if (path === "/api/notifications/preview" && method === "POST") {
      if (!this.connected[user] || !this.model) {
        await this.respond(route, { detail: "Model unavailable" }, 503);
        return;
      }
      const notice = {
        id: randomUUID(),
        content: "モック通知です。今日のタスクを確認しましょう。",
        created_at: "2026-10-19T00:00:00+00:00"
      };
      this.notifications[user].push(notice);
      await this.respond(route, { content: notice.content });
      return;
    }
    if (path === "/api/threads" && method === "POST") {
      const id = randomUUID();
      this.threads.set(id, { userId: user, messages: [], replies: new Map() });
      await this.respond(route, { id }, 201);
      return;
    }

    const parts = path.split("/").filter(Boolean);
    if (parts.length === 4 && parts[0] === "api" && parts[1] === "tasks" &&
      parts[3] === "complete" && method === "POST") {
      const task = this.tasks[user].find((item) => item.id === parts[2]);
      if (!task) await this.respond(route, { detail: "Task not found" }, 404);
      else {
        task.completed = true;
        await this.respond(route, task);
      }
      return;
    }
    if (parts.length === 4 && parts[0] === "api" && parts[1] === "threads") {
      const thread = this.threads.get(parts[2]!);
      if (!thread || thread.userId !== user) {
        await this.respond(route, { detail: "Thread not found" }, 404);
        return;
      }
      if (parts[3] === "messages" && method === "GET") {
        await this.respond(route, thread.messages);
        return;
      }
      if (parts[3] === "chat" && method === "POST") {
        if (!this.connected[user] || !this.model) {
          await this.respond(route, { detail: "Model unavailable" }, 503);
          return;
        }
        const id = String(body?.request_id);
        expect(id).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i);
        const pending = thread.messages.find((message) => message.pending);
        if (pending && pending.request_id !== id) {
          await this.respond(route, { detail: "Another request is pending" }, 409);
          return;
        }
        if (this.failNextChat) {
          this.failNextChat = false;
          thread.messages.push({ role: "user", content: String(body?.message), request_id: id, pending: true });
          await this.respond(route, { detail: "Synthetic temporary failure" }, 503);
          return;
        }
        this.chatStarted?.();
        if (this.chatGate) await this.chatGate;
        if (!thread.replies.has(id)) {
          thread.messages = thread.messages.filter((message) => !message.pending);
          thread.messages.push(
            { role: "user", content: String(body?.message) },
            { role: "assistant", content: this.reply }
          );
          thread.replies.set(id, this.reply);
          if (this.preferencesAfterChat) {
            this.preferences[user] = { ...this.preferences[user], ...this.preferencesAfterChat };
            this.preferencesAfterChat = null;
          }
        }
        if (!request.failure()) await this.respond(route, { reply: thread.replies.get(id) });
        this.completedChats.push(user);
        return;
      }
    }
    this.unhandled.push(method + " " + path);
    await this.respond(route, { detail: "Unexpected mock request" }, 404);
  }
}

const test = base.extend<{ api: MockApi }>({
  api: async ({ page }, use) => {
    const api = new MockApi();
    await page.route("**/api/**", (route) => api.handle(route));
    await use(api);
    expect(api.unhandled).toEqual([]);
    expect(api.csrfErrors).toEqual([]);
  }
});

async function openApp(page: Page): Promise<void> {
  await page.goto("/");
  await expect(page.locator("#runtime-status")).toHaveText("準備できました");
  await expect(page.locator("#error-banner")).toBeHidden();
  await page.addStyleTag({ content:
    'body::before { content: "UIテスト · すべて合成データ · 実際のOAuth・モデルは未使用"; display: block; padding: 8px 16px; text-align: center; background: #233a32; color: white; font: 12px sans-serif; }'
  });
}

async function sendMessage(page: Page, message: string): Promise<void> {
  await page.getByLabel("メッセージ", { exact: true }).fill(message);
  await page.getByRole("button", { name: "送信", exact: true }).click();
}

test("direct tasks and preferences work without OAuth or a model", async ({ page, api }) => {
  await openApp(page);
  await expect(page.getByRole("button", { name: "ChatGPTにログイン", exact: true })).toBeEnabled();
  await expect(page.locator("#chat-message")).toBeDisabled();
  await expect(page.locator("#preview-notification")).toBeDisabled();
  await expect(page.locator("#task-title")).toBeEnabled();
  await expect(page.locator("#tone")).toBeEnabled();
  await expect(page.locator("#model-setup")).toBeHidden();
  await expect(page.locator("#task-list")).toContainText("Aliceの見積もり");

  await page.getByLabel("新しいタスク", { exact: true }).fill("資料をまとめる");
  await page.getByRole("button", { name: "追加", exact: true }).click();
  await expect(page.locator("#task-list").getByText("資料をまとめる", { exact: true })).toBeVisible();
  await expect(page.locator("#task-count")).toHaveText("残り 2 件");
  await page.getByRole("button", { name: "資料をまとめるを完了にする", exact: true }).click();
  await expect(page.locator(".task-row.completed")).toContainText("資料をまとめる");
  await expect(page.locator("#task-count")).toHaveText("残り 1 件");

  await page.locator("#tone").selectOption("friendly");
  await page.locator("#notification-hour").selectOption("14");
  await page.getByRole("button", { name: "設定を保存", exact: true }).click();
  await expect(page.locator("#preference-status")).toHaveText("保存済み");
  expect(api.preferences.alice).toEqual({ tone: "friendly", notification_hour: 14, timezone: "Asia/Tokyo" });
  expect(api.model).toBeNull();
  for (const call of api.calls.filter((item) => item.method !== "GET" && item.method !== "HEAD")) {
    expect(call.csrf).toBe("synthetic-csrf-" + call.userId);
  }
});

test("profile switches clear the previous tasks and chat", async ({ page, api }) => {
  api.connected = { alice: true, bob: true };
  api.model = "mock-model-small";
  await openApp(page);
  await sendMessage(page, "Aliceだけの相談");
  await expect(page.locator("#chat-log")).toContainText(api.reply);
  await expect(page.locator("#runtime-status")).toHaveText("準備できました");

  await page.locator("#demo-user").selectOption("bob");
  await expect(page.locator("#runtime-status")).toHaveText("準備できました");
  await expect(page.locator("#task-list")).toContainText("Bobの買い物");
  await expect(page.locator("#task-list")).not.toContainText("Aliceの見積もり");
  await expect(page.locator("#chat-log")).not.toContainText("Aliceだけの相談");
  await expect(page.locator("#chat-log")).not.toContainText(api.reply);
  await expect(page.locator("#chat-message")).toHaveValue("");
  await expect(page.locator("#tone")).toHaveValue("friendly");
  await expect(page.locator("#notification-hour")).toHaveValue("18");
  expect(api.threads.size).toBe(2);

  await sendMessage(page, "Bobの相談");
  await expect(page.locator("#chat-log")).toContainText(api.reply);
  const calls = api.calls.filter((call) => call.path.endsWith("/chat"));
  expect(calls.map((call) => call.userId)).toEqual(["alice", "bob"]);
  expect(calls[0]!.path).not.toBe(calls[1]!.path);
});

test("a late Alice reply cannot appear after switching to Bob", async ({ page, api }) => {
  api.connected = { alice: true, bob: true };
  api.model = "mock-model-small";
  let release!: () => void;
  let started!: () => void;
  api.chatGate = new Promise<void>((resolve) => { release = resolve; });
  const chatStarted = new Promise<void>((resolve) => { started = resolve; });
  api.chatStarted = started;
  await openApp(page);
  await sendMessage(page, "Aliceの遅い相談");
  await chatStarted;
  await page.locator("#demo-user").selectOption("bob");
  await expect(page.locator("#task-list")).toContainText("Bobの買い物");
  await expect(page.locator("#runtime-status")).toHaveText("準備できました");
  release();
  await expect.poll(() => api.completedChats).toContain("alice");
  await expect(page.locator("#chat-log")).not.toContainText("Aliceの遅い相談");
  await expect(page.locator("#chat-log")).not.toContainText(api.reply);
  await expect(page.locator("#chat-message")).toHaveValue("");
  await expect(page.locator("#error-banner")).toBeHidden();
});

test("reload and profile changes restore each profile's thread", async ({ page, api }) => {
  api.connected = { alice: true, bob: true };
  api.model = "mock-model-small";
  await openApp(page);
  await sendMessage(page, "Aliceの保存される会話");
  await expect(page.locator("#chat-log")).toContainText(api.reply);
  await expect(page.locator("#runtime-status")).toHaveText("準備できました");
  const aliceThread = [...api.threads.keys()][0]!;
  await page.reload();
  await expect(page.locator("#runtime-status")).toHaveText("準備できました");
  await expect(page.locator("#chat-log")).toContainText("Aliceの保存される会話");
  expect(api.threads.size).toBe(1);

  await page.locator("#demo-user").selectOption("bob");
  await expect(page.locator("#runtime-status")).toHaveText("準備できました");
  await sendMessage(page, "Bobの保存される会話");
  await expect(page.locator("#chat-log")).toContainText(api.reply);
  await expect(page.locator("#runtime-status")).toHaveText("準備できました");
  const bobThread = [...api.threads.entries()].find(([, thread]) => thread.userId === "bob")![0];
  await page.locator("#demo-user").selectOption("alice");
  await expect(page.locator("#chat-log")).toContainText("Aliceの保存される会話");
  await expect(page.locator("#chat-log")).not.toContainText("Bobの保存される会話");
  await expect(page.locator("#runtime-status")).toHaveText("準備できました");
  await page.locator("#demo-user").selectOption("bob");
  await expect(page.locator("#chat-log")).toContainText("Bobの保存される会話");
  await expect(page.locator("#chat-log")).not.toContainText("Aliceの保存される会話");
  expect(api.threads.size).toBe(2);
  expect(await page.evaluate(() => Object.fromEntries(Object.entries(sessionStorage)))).toEqual({
    "taskmate.thread.alice": aliceThread,
    "taskmate.thread.bob": bobThread
  });
});

test("model catalog is explicit and never selects a default", async ({ page, api }) => {
  api.connected.alice = true;
  await openApp(page);
  await expect(page.locator("#load-models")).toBeEnabled();
  await expect(page.locator("#model-help")).toContainText("TASKMATE_MODEL");
  await expect(page.locator("#chat-message")).toBeDisabled();
  await expect(page.locator("#preview-notification")).toBeDisabled();
  expect(api.calls.some((call) => call.path === "/api/models")).toBe(false);

  await page.getByRole("button", { name: "一覧を取得", exact: true }).click();
  await expect(page.locator("#model-list li")).toHaveCount(2);
  for (const model of api.models) {
    await expect(page.locator("#model-list").getByText(model, { exact: true })).toBeVisible();
  }
  await expect(page.locator("#chat-message")).toBeDisabled();
  await expect(page.locator("#task-title")).toBeEnabled();
  expect(api.model).toBeNull();
  expect(api.calls.filter((call) => call.path === "/api/models")).toHaveLength(1);
});

test("model content, task titles, and notifications render as text", async ({ page, api }) => {
  const payload = '<img src=x onerror="document.documentElement.dataset.xss = 1">';
  api.connected.alice = true;
  api.model = "mock-model-small";
  api.tasks.alice[0]!.title = payload;
  api.reply = payload;
  api.notifications.alice = [{ id: "xss-notice", content: payload, created_at: "2026-10-19T00:00:00+00:00" }];
  await openApp(page);
  await expect(page.locator("#task-list .task-title")).toHaveText(payload);
  await expect(page.locator("#notification-list .notification-content")).toHaveText(payload);
  await sendMessage(page, "安全な表示を確認");
  await expect(page.locator("#chat-log .assistant .message-content")).toHaveText(payload);
  await expect(page.locator("#task-list img, #chat-log img, #notification-list img")).toHaveCount(0);
  expect(await page.locator("html").getAttribute("data-xss")).toBeNull();
});

test("failed chats retry the same UUIDv4 when randomUUID is unavailable", async ({ page, api }) => {
  api.connected.alice = true;
  api.model = "mock-model-small";
  api.failNextChat = true;
  await page.addInitScript(() => {
    Object.defineProperty(window.crypto, "randomUUID", { value: undefined, configurable: true });
  });
  await openApp(page);
  await sendMessage(page, "もう一度確認する");
  await expect(page.getByRole("button", { name: "再試行", exact: true })).toBeEnabled();
  await expect(page.locator("#chat-message")).toHaveAttribute("readonly", "");
  await page.getByRole("button", { name: "再試行", exact: true }).click();
  await expect(page.locator("#chat-log")).toContainText(api.reply);
  const calls = api.calls.filter((call) => call.path.endsWith("/chat"));
  expect(calls).toHaveLength(2);
  const id = String(calls[0]!.body?.request_id);
  expect(id).toBe(calls[1]!.body?.request_id);
  expect(id).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/);
  await expect(page.locator("#chat-log .user")).toHaveCount(1);
});

test("reload restores a pending command without duplicating its user message", async ({ page, api }) => {
  api.connected.alice = true;
  api.model = "mock-model-small";
  api.failNextChat = true;
  await openApp(page);
  await sendMessage(page, "途中の相談を再開する");
  await expect(page.getByRole("button", { name: "再試行", exact: true })).toBeEnabled();
  const first = api.calls.find((call) => call.path.endsWith("/chat"))!;
  await page.reload();
  await expect(page.locator("#runtime-status")).toHaveText("準備できました");
  await expect(page.getByRole("button", { name: "再試行", exact: true })).toBeEnabled();
  await expect(page.locator("#chat-message")).toHaveValue("途中の相談を再開する");
  await expect(page.locator("#chat-log .user")).toHaveCount(1);
  expect(api.threads.size).toBe(1);
  await page.getByRole("button", { name: "再試行", exact: true }).click();
  await expect(page.locator("#chat-log")).toContainText(api.reply);
  await expect(page.locator("#chat-log .user")).toHaveCount(1);
  const calls = api.calls.filter((call) => call.path.endsWith("/chat"));
  expect(calls).toHaveLength(2);
  expect(calls[1]!.body).toEqual(first.body);
});

test("UTC and an agent-selected Europe/London timezone stay editable", async ({ page, api }) => {
  api.connected.alice = true;
  api.model = "mock-model-small";
  api.preferences.alice.timezone = "UTC";
  api.notifications.alice = [{
    id: "timezone-notice",
    content: "タイムゾーンを確認するためのモック通知です。",
    created_at: "2026-10-19T00:00:00+00:00"
  }];
  await openApp(page);
  await expect(page.locator("#timezone")).toHaveValue("UTC");
  await expect(page.locator("#timezone")).toBeEnabled();
  await expect(page.locator("#notification-list .notification-time")).toContainText("UTC");
  api.preferencesAfterChat = { timezone: "Europe/London" };
  await sendMessage(page, "通知のタイムゾーンをロンドンに変えて");
  await expect(page.locator("#chat-log")).toContainText(api.reply);
  await expect(page.locator("#runtime-status")).toHaveText("準備できました");
  await expect(page.locator("#timezone")).toHaveValue("Europe/London");
  await expect(page.locator("#timezone")).toBeEnabled();
  await expect(page.locator("#tone")).toBeEnabled();
  await expect(page.locator("#timezone option[value='Europe/London']")).toHaveText("Europe/London");
  await expect(page.locator("#notification-list .notification-time")).toContainText("Europe/London");
  await expect(page.locator("#error-banner")).toBeHidden();
});

test("disconnect reports unconfirmed remote revocation", async ({ page, api }) => {
  api.connected.alice = true;
  api.model = "mock-model-small";
  await openApp(page);
  await page.getByRole("button", { name: "接続を解除", exact: true }).click();
  await expect(page.locator("#connection-state")).toHaveText("ChatGPT 未接続");
  await expect(page.getByRole("button", { name: "ChatGPTにログイン", exact: true })).toBeVisible();
  await expect(page.locator("#error-banner")).toContainText("失効は確認できませんでした");
  await expect(page.locator("#chat-message")).toBeDisabled();
  await expect(page.locator("#task-title")).toBeEnabled();
});

test("synthetic dashboard screenshot", async ({ page, api }, testInfo) => {
  api.connected.alice = true;
  api.model = "mock-model-small";
  api.tasks.alice = [
    { id: "demo-1", title: "見積もりを送る", completed: false },
    { id: "demo-2", title: "打ち合わせのメモを整理する", completed: false },
    { id: "demo-3", title: "机の上を片づける", completed: true }
  ];
  api.notifications.alice = [{
    id: "demo-notice",
    content: "今日は2件のタスクが残っています。ひとつずつ進めましょう。",
    created_at: "2026-10-19T00:00:00+00:00"
  }];
  api.reply = "まずは「見積もりを送る」から始めましょう。\n送信前に、金額と宛先を確認すると安心です。";
  await page.setViewportSize({ width: 1440, height: 1040 });
  await openApp(page);
  await page.getByRole("button", { name: "一覧を取得", exact: true }).click();
  await expect(page.locator("#model-list li")).toHaveCount(2);
  await sendMessage(page, "今日のタスク、どこから手をつけよう？");
  await expect(page.locator("#chat-log")).toContainText(api.reply);
  await expect(page.locator("#runtime-status")).toHaveText("準備できました");
  await page.screenshot({ path: testInfo.outputPath("taskmate-demo.png"), fullPage: true });
});
