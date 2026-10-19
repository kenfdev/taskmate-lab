import { randomBytes, randomUUID } from "node:crypto";
import { beforeEach, describe, expect, it, vi } from "vite-plus/test";
import { createApp, settingsFromEnv, type AppServices } from "../src/app.js";
import {
  DomainError, chatSchema, defaultPreferences, preferencesSchema, uuidSchema,
  type ChatMessage, type Preferences, type Session, type Task, type UserId,
} from "../src/domain.js";

const ORIGIN = "http://127.0.0.1:8765";
function harness(model: string | null = "mock-model", address = "127.0.0.1") {
  const sessions = new Map<string, Session>();
  const tasks = new Map<UserId, Task[]>([["alice", []], ["bob", []]]);
  const preferences = new Map<UserId, Preferences>();
  const threads = new Map<string, { user: UserId; messages: ChatMessage[] }>();
  const connected = new Set<UserId>();
  const pendingOAuth = new Set<string>();
  const status = (user: string) => ({
    connected: connected.has(user as UserId), plan_enabled: connected.has(user as UserId), email: null, client_id: null,
  });
  const owned = (user: UserId, id: string) => {
    uuidSchema.parse(id);
    const thread = threads.get(id);
    if (!thread || thread.user !== user) throw new DomainError(404, "Thread not found.");
    return thread;
  };
  const complete = vi.fn(async (user: string, sessionId: string, callback: { state: string }) => {
    if (!pendingOAuth.delete(callback.state + ":" + sessionId)) throw new DomainError(400, "Bad OAuth state.");
    connected.add(user as UserId);
    return status(user);
  });
  const services: AppServices = {
    settings: { databaseUrl: "unused-test", encryptionKey: "unused-test", model: model ?? undefined },
    repo: {
      async session(token) { return token ? sessions.get(token) ?? null : null; },
      async newSession(user, oldHash) {
        if (oldHash) for (const [token, value] of sessions) if (value.token_hash === oldHash) sessions.delete(token);
        const token = randomBytes(32).toString("base64url");
        const session = { token_hash: "hash-" + token, user_id: user, csrf_token: randomBytes(32).toString("base64url") };
        sessions.set(token, session);
        return { token, session };
      },
      async listTasks(user) { return tasks.get(user)!; },
      async createTask(user, title) {
        const task = { id: randomUUID(), title, completed: false, created_at: new Date().toISOString() };
        tasks.get(user)!.push(task); return task;
      },
      async completeTask(user, id) {
        uuidSchema.parse(id);
        const task = tasks.get(user)!.find((item) => item.id === id);
        if (!task) throw new DomainError(404, "Task not found.");
        task.completed = true; return task;
      },
      async getPreferences(user) { return preferences.get(user) ?? { ...defaultPreferences }; },
      async savePreferences(user, input) {
        const value = preferencesSchema.parse(input); preferences.set(user, value); return value;
      },
      async createThread(user) {
        const id = randomUUID(); threads.set(id, { user, messages: [] }); return { id };
      },
      async messages(user, id) { return owned(user, id).messages; },
      async notifications() { return []; },
    },
    auth: {
      async begin(_user, sessionId) {
        pendingOAuth.add("state-good:" + sessionId);
        return "https://auth.openai.com/authorize?state=state-good";
      },
      complete,
      async status(user) { return status(user); },
      async accessToken(user) {
        if (!status(user).connected) throw new DomainError(401, "Connect ChatGPT.");
        return "mock-access-token";
      },
      async disconnect(user) { connected.delete(user as UserId); return false; },
    },
    chat: {
      async availableModels(user) { await services.auth.accessToken(user); return ["account-model-1"]; },
      async chat(user, id, input) {
        const request = chatSchema.parse(input); const thread = owned(user, id);
        if (!model) throw new DomainError(503, "Set TASKMATE_MODEL.");
        await services.auth.accessToken(user);
        thread.messages.push({ role: "user", content: request.message }, { role: "assistant", content: "Reply" });
        return { reply: "Reply", request_id: request.request_id };
      },
      async reminder(user) { await services.auth.accessToken(user); return "Model reminder"; },
    },
    notifications: {
      async preview(user) {
        if (!model) throw new DomainError(503, "Set TASKMATE_MODEL.");
        await services.auth.accessToken(user);
        return { id: randomUUID(), kind: "preview", content: "Model reminder", created_at: new Date().toISOString() };
      },
    },
  };
  const app = createApp(services, { getClientAddress: () => address });
  let cookie = ""; let csrf = "";
  async function call(path: string, method = "GET", input?: unknown, extra: Record<string, string> = {}) {
    const headers: Record<string, string> = {
      host: "127.0.0.1:8765", cookie, "x-csrf-token": csrf,
      ...(method !== "GET" ? { origin: ORIGIN, "content-type": "application/json" } : {}), ...extra,
    };
    const response = await app.request(ORIGIN + path, {
      method, headers, ...(input === undefined ? {} : { body: JSON.stringify(input) }),
    });
    const newCookie = response.headers.get("set-cookie");
    if (newCookie) cookie = newCookie.split(";")[0]!;
    if (path === "/api/session" && response.ok) {
      const body = await response.clone().json() as { csrf_token: string };
      csrf = body.csrf_token;
    }
    return response;
  }
  return {
    app, services, call, connected, threads, pendingOAuth, complete,
    cookie: () => cookie, csrf: () => csrf,
  };
}

describe("Hono local-only API", () => {
  let h: ReturnType<typeof harness>;
  beforeEach(async () => { h = harness(); await h.call("/api/session"); });

  it("starts without a selected model and uses an HttpOnly Lax cookie for OAuth", async () => {
    const first = harness(null);
    const response = await first.call("/api/session");
    expect(response.status).toBe(200);
    expect(await response.json()).toMatchObject({ user_id: "alice", chatgpt_connected: false, model: null });
    expect(response.headers.get("set-cookie")).toContain("HttpOnly");
    expect(response.headers.get("set-cookie")).toContain("SameSite=Lax");
    expect(response.headers.get("set-cookie")).not.toContain("SameSite=Strict");
    expect(response.headers.get("cache-control")).toBe("no-store");
    expect(response.headers.get("referrer-policy")).toBe("no-referrer");
  });
  it("requires DATABASE_URL and an explicit encryption key but not a model", () => {
    expect(() => settingsFromEnv({})).toThrow("DATABASE_URL");
    expect(settingsFromEnv({ DATABASE_URL: "postgres://local", TASKMATE_ENCRYPTION_KEY: "key" }).model).toBeUndefined();
  });
  it.each([
    ["/api/session", "POST", { user_id: "bob" }],
    ["/api/oauth/start", "POST", {}],
    ["/api/oauth/disconnect", "POST", {}],
    ["/api/tasks", "POST", { title: "Unsafe" }],
    ["/api/tasks/" + randomUUID() + "/complete", "POST", {}],
    ["/api/preferences", "PUT", defaultPreferences],
    ["/api/threads", "POST", {}],
    ["/api/threads/" + randomUUID() + "/chat", "POST", { message: "Hi", request_id: randomUUID() }],
    ["/api/notifications/preview", "POST", {}],
  ])("requires CSRF for %s %s", async (path, method, input) => {
    expect((await h.call(path as string, method as string, input, { "x-csrf-token": "" })).status).toBe(403);
  });
  it.each(["/api/tasks", "/api/preferences", "/api/notifications", "/api/models", "/api/threads/" + randomUUID() + "/messages"])(
    "binds scoped GET %s to its current tab token", async (path) => {
      expect((await h.call(path, "GET", undefined, { "x-csrf-token": "" })).status).toBe(403);
    });
  it("rejects wrong Host, Origin and nonloopback socket peers", async () => {
    expect((await h.call("/api/session", "GET", undefined, { host: "evil.example:8765" })).status).toBe(403);
    expect((await h.call("/api/tasks", "POST", { title: "Unsafe" }, { origin: "https://evil.example" })).status).toBe(403);
    expect((await h.call("/api/tasks", "POST", { title: "Unsafe" }, { origin: "" })).status).toBe(403);
    expect((await harness("mock", "192.0.2.2").call("/api/session")).status).toBe(403);
  });
  it("rejects cross-site reads even when cookies and CSRF are forged in the test", async () => {
    expect((await h.call("/api/tasks", "GET", undefined, { "sec-fetch-site": "cross-site" })).status).toBe(403);
  });
  it("rotates session and CSRF and blocks a stale tab after a shared-cookie profile change", async () => {
    const oldCookie = h.cookie(); const oldCsrf = h.csrf();
    await h.call("/api/tasks", "POST", { title: "Alice only" });
    const switched = await h.call("/api/session", "POST", { user_id: "bob" });
    expect((await switched.json()).user_id).toBe("bob");
    expect(h.cookie()).not.toBe(oldCookie); expect(h.csrf()).not.toBe(oldCsrf);
    expect((await h.call("/api/tasks", "GET", undefined, { "x-csrf-token": oldCsrf })).status).toBe(403);
    expect((await h.call("/api/tasks", "GET", undefined, { cookie: oldCookie, "x-csrf-token": oldCsrf })).status).toBe(401);
    expect(await (await h.call("/api/tasks")).json()).toEqual([]);
    await h.call("/api/session", "POST", { user_id: "alice" });
    expect(await (await h.call("/api/tasks")).json()).toMatchObject([{ title: "Alice only" }]);
  });
  it("validates strict task and preference DTOs", async () => {
    for (const title of [" ", "", "x".repeat(241)]) expect((await h.call("/api/tasks", "POST", { title })).status).toBe(422);
    expect((await h.call("/api/tasks", "POST", { title: "Okay", user_id: "bob" })).status).toBe(422);
    expect((await h.call("/api/preferences", "PUT", { ...defaultPreferences, notification_hour: true })).status).toBe(422);
    expect((await h.call("/api/preferences", "PUT", { ...defaultPreferences, timezone: "invalid/zone" })).status).toBe(422);
    expect(await (await h.call("/api/preferences")).json()).toEqual(defaultPreferences);
  });
  it("bounds JSON bodies", async () => {
    expect((await h.call("/api/tasks", "POST", { title: "x".repeat(40_000) })).status).toBe(413);
  });
  it("checks callback state before consuming the pending OAuth attempt", async () => {
    expect((await h.call("/api/oauth/start", "POST", {})).status).toBe(200);
    expect(h.pendingOAuth.size).toBe(1);
    for (const path of ["/auth/callback?code=fake", "/auth/callback?state=&code=fake", "/auth/callback?state=" + "s".repeat(1025)]) {
      expect((await h.call(path)).status).toBe(400);
    }
    expect(h.complete).not.toHaveBeenCalled();
    expect(h.pendingOAuth.size).toBe(1);
    const response = await h.call("/auth/callback?state=state-good&code=fake&client_id=fake-client");
    expect(response.status).toBe(303); expect(response.headers.get("location")).toBe("/");
    expect(h.complete).toHaveBeenCalledOnce();
    expect(h.pendingOAuth.size).toBe(0);
  });
  it("requires a bound session for callbacks and never echoes callback secrets", async () => {
    const response = await harness().call("/auth/callback?state=secret-state&code=secret-code");
    expect(response.status).toBe(401);
    expect(await response.text()).not.toContain("secret-code");
  });
  it("exposes catalogue IDs only through the profile credential and reports revocation uncertainty", async () => {
    expect((await h.call("/api/models")).status).toBe(401);
    h.connected.add("alice");
    expect(await (await h.call("/api/models")).json()).toEqual({ models: ["account-model-1"] });
    const response = await h.call("/api/oauth/disconnect", "POST", {});
    expect(await response.json()).toEqual({ disconnected: true, remote_revocation_confirmed: false });
  });
  it("restores one pending command and denies cross-profile thread access", async () => {
    const thread = await (await h.call("/api/threads", "POST", {})).json() as { id: string };
    const requestId = randomUUID();
    h.threads.get(thread.id)!.messages.push({ role: "user", content: "Retry me", request_id: requestId, pending: true });
    expect(await (await h.call("/api/threads/" + thread.id + "/messages")).json()).toEqual([
      { role: "user", content: "Retry me", request_id: requestId, pending: true },
    ]);
    await h.call("/api/session", "POST", { user_id: "bob" });
    expect((await h.call("/api/threads/" + thread.id + "/messages")).status).toBe(404);
  });
  it("generates a model preview only when connected and configured", async () => {
    expect((await h.call("/api/notifications/preview", "POST", {})).status).toBe(401);
    h.connected.add("alice");
    expect(await (await h.call("/api/notifications/preview", "POST", {})).json()).toMatchObject({ kind: "preview", content: "Model reminder" });
    const noModel = harness(null); await noModel.call("/api/session"); noModel.connected.add("alice");
    expect((await noModel.call("/api/notifications/preview", "POST", {})).status).toBe(503);
  });
  it("returns generic errors without serializing exceptions or attached provider data", async () => {
    h.services.repo.listTasks = async () => { throw Object.assign(new Error("secret-token"), { response: "private body" }); };
    const response = await h.call("/api/tasks");
    expect(response.status).toBe(500);
    expect(await response.text()).toBe('{"detail":"The local service failed. Retry the same request."}');
  });
});
