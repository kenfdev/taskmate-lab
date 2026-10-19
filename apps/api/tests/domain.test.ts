import { randomUUID } from "node:crypto";
import pg from "pg";
import { AIMessage, type BaseMessage, type MessageContent } from "@langchain/core/messages";
import { BaseChatModel, type BindToolsInput } from "@langchain/core/language_models/chat_models";
import type { ChatResult } from "@langchain/core/outputs";
import { PostgresSaver } from "@langchain/langgraph-checkpoint-postgres";
import { PostgresStore } from "@langchain/langgraph-checkpoint-postgres/store";
import { afterAll, beforeAll, beforeEach, describe, expect, it, vi } from "vite-plus/test";
import {
  DomainRepository, chatSchema, defaultPreferences, preferencesSchema, titleSchema,
} from "../src/domain.js";
import { AgentService, createGraph, domainTools, allowedToolNames, type GraphPort, type RunOptions } from "../src/agent.js";
import { ChatGPTResponsesModel } from "../src/model.js";
import { NotificationService, localDayAndHour } from "../src/worker.js";

describe("domain validation and local scheduling", () => {
  it("trims titles and rejects whitespace-only or oversized input", () => {
    expect(titleSchema.parse("  Buy milk  ")).toBe("Buy milk");
    for (const value of [" ", "", "x".repeat(241)]) expect(titleSchema.safeParse(value).success).toBe(false);
  });
  it.each([true, -1, 24, 1.5, "9"])("rejects invalid notification hour %s", (hour) => {
    expect(preferencesSchema.safeParse({ ...defaultPreferences, notification_hour: hour }).success).toBe(false);
  });
  it("validates IANA zones and strict preference keys", () => {
    expect(defaultPreferences.timezone).toBe("Asia/Tokyo");
    expect(preferencesSchema.safeParse({ ...defaultPreferences, timezone: "Mars/Station" }).success).toBe(false);
    expect(preferencesSchema.safeParse({ ...defaultPreferences, user_id: "bob" }).success).toBe(false);
  });
  it("requires UUID request IDs and nonblank bounded chat text", () => {
    expect(chatSchema.safeParse({ request_id: randomUUID(), message: " " }).success).toBe(false);
    expect(chatSchema.safeParse({ request_id: "bad", message: "hello" }).success).toBe(false);
    expect(chatSchema.safeParse({ request_id: randomUUID(), message: "x".repeat(8001) }).success).toBe(false);
  });
  it("uses local Tokyo date and handles repeated DST hours with one date", () => {
    expect(localDayAndHour(new Date("2026-10-19T00:00:00Z"), "Asia/Tokyo"))
      .toEqual({ day: "2026-10-19", hour: 9 });
    expect(localDayAndHour(new Date("2026-11-01T05:30:00Z"), "America/New_York"))
      .toEqual(localDayAndHour(new Date("2026-11-01T06:30:00Z"), "America/New_York"));
  });
});

const databaseUrl = process.env.TEST_DATABASE_URL;
describe.skipIf(!databaseUrl)("PostgreSQL boundaries and durable agent effects", () => {
  const schema = "tm_test_" + randomUUID().replaceAll("-", "");
  let admin: pg.Pool;
  let pool: pg.Pool;
  let store: PostgresStore;
  let saver: PostgresSaver;
  let repo: DomainRepository;
  const auth = { status: vi.fn(async () => ({ connected: true })), accessToken: vi.fn(async () => "mock-access-token") };

  beforeAll(async () => {
    admin = new pg.Pool({ connectionString: databaseUrl });
    // Generated identifier, never supplied by a request.
    await admin.query('CREATE SCHEMA "' + schema + '"');
    pool = new pg.Pool({ connectionString: databaseUrl, max: 16, options: "-c search_path=" + schema });
    store = new PostgresStore({ connectionOptions: { connectionString: databaseUrl, max: 4 }, schema, ensureTables: false });
    saver = new PostgresSaver(pool, undefined, { schema });
    repo = new DomainRepository(pool, store);
    await repo.setup(); await saver.setup(); await store.setup();
  }, 60_000);
  beforeEach(async () => {
    await pool.query("TRUNCATE tm_chat_requests,tm_tasks,tm_threads,tm_sessions,tm_notifications RESTART IDENTITY CASCADE");
    await store.delete(["users", "alice", "preferences"], "profile");
    await store.delete(["users", "bob", "preferences"], "profile");
    auth.status.mockResolvedValue({ connected: true });
  });
  afterAll(async () => {
    await store?.stop(); await pool?.end();
    if (admin) { await admin.query('DROP SCHEMA IF EXISTS "' + schema + '" CASCADE'); await admin.end(); }
  }, 60_000);

  it("scopes tasks and completion to their owners", async () => {
    const alice = await repo.createTask("alice", "Alice private");
    await repo.createTask("bob", "Bob private");
    expect((await repo.listTasks("alice")).map((task) => task.title)).toEqual(["Alice private"]);
    await expect(repo.completeTask("bob", alice.id)).rejects.toMatchObject({ status: 404 });
    expect((await repo.completeTask("alice", alice.id)).completed).toBe(true);
    expect((await repo.completeTask("alice", alice.id)).completed).toBe(true);
  });
  it("deduplicates tool creates across repository instances and rejects changed data", async () => {
    const thread = await repo.createThread("alice");
    const effect = { threadId: thread.id, toolCallId: "call-1" };
    const first = await repo.createTask("alice", "One durable task", effect);
    const second = await new DomainRepository(pool, store).createTask("alice", "One durable task", effect);
    expect(second.id).toBe(first.id);
    await expect(repo.createTask("alice", "Different", effect)).rejects.toMatchObject({ status: 409 });
    await expect(repo.createTask("bob", "Stolen", effect)).rejects.toMatchObject({ status: 404 });
    expect(await repo.listTasks("alice")).toHaveLength(1);
  });
  it("persists preferences in PostgresStore across threads, isolates profiles", async () => {
    await repo.createThread("alice"); await repo.createThread("alice");
    const preferences = { tone: "friendly" as const, notification_hour: 15, timezone: "Europe/London" };
    await repo.savePreferences("alice", preferences);
    const another = new PostgresStore({ connectionOptions: databaseUrl!, schema, ensureTables: false });
    try {
      expect(await new DomainRepository(pool, another).getPreferences("alice")).toEqual(preferences);
      expect(await new DomainRepository(pool, another).getPreferences("bob")).toEqual(defaultPreferences);
    } finally { await another.stop(); }
  });
  it("rotates cookie secrets and removes the previous server session", async () => {
    const first = await repo.newSession("alice");
    const second = await repo.newSession("bob", first.session.token_hash);
    expect(first.token).not.toBe(second.token);
    expect(second.session.csrf_token).not.toBe(first.session.csrf_token);
    expect(await repo.session(first.token)).toBeNull();
    expect((await repo.session(second.token))?.user_id).toBe("bob");
    expect(await repo.session("not a token")).toBeNull();
  });
  it("uses PostgreSQL locks across independent repository instances and releases on failure", async () => {
    const other = new DomainRepository(pool, store);
    await repo.withLock("test-lock", async () => {
      await expect(other.withLock("test-lock", async () => "bad")).rejects.toMatchObject({ status: 409 });
    });
    await expect(repo.withLock("test-lock", async () => { throw new Error("test failure"); })).rejects.toThrow("test failure");
    expect(await other.withLock("test-lock", async () => "released")).toBe("released");
  });
  it("returns the pending command once with its durable request identity", async () => {
    const thread = await repo.createThread("alice"); const requestId = randomUUID();
    await repo.beginRequest("alice", thread.id, requestId, "Pending message");
    expect(await repo.messages("alice", thread.id)).toEqual([
      { role: "user", content: "Pending message", request_id: requestId, pending: true },
    ]);
    await expect(repo.messages("bob", thread.id)).rejects.toMatchObject({ status: 404 });
    await repo.finishRequest("alice", thread.id, requestId, "Finished");
    expect(await repo.messages("alice", thread.id)).toEqual([
      { role: "user", content: "Pending message" }, { role: "assistant", content: "Finished" },
    ]);
  });
  it("resumes failed tool work, keeps identity trusted, and caches completed requests", async () => {
    const thread = await repo.createThread("alice"); const requestId = randomUUID();
    let accepted = false; let failed = false; let invocations = 0;
    const graph: GraphPort = {
      async getState() { return { values: accepted ? { requestId } : {}, next: accepted ? ["tools"] : [] }; },
      async invoke(input, options) {
        invocations++;
        expect(options.context).toEqual({ userId: "alice", threadId: thread.id });
        if (accepted) expect(input).toBeNull();
        else { expect(input?.requestId).toBe(requestId); accepted = true; }
        await repo.createTask("alice", "Exactly once", { threadId: thread.id, toolCallId: "stable-call" });
        if (!failed) { failed = true; throw new Error("simulated crash after durable task insert"); }
        return { requestId, messages: [new AIMessage("Task saved.")] };
      },
    };
    const chat = new AgentService(repo, auth, "mock-model", saver, store, { graphFactory: () => graph });
    await expect(chat.chat("alice", thread.id, { request_id: requestId, message: "Save it" })).rejects.toThrow("simulated crash");
    await expect(chat.chat("alice", thread.id, { request_id: randomUUID(), message: "New turn" })).rejects.toMatchObject({ status: 409 });
    expect((await chat.chat("alice", thread.id, { request_id: requestId, message: "Save it" })).reply).toBe("Task saved.");
    expect(await repo.listTasks("alice")).toHaveLength(1);
    auth.status.mockResolvedValue({ connected: false });
    expect((await chat.chat("alice", thread.id, { request_id: requestId, message: "Save it" })).reply).toBe("Task saved.");
    expect(invocations).toBe(2);
    await expect(chat.chat("alice", thread.id, { request_id: requestId, message: "Changed" })).rejects.toMatchObject({ status: 409 });
  });
  it("recovers a completed checkpoint without calling the model again", async () => {
    const thread = await repo.createThread("alice"); const requestId = randomUUID();
    await repo.beginRequest("alice", thread.id, requestId, "Recover");
    const invoke = vi.fn();
    const graph: GraphPort = {
      getState: async () => ({ values: { requestId, messages: [new AIMessage("Already done.")] }, next: [] }), invoke,
    };
    const chat = new AgentService(repo, auth, "mock-model", saver, store, { graphFactory: () => graph });
    expect((await chat.chat("alice", thread.id, { request_id: requestId, message: "Recover" })).reply).toBe("Already done.");
    expect(invoke).not.toHaveBeenCalled();
  });
  it("does not create running commands before connection/model prerequisites", async () => {
    const thread = await repo.createThread("alice"); const requestId = randomUUID();
    const noModel = new AgentService(repo, auth, undefined, saver, store);
    await expect(noModel.chat("alice", thread.id, { request_id: requestId, message: "Hello" })).rejects.toMatchObject({ status: 503 });
    expect(await repo.request("alice", thread.id, requestId)).toBeNull();
    auth.status.mockResolvedValue({ connected: false });
    const noLogin = new AgentService(repo, auth, "mock-model", saver, store);
    await expect(noLogin.chat("alice", thread.id, { request_id: requestId, message: "Hello" })).rejects.toMatchObject({ status: 401 });
    expect(await repo.request("alice", thread.id, requestId)).toBeNull();
  });
  it("never releases the advisory lock while an awaited graph invocation is active", async () => {
    const thread = await repo.createThread("alice"); const requestId = randomUUID();
    let unblock!: () => void; let started!: () => void;
    const waiting = new Promise<void>((resolve) => { unblock = resolve; });
    const entered = new Promise<void>((resolve) => { started = resolve; });
    const graph: GraphPort = {
      getState: async () => ({ values: {}, next: [] }),
      invoke: async () => { started(); await waiting; return { messages: [new AIMessage("Done")] }; },
    };
    const chat = new AgentService(repo, auth, "mock-model", saver, store, { graphFactory: () => graph });
    const pending = chat.chat("alice", thread.id, { request_id: requestId, message: "Hold" });
    await entered;
    try {
      await expect(repo.withLock("chat:alice:" + thread.id, async () => "unsafe")).rejects.toMatchObject({ status: 409 });
    } finally { unblock(); }
    await pending;
  });
  it("writes only one daily in-app reminder per user and local date", async () => {
    await repo.createTask("alice", "Alice task"); await repo.createTask("bob", "Bob task");
    const reminder = vi.fn(async (user: string) => "Reminder for " + user);
    const service = new NotificationService(repo, { reminder });
    await service.tick(new Date("2026-10-19T00:00:00Z"));
    await service.tick(new Date("2026-10-19T10:00:00Z"));
    expect(reminder).toHaveBeenCalledTimes(2);
    expect(await repo.notifications("alice")).toHaveLength(1);
    expect((await repo.notifications("bob"))[0]?.content).toBe("Reminder for bob");
    await service.tick(new Date("2026-10-20T00:00:00Z"));
    expect(await repo.notifications("alice")).toHaveLength(2);
  });
  it("deduplicates model previews by owned task/preferences snapshot", async () => {
    await repo.createTask("alice", "Preview task");
    const reminder = vi.fn(async () => "Friendly model text");
    const service = new NotificationService(repo, { reminder });
    const first = await service.preview("alice"); const second = await service.preview("alice");
    expect(second.id).toBe(first.id); expect(reminder).toHaveBeenCalledOnce();
    await repo.savePreferences("alice", { ...defaultPreferences, tone: "friendly" });
    expect((await service.preview("alice")).id).not.toBe(first.id);
    expect(await repo.notifications("bob")).toHaveLength(0);
  });


  it("denies a fabricated built-in tool call at graph execution even from a custom model", async () => {
    class FabricatingModel extends BaseChatModel {
      calls = 0;
      constructor() { super({}); }
      override _llmType(): string { return "taskmate-test"; }
      override bindTools(_tools: BindToolsInput[]) { return this; }
      override async getNumTokens(content: MessageContent): Promise<number> { return Math.ceil(JSON.stringify(content).length / 4); }
      override async _generate(messages: BaseMessage[]): Promise<ChatResult> {
        this.calls++;
        const message = this.calls === 1 ? new AIMessage({
          content: "", tool_calls: [{
            type: "tool_call", id: "blocked-call", name: "write_file",
            args: { file_path: "/blocked.txt", content: "must not be written" },
          }],
        }) : new AIMessage("That action is unavailable.");
        if (this.calls > 1) expect(JSON.stringify(messages)).toContain("This tool is not allowed");
        return { generations: [{ text: typeof message.content === "string" ? message.content : "", message }] };
      }
    }
    const model = new FabricatingModel();
    const chat = new AgentService(repo, auth, "mock-model", saver, store, { modelFactory: () => model });
    const thread = await repo.createThread("alice"); const requestId = randomUUID();
    expect((await chat.chat("alice", thread.id, { request_id: requestId, message: "Attempt an unadvertised file write" })).reply)
      .toBe("That action is unavailable.");
    const graph = createGraph(repo, model, saver, store);
    const state = await graph.getState({
      configurable: { thread_id: "alice:" + thread.id }, context: { userId: "alice", threadId: thread.id },
      recursionLimit: 32, durability: "sync", signal: AbortSignal.timeout(10_000),
    });
    expect((state.values as { files?: Record<string, unknown> }).files ?? {}).not.toHaveProperty("/blocked.txt");
    expect(await repo.listTasks("alice")).toHaveLength(0);
  }, 60_000);

  // Actual adapter + actual DeepAgents + actual PostgreSQL checkpoint/store.
  // Only the HTTP wire is fake. A missing adapter/graph integration cannot pass.
  it("executes a tool through ChatGPT Responses and restores the real checkpoint", async () => {
    const requests: Record<string, unknown>[] = [];
    let turns = 0;
    const fakeFetch: typeof globalThis.fetch = async (input, init) => {
      const url = String(input instanceof Request ? input.url : input);
      if (url.endsWith("/v1/models")) return Response.json({ models: [{ slug: "mock-task-model", visibility: "list" }] });
      if (!url.endsWith("/v1/responses")) throw new Error("Unexpected provider path");
      expect(new Headers(init?.headers).get("authorization")).toBe("Bearer mock-access-token");
      const body = JSON.parse(String(init?.body)) as Record<string, unknown>;
      requests.push(body);
      turns++;
      const output = turns === 1 ? [{
        type: "function_call", id: "fc_mock_1", call_id: "call_mock_1",
        name: "create_task", arguments: JSON.stringify({ title: "Actual adapter task" }), status: "completed",
      }] : [{
        type: "message", id: "msg_mock_1", role: "assistant", status: "completed",
        content: [{ type: "output_text", text: "Saved your task.", annotations: [] }],
      }];
      return new Response("event: response.completed\ndata: " + JSON.stringify({
        type: "response.completed", response: {
          id: "resp_mock_" + turns, object: "response", model: "mock-task-model", status: "completed", output,
          usage: { input_tokens: 20, output_tokens: 10, total_tokens: 30 },
        },
      }) + "\n\n", { headers: { "content-type": "text/event-stream" } });
    };
    const modelFactory = () => new ChatGPTResponsesModel({
      model: "mock-task-model", accessTokenGetter: () => auth.accessToken(), fetch: fakeFetch,
    });
    const chat = new AgentService(repo, auth, "mock-task-model", saver, store, { modelFactory });
    const thread = await repo.createThread("alice"); const requestId = randomUUID();
    const result = await chat.chat("alice", thread.id, { request_id: requestId, message: "Create a task" });
    expect(result.reply).toBe("Saved your task.");
    expect(turns).toBe(2);
    expect((await repo.listTasks("alice")).map((task) => task.title)).toEqual(["Actual adapter task"]);
    expect(await repo.listTasks("bob")).toHaveLength(0);
    const namespaces = requests[0]?.tools as Array<{ type: string; name: string; tools: Array<{ name: string; parameters: { properties: Record<string, unknown> } }> }>;
    expect(namespaces[0]?.type).toBe("namespace");
    expect(namespaces[0]?.name).toBe("taskmate");
    const firstTools = namespaces[0]!.tools;
    const create = firstTools.find((tool) => tool.name === "create_task");
    expect(create?.parameters.properties).not.toHaveProperty("user_id");
    expect(create?.parameters.properties).not.toHaveProperty("userId");
    expect(firstTools.map((tool) => tool.name).sort()).toEqual([...allowedToolNames].sort());
    expect(JSON.stringify(requests[1]?.input)).toContain("function_call_output");
    expect(JSON.stringify(requests[1]?.input)).toContain("call_mock_1");
    const restored = createGraph(repo, modelFactory(), new PostgresSaver(pool, undefined, { schema }), store);
    const options: RunOptions = {
      configurable: { thread_id: "alice:" + thread.id }, context: { userId: "alice", threadId: thread.id },
      recursionLimit: 32, durability: "sync", signal: AbortSignal.timeout(10_000),
    };
    const snapshot = await restored.getState(options);
    expect(snapshot.next).toEqual([]);
    expect(snapshot.values).toMatchObject({ requestId });
    expect((snapshot.values as { files?: Record<string, unknown> }).files ?? {}).not.toHaveProperty("/blocked.txt");
    expect((await chat.chat("alice", thread.id, { request_id: requestId, message: "Create a task" })).reply).toBe(result.reply);
    expect(turns).toBe(2);
    const tools = domainTools(repo);
    expect(tools.map((tool) => tool.name)).toContain("save_preferences");
  }, 60_000);
});
