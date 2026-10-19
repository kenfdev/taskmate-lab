import { describe, expect, it, vi } from "vite-plus/test";
import {
  AIMessage,
  HumanMessage,
  SystemMessage,
  ToolMessage,
  mapChatMessagesToStoredMessages,
  mapStoredMessagesToChatMessages,
} from "@langchain/core/messages";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import {
  ChatGPTResponsesModel,
  RESPONSES_OUTPUT_KEY,
  ResponsesError,
  type ResponsesCallOptions,
} from "../src/model.js";

type JsonObject = Record<string, unknown>;
const MODEL = "catalog-model";
const TOKEN = "test-only-access-token";
const MODELS_URL = "https://api.openai.com/v1/models";
const RESPONSES_URL = "https://api.openai.com/v1/responses";
const functionTool = {
  type: "function",
  function: {
    name: "lookup",
    description: "Look up a local task.",
    parameters: { type: "object", properties: { query: { type: "string" } }, required: ["query"] },
  },
};

function catalog(models: unknown = [{ slug: MODEL, visibility: "list" }]): Response {
  return Response.json({ models });
}

function message(text = "Hello"): JsonObject {
  return {
    type: "message",
    id: "msg_1",
    role: "assistant",
    status: "completed",
    content: [{ type: "output_text", text, annotations: [] }],
  };
}

function completed(output: JsonObject[] = [message()]): JsonObject {
  return {
    type: "response.completed",
    response: {
      id: "resp_1",
      model: MODEL,
      status: "completed",
      output,
      usage: { input_tokens: 8, output_tokens: 4, total_tokens: 12 },
    },
  };
}

function encodeEvent(event: JsonObject): string {
  return "data: " + JSON.stringify(event) + "\n\n";
}

function sse(events: JsonObject[] = [completed()]): Response {
  return new Response(events.map(encodeEvent).join(""), {
    headers: { "Content-Type": "text/event-stream; charset=utf-8" },
  });
}

function wireHarness(responses: Response[] = [sse()], catalogFactory = () => catalog()) {
  const requests: Array<{ url: string; init: RequestInit | undefined }> = [];
  const fetch = vi.fn<typeof globalThis.fetch>(async (input, init) => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    requests.push({ url, init });
    if (url === MODELS_URL) return catalogFactory();
    if (url !== RESPONSES_URL) throw new Error("Unexpected test URL");
    const next = responses.shift();
    if (!next) throw new Error("No mocked inference response remains");
    return next;
  });
  const accessTokenGetter = vi.fn(async () => TOKEN);
  const model = new ChatGPTResponsesModel({ model: MODEL, accessTokenGetter, fetch });
  return {
    model,
    fetch,
    requests,
    accessTokenGetter,
    bodies(): JsonObject[] {
      return requests
        .filter((request) => request.url === RESPONSES_URL)
        .map((request) => {
          const body = request.init?.body;
          if (typeof body !== "string") throw new Error("Expected a JSON string request body");
          return JSON.parse(body) as JsonObject;
        });
    },
  };
}

async function rejection(promise: Promise<unknown>): Promise<unknown> {
  try {
    await promise;
  } catch (error: unknown) {
    return error;
  }
  throw new Error("Expected a rejection");
}

describe("ChatGPT account catalog", () => {
  it("lists visible account models before a selection, preserving order and removing duplicates", async () => {
    const fetch = vi.fn<typeof globalThis.fetch>(async () =>
      catalog([
        { slug: "second", visibility: "list" },
        { slug: "hidden", visibility: "hidden" },
        { slug: "first", visibility: "list" },
        { slug: "second", visibility: "list" },
      ]),
    );
    const model = new ChatGPTResponsesModel({ accessTokenGetter: async () => TOKEN, fetch });
    expect(await model.availableModels()).toEqual(["second", "first"]);
    expect(fetch).toHaveBeenCalledWith(
      MODELS_URL,
      expect.objectContaining({
        headers: { Authorization: "Bearer " + TOKEN },
        redirect: "error",
      }),
    );
    await expect(model.invoke("hello")).rejects.toThrow("Select a model");
    expect(fetch).toHaveBeenCalledTimes(1);
  });

  it.each([
    { data: [{ id: MODEL }] },
    { models: {} },
    { models: [null] },
    { models: [{ slug: "", visibility: "list" }] },
  ])("rejects unsupported or malformed catalog schemas: %j", async (payload) => {
    const model = new ChatGPTResponsesModel({
      accessTokenGetter: async () => TOKEN,
      fetch: async () => Response.json(payload),
    });
    await expect(model.availableModels()).rejects.toBeInstanceOf(ResponsesError);
  });

  it("checks the current account catalog before every inference and never guesses a model", async () => {
    const harness = wireHarness([], () => catalog([{ slug: "other", visibility: "list" }]));
    await expect(harness.model.invoke("hi")).rejects.toThrow("visible catalog");
    expect(harness.requests.map((request) => request.url)).toEqual([MODELS_URL]);
  });
});

describe("Responses wire contract", () => {
  it("uses the official OAuth endpoints, converts system to developer, and keeps local config off the wire", async () => {
    const harness = wireHarness();
    const output = await harness.model.invoke(
      [new SystemMessage("Be helpful"), new HumanMessage("Hi")],
      { metadata: { tag: "local-only" }, tags: ["local"] },
    );
    expect(output.content).toBe("Hello");
    expect(output.response_metadata).toMatchObject({ response_id: "resp_1", model_name: MODEL });
    expect(output.usage_metadata).toEqual({ input_tokens: 8, output_tokens: 4, total_tokens: 12 });
    expect(harness.accessTokenGetter).toHaveBeenCalledTimes(1);
    expect(harness.requests.map((request) => request.url)).toEqual([MODELS_URL, RESPONSES_URL]);
    for (const request of harness.requests) {
      expect(new Headers(request.init?.headers).get("Authorization")).toBe("Bearer " + TOKEN);
      expect(request.init?.redirect).toBe("error");
    }
    expect(harness.bodies()).toEqual([
      {
        model: MODEL,
        input: [
          { role: "developer", content: "Be helpful" },
          { role: "user", content: "Hi" },
        ],
        store: false,
        stream: true,
        include: ["reasoning.encrypted_content"],
      },
    ]);
    const request = harness.requests[1];
    expect(request?.init?.method).toBe("POST");
    expect(new Headers(request?.init?.headers).get("Accept")).toBe("text/event-stream");
    expect(JSON.stringify(harness.model)).not.toContain(TOKEN);
    expect(JSON.stringify(harness.model)).not.toContain("accessTokenGetter");
  });

  it("binds a real LangChain tool under a namespace without mutating the unbound model", async () => {
    const harness = wireHarness([sse(), sse()]);
    const lookup = tool(async ({ query }) => query, {
      name: "lookup",
      description: "Look up a local task.",
      schema: z.object({ query: z.string() }),
    });
    const bound = harness.model.bindTools([lookup], { strict: true, tool_choice: "any" });
    await bound.invoke("Search");
    await harness.model.invoke("No tools");
    expect(harness.bodies()[0]).toMatchObject({
      tool_choice: "required",
      tools: [
        {
          type: "namespace",
          name: "taskmate",
          tools: [
            {
              type: "function",
              name: "lookup",
              strict: true,
              parameters: { type: "object", properties: { query: { type: "string" } } },
            },
          ],
        },
      ],
    });
    expect(harness.bodies()[1]).not.toHaveProperty("tools");
    expect(functionTool.function).not.toHaveProperty("strict");
  });

  it("round-trips function outputs and completed encrypted reasoning through stored LangChain messages", async () => {
    const reasoning = {
      type: "reasoning",
      id: "rs_1",
      summary: [],
      encrypted_content: "completed-ciphertext",
    };
    const call = {
      type: "function_call",
      id: "fc_1",
      call_id: "call_1",
      namespace: "taskmate",
      name: "lookup",
      arguments: '{"query":"today"}',
      status: "completed",
    };
    const harness = wireHarness([
      sse([
        {
          type: "response.output_item.added",
          output_index: 0,
          item: { ...reasoning, encrypted_content: "partial" },
        },
        { type: "response.output_item.done", output_index: 0, item: reasoning },
        { type: "response.output_item.done", output_index: 1, item: call },
        completed([{ ...reasoning, encrypted_content: "stale" }, call]),
      ]),
      sse([completed([message("One task")])]),
    ]);
    const bound = harness.model.bindTools([functionTool]);
    const first = await bound.invoke([new HumanMessage("What is due?")]);
    expect(first.content).toBe("");
    expect(first.tool_calls).toEqual([
      { type: "tool_call", id: "call_1", name: "lookup", args: { query: "today" } },
    ]);
    expect(first.additional_kwargs[RESPONSES_OUTPUT_KEY]).toEqual([reasoning, call]);
    const restored = mapStoredMessagesToChatMessages(
      mapChatMessagesToStoredMessages([
        new HumanMessage("What is due?"),
        first,
        new ToolMessage({ content: "One task", tool_call_id: "call_1" }),
      ]),
    );
    expect((await bound.invoke(restored)).content).toBe("One task");
    expect(harness.bodies()[1]?.input).toEqual([
      { role: "user", content: "What is due?" },
      reasoning,
      call,
      { type: "function_call_output", call_id: "call_1", output: "One task" },
    ]);
    expect(harness.bodies()[1]).not.toHaveProperty("previous_response_id");
    expect(harness.accessTokenGetter).toHaveBeenCalledTimes(2);
  });

  it("respects middleware edits to saved function arguments while retaining opaque reasoning", async () => {
    const reasoning = { type: "reasoning", id: "rs_1", summary: [], encrypted_content: "opaque" };
    const call = {
      type: "function_call",
      id: "fc_1",
      call_id: "call_1",
      namespace: "taskmate",
      name: "lookup",
      arguments: '{"query":"long original"}',
    };
    const harness = wireHarness([sse([completed([reasoning, call])]), sse()]);
    const bound = harness.model.bindTools([functionTool]);
    const first = await bound.invoke("Search");
    const currentCall = first.tool_calls?.[0];
    if (!currentCall) throw new Error("Expected a call");
    currentCall.args = { query: "[trimmed]" };
    await bound.invoke([
      new HumanMessage("Search"),
      first,
      new ToolMessage({
        content: "done",
        tool_call_id: "call_1",
      }),
    ]);
    expect(harness.bodies()[1]?.input).toEqual([
      { role: "user", content: "Search" },
      reasoning,
      { ...call, arguments: '{"query":"[trimmed]"}' },
      { type: "function_call_output", call_id: "call_1", output: "done" },
    ]);
  });

  it("round-trips custom tool input and output without attempting to parse its free-form input as JSON", async () => {
    const custom = { type: "custom", name: "run_text", format: { type: "text" } };
    const call = {
      type: "custom_tool_call",
      id: "ct_1",
      call_id: "call_1",
      namespace: "taskmate",
      name: "run_text",
      input: "plain text, not JSON",
    };
    const harness = wireHarness([sse([completed([call])]), sse()]);
    const bound = harness.model.bindTools([custom]);
    const first = await bound.invoke("Use the local tool");
    expect(first.tool_calls?.[0]?.args).toEqual({ __arg1: "plain text, not JSON" });
    await bound.invoke([first, new ToolMessage({ content: "ok", tool_call_id: "call_1" })]);
    expect(harness.bodies()[1]?.input).toEqual([
      call,
      { type: "custom_tool_call_output", call_id: "call_1", output: "ok" },
    ]);
  });

  it("recognizes the exact taskmate prefix when no namespace field is supplied", async () => {
    const harness = wireHarness([
      sse([
        completed([
          {
            type: "function_call",
            name: "taskmate.lookup",
            call_id: "call_1",
            arguments: "{}",
          },
        ]),
      ]),
    ]);
    const result = await harness.model.bindTools([functionTool]).invoke("Search");
    expect(result.tool_calls?.[0]?.name).toBe("lookup");
  });

  it.each([
    { name: "lookup", namespace: "other", arguments: "{}" },
    { name: "other.lookup", arguments: "{}" },
    { name: "unknown", namespace: "taskmate", arguments: "{}" },
    { name: "lookup", namespace: "taskmate", arguments: "[]" },
    { name: "lookup", namespace: "taskmate", arguments: "private invalid JSON" },
  ])("rejects unsafe or invalid function calls: %j", async (fields) => {
    const harness = wireHarness([
      sse([
        completed([
          {
            type: "function_call",
            call_id: "call_1",
            ...fields,
          },
        ]),
      ]),
    ]);
    const error = await rejection(harness.model.bindTools([functionTool]).invoke("Search"));
    expect(error).toBeInstanceOf(ResponsesError);
    expect(String(error)).not.toContain("private invalid JSON");
  });

  it("rejects orphan or repeated tool results before obtaining a token", async () => {
    const harness = wireHarness();
    await expect(
      harness.model.invoke([new ToolMessage({ content: "x", tool_call_id: "missing" })]),
    ).rejects.toThrow("preceding unanswered");
    const call = new AIMessage({
      content: "",
      tool_calls: [{ id: "call_1", name: "lookup", args: {} }],
    });
    await expect(
      harness.model.invoke([
        call,
        new ToolMessage({ content: "x", tool_call_id: "call_1" }),
        new ToolMessage({ content: "x", tool_call_id: "call_1" }),
      ]),
    ).rejects.toThrow("preceding unanswered");
    expect(harness.accessTokenGetter).not.toHaveBeenCalled();
    expect(harness.fetch).not.toHaveBeenCalled();
  });

  it.each([
    { store: true },
    { store: false },
    { store: "enabled" },
    { stream: false },
    { previous_response_id: "resp_old" },
    { temperature: 1 },
    { max_output_tokens: 100 },
    { stop: ["x"] },
    { background: true },
    { additional_tools: [] },
    { api_key: "test-only-key" },
    { outputVersion: "v1" },
    { maxRetries: 2 },
  ])("rejects unsupported request options before network access: %j", async (options) => {
    const harness = wireHarness();
    await expect(
      harness.model.invoke("Hi", options as Partial<ResponsesCallOptions>),
    ).rejects.toBeInstanceOf(ResponsesError);
    expect(harness.fetch).not.toHaveBeenCalled();
    expect(harness.accessTokenGetter).not.toHaveBeenCalled();
  });

  it("accepts local LangGraph runtime options without leaking them into inference", async () => {
    const harness = wireHarness();
    const graphStore = { namespace: ["private-profile"], get: vi.fn(), put: vi.fn() };
    const writer = vi.fn();
    const interrupt = vi.fn();
    await harness.model.invoke("Hi", {
      context: { userId: "private-profile" },
      control: { local: "control-marker" },
      durability: "sync",
      executionInfo: { local: "execution-marker" },
      interrupt,
      store: graphStore,
      writer,
    });
    expect(harness.bodies()).toEqual([
      {
        model: MODEL,
        input: [{ role: "user", content: "Hi" }],
        store: false,
        stream: true,
        include: ["reasoning.encrypted_content"],
      },
    ]);
    const body = JSON.stringify(harness.bodies());
    expect(body).not.toContain("private-profile");
    expect(body).not.toContain("control-marker");
    expect(body).not.toContain("execution-marker");
    expect(graphStore.get).not.toHaveBeenCalled();
    expect(graphStore.put).not.toHaveBeenCalled();
    expect(writer).not.toHaveBeenCalled();
    expect(interrupt).not.toHaveBeenCalled();
  });

  it("passes only supported inference settings", async () => {
    const harness = wireHarness();
    await harness.model.invoke("Hi", {
      instructions: "Be brief",
      reasoning: { effort: "low" },
      text: { verbosity: "low" },
      parallel_tool_calls: false,
      tool_choice: "none",
    });
    expect(harness.bodies()[0]).toMatchObject({
      instructions: "Be brief",
      reasoning: { effort: "low" },
      text: { verbosity: "low" },
      parallel_tool_calls: false,
      tool_choice: "none",
    });
  });

  it("estimates tokens locally without a tokenizer download or an access token", async () => {
    const harness = wireHarness();
    expect(await harness.model.getNumTokens("12345678")).toBe(2);
    expect(await harness.model.getNumTokensFromMessages([new HumanMessage("1234")])).toBe(4);
    expect(harness.fetch).not.toHaveBeenCalled();
    expect(harness.accessTokenGetter).not.toHaveBeenCalled();
  });
});

describe("SSE parsing, failures, and cancellation", () => {
  it("handles byte-split UTF-8, CRLF, comments, and multiline event data", async () => {
    const final = JSON.stringify(completed([message("日本語 😀")]), null, 2);
    const wire =
      ": heartbeat\r\n\r\n" +
      final
        .split("\n")
        .map((line) => "data: " + line + "\r\n")
        .join("") +
      "\r\n";
    const bytes = new TextEncoder().encode(wire);
    let offset = 0;
    const stream = new ReadableStream<Uint8Array>({
      pull(controller) {
        if (offset >= bytes.length) controller.close();
        else controller.enqueue(bytes.slice(offset, ++offset));
      },
    });
    const harness = wireHarness([
      new Response(stream, { headers: { "Content-Type": "text/event-stream" } }),
    ]);
    expect((await harness.model.invoke("Hi")).content).toBe("日本語 😀");
  });

  it("forwards text deltas to callbacks while keeping a completed message result", async () => {
    const harness = wireHarness([
      sse([
        { type: "response.output_text.delta", delta: "He" },
        { type: "response.output_text.delta", delta: "llo" },
        completed(),
      ]),
    ]);
    const deltas: string[] = [];
    const result = await harness.model.invoke("Hi", {
      callbacks: [
        {
          handleLLMNewToken(token: string) {
            deltas.push(token);
          },
        },
      ],
    });
    expect(deltas).toEqual(["He", "llo"]);
    expect(result.content).toBe("Hello");
  });

  it.each(["error", "response.failed", "response.incomplete", "response.cancelled"])(
    "redacts provider content from %s events",
    async (type) => {
      const harness = wireHarness([
        sse([{ type, message: TOKEN, error: { message: "private prompt" } }]),
      ]);
      const error = await rejection(harness.model.invoke("Hi"));
      expect(error).toBeInstanceOf(ResponsesError);
      expect(String(error)).not.toContain(TOKEN);
      expect(String(error)).not.toContain("private prompt");
    },
  );

  it.each([401, 403, 429, 500])(
    "sanitizes HTTP %i errors without automatic retries",
    async (status) => {
      const harness = wireHarness([new Response("private prompt " + TOKEN, { status })]);
      const error = await rejection(harness.model.invoke("Hi"));
      expect(error).toBeInstanceOf(ResponsesError);
      expect((error as ResponsesError).statusCode).toBe(status);
      expect(String(error)).not.toContain(TOKEN);
      expect(String(error)).not.toContain("private prompt");
      expect(harness.fetch).toHaveBeenCalledTimes(2);
    },
  );

  it.each([
    "data: private invalid JSON\n\n",
    "data: {}\n\n",
    "data: [DONE]\n\n",
    "data: " + JSON.stringify(completed()),
    "",
  ])("rejects malformed or incomplete SSE without exposing its data", async (wire) => {
    const harness = wireHarness([
      new Response(wire, { headers: { "Content-Type": "text/event-stream" } }),
    ]);
    const error = await rejection(harness.model.invoke("Hi"));
    expect(error).toBeInstanceOf(ResponsesError);
    expect(String(error)).not.toContain("private invalid JSON");
  });

  it.each([{ model: { private: TOKEN } }, { id: { private: TOKEN } }, { model: "" }])(
    "rejects invalid optional response metadata: %j",
    async (metadata) => {
      const event = {
        type: "response.completed",
        response: { status: "completed", output: [message()], ...metadata },
      };
      const harness = wireHarness([sse([event])]);
      const error = await rejection(harness.model.invoke("Hi"));
      expect(error).toBeInstanceOf(ResponsesError);
      expect(String(error)).not.toContain(TOKEN);
    },
  );

  it("allows omitted optional response metadata without inventing model identifiers", async () => {
    const harness = wireHarness([
      sse([
        {
          type: "response.completed",
          response: { status: "completed", output: [message()] },
        },
      ]),
    ]);
    const result = await harness.model.invoke("Hi");
    expect(result.content).toBe("Hello");
    expect(result.response_metadata).not.toHaveProperty("model_name");
    expect(result.response_metadata).not.toHaveProperty("response_id");
  });

  it("rejects JSON/nonstream responses", async () => {
    const harness = wireHarness([Response.json({ private: TOKEN })]);
    await expect(harness.model.invoke("Hi")).rejects.toThrow("Expected an event stream");
  });

  it("rejects output-item index/identity mismatches", async () => {
    const harness = wireHarness([
      sse([
        { type: "response.output_item.done", output_index: 0, item: { ...message(), id: "wrong" } },
        completed(),
      ]),
    ]);
    await expect(harness.model.invoke("Hi")).rejects.toThrow("does not match");
  });

  it("cancels an active reader with the RunnableConfig signal and sanitizes the abort reason", async () => {
    let readStarted: (() => void) | undefined;
    const ready = new Promise<void>((resolve) => {
      readStarted = resolve;
    });
    const cancel = vi.fn();
    const stream = new ReadableStream<Uint8Array>(
      {
        pull() {
          readStarted?.();
        },
        cancel,
      },
      { highWaterMark: 0 },
    );
    const harness = wireHarness([
      new Response(stream, { headers: { "Content-Type": "text/event-stream" } }),
    ]);
    const controller = new AbortController();
    const pending = harness.model.invoke("Hi", { signal: controller.signal });
    await ready;
    controller.abort(new Error(TOKEN));
    const error = await rejection(pending);
    expect(error).toMatchObject({ name: "AbortError" });
    expect(String(error)).not.toContain(TOKEN);
    expect(cancel).toHaveBeenCalled();
    expect(harness.requests.at(-1)?.init?.signal?.aborted).toBe(true);
  });

  it("honors a signal while the token getter is pending", async () => {
    const fetch = vi.fn<typeof globalThis.fetch>();
    let entered: (() => void) | undefined;
    const ready = new Promise<void>((resolve) => {
      entered = resolve;
    });
    const model = new ChatGPTResponsesModel({
      model: MODEL,
      fetch,
      accessTokenGetter: () => {
        entered?.();
        return new Promise<string>(() => undefined);
      },
    });
    const controller = new AbortController();
    const pending = model.availableModels({ signal: controller.signal });
    await ready;
    controller.abort(new Error(TOKEN));
    expect(await rejection(pending)).toMatchObject({ name: "AbortError" });
    expect(fetch).not.toHaveBeenCalled();
  });

  it("redacts rejected transport errors and never falls back to another credential", async () => {
    const fetch = vi.fn<typeof globalThis.fetch>(async () => {
      throw new Error(TOKEN);
    });
    const model = new ChatGPTResponsesModel({
      model: MODEL,
      accessTokenGetter: async () => TOKEN,
      fetch,
    });
    const error = await rejection(model.invoke("Hi"));
    expect(error).toBeInstanceOf(ResponsesError);
    expect(String(error)).not.toContain(TOKEN);
    expect(fetch).toHaveBeenCalledTimes(1);
    const noToken = new ChatGPTResponsesModel({
      model: MODEL,
      accessTokenGetter: async () => "",
      fetch,
    });
    await expect(noToken.invoke("Hi")).rejects.toThrow("No valid ChatGPT access token");
    expect(fetch).toHaveBeenCalledTimes(1);
  });
});
