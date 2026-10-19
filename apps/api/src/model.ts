import {
  BaseChatModel,
  type BaseChatModelCallOptions,
  type BindToolsInput,
} from "@langchain/core/language_models/chat_models";
import type { BaseLanguageModelInput } from "@langchain/core/language_models/base";
import type { CallbackManagerForLLMRun } from "@langchain/core/callbacks/manager";
import {
  AIMessage,
  AIMessageChunk,
  ChatMessage,
  ToolMessage,
  type BaseMessage,
  type MessageContent,
  type ToolCall,
} from "@langchain/core/messages";
import type { ChatResult } from "@langchain/core/outputs";
import type { Runnable } from "@langchain/core/runnables";
import { convertToOpenAITool } from "@langchain/core/utils/function_calling";

type JsonObject = Record<string, unknown>;
type ToolKind = "function" | "custom";
type ToolSpec = JsonObject & { type: ToolKind; name: string };

export const RESPONSES_OUTPUT_KEY = "taskmate_responses_output";
const MODELS_URL = "https://api.openai.com/v1/models";
const RESPONSES_URL = "https://api.openai.com/v1/responses";
const NAMESPACE = "taskmate";
const MAX_EVENT_CHARS = 16 * 1024 * 1024;
const OUTPUT_TYPES = new Set(["message", "reasoning", "function_call", "custom_tool_call"]);
const WIRE_OPTIONS = new Set([
  "instructions", "reasoning", "text", "parallel_tool_calls", "tool_choice", "tools",
]);
const LOCAL_OPTIONS = new Set([
  "signal", "outputVersion", "ls_structured_output_format", "maxRetries", "promptIndex",
]);

export interface ResponsesCallOptions extends BaseChatModelCallOptions {
  tools?: BindToolsInput[];
  instructions?: string;
  reasoning?: JsonObject;
  text?: JsonObject;
  parallel_tool_calls?: boolean;
  strict?: boolean;
}

export interface ChatGPTResponsesModelOptions {
  model?: string;
  accessTokenGetter: () => Promise<string>;
  fetch?: typeof globalThis.fetch;
  timeoutMs?: number;
}

export class ResponsesError extends Error {
  constructor(message: string, readonly statusCode?: number) {
    super(message);
    this.name = "ResponsesError";
  }
}

function isObject(value: unknown): value is JsonObject {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function object(value: unknown, message: string): JsonObject {
  if (!isObject(value)) throw new ResponsesError(message);
  return value;
}

function nonemptyString(value: unknown, message: string): string {
  if (typeof value !== "string" || !value) throw new ResponsesError(message);
  return value;
}

function parseJson(value: string): unknown {
  try {
    return JSON.parse(value) as unknown;
  } catch {
    throw new ResponsesError("The OpenAI response contained invalid JSON.");
  }
}

function abortError(): DOMException {
  return new DOMException("The model request was cancelled.", "AbortError");
}

function checkAbort(signal: AbortSignal): void {
  if (signal.aborted) throw abortError();
}

function waitFor<T>(promise: Promise<T>, signal: AbortSignal): Promise<T> {
  if (signal.aborted) return Promise.reject(abortError());
  return new Promise<T>((resolve, reject) => {
    const onAbort = () => reject(abortError());
    signal.addEventListener("abort", onAbort, { once: true });
    promise.then(
      (value) => {
        signal.removeEventListener("abort", onAbort);
        resolve(value);
      },
      (error: unknown) => {
        signal.removeEventListener("abort", onAbort);
        reject(error);
      },
    );
  });
}

function checkStatus(response: Response): void {
  if (response.ok) return;
  const messages: Record<number, string> = {
    401: "ChatGPT sign-in expired or was revoked. Sign in again.",
    403: "This ChatGPT account cannot access the requested resource.",
    429: "The ChatGPT account reached a usage or rate limit.",
  };
  throw new ResponsesError(
    messages[response.status] ?? "The OpenAI request failed.",
    response.status,
  );
}

function textContent(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) throw new ResponsesError("This PoC accepts text content only.");
  return content.map((block: unknown) => {
    if (typeof block === "string") return block;
    if (
      isObject(block) &&
      ["text", "input_text", "output_text"].includes(String(block.type)) &&
      typeof block.text === "string"
    ) return block.text;
    throw new ResponsesError("This PoC accepts text content only.");
  }).join("");
}

function toolSpecs(tools: BindToolsInput[], strict?: boolean): ToolSpec[] {
  const names = new Set<string>();
  return tools.map((input) => {
    const converted = object(
      isObject(input) && input.type === "custom" ? input : convertToOpenAITool(input),
      "Invalid tool definition.",
    );
    const kind = converted.type;
    if (kind !== "function" && kind !== "custom") {
      throw new ResponsesError("Only local function and custom tools are supported.");
    }
    const definition = object(converted.function ?? converted, "Invalid tool definition.");
    const name = nonemptyString(definition.name, "A tool name is required.");
    if (!/^[A-Za-z0-9_-]{1,64}$/.test(name) || names.has(name)) {
      throw new ResponsesError("Tool names must be unique simple identifiers up to 64 characters.");
    }
    names.add(name);
    const spec: ToolSpec = { type: kind, name };
    if (definition.description !== undefined) {
      if (typeof definition.description !== "string") {
        throw new ResponsesError("A tool description must be a string.");
      }
      spec.description = definition.description;
    }
    if (kind === "function") {
      spec.parameters = structuredClone(object(
        definition.parameters ?? { type: "object", properties: {} },
        "Function parameters must be a JSON schema object.",
      ));
      const strictValue = strict ?? definition.strict;
      if (strictValue !== undefined) {
        if (typeof strictValue !== "boolean") {
          throw new ResponsesError("The tool strict option must be a boolean.");
        }
        spec.strict = strictValue;
      }
    } else if (definition.format !== undefined) {
      spec.format = structuredClone(object(definition.format, "Invalid custom tool format."));
    }
    return spec;
  });
}

function callName(item: JsonObject): string {
  if (item.namespace !== undefined && item.namespace !== null && item.namespace !== NAMESPACE) {
    throw new ResponsesError("The model returned a tool outside the bound namespace.");
  }
  let name = nonemptyString(item.name, "The model returned an invalid tool name.");
  if (item.namespace == null && name.startsWith(NAMESPACE + ".")) {
    name = name.slice(NAMESPACE.length + 1);
  }
  return name;
}

function savedItems(message: AIMessage): JsonObject[] | undefined {
  const saved: unknown = message.additional_kwargs[RESPONSES_OUTPUT_KEY];
  if (saved === undefined) return undefined;
  if (!Array.isArray(saved)) throw new ResponsesError("Invalid saved Responses output.");
  const items = saved.map((item: unknown) => {
    const parsed = object(item, "Invalid saved Responses output.");
    if (!OUTPUT_TYPES.has(String(parsed.type))) {
      throw new ResponsesError("Invalid saved Responses output.");
    }
    return structuredClone(parsed);
  });
  const calls = new Map((message.tool_calls ?? []).map((call) => [call.id, call]));
  if (calls.size !== (message.tool_calls ?? []).length) {
    throw new ResponsesError("Tool call IDs must be unique.");
  }
  for (const item of items) {
    if (item.type !== "function_call" && item.type !== "custom_tool_call") continue;
    const id = nonemptyString(item.call_id, "A saved tool call needs its call ID.");
    const call = calls.get(id);
    if (!call || call.name !== callName(item)) {
      throw new ResponsesError("Saved Responses calls do not match the message tool calls.");
    }
    calls.delete(id);
    if (item.type === "function_call") {
      const args = parseJson(nonemptyString(item.arguments, "Invalid saved tool arguments."));
      if (JSON.stringify(args) !== JSON.stringify(call.args)) {
        item.arguments = JSON.stringify(call.args);
      }
    } else {
      const input = call.args.__arg1;
      if (typeof input !== "string" || Object.keys(call.args).length !== 1) {
        throw new ResponsesError("Custom tool input must remain a string.");
      }
      item.input = input;
    }
  }
  if (calls.size) {
    throw new ResponsesError("Saved Responses calls do not match the message tool calls.");
  }
  return items;
}

function history(messages: BaseMessage[]): JsonObject[] {
  const result: JsonObject[] = [];
  const callKinds = new Map<string, string>();
  const answered = new Set<string>();
  for (const message of messages) {
    if (ToolMessage.isInstance(message)) {
      const kind = callKinds.get(message.tool_call_id);
      if (!kind || answered.has(message.tool_call_id)) {
        throw new ResponsesError("A tool result must match one preceding unanswered tool call.");
      }
      answered.add(message.tool_call_id);
      result.push({
        type: kind + "_output",
        call_id: message.tool_call_id,
        output: textContent(message.content),
      });
      continue;
    }
    if (AIMessage.isInstance(message)) {
      let items = savedItems(message);
      if (items === undefined) {
        items = [];
        const text = textContent(message.content);
        if (text) items.push({ role: "assistant", content: text });
        for (const call of message.tool_calls ?? []) {
          items.push({
            type: "function_call",
            namespace: NAMESPACE,
            name: call.name,
            call_id: nonemptyString(call.id, "Every tool call needs its original call ID."),
            arguments: JSON.stringify(call.args),
          });
        }
      }
      for (const item of items) {
        if (item.type !== "function_call" && item.type !== "custom_tool_call") continue;
        const id = nonemptyString(item.call_id, "Every tool call needs its original call ID.");
        if (callKinds.has(id)) throw new ResponsesError("Tool call IDs must be unique.");
        callKinds.set(id, item.type);
      }
      result.push(...items);
      continue;
    }
    let role: string;
    if (message.type === "system") role = "developer";
    else if (message.type === "human") role = "user";
    else if (ChatMessage.isInstance(message) && ["system", "developer", "user"].includes(message.role)) {
      role = message.role === "system" ? "developer" : message.role;
    } else {
      throw new ResponsesError("Unsupported message role.");
    }
    result.push({ role, content: textContent(message.content) });
  }
  return result;
}

async function* events(response: Response, signal: AbortSignal): AsyncGenerator<JsonObject> {
  if (!response.body) throw new ResponsesError("The inference response has no stream.");
  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8", { fatal: true });
  let buffer = "";
  let data: string[] = [];
  let size = 0;
  const onAbort = () => { void reader.cancel().catch(() => undefined); };
  signal.addEventListener("abort", onAbort, { once: true });
  try {
    while (true) {
      checkAbort(signal);
      const chunk = await reader.read();
      checkAbort(signal);
      buffer += decoder.decode(chunk.value, { stream: !chunk.done });
      if (buffer.length + size > MAX_EVENT_CHARS) {
        throw new ResponsesError("The inference stream exceeded its event-size limit.");
      }
      while (true) {
        const index = buffer.search(/[\r\n]/);
        if (index < 0) break;
        if (buffer[index] === "\r" && index === buffer.length - 1 && !chunk.done) break;
        const width = buffer[index] === "\r" && buffer[index + 1] === "\n" ? 2 : 1;
        const line = buffer.slice(0, index);
        buffer = buffer.slice(index + width);
        if (line.startsWith("data:")) {
          const value = line.slice(5).replace(/^ /, "");
          size += value.length + 1;
          data.push(value);
        } else if (line === "" && data.length) {
          const value = data.join("\n");
          data = [];
          size = 0;
          if (value === "[DONE]") return;
          const event = object(parseJson(value), "Invalid inference stream event.");
          nonemptyString(event.type, "Invalid inference stream event.");
          yield event;
        }
      }
      if (chunk.done) break;
    }
    if (buffer || data.length) {
      throw new ResponsesError("The inference stream ended inside an event.");
    }
  } finally {
    signal.removeEventListener("abort", onAbort);
    await reader.cancel().catch(() => undefined);
    reader.releaseLock();
  }
}

function completedMessage(response: JsonObject, kinds: Map<string, ToolKind>): AIMessageChunk {
  if (!Array.isArray(response.output)) {
    throw new ResponsesError("The completed response is missing its output.");
  }
  const output = response.output.map((item: unknown) => object(item, "Invalid response output."));
  const text: string[] = [];
  const calls: ToolCall[] = [];
  const ids = new Set<string>();
  for (const item of output) {
    if (!OUTPUT_TYPES.has(String(item.type))) {
      throw new ResponsesError("The response contains an unsupported output item.");
    }
    if (item.type === "message") {
      if (!Array.isArray(item.content)) throw new ResponsesError("Invalid response message content.");
      for (const raw of item.content) {
        const block = object(raw, "Invalid response message content.");
        const field = block.type === "output_text" ? "text" : block.type === "refusal" ? "refusal" : undefined;
        const value = field ? block[field] : undefined;
        if (typeof value !== "string") {
          throw new ResponsesError("Unsupported response message content.");
        }
        text.push(value);
      }
    } else if (item.type === "function_call" || item.type === "custom_tool_call") {
      const name = callName(item);
      const kind = item.type === "function_call" ? "function" : "custom";
      if (kinds.get(name) !== kind) throw new ResponsesError("The model returned an unbound tool.");
      const id = nonemptyString(item.call_id, "The model returned an invalid tool call ID.");
      if (ids.has(id)) throw new ResponsesError("The model returned duplicate tool call IDs.");
      ids.add(id);
      let args: JsonObject;
      if (kind === "function") {
        args = object(
          parseJson(nonemptyString(item.arguments, "Invalid function arguments.")),
          "Function arguments must be a JSON object.",
        );
      } else {
        if (typeof item.input !== "string") throw new ResponsesError("Invalid custom tool input.");
        args = { __arg1: item.input };
      }
      calls.push({ type: "tool_call", id, name, args });
    }
  }
  const usage = isObject(response.usage) ? response.usage : undefined;
  const input = usage?.input_tokens;
  const outputCount = usage?.output_tokens;
  const total = usage?.total_tokens;
  const usageMetadata = (
    typeof input === "number" && Number.isSafeInteger(input) && input >= 0 &&
    typeof outputCount === "number" && Number.isSafeInteger(outputCount) && outputCount >= 0 &&
    typeof total === "number" && Number.isSafeInteger(total) && total >= 0
  ) ? { input_tokens: input, output_tokens: outputCount, total_tokens: total } : undefined;
  return new AIMessageChunk({
    content: text.join(""),
    tool_calls: calls,
    additional_kwargs: { [RESPONSES_OUTPUT_KEY]: structuredClone(output) },
    response_metadata: {
      response_id: response.id,
      model_name: response.model,
      finish_reason: calls.length ? "tool_calls" : "stop",
    },
    ...(usageMetadata ? { usage_metadata: usageMetadata } : {}),
  });
}

export class ChatGPTResponsesModel extends BaseChatModel<ResponsesCallOptions> {
  readonly model: string | undefined;
  #accessTokenGetter: () => Promise<string>;
  #fetch: typeof globalThis.fetch;
  #timeoutMs: number;

  constructor(options: ChatGPTResponsesModelOptions) {
    super({ cache: false, maxRetries: 0, outputVersion: "v0", disableStreaming: true });
    this.disableStreaming = true;
    this.model = options.model;
    this.#accessTokenGetter = options.accessTokenGetter;
    this.#fetch = options.fetch ?? globalThis.fetch;
    this.#timeoutMs = options.timeoutMs ?? 180_000;
    if (!Number.isSafeInteger(this.#timeoutMs) || this.#timeoutMs < 1 || this.#timeoutMs > 2_147_483_647) {
      throw new ResponsesError("timeoutMs must be a positive integer within the timer limit.");
    }
  }

  _llmType(): string {
    return "chatgpt-plan-responses";
  }

  override invocationParams(): JsonObject {
    return { model: this.model };
  }

  override async getNumTokens(content: MessageContent): Promise<number> {
    return Math.ceil(textContent(content).length / 4);
  }

  async getNumTokensFromMessages(messages: BaseMessage[]): Promise<number> {
    return messages.reduce((sum, message) => {
      const calls = AIMessage.isInstance(message) ? JSON.stringify(message.tool_calls ?? []) : "";
      return sum + 3 + Math.ceil((textContent(message.content).length + calls.length) / 4);
    }, 0);
  }

  override bindTools(
    tools: BindToolsInput[],
    kwargs: Partial<ResponsesCallOptions> = {},
  ): Runnable<BaseLanguageModelInput, AIMessageChunk, ResponsesCallOptions> {
    const { strict, ...options } = kwargs;
    return this.withConfig({ ...options, tools: toolSpecs(tools, strict) });
  }

  async #token(signal: AbortSignal): Promise<string> {
    const token = await waitFor(this.#accessTokenGetter(), signal);
    if (typeof token !== "string" || !token || /\s/.test(token)) {
      throw new ResponsesError("No valid ChatGPT access token is available. Sign in again.");
    }
    return token;
  }

  async #catalog(token: string, signal: AbortSignal): Promise<string[]> {
    const response = await waitFor(this.#fetch(MODELS_URL, {
      headers: { Authorization: "Bearer " + token },
      redirect: "error",
      signal,
    }), signal);
    try {
      checkStatus(response);
      const payload = object(await waitFor(response.json() as Promise<unknown>, signal), "Invalid model catalog.");
      if (!Array.isArray(payload.models)) throw new ResponsesError("The model catalog has no models array.");
      const models: string[] = [];
      for (const raw of payload.models) {
        const entry = object(raw, "The model catalog contains an invalid entry.");
        if (entry.visibility === "list") {
          const slug = nonemptyString(entry.slug, "The model catalog contains an invalid slug.");
          if (!models.includes(slug)) models.push(slug);
        }
      }
      return models;
    } finally {
      await response.body?.cancel().catch(() => undefined);
    }
  }

  #failure(error: unknown, signal: AbortSignal, callerSignal?: AbortSignal): never {
    if (callerSignal?.aborted) throw abortError();
    if (signal.aborted) throw new ResponsesError("The OpenAI request timed out.");
    if (error instanceof ResponsesError) throw error;
    throw new ResponsesError("The OpenAI request could not be completed.");
  }

  async availableModels(options: { signal?: AbortSignal } = {}): Promise<string[]> {
    const timeout = AbortSignal.timeout(this.#timeoutMs);
    const signal = options.signal ? AbortSignal.any([options.signal, timeout]) : timeout;
    try {
      checkAbort(signal);
      return await this.#catalog(await this.#token(signal), signal);
    } catch (error: unknown) {
      return this.#failure(error, signal, options.signal);
    }
  }

  async _generate(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
    runManager?: CallbackManagerForLLMRun,
  ): Promise<ChatResult> {
    const unsupported = Object.entries(options).filter(([key, value]) =>
      value !== undefined && !WIRE_OPTIONS.has(key) && !LOCAL_OPTIONS.has(key),
    ).map(([key]) => key);
    if (unsupported.length) {
      throw new ResponsesError("Unsupported ChatGPT plan options: " + unsupported.sort().join(", "));
    }
    if (options.outputVersion !== undefined && options.outputVersion !== "v0") {
      throw new ResponsesError("This adapter uses message output version v0.");
    }
    if (options.maxRetries !== undefined && options.maxRetries !== 0) {
      throw new ResponsesError("Automatic inference retries are disabled.");
    }
    if (!this.model) throw new ResponsesError("Select a model from availableModels() before inference.");
    const specs = toolSpecs(options.tools ?? []);
    const kinds = new Map<string, ToolKind>(specs.map((spec) => [spec.name, spec.type]));
    const body: JsonObject = {
      model: this.model,
      input: history(messages),
      store: false,
      stream: true,
      include: ["reasoning.encrypted_content"],
    };
    for (const key of ["instructions", "reasoning", "text", "parallel_tool_calls"] as const) {
      if (options[key] !== undefined) body[key] = options[key];
    }
    if (options.tool_choice !== undefined) {
      const choice = options.tool_choice === "any" ? "required" : options.tool_choice;
      if (typeof choice !== "string" || !["auto", "none", "required"].includes(choice)) {
        throw new ResponsesError("tool_choice must be auto, none, required, or any.");
      }
      if (choice === "required" && specs.length === 0) {
        throw new ResponsesError("A required tool choice needs at least one tool.");
      }
      body.tool_choice = choice;
    }
    if (specs.length) {
      body.tools = [{
        type: "namespace",
        name: NAMESPACE,
        description: "Tools executed by the local Taskmate agent.",
        tools: specs,
      }];
    }
    const timeout = AbortSignal.timeout(this.#timeoutMs);
    const signal = options.signal ? AbortSignal.any([options.signal, timeout]) : timeout;
    let response: Response | undefined;
    try {
      checkAbort(signal);
      const token = await this.#token(signal);
      if (!(await this.#catalog(token, signal)).includes(this.model)) {
        throw new ResponsesError("The selected model is not in this ChatGPT account's visible catalog.");
      }
      response = await waitFor(this.#fetch(RESPONSES_URL, {
        method: "POST",
        headers: {
          Authorization: "Bearer " + token,
          "Content-Type": "application/json",
          Accept: "text/event-stream",
        },
        body: JSON.stringify(body),
        redirect: "error",
        signal,
      }), signal);
      checkStatus(response);
      const contentType = response.headers.get("content-type")?.split(";")[0]?.trim();
      if (contentType !== "text/event-stream") {
        throw new ResponsesError("Expected an event stream from the Responses endpoint.");
      }
      const completedItems = new Map<number, JsonObject>();
      for await (const event of events(response, signal)) {
        if (["error", "response.failed", "response.incomplete", "response.cancelled"].includes(String(event.type))) {
          throw new ResponsesError("The inference stream reported an unsuccessful response.");
        }
        if (event.type === "response.output_text.delta" && typeof event.delta === "string") {
          await runManager?.handleLLMNewToken(event.delta);
        }
        if (event.type === "response.output_item.done") {
          const index = event.output_index;
          if (typeof index !== "number" || !Number.isSafeInteger(index) || index < 0) {
            throw new ResponsesError("Invalid completed output index.");
          }
          completedItems.set(index, object(event.item, "Invalid completed output item."));
        }
        if (event.type === "response.completed") {
          const result = structuredClone(object(event.response, "Invalid inference completion."));
          if (result.status !== "completed" || !Array.isArray(result.output)) {
            throw new ResponsesError("Invalid inference completion.");
          }
          for (const [index, item] of completedItems) {
            const finalItem = object(result.output[index], "The completed output does not match the stream.");
            if (finalItem.id !== item.id) {
              throw new ResponsesError("The completed output does not match the stream.");
            }
            result.output[index] = { ...finalItem, ...item };
          }
          const message = completedMessage(result, kinds);
          return { generations: [{ text: textContent(message.content), message }] };
        }
      }
      throw new ResponsesError("The inference stream ended before a completed response.");
    } catch (error: unknown) {
      return this.#failure(error, signal, options.signal);
    } finally {
      await response?.body?.cancel().catch(() => undefined);
    }
  }
}
