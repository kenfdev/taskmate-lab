import { useEffect, useRef, useState } from "react";

type DemoUser = "alice" | "bob";
type Tone = "concise" | "friendly";
type Phase = "loading" | "ready" | "failed";
type Id = string | number;
type Session = {
  user_id: DemoUser;
  csrf_token: string;
  chatgpt_connected: boolean;
  model: string | null;
};
type Task = { id: Id; title: string; completed: boolean };
type Preferences = { tone: Tone; notification_hour: number; timezone: string };
type Message = { role: "user" | "assistant"; content: string };
type ChatDraft = { message: string; request_id: string; failed: boolean };
type Notification = { id: Id; content: string; created_at: string };

const busyLabels = {
  tasks: "タスクを保存中…",
  chat: "返答を待っています…",
  thread: "会話を準備中…",
  preferences: "設定を保存中…",
  notifications: "お知らせを更新中…",
  preview: "通知文を作成中…",
  models: "モデル一覧を取得中…",
  oauth: "接続を更新中…"
} as const;
type Action = keyof typeof busyLabels;

type RequestScope = {
  active: boolean;
  controller: AbortController;
  phase: Phase;
  csrf: string | null;
  userId: DemoUser | null;
  pending: Set<Action>;
  taskReadVersion: number;
};

type UiState = {
  phase: Phase;
  session: Session | null;
  pending: Action[];
  error: string | null;
  tasks: Task[];
  tasksLoaded: boolean;
  taskTitle: string;
  preferences: Preferences | null;
  preferenceForm: Preferences;
  preferenceDirty: boolean;
  notifications: Notification[];
  notificationsLoaded: boolean;
  models: string[];
  modelsLoaded: boolean;
  thread: Id | null;
  messages: Message[];
  chatDraft: ChatDraft | null;
  chatText: string;
  preview: string | null;
};

function initialState(): UiState {
  return {
    phase: "loading",
    session: null,
    pending: [],
    error: null,
    tasks: [],
    tasksLoaded: false,
    taskTitle: "",
    preferences: null,
    preferenceForm: { tone: "concise", notification_hour: 9, timezone: "Asia/Tokyo" },
    preferenceDirty: false,
    notifications: [],
    notificationsLoaded: false,
    models: [],
    modelsLoaded: false,
    thread: null,
    messages: [],
    chatDraft: null,
    chatText: "",
    preview: null
  };
}

class HttpError extends Error {
  constructor(readonly status: number) {
    super("HTTP " + status);
  }
}

function ensure(condition: unknown): asserts condition {
  if (!condition) throw new Error("Unexpected API response");
}

function record(value: unknown): Record<string, unknown> {
  ensure(value !== null && typeof value === "object" && !Array.isArray(value));
  return value as Record<string, unknown>;
}

function readId(value: unknown): Id {
  ensure(typeof value === "string" || (typeof value === "number" && Number.isFinite(value)));
  return value;
}

function readSession(value: unknown): Session {
  const data = record(value);
  ensure(data.user_id === "alice" || data.user_id === "bob");
  ensure(typeof data.csrf_token === "string" && data.csrf_token.length > 0);
  ensure(typeof data.chatgpt_connected === "boolean");
  ensure(data.model === null || typeof data.model === "string");
  return {
    user_id: data.user_id,
    csrf_token: data.csrf_token,
    chatgpt_connected: data.chatgpt_connected,
    model: data.model
  };
}

function readTasks(value: unknown): Task[] {
  ensure(Array.isArray(value));
  return value.map((item: unknown) => {
    const data = record(item);
    ensure(typeof data.title === "string" && typeof data.completed === "boolean");
    return { id: readId(data.id), title: data.title, completed: data.completed };
  });
}

function readPreferences(value: unknown): Preferences {
  const data = record(value);
  ensure(data.tone === "concise" || data.tone === "friendly");
  ensure(typeof data.notification_hour === "number" && Number.isInteger(data.notification_hour) &&
    data.notification_hour >= 0 && data.notification_hour <= 23);
  ensure(typeof data.timezone === "string" && data.timezone.trim().length > 0);
  return { tone: data.tone, notification_hour: data.notification_hour, timezone: data.timezone };
}

function readNotifications(value: unknown): Notification[] {
  ensure(Array.isArray(value));
  return value.map((item: unknown) => {
    const data = record(item);
    ensure(typeof data.content === "string" && typeof data.created_at === "string");
    return { id: readId(data.id), content: data.content, created_at: data.created_at };
  });
}

function readMessages(value: unknown): { messages: Message[]; draft: ChatDraft | null } {
  ensure(Array.isArray(value));
  const messages: Message[] = [];
  let draft: ChatDraft | null = null;
  for (const item of value) {
    const data = record(item);
    ensure(typeof data.role === "string" && typeof data.content === "string");
    if (data.role !== "user" && data.role !== "assistant") continue;
    if (data.pending === true) {
      ensure(draft === null && data.role === "user" && typeof data.request_id === "string" &&
        /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(data.request_id));
      draft = { message: data.content, request_id: data.request_id, failed: true };
    } else {
      messages.push({ role: data.role, content: data.content });
    }
  }
  return { messages, draft };
}

function readModels(value: unknown): string[] {
  const data = record(value);
  ensure(Array.isArray(data.models));
  const models = data.models.map((model: unknown) => {
    ensure(typeof model === "string" && model.length > 0);
    return model;
  });
  return [...new Set(models)];
}

function readText(value: unknown, field: string): string {
  const text = record(value)[field];
  ensure(typeof text === "string");
  return text;
}

function assertActive(scope: RequestScope): void {
  if (!scope.active || scope.controller.signal.aborted) {
    throw new DOMException("Stale profile request", "AbortError");
  }
}

async function api(
  scope: RequestScope,
  path: string,
  options: { method?: "GET" | "POST" | "PUT"; body?: unknown } = {}
): Promise<unknown> {
  assertActive(scope);
  const method = options.method ?? "GET";
  const headers: Record<string, string> = { Accept: "application/json" };
  if (method !== "GET") {
    ensure(typeof scope.csrf === "string");
    headers["X-CSRF-Token"] = scope.csrf;
  }
  if (options.body !== undefined) headers["Content-Type"] = "application/json";
  const response = await fetch(path, {
    method,
    credentials: "same-origin",
    cache: "no-store",
    signal: scope.controller.signal,
    headers,
    body: options.body === undefined ? undefined : JSON.stringify(options.body)
  });
  assertActive(scope);
  if (!response.ok) throw new HttpError(response.status);
  if (response.status === 204) return null;
  const raw = await response.text();
  assertActive(scope);
  return raw ? JSON.parse(raw) as unknown : null;
}

function isAbort(error: unknown): boolean {
  return error instanceof Error && error.name === "AbortError";
}

function errorText(error: unknown): string {
  if (error instanceof HttpError) {
    if (error.status === 401) return "接続を確認できませんでした。ChatGPTへのログイン状態を確認してください。";
    if (error.status === 403) return "セッションを確認できませんでした。画面を読み込み直してください。";
    if (error.status === 409) return "処理が進行中か、状態が変更されています。少し待ってからもう一度お試しください。";
    if (error.status === 429) return "リクエストが混み合っています。少し待ってからもう一度お試しください。";
    if (error.status === 400 || error.status === 422) return "入力内容を確認してください。保存できませんでした。";
    if (error.status === 503) return "現在この機能を利用できません。サーバーの接続設定を確認してください。";
    return "サーバーで処理を完了できませんでした。もう一度お試しください。";
  }
  if (error instanceof TypeError) return "サーバーに接続できませんでした。起動状態やネットワークを確認してください。";
  return "データを読み取れませんでした。画面を読み込み直してください。";
}

function savedThread(userId: DemoUser): string | null {
  try {
    return sessionStorage.getItem("taskmate.thread." + userId);
  } catch {
    return null;
  }
}

function rememberThread(userId: DemoUser, threadId: Id): void {
  try {
    sessionStorage.setItem("taskmate.thread." + userId, String(threadId));
  } catch {
    return;
  }
}

function requestId(): string {
  if (typeof crypto.randomUUID === "function") return crypto.randomUUID();
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  bytes[6] = (bytes[6]! & 0x0f) | 0x40;
  bytes[8] = (bytes[8]! & 0x3f) | 0x80;
  const hex = Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
  return [hex.slice(0, 8), hex.slice(8, 12), hex.slice(12, 16), hex.slice(16, 20), hex.slice(20)].join("-");
}

function notificationTime(value: string, profileTimezone: string): { iso?: string; text: string } {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return { text: "時刻を確認できません" };
  let timezone = profileTimezone;
  const options: Intl.DateTimeFormatOptions = {
    month: "short", day: "numeric", hour: "2-digit", minute: "2-digit"
  };
  let formatter: Intl.DateTimeFormat;
  try {
    formatter = new Intl.DateTimeFormat("ja-JP", { ...options, timeZone: timezone });
  } catch {
    timezone = "UTC";
    formatter = new Intl.DateTimeFormat("ja-JP", { ...options, timeZone: timezone });
  }
  const label = timezone === "Asia/Tokyo" ? "日本時間" : timezone;
  return { iso: date.toISOString(), text: formatter.format(date) + " · " + label };
}

function Bubble({ role, content, status }: Message & { status?: "pending" | "failed" }) {
  return (
    <article className={"message " + role + (status ? " pending" + (status === "failed" ? " failed" : "") : "")}>
      <span className="message-label">{role === "user" ? "あなた" : "Taskmate"}</span>
      <p className="message-content">{content}</p>
    </article>
  );
}

export default function App() {
  const [state, setState] = useState<UiState>(initialState);
  const scopeRef = useRef<RequestScope | null>(null);

  function current(scope: RequestScope): boolean {
    return scopeRef.current === scope && scope.active && !scope.controller.signal.aborted;
  }

  function update(scope: RequestScope, change: (previous: UiState) => UiState): void {
    if (!current(scope)) return;
    setState((previous) => current(scope) ? change(previous) : previous);
  }

  function reportError(scope: RequestScope, error: unknown): void {
    if (!isAbort(error)) update(scope, (previous) => ({ ...previous, error: errorText(error) }));
  }

  async function loadTasks(scope: RequestScope): Promise<void> {
    const version = ++scope.taskReadVersion;
    const tasks = readTasks(await api(scope, "/api/tasks"));
    if (version !== scope.taskReadVersion) return;
    update(scope, (previous) => ({ ...previous, tasks, tasksLoaded: true }));
  }

  async function loadPreferences(scope: RequestScope, force = false): Promise<void> {
    const preferences = readPreferences(await api(scope, "/api/preferences"));
    update(scope, (previous) => ({
      ...previous,
      preferences,
      preferenceForm: force || !previous.preferenceDirty ? preferences : previous.preferenceForm,
      preferenceDirty: force ? false : previous.preferenceDirty
    }));
  }

  async function loadNotifications(scope: RequestScope): Promise<void> {
    const notifications = readNotifications(await api(scope, "/api/notifications"));
    update(scope, (previous) => ({ ...previous, notifications, notificationsLoaded: true }));
  }

  function applyMessages(scope: RequestScope, thread: Id, value: unknown): void {
    const { messages, draft } = readMessages(value);
    update(scope, (previous) => ({
      ...previous,
      thread,
      messages,
      chatDraft: draft,
      chatText: draft?.message ?? ""
    }));
  }

  async function createThread(scope: RequestScope): Promise<void> {
    const thread = readId(record(await api(scope, "/api/threads", { method: "POST" })).id);
    ensure(scope.userId !== null);
    rememberThread(scope.userId, thread);
    update(scope, (previous) => ({
      ...previous, thread, messages: [], chatDraft: null, chatText: ""
    }));
    applyMessages(scope, thread, await api(scope, "/api/threads/" + encodeURIComponent(thread) + "/messages"));
  }

  async function restoreThread(scope: RequestScope): Promise<void> {
    ensure(scope.userId !== null);
    const thread = savedThread(scope.userId);
    if (!thread) {
      await createThread(scope);
      return;
    }
    try {
      applyMessages(scope, thread, await api(scope, "/api/threads/" + encodeURIComponent(thread) + "/messages"));
    } catch (error) {
      if (!(error instanceof HttpError) || error.status !== 404) throw error;
      await createThread(scope);
    }
  }

  async function initialize(userId?: DemoUser): Promise<void> {
    const previous = scopeRef.current;
    const csrf = previous?.csrf ?? null;
    if (userId && (!csrf || previous?.phase !== "ready")) return;
    if (previous) {
      previous.active = false;
      previous.controller.abort();
    }
    const scope: RequestScope = {
      active: true,
      controller: new AbortController(),
      phase: "loading",
      csrf,
      userId: null,
      pending: new Set(),
      taskReadVersion: 0
    };
    scopeRef.current = scope;
    setState(initialState());
    try {
      let session: Session;
      try {
        session = readSession(await api(scope, "/api/session",
          userId ? { method: "POST", body: { user_id: userId } } : {}));
      } catch (error) {
        if (!userId || isAbort(error)) throw error;
        session = readSession(await api(scope, "/api/session"));
        reportError(scope, error);
      }
      scope.csrf = session.csrf_token;
      scope.userId = session.user_id;
      update(scope, (previousState) => ({ ...previousState, session }));
      const results = await Promise.allSettled([
        loadTasks(scope),
        loadPreferences(scope),
        loadNotifications(scope),
        restoreThread(scope)
      ]);
      assertActive(scope);
      scope.phase = "ready";
      update(scope, (previousState) => ({ ...previousState, phase: "ready" }));
      const failure = results.find((result) => result.status === "rejected");
      if (failure?.status === "rejected") reportError(scope, failure.reason);
    } catch (error) {
      if (!current(scope)) return;
      scope.phase = "failed";
      update(scope, (previousState) => ({ ...previousState, phase: "failed" }));
      reportError(scope, error);
    }
  }

  useEffect(() => {
    void initialize();
    const handlePageShow = (event: PageTransitionEvent) => {
      if (event.persisted) void initialize();
    };
    window.addEventListener("pageshow", handlePageShow);
    return () => {
      const scope = scopeRef.current;
      if (scope) {
        scope.active = false;
        scope.controller.abort();
      }
      window.removeEventListener("pageshow", handlePageShow);
    };
  }, []);

  async function run(action: Action, work: (scope: RequestScope) => Promise<void>): Promise<void> {
    const scope = scopeRef.current;
    if (!scope || scope.phase !== "ready" || scope.pending.has(action)) return;
    scope.pending.add(action);
    update(scope, (previous) => ({ ...previous, pending: [...scope.pending], error: null }));
    try {
      await work(scope);
    } catch (error) {
      reportError(scope, error);
    } finally {
      if (current(scope)) {
        scope.pending.delete(action);
        update(scope, (previous) => ({ ...previous, pending: [...scope.pending] }));
      }
    }
  }

  function changePreference(change: Partial<Preferences>): void {
    const scope = scopeRef.current;
    if (!scope || scope.phase !== "ready") return;
    update(scope, (previous) => ({
      ...previous,
      preferenceForm: { ...previous.preferenceForm, ...change },
      preferenceDirty: true
    }));
  }

  async function sendChat(): Promise<void> {
    if (chatDisabled || (!state.chatDraft && !state.chatText.trim()) || state.thread === null) return;
    const thread = state.thread;
    const draft = state.chatDraft ?? {
      message: state.chatText.trim(),
      request_id: requestId(),
      failed: false
    };
    await run("chat", async (scope) => {
      update(scope, (previous) => ({ ...previous, chatDraft: { ...draft, failed: false } }));
      let reply: string;
      try {
        reply = readText(await api(scope, "/api/threads/" + encodeURIComponent(thread) + "/chat", {
          method: "POST",
          body: { message: draft.message, request_id: draft.request_id }
        }), "reply");
      } catch (error) {
        update(scope, (previous) => ({ ...previous, chatDraft: { ...draft, failed: true } }));
        throw error;
      }
      update(scope, (previous) => ({
        ...previous,
        messages: [...previous.messages,
          { role: "user", content: draft.message },
          { role: "assistant", content: reply }],
        chatDraft: null,
        chatText: ""
      }));
      const reads = [loadTasks(scope)];
      if (!state.preferenceDirty) reads.push(loadPreferences(scope));
      const results = await Promise.allSettled(reads);
      assertActive(scope);
      const failure = results.find((result) => result.status === "rejected");
      if (failure?.status === "rejected") reportError(scope, failure.reason);
    });
  }

  const pending = new Set(state.pending);
  const ready = state.phase === "ready";
  const connected = state.session?.chatgpt_connected === true;
  const modelConfigured = Boolean(state.session?.model?.trim());
  const chatting = pending.has("chat");
  const tasksDisabled = !ready || !state.tasksLoaded || pending.has("tasks");
  const preferencesDisabled = !ready || !state.preferences || pending.has("preferences") || chatting;
  const chatDisabled = !ready || !connected || !modelConfigured || state.thread === null ||
    chatting || pending.has("thread") || pending.has("oauth") || pending.has("preferences");
  const previewDisabled = !ready || !connected || !modelConfigured ||
    pending.has("notifications") || pending.has("preview") || pending.has("oauth");
  const timezoneOptions = [...new Set([
    "Asia/Tokyo", "UTC", state.preferenceForm.timezone, state.preferences?.timezone ?? "Asia/Tokyo"
  ])];
  const tasks = [...state.tasks].sort((a, b) => Number(a.completed) - Number(b.completed));
  const firstPending = state.pending[0];
  const runtimeStatus = state.phase === "loading" ? "読み込み中…" :
    state.phase === "failed" ? "読み込みを確認してください" :
      firstPending ? busyLabels[firstPending] : "準備できました";

  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">t</span>
          <div>
            <p className="brand-name">Taskmate <span className="brand-name-light">Lab</span></p>
            <p className="brand-subtitle">毎日の小さなタスクに、相談相手を。</p>
          </div>
        </div>
        <div className="account-controls">
          <label className="demo-label" htmlFor="demo-user">
            <span className="badge">デモユーザー</span>
            <select id="demo-user" value={state.session?.user_id ?? "alice"}
              disabled={!ready || pending.has("oauth")}
              onChange={(event) => void initialize(event.target.value as DemoUser)}>
              <option value="alice">Alice</option>
              <option value="bob">Bob</option>
            </select>
          </label>
          <div className="connection">
            <span id="connection-state" className={"connection-state" + (connected ? " connected" : "")}>
              {!state.session ? "接続を確認中" : connected ? "ChatGPT 接続済み" : "ChatGPT 未接続"}
            </span>
            <button id="connect" type="button" hidden={connected} disabled={!ready || pending.has("oauth")}
              onClick={() => void run("oauth", async (scope) => {
                const authorizationUrl = readText(await api(scope, "/api/oauth/start", { method: "POST" }), "authorization_url");
                const url = new URL(authorizationUrl);
                ensure(!url.username && !url.password && !url.hash &&
                  url.origin === "https://auth.openai.com" && url.pathname === "/api/accounts/authorize");
                window.location.assign(url.href);
              })}>ChatGPTにログイン</button>
            <button id="disconnect" type="button" className="quiet" hidden={!connected}
              disabled={!ready || pending.has("oauth") || chatting}
              onClick={() => void run("oauth", async (scope) => {
                const result = record(await api(scope, "/api/oauth/disconnect", { method: "POST" }));
                ensure(result.disconnected === true && typeof result.remote_revocation_confirmed === "boolean");
                const session = readSession(await api(scope, "/api/session"));
                if (session.user_id !== scope.userId) {
                  await initialize();
                  return;
                }
                scope.csrf = session.csrf_token;
                update(scope, (previous) => ({
                  ...previous,
                  session,
                  models: [],
                  modelsLoaded: false,
                  error: result.remote_revocation_confirmed ? null :
                    "このアプリの接続は解除しました。ChatGPT側での認証情報の失効は確認できませんでした。"
                }));
              })}>接続を解除</button>
          </div>
        </div>
      </header>

      <p className="demo-note">Alice / Bob の切り替えはデータ分離を試すデモです。本人確認やアクセス制御を行うログイン機能ではありません。</p>
      <div className="intro">
        <div>
          <h1>今日のことを、ひとつずつ。</h1>
          <p>タスクを整理して、会話で次の一歩を決めましょう。</p>
        </div>
        <p id="runtime-status" className="runtime-status" role="status" aria-live="polite">{runtimeStatus}</p>
      </div>
      <div id="error-banner" className="error" role="alert" hidden={!state.error}>
        <p id="error-message">{state.error}</p>
        <button id="reload" type="button" disabled={state.phase === "loading"}
          onClick={() => void initialize()}>読み込み直す</button>
      </div>

      <main className="workspace" id="workspace" aria-busy={!ready}>
        <section className="panel" aria-labelledby="tasks-heading">
          <div className="panel-header">
            <h2 id="tasks-heading">やること</h2>
            <span id="task-count" className="badge">{state.tasksLoaded
              ? "残り " + state.tasks.filter((task) => !task.completed).length + " 件" : "読み込み中"}</span>
          </div>
          <div className="panel-body">
            <form id="task-form" onSubmit={(event) => {
              event.preventDefault();
              const title = state.taskTitle.trim();
              if (!title || tasksDisabled) return;
              void run("tasks", async (scope) => {
                await api(scope, "/api/tasks", { method: "POST", body: { title } });
                update(scope, (previous) => ({ ...previous, taskTitle: "" }));
                await loadTasks(scope);
              });
            }}>
              <fieldset id="task-fields" className="task-form" disabled={tasksDisabled}>
                <label htmlFor="task-title">
                  <span>新しいタスク</span>
                  <input id="task-title" name="title" placeholder="まず、何から始める？"
                    maxLength={240} autoComplete="off" required value={state.taskTitle}
                    onChange={(event) => setState((previous) => ({ ...previous, taskTitle: event.target.value }))} />
                </label>
                <button id="add-task" className="primary" type="submit"
                  disabled={tasksDisabled || !state.taskTitle.trim()}>追加</button>
              </fieldset>
            </form>
            <ul id="task-list" className="task-list" aria-label="タスク一覧">
              {tasks.map((task) => (
                <li key={task.id} className={"task-row" + (task.completed ? " completed" : "")}>
                  <button type="button" className={"task-check" + (task.completed ? " done" : "")}
                    disabled={tasksDisabled || task.completed}
                    aria-label={task.completed ? task.title + " 完了済み" : task.title + "を完了にする"}
                    onClick={() => void run("tasks", async (scope) => {
                      await api(scope, "/api/tasks/" + encodeURIComponent(task.id) + "/complete", { method: "POST" });
                      await loadTasks(scope);
                    })}>{task.completed ? "✓" : ""}</button>
                  <span className="task-title">{task.title}</span>
                </li>
              ))}
            </ul>
            <div id="tasks-empty" className="empty" hidden={state.tasks.length > 0}>
              <strong>{state.tasksLoaded ? "まだタスクはありません" : "タスクを読み込んでいます"}</strong>
              {state.tasksLoaded && <p>小さなことから、ひとつ追加してみましょう。</p>}
            </div>
          </div>
        </section>

        <section className="panel chat-panel" aria-labelledby="chat-heading">
          <div className="panel-header">
            <div>
              <h2 id="chat-heading">Taskmateに相談</h2>
              <p id="model-label" className="panel-description">
                {modelConfigured ? "モデル · " + state.session?.model : "モデル未設定"}
              </p>
            </div>
            <button id="new-thread" type="button" className="quiet"
              disabled={!ready || chatting || pending.has("thread")}
              onClick={() => void run("thread", createThread)}>新しい会話</button>
          </div>
          <div id="chat-log" className="chat-log" role="log" aria-label="会話"
            aria-live="polite" aria-relevant="additions text" aria-busy={chatting || pending.has("thread")}>
            {!state.messages.length && !state.chatDraft ? (
              <div className="empty chat-empty">
                <span className="brand-mark" aria-hidden="true">t</span>
                <strong>次の一歩を、一緒に。</strong>
                <p>{connected && modelConfigured
                  ? "タスクの整理や進め方など、気になっていることを話してください。"
                  : "ChatGPTに接続し、モデルを設定すると会話を始められます。タスクはそのまま使えます。"}</p>
              </div>
            ) : state.messages.map((message, index) => <Bubble key={index} {...message} />)}
            {state.chatDraft && <>
              <Bubble role="user" content={state.chatDraft.message} />
              <Bubble role="assistant" status={state.chatDraft.failed ? "failed" : "pending"}
                content={state.chatDraft.failed
                  ? "送信結果を確認できませんでした。「再試行」で確認できます。新しい会話を始めることもできます。"
                  : "考えています…"} />
            </>}
          </div>
          <form id="chat-form" className="composer" onSubmit={(event) => {
            event.preventDefault();
            void sendChat();
          }}>
            <p id="connection-help" className="connection-help" hidden={connected}>
              会話を始めるには、上の「ChatGPTにログイン」から接続してください。
            </p>
            <p id="model-help" className="connection-help" hidden={!connected || modelConfigured}>
              利用できるモデルを設定欄で確認し、TASKMATE_MODEL に設定して再起動してください。
            </p>
            <fieldset id="chat-fields" disabled={chatDisabled}>
              <label htmlFor="chat-message">メッセージ</label>
              <textarea id="chat-message" name="message" maxLength={8000} required
                placeholder="今日のタスク、どこから手をつけよう？" value={state.chatText}
                readOnly={Boolean(state.chatDraft)}
                onChange={(event) => setState((previous) => ({ ...previous, chatText: event.target.value }))}
                onKeyDown={(event) => {
                  if ((event.metaKey || event.ctrlKey) && event.key === "Enter" && !event.nativeEvent.isComposing) {
                    event.preventDefault();
                    void sendChat();
                  }
                }} />
              <div className="composer-footer">
                <span id="chat-hint" className="hint">{state.chatDraft
                  ? "再試行では同じリクエストIDを使います" : "⌘ / Ctrl + Enter で送信"}</span>
                <button id="send-chat" className="primary" type="submit"
                  disabled={chatDisabled || (!state.chatDraft && !state.chatText.trim())}>
                  {chatting ? "送信中…" : state.chatDraft ? "再試行" : "送信"}
                </button>
              </div>
            </fieldset>
          </form>
        </section>

        <aside className="sidebar" aria-label="設定と通知">
          <section className="panel" aria-labelledby="preferences-heading">
            <div className="panel-header">
              <div>
                <h2 id="preferences-heading">あなたに合わせる</h2>
                <p className="panel-description">ユーザーごとに設定を保存します。</p>
              </div>
            </div>
            <div className="panel-body">
              <form id="preferences-form" onSubmit={(event) => {
                event.preventDefault();
                if (preferencesDisabled || !state.preferenceDirty) return;
                const preferences = { ...state.preferenceForm };
                void run("preferences", async (scope) => {
                  await api(scope, "/api/preferences", { method: "PUT", body: preferences });
                  await loadPreferences(scope, true);
                });
              }}>
                <fieldset id="preference-fields" className="preferences" disabled={preferencesDisabled}>
                  <label htmlFor="tone">
                    <span>返答の雰囲気</span>
                    <select id="tone" name="tone" value={state.preferenceForm.tone}
                      onChange={(event) => changePreference({ tone: event.target.value as Tone })}>
                      <option value="concise">要点を短く</option>
                      <option value="friendly">親しみやすく</option>
                    </select>
                  </label>
                  <div className="field-pair">
                    <label htmlFor="notification-hour">
                      <span>通知の時刻</span>
                      <select id="notification-hour" name="notification_hour"
                        value={state.preferenceForm.notification_hour}
                        onChange={(event) => changePreference({ notification_hour: Number(event.target.value) })}>
                        {Array.from({ length: 24 }, (_, hour) =>
                          <option key={hour} value={hour}>{String(hour).padStart(2, "0") + ":00"}</option>)}
                      </select>
                    </label>
                    <label htmlFor="timezone">
                      <span>タイムゾーン</span>
                      <select id="timezone" name="timezone" value={state.preferenceForm.timezone}
                        onChange={(event) => changePreference({ timezone: event.target.value })}>
                        {timezoneOptions.map((timezone) => <option key={timezone} value={timezone}>
                          {timezone === "Asia/Tokyo" ? "日本時間" : timezone}
                        </option>)}
                      </select>
                    </label>
                  </div>
                  <div className="save-row">
                    <span id="preference-status" className="hint">{!state.preferences ? "読み込み中" :
                      state.preferenceDirty ? "未保存の変更があります" : "保存済み"}</span>
                    <button id="save-preferences" type="submit"
                      disabled={preferencesDisabled || !state.preferenceDirty}>設定を保存</button>
                  </div>
                </fieldset>
              </form>
              <section id="model-setup" className="model-setup" aria-labelledby="model-setup-heading" hidden={!connected}>
                <div className="model-setup-heading">
                  <h3 id="model-setup-heading">利用できるモデル</h3>
                  <button id="load-models" className="quiet" type="button"
                    disabled={!ready || !connected || pending.has("models") || pending.has("oauth")}
                    onClick={() => void run("models", async (scope) => {
                      const models = readModels(await api(scope, "/api/models"));
                      update(scope, (previous) => ({ ...previous, models, modelsLoaded: true }));
                    })}>{pending.has("models") ? "取得中…" : "一覧を取得"}</button>
                </div>
                <p className="hint">アカウントのモデルIDを確認し、使うIDを TASKMATE_MODEL に設定して再起動してください。</p>
                <p id="model-catalog-status" className="hint model-catalog-status" role="status">
                  {!state.modelsLoaded ? "モデルは自動選択されません。" :
                    state.models.length ? state.models.length + " 件取得しました。使うモデルIDを選んでサーバーに設定してください。" :
                      "このアカウントで利用できるモデルIDを取得できませんでした。"}
                </p>
                <ul id="model-list" className="model-list" aria-label="利用可能なモデルID">
                  {state.models.map((model) => <li key={model}><code>{model}</code></li>)}
                </ul>
              </section>
            </div>
          </section>

          <section className="panel" aria-labelledby="notifications-heading">
            <div className="panel-header">
              <div>
                <h2 id="notifications-heading">お知らせ</h2>
                <p className="panel-description">必要なときに更新できます。</p>
              </div>
              <button id="refresh-notifications" type="button" className="quiet"
                disabled={!ready || pending.has("notifications") || pending.has("preview")}
                onClick={() => void run("notifications", loadNotifications)}>更新</button>
            </div>
            <div className="panel-body">
              <ul id="notification-list" className="notification-list" aria-label="通知一覧">
                {state.notifications.map((notification) => {
                  const time = notificationTime(notification.created_at, state.preferences?.timezone ?? "Asia/Tokyo");
                  return (
                    <li key={notification.id} className="notification-item">
                      <p className="notification-content">{notification.content}</p>
                      <time className="notification-time" dateTime={time.iso}>{time.text}</time>
                    </li>
                  );
                })}
              </ul>
              <p id="notifications-empty" className="empty" hidden={state.notifications.length > 0}>
                {state.notificationsLoaded ? "今のところ、お知らせはありません。" : "お知らせを読み込んでいます"}
              </p>
              <button id="preview-notification" className="wide" type="button" disabled={previewDisabled}
                onClick={() => void run("preview", async (scope) => {
                  const preview = readText(await api(scope, "/api/notifications/preview", { method: "POST" }), "content");
                  update(scope, (previous) => ({ ...previous, preview }));
                  await loadNotifications(scope);
                })}>通知文をプレビュー</button>
              <div id="preview" className="preview" hidden={state.preview === null}>
                <h3>通知プレビュー</h3>
                <p id="preview-content">{state.preview}</p>
              </div>
              <p className="hint notification-note">この画面では通知文を確認できます。外部への配信は行いません。</p>
            </div>
          </section>
        </aside>
      </main>
      <footer className="page-footer">Taskmate Lab · ローカル検証用</footer>
    </div>
  );
}
