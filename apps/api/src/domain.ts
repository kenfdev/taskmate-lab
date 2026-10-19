import { createHash, randomBytes, randomUUID } from "node:crypto";
import type { Pool } from "pg";
import { z } from "zod";

/** The local demo selector is deliberately not production authentication. */
export const userIdSchema = z.enum(["alice", "bob"]);
export type UserId = z.infer<typeof userIdSchema>;
export const titleSchema = z.string().trim().min(1).max(240);
export const uuidSchema = z.string().uuid();
export const preferencesSchema = z.object({
  tone: z.enum(["concise", "friendly"]),
  notification_hour: z.number().int().min(0).max(23),
  timezone: z.string().min(1).max(100).refine((value) => {
    try { new Intl.DateTimeFormat("en-US", { timeZone: value }).format(); return true; }
    catch { return false; }
  }, "Use a valid IANA timezone"),
}).strict();
export const chatSchema = z.object({
  message: z.string().trim().min(1).max(8000),
  request_id: uuidSchema,
}).strict();
export type Preferences = z.infer<typeof preferencesSchema>;
export const defaultPreferences: Preferences = {
  tone: "concise", notification_hour: 9, timezone: "Asia/Tokyo",
};
export class DomainError extends Error {
  constructor(public readonly status: 400 | 401 | 403 | 404 | 409 | 422 | 503, message: string) {
    super(message); this.name = "DomainError";
  }
}
export interface Task {
  id: string; title: string; completed: boolean; created_at: Date | string;
}
export interface Session { token_hash: string; user_id: UserId; csrf_token: string }
export interface ChatRequest {
  request_id: string; message: string; status: "running" | "completed"; reply: string | null;
}
export interface ChatMessage {
  role: "user" | "assistant"; content: string; request_id?: string; pending?: true;
}
export interface Notification {
  id: string; kind: "daily" | "preview"; content: string; created_at: Date | string;
}
export interface PreferenceStore {
  get(namespace: string[], key: string): Promise<{ value: Record<string, unknown> } | null>;
  put(namespace: string[], key: string, value: Record<string, unknown>): Promise<void>;
}
export function digest(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}
function lockId(key: string): string {
  return createHash("sha256").update(key).digest().readBigInt64BE().toString();
}

export class DomainRepository {
  constructor(public readonly pool: Pool, private readonly store: PreferenceStore) {}

  async setup(): Promise<void> {
    await this.pool.query(`
      CREATE TABLE IF NOT EXISTS tm_threads (
        user_id text NOT NULL CHECK (user_id IN ('alice','bob')),
        id uuid NOT NULL, created_at timestamptz NOT NULL DEFAULT now(),
        PRIMARY KEY (user_id,id)
      );
      CREATE TABLE IF NOT EXISTS tm_tasks (
        user_id text NOT NULL CHECK (user_id IN ('alice','bob')),
        id uuid NOT NULL, title text NOT NULL CHECK (length(title) BETWEEN 1 AND 240),
        completed boolean NOT NULL DEFAULT false,
        created_at timestamptz NOT NULL DEFAULT now(),
        thread_id uuid, tool_call_id text,
        PRIMARY KEY (user_id,id), UNIQUE (user_id,thread_id,tool_call_id),
        FOREIGN KEY (user_id,thread_id) REFERENCES tm_threads(user_id,id),
        CHECK ((thread_id IS NULL) = (tool_call_id IS NULL))
      );
      CREATE TABLE IF NOT EXISTS tm_sessions (
        token_hash text PRIMARY KEY,
        user_id text NOT NULL CHECK (user_id IN ('alice','bob')),
        csrf_token text NOT NULL, expires_at timestamptz NOT NULL
      );
      CREATE TABLE IF NOT EXISTS tm_chat_requests (
        sequence bigserial UNIQUE,
        user_id text NOT NULL, thread_id uuid NOT NULL, request_id uuid NOT NULL,
        message text NOT NULL, status text NOT NULL CHECK (status IN ('running','completed')),
        reply text, created_at timestamptz NOT NULL DEFAULT now(),
        PRIMARY KEY (user_id,thread_id,request_id),
        FOREIGN KEY (user_id,thread_id) REFERENCES tm_threads(user_id,id),
        CHECK ((status='completed') = (reply IS NOT NULL))
      );
      CREATE UNIQUE INDEX IF NOT EXISTS tm_one_running_chat
        ON tm_chat_requests(user_id,thread_id) WHERE status='running';
      CREATE TABLE IF NOT EXISTS tm_notifications (
        user_id text NOT NULL CHECK (user_id IN ('alice','bob')),
        id uuid NOT NULL, dedupe_key text NOT NULL,
        kind text NOT NULL CHECK (kind IN ('daily','preview')),
        content text NOT NULL, created_at timestamptz NOT NULL DEFAULT now(),
        PRIMARY KEY (user_id,id), UNIQUE (user_id,dedupe_key)
      );
    `);
  }

  /** Transaction-scoped DB locks survive process concurrency and release on crash.
   * Work uses other pool connections; never start with a pool smaller than 4.
   * Do not race work against a timeout: the lock must outlive all its effects.
   */
  async withLock<T>(key: string, work: () => Promise<T>): Promise<T> {
    const client = await this.pool.connect();
    try {
      await client.query("BEGIN");
      const result = await client.query<{ locked: boolean }>(
        "SELECT pg_try_advisory_xact_lock($1::bigint) AS locked", [lockId(key)]);
      if (!result.rows[0]?.locked) throw new DomainError(409, "This operation is already running. Retry shortly.");
      const value = await work();
      await client.query("COMMIT");
      return value;
    } catch (error) {
      await client.query("ROLLBACK").catch(() => undefined);
      throw error;
    } finally { client.release(); }
  }

  async listTasks(user: UserId): Promise<Task[]> {
    const result = await this.pool.query<Task>(
      "SELECT id,title,completed,created_at FROM tm_tasks WHERE user_id=$1 ORDER BY created_at,id", [user]);
    return result.rows;
  }
  async createTask(user: UserId, title: string, effect?: { threadId: string; toolCallId: string }): Promise<Task> {
    title = titleSchema.parse(title);
    if (effect) {
      uuidSchema.parse(effect.threadId);
      if (!effect.toolCallId || effect.toolCallId.length > 500) throw new DomainError(422, "Missing tool call identity.");
      await this.ownedThread(user, effect.threadId);
    }
    const result = await this.pool.query<Task>(
      `INSERT INTO tm_tasks(user_id,id,title,thread_id,tool_call_id) VALUES($1,$2,$3,$4,$5)
       ON CONFLICT (user_id,thread_id,tool_call_id) DO NOTHING
       RETURNING id,title,completed,created_at`,
      [user, randomUUID(), title, effect?.threadId ?? null, effect?.toolCallId ?? null]);
    if (result.rows[0]) return result.rows[0];
    const existing = await this.pool.query<Task>(
      `SELECT id,title,completed,created_at FROM tm_tasks
       WHERE user_id=$1 AND thread_id=$2 AND tool_call_id=$3`,
      [user, effect?.threadId, effect?.toolCallId]);
    const task = existing.rows[0];
    if (!task || task.title !== title) throw new DomainError(409, "Tool call identity was reused with different task data.");
    return task;
  }
  async completeTask(user: UserId, id: string): Promise<Task> {
    uuidSchema.parse(id);
    const result = await this.pool.query<Task>(
      `UPDATE tm_tasks SET completed=true WHERE user_id=$1 AND id=$2
       RETURNING id,title,completed,created_at`, [user, id]);
    const task = result.rows[0];
    if (!task) throw new DomainError(404, "Task not found.");
    return task;
  }
  async getPreferences(user: UserId): Promise<Preferences> {
    const item = await this.store.get(["users", user, "preferences"], "profile");
    return item ? preferencesSchema.parse(item.value) : { ...defaultPreferences };
  }
  async savePreferences(user: UserId, input: unknown): Promise<Preferences> {
    const value = preferencesSchema.parse(input);
    await this.store.put(["users", user, "preferences"], "profile", value);
    return value;
  }
  async newSession(user: UserId, oldHash?: string): Promise<{ token: string; session: Session }> {
    const token = randomBytes(32).toString("base64url");
    const session = { token_hash: digest(token), user_id: user, csrf_token: randomBytes(32).toString("base64url") };
    const client = await this.pool.connect();
    try {
      await client.query("BEGIN");
      if (oldHash) await client.query("DELETE FROM tm_sessions WHERE token_hash=$1", [oldHash]);
      await client.query(
        "INSERT INTO tm_sessions(token_hash,user_id,csrf_token,expires_at) VALUES($1,$2,$3,now()+interval '7 days')",
        [session.token_hash, user, session.csrf_token]);
      await client.query("DELETE FROM tm_sessions WHERE expires_at <= now()");
      await client.query("COMMIT");
    } catch (error) {
      await client.query("ROLLBACK").catch(() => undefined); throw error;
    } finally { client.release(); }
    return { token, session };
  }
  async session(token: string | undefined): Promise<Session | null> {
    if (!token || !/^[A-Za-z0-9_-]{43}$/.test(token)) return null;
    const result = await this.pool.query<Session>(
      "SELECT token_hash,user_id,csrf_token FROM tm_sessions WHERE token_hash=$1 AND expires_at>now()", [digest(token)]);
    return result.rows[0] ?? null;
  }
  async createThread(user: UserId): Promise<{ id: string }> {
    const id = randomUUID();
    await this.pool.query("INSERT INTO tm_threads(user_id,id) VALUES($1,$2)", [user, id]);
    return { id };
  }
  async ownedThread(user: UserId, id: string): Promise<void> {
    uuidSchema.parse(id);
    const result = await this.pool.query("SELECT 1 FROM tm_threads WHERE user_id=$1 AND id=$2", [user, id]);
    if (!result.rowCount) throw new DomainError(404, "Thread not found.");
  }
  async request(user: UserId, thread: string, id: string): Promise<ChatRequest | null> {
    const result = await this.pool.query<ChatRequest>(
      `SELECT request_id,message,status,reply FROM tm_chat_requests
       WHERE user_id=$1 AND thread_id=$2 AND request_id=$3`, [user, thread, id]);
    return result.rows[0] ?? null;
  }
  /** Caller holds the per-thread lock. A failed command stays resumable. */
  async beginRequest(user: UserId, thread: string, id: string, message: string): Promise<void> {
    const running = await this.pool.query<{ request_id: string }>(
      "SELECT request_id FROM tm_chat_requests WHERE user_id=$1 AND thread_id=$2 AND status='running'", [user, thread]);
    if (running.rows[0] && running.rows[0].request_id !== id)
      throw new DomainError(409, "Retry the unfinished message before sending a new one.");
    await this.pool.query(
      `INSERT INTO tm_chat_requests(user_id,thread_id,request_id,message,status) VALUES($1,$2,$3,$4,'running')
       ON CONFLICT (user_id,thread_id,request_id) DO NOTHING`, [user, thread, id, message]);
  }
  async finishRequest(user: UserId, thread: string, id: string, reply: string): Promise<void> {
    await this.pool.query(
      "UPDATE tm_chat_requests SET status='completed',reply=$4 WHERE user_id=$1 AND thread_id=$2 AND request_id=$3",
      [user, thread, id, reply]);
  }
  async messages(user: UserId, thread: string): Promise<ChatMessage[]> {
    await this.ownedThread(user, thread);
    const result = await this.pool.query<ChatRequest>(
      `SELECT request_id,message,status,reply FROM tm_chat_requests
       WHERE user_id=$1 AND thread_id=$2 ORDER BY sequence`, [user, thread]);
    return result.rows.flatMap((row): ChatMessage[] => row.status === "completed"
      ? [{ role: "user", content: row.message }, { role: "assistant", content: row.reply! }]
      : [{ role: "user", content: row.message, request_id: row.request_id, pending: true }]);
  }
  async notification(user: UserId, key: string): Promise<Notification | null> {
    const result = await this.pool.query<Notification>(
      "SELECT id,kind,content,created_at FROM tm_notifications WHERE user_id=$1 AND dedupe_key=$2", [user, key]);
    return result.rows[0] ?? null;
  }
  async addNotification(user: UserId, key: string, kind: Notification["kind"], content: string): Promise<Notification> {
    await this.pool.query(
      `INSERT INTO tm_notifications(user_id,id,dedupe_key,kind,content) VALUES($1,$2,$3,$4,$5)
       ON CONFLICT (user_id,dedupe_key) DO NOTHING`, [user, randomUUID(), key, kind, content]);
    const notice = await this.notification(user, key);
    if (!notice) throw new Error("Notification persistence failed.");
    return notice;
  }
  async notifications(user: UserId): Promise<Notification[]> {
    const result = await this.pool.query<Notification>(
      "SELECT id,kind,content,created_at FROM tm_notifications WHERE user_id=$1 ORDER BY created_at DESC,id DESC LIMIT 50", [user]);
    return result.rows;
  }
}
