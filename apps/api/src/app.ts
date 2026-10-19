import { timingSafeEqual } from "node:crypto";
import { Hono, type Context } from "hono";
import { bodyLimit } from "hono/body-limit";
import { getCookie, setCookie } from "hono/cookie";
import { getConnInfo } from "@hono/node-server/conninfo";
import { serveStatic } from "@hono/node-server/serve-static";
import pg from "pg";
import { PostgresSaver } from "@langchain/langgraph-checkpoint-postgres";
import { PostgresStore } from "@langchain/langgraph-checkpoint-postgres/store";
import { z } from "zod";
import { AuthService, PostgresAuthRepository, AuthError } from "./auth.js";
import { AgentService } from "./agent.js";
import { ResponsesError } from "./model.js";
import {
  DomainError,
  DomainRepository,
  titleSchema,
  userIdSchema,
  preferencesSchema,
  type Session,
} from "./domain.js";
import { NotificationService } from "./worker.js";

const ORIGIN = "http://127.0.0.1:8765";
const COOKIE = "taskmate_session";
export interface Settings {
  databaseUrl: string;
  encryptionKey: string;
  model?: string;
}
export function settingsFromEnv(env: NodeJS.ProcessEnv = process.env): Settings {
  if (!env.DATABASE_URL || !env.TASKMATE_ENCRYPTION_KEY)
    throw new Error("Set DATABASE_URL and TASKMATE_ENCRYPTION_KEY in .env before starting.");
  return {
    databaseUrl: env.DATABASE_URL,
    encryptionKey: env.TASKMATE_ENCRYPTION_KEY,
    model: env.TASKMATE_MODEL?.trim() || undefined,
  };
}
type AuthApi = Pick<AuthService, "begin" | "complete" | "status" | "accessToken" | "disconnect">;
type ChatApi = Pick<AgentService, "chat" | "availableModels" | "reminder">;
type RepoApi = Pick<
  DomainRepository,
  | "session"
  | "newSession"
  | "listTasks"
  | "createTask"
  | "completeTask"
  | "getPreferences"
  | "savePreferences"
  | "createThread"
  | "messages"
  | "notifications"
>;
export interface AppServices {
  settings: Settings;
  repo: RepoApi;
  auth: AuthApi;
  chat: ChatApi;
  notifications: Pick<NotificationService, "preview">;
}
export interface Services extends AppServices {
  repo: DomainRepository;
  chat: AgentService;
  close(): Promise<void>;
}
export async function buildServices(settings = settingsFromEnv()): Promise<Services> {
  // This local learning app never exports chat/token traces to telemetry services.
  process.env.LANGSMITH_TRACING = "false";
  process.env.LANGCHAIN_TRACING_V2 = "false";
  process.env.LANGCHAIN_TRACING = "false";
  const pool = new pg.Pool({
    connectionString: settings.databaseUrl,
    max: 16,
    connectionTimeoutMillis: 5000,
  });
  pool.on("error", () => console.error("An idle PostgreSQL connection failed."));
  const store = new PostgresStore({
    connectionOptions: {
      connectionString: settings.databaseUrl,
      max: 4,
      connectionTimeoutMillis: 5000,
    },
    schema: "public",
    ensureTables: false,
  });
  let closed = false;
  const close = async () => {
    if (closed) return;
    closed = true;
    await store.stop();
    await pool.end();
  };
  try {
    const checkpointer = new PostgresSaver(pool);
    const repo = new DomainRepository(pool, store);
    const authRepo = new PostgresAuthRepository(pool);
    const auth = new AuthService({
      repo: authRepo,
      encryptionKey: settings.encryptionKey,
      redirectUri: ORIGIN + "/auth/callback",
    });
    // Serialize migrations across API and worker startup. Library setup uses other
    // pool connections because checkpoint migrations may not run in this transaction.
    const migration = await pool.connect();
    try {
      await migration.query("BEGIN");
      await migration.query("SELECT pg_advisory_xact_lock(87658765)");
      await repo.setup();
      await authRepo.setup();
      await checkpointer.setup();
      await store.setup();
      await migration.query("COMMIT");
    } catch (error) {
      await migration.query("ROLLBACK").catch(() => undefined);
      throw error;
    } finally {
      migration.release();
    }
    const chat = new AgentService(repo, auth, settings.model, checkpointer, store);
    const notifications = new NotificationService(repo, chat);
    return { settings, repo, auth, chat, notifications, close };
  } catch (error) {
    await close();
    throw error;
  }
}

type Env = { Variables: { session: Session } };
export interface AppOptions {
  staticRoot?: string;
  /** Test injection only. Production derives the peer from the Node socket. */
  getClientAddress?: (context: Context) => string | undefined;
}
function constantTimeEqual(left: string, right: string): boolean {
  const a = Buffer.from(left);
  const b = Buffer.from(right);
  return a.length === b.length && timingSafeEqual(a, b);
}
function sessionCookie(c: Context, token: string): void {
  setCookie(c, COOKIE, token, {
    path: "/",
    httpOnly: true,
    sameSite: "Lax",
    secure: false,
    maxAge: 7 * 86400,
  });
}
export function createApp(services: AppServices, options: AppOptions = {}) {
  const app = new Hono<Env>();
  app.onError((error, c) => {
    if (error instanceof DomainError) return c.json({ detail: error.message }, error.status);
    if (error instanceof AuthError)
      return c.json({ detail: error.publicMessage, code: error.code }, 401);
    if (error instanceof ResponsesError) return c.json({ detail: error.message }, 502);
    if (error instanceof z.ZodError) return c.json({ detail: "Invalid request." }, 422);
    if (error instanceof SyntaxError) return c.json({ detail: "Invalid JSON." }, 400);
    // Never stringify unknown errors: provider responses and callback codes may be attached.
    return c.json({ detail: "The local service failed. Retry the same request." }, 500);
  });
  app.use("*", async (c, next) => {
    c.header("Cache-Control", "no-store");
    c.header("Referrer-Policy", "no-referrer");
    c.header("X-Content-Type-Options", "nosniff");
    c.header("X-Frame-Options", "DENY");
    c.header("Cross-Origin-Resource-Policy", "same-origin");
    const address = options.getClientAddress
      ? options.getClientAddress(c)
      : getConnInfo(c).remote.address;
    if (
      !["127.0.0.1", "::1", "::ffff:127.0.0.1"].includes(address ?? "") ||
      c.req.header("host") !== "127.0.0.1:8765" ||
      new URL(c.req.url).origin !== ORIGIN
    )
      throw new DomainError(403, "This app is available only at http://127.0.0.1:8765.");
    const origin = c.req.header("origin");
    if (origin && origin !== ORIGIN)
      throw new DomainError(403, "Cross-origin requests are not allowed.");
    if (c.req.path.startsWith("/api/") && c.req.header("sec-fetch-site") === "cross-site")
      throw new DomainError(403, "Cross-site requests are not allowed.");
    await next();
  });
  app.use(
    "/api/*",
    bodyLimit({ maxSize: 32768, onError: (c) => c.json({ detail: "Request is too large." }, 413) }),
  );
  app.use("/api/*", async (c, next) => {
    const session = await services.repo.session(getCookie(c, COOKIE));
    const bootstrap = c.req.path === "/api/session" && c.req.method === "GET";
    if (!session && !bootstrap)
      throw new DomainError(401, "Open the app to start a local session.");
    if (session) c.set("session", session);
    const mutation = !["GET", "HEAD", "OPTIONS"].includes(c.req.method);
    // The read token also binds each tab's displayed profile to the shared cookie.
    // A profile switch in another tab must not silently read that profile's data.
    if (
      !bootstrap &&
      (!session || !constantTimeEqual(c.req.header("x-csrf-token") ?? "", session.csrf_token))
    )
      throw new DomainError(403, "Refresh the app and retry with its CSRF token.");
    if (mutation && c.req.header("origin") !== ORIGIN)
      throw new DomainError(403, "Cross-origin requests are not allowed.");
    await next();
  });
  const sessionView = async (session: Session) => ({
    user_id: session.user_id,
    csrf_token: session.csrf_token,
    chatgpt_connected: (await services.auth.status(session.user_id)).connected,
    model: services.settings.model ?? null,
  });
  app.get("/api/session", async (c) => {
    let session = c.get("session");
    if (!session) {
      const created = await services.repo.newSession("alice");
      session = created.session;
      sessionCookie(c, created.token);
    }
    return c.json(await sessionView(session));
  });
  app.post("/api/session", async (c) => {
    const input = z
      .object({ user_id: userIdSchema })
      .strict()
      .parse(await c.req.json());
    const created = await services.repo.newSession(input.user_id, c.get("session").token_hash);
    sessionCookie(c, created.token);
    return c.json(await sessionView(created.session));
  });
  app.post("/api/oauth/start", async (c) => {
    const session = c.get("session");
    return c.json({
      authorization_url: await services.auth.begin(session.user_id, session.token_hash),
    });
  });
  app.get("/auth/callback", async (c) => {
    const session = await services.repo.session(getCookie(c, COOKIE));
    if (!session)
      throw new DomainError(401, "The browser session expired. Open the app and connect again.");
    const state = c.req.query("state");
    if (!state || state.length > 512)
      throw new DomainError(400, "The authorization callback is missing valid state.");
    await services.auth.complete(session.user_id, session.token_hash, {
      state,
      code: c.req.query("code"),
      clientId: c.req.query("client_id"),
      error: c.req.query("error"),
    });
    return c.redirect("/", 303);
  });
  app.post("/api/oauth/disconnect", async (c) =>
    c.json({
      disconnected: true,
      remote_revocation_confirmed: await services.auth.disconnect(c.get("session").user_id),
    }),
  );
  app.get("/api/models", async (c) =>
    c.json({ models: await services.chat.availableModels(c.get("session").user_id) }),
  );
  app.get("/api/tasks", async (c) =>
    c.json(await services.repo.listTasks(c.get("session").user_id)),
  );
  app.post("/api/tasks", async (c) => {
    const input = z
      .object({ title: titleSchema })
      .strict()
      .parse(await c.req.json());
    return c.json(await services.repo.createTask(c.get("session").user_id, input.title), 201);
  });
  app.post("/api/tasks/:id/complete", async (c) =>
    c.json(await services.repo.completeTask(c.get("session").user_id, c.req.param("id"))),
  );
  app.get("/api/preferences", async (c) =>
    c.json(await services.repo.getPreferences(c.get("session").user_id)),
  );
  app.put("/api/preferences", async (c) =>
    c.json(
      await services.repo.savePreferences(
        c.get("session").user_id,
        preferencesSchema.parse(await c.req.json()),
      ),
    ),
  );
  app.post("/api/threads", async (c) =>
    c.json(await services.repo.createThread(c.get("session").user_id), 201),
  );
  app.get("/api/threads/:id/messages", async (c) =>
    c.json(await services.repo.messages(c.get("session").user_id, c.req.param("id"))),
  );
  app.post("/api/threads/:id/chat", async (c) =>
    c.json(
      await services.chat.chat(c.get("session").user_id, c.req.param("id"), await c.req.json()),
    ),
  );
  app.get("/api/notifications", async (c) =>
    c.json(await services.repo.notifications(c.get("session").user_id)),
  );
  app.post("/api/notifications/preview", async (c) =>
    c.json(await services.notifications.preview(c.get("session").user_id), 201),
  );
  app.all("/api/*", (c) => c.json({ detail: "Endpoint not found." }, 404));
  app.all("/auth/*", (c) => c.json({ detail: "Endpoint not found." }, 404));
  if (options.staticRoot) {
    app.use("*", serveStatic({ root: options.staticRoot }));
    app.get("*", serveStatic({ root: options.staticRoot, path: "index.html" }));
  }
  app.notFound((c) => c.json({ detail: "Not found." }, 404));
  return app;
}
