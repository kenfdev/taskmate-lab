import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { randomBytes, randomUUID } from "node:crypto";
import { createConnection } from "node:net";
import { setTimeout as delay } from "node:timers/promises";
import { fileURLToPath } from "node:url";

// Run only against a disposable test database, after building both workspace apps.
// Node 24 executes this TypeScript file directly; it never calls OAuth or a model.
const databaseUrl = process.env.TEST_DATABASE_URL;
assert.ok(databaseUrl, "TEST_DATABASE_URL must name a disposable smoke-test database.");
const origin = "http://127.0.0.1:8765";
const key = randomBytes(32).toString("base64");
const root = fileURLToPath(new URL("../", import.meta.url));
const entry = fileURLToPath(new URL("../apps/api/dist/index.js", import.meta.url));

async function portIsOpen(): Promise<boolean> {
  return new Promise((resolve) => {
    const socket = createConnection({ host: "127.0.0.1", port: 8765 });
    let settled = false;
    const finish = (open: boolean) => {
      if (settled) return;
      settled = true;
      socket.destroy();
      resolve(open);
    };
    socket.once("connect", () => finish(true));
    socket.once("error", () => finish(false));
    socket.setTimeout(1000, () => finish(false));
  });
}
assert.equal(
  await portIsOpen(),
  false,
  "Port 8765 is already in use; refusing to contact another service.",
);

const child = spawn(process.execPath, [entry], {
  cwd: root,
  env: {
    ...process.env,
    DATABASE_URL: databaseUrl,
    TASKMATE_ENCRYPTION_KEY: key,
    TASKMATE_MODEL: "",
    LANGSMITH_TRACING: "false",
    LANGCHAIN_TRACING_V2: "false",
    LANGCHAIN_TRACING: "false",
  },
  stdio: ["ignore", "pipe", "pipe"],
});
let exited = false;
let spawnError: Error | undefined;
let output = "";
const recordOutput = (data: Buffer) => {
  // Keep diagnostics bounded and redact the two values this script supplied.
  output = (output + data.toString("utf8")).slice(-8000);
};
child.stdout.on("data", recordOutput);
child.stderr.on("data", recordOutput);
const stopped = new Promise<void>((resolve) => {
  child.once("exit", () => {
    exited = true;
    resolve();
  });
  child.once("error", (error) => {
    spawnError = error;
    exited = true;
    resolve();
  });
});
let cookie = "";
let csrf = "";
function captureCookie(response: Response): void {
  const value = response.headers.get("set-cookie");
  if (value) cookie = value.split(";")[0] ?? "";
}
async function request(
  path: string,
  method = "GET",
  body?: unknown,
  extraHeaders: Record<string, string> = {},
): Promise<Response> {
  assert.equal(exited, false, "The API child exited before the request.");
  return fetch(origin + path, {
    method,
    headers: {
      cookie,
      "x-csrf-token": csrf,
      ...(method === "GET" ? {} : { origin, "content-type": "application/json" }),
      ...extraHeaders,
    },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
    signal: AbortSignal.timeout(5000),
    redirect: "error",
  });
}
async function object(response: Response): Promise<Record<string, unknown>> {
  const value: unknown = await response.json();
  assert.ok(value && typeof value === "object" && !Array.isArray(value), "Expected a JSON object.");
  return value as Record<string, unknown>;
}
async function list(response: Response): Promise<Array<Record<string, unknown>>> {
  const value: unknown = await response.json();
  assert.ok(Array.isArray(value), "Expected a JSON array.");
  return value as Array<Record<string, unknown>>;
}
try {
  let bootstrap: Response | undefined;
  const deadline = Date.now() + 30_000;
  while (Date.now() < deadline && !exited) {
    try {
      const response = await fetch(origin + "/api/session", {
        signal: AbortSignal.timeout(1000),
        redirect: "error",
      });
      if (response.ok) {
        bootstrap = response;
        break;
      }
    } catch {
      /* The child may still be starting or migrating the test database. */
    }
    await delay(100);
  }
  assert.ok(bootstrap, "The built API did not become ready within 30 seconds.");
  await delay(100); // Observe a possible bind failure before any CRUD request.
  assert.equal(exited, false, "The API child exited during startup.");
  captureCookie(bootstrap);
  const session = await object(bootstrap);
  assert.equal(session.user_id, "alice");
  assert.equal(session.model, null);
  assert.equal(session.chatgpt_connected, false);
  assert.equal(typeof session.csrf_token, "string");
  csrf = session.csrf_token as string;
  assert.ok(cookie && csrf);

  const index = await request("/");
  assert.equal(index.status, 200);
  assert.match(index.headers.get("content-type") ?? "", /text\/html/);
  const html = await index.text();
  assert.match(html, /id=["']root["']/);
  const asset = html.match(/<script\b[^>]*\bsrc=["']([^"']+)["']/)?.[1];
  assert.ok(asset, "The built HTML must reference a JavaScript bundle.");
  const assetUrl = new URL(asset, origin);
  assert.equal(assetUrl.origin, origin);
  const bundle = await request(assetUrl.pathname);
  assert.equal(bundle.status, 200);
  assert.match(bundle.headers.get("content-type") ?? "", /javascript/);
  assert.equal((await request("/api/does-not-exist")).status, 404);
  assert.equal(
    (await request("/api/session", "GET", undefined, { host: "localhost:8765" })).status,
    403,
  );

  const title = "Smoke task " + randomUUID();
  const createdResponse = await request("/api/tasks", "POST", { title });
  assert.equal(createdResponse.status, 201);
  const task = await object(createdResponse);
  assert.equal(task.title, title);
  assert.equal(typeof task.id, "string");
  const id = task.id as string;
  const ownTasks = await list(await request("/api/tasks"));
  assert.ok(ownTasks.some((item) => item.id === id));
  const completed = await request("/api/tasks/" + id + "/complete", "POST", {});
  assert.equal(completed.status, 200);
  assert.equal((await object(completed)).completed, true);
  assert.equal(
    (await request("/api/tasks", "POST", { title: "CSRF" }, { "x-csrf-token": "" })).status,
    403,
  );
  assert.equal(
    (
      await request(
        "/api/tasks",
        "POST",
        { title: "Origin must reject" },
        { origin: "https://example.invalid" },
      )
    ).status,
    403,
  );

  const staleCsrf = csrf;
  const switchResponse = await request("/api/session", "POST", { user_id: "bob" });
  assert.equal(switchResponse.status, 200);
  captureCookie(switchResponse);
  const bob = await object(switchResponse);
  assert.equal(bob.user_id, "bob");
  assert.equal(typeof bob.csrf_token, "string");
  csrf = bob.csrf_token as string;
  assert.notEqual(csrf, staleCsrf);
  assert.equal(
    (await request("/api/tasks", "GET", undefined, { "x-csrf-token": staleCsrf })).status,
    403,
  );
  const bobTasks = await list(await request("/api/tasks"));
  assert.equal(bobTasks.some((item) => item.id === id), false);
  assert.equal((await request("/api/tasks/" + id + "/complete", "POST", {})).status, 404);
  console.log("Built-service smoke passed.");
} catch (error) {
  const safeOutput = output
    .replaceAll(databaseUrl, "[TEST_DATABASE_URL]")
    .replaceAll(key, "[TEST_KEY]");
  if (safeOutput) console.error("Built API diagnostics:\n" + safeOutput);
  if (spawnError) console.error("API process could not be spawned.");
  throw error;
} finally {
  if (!exited) child.kill("SIGTERM");
  await Promise.race([stopped, delay(5000, undefined, { ref: false })]);
  if (!exited) {
    child.kill("SIGKILL");
    await stopped;
  }
}
