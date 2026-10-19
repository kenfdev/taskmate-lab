import { createHash, randomBytes, randomUUID } from "node:crypto";
import { inspect } from "node:util";
import { setTimeout as delay } from "node:timers/promises";
import { exportJWK, generateKeyPair, SignJWT, type JWK } from "jose";
import { Pool } from "pg";
import { beforeAll, beforeEach, describe, expect, it } from "vite-plus/test";
import {
  AUTHORIZE, DISCOVERY, ISSUER, JWKS, REVOKE, TOKEN,
  AuthError, AuthService, PostgresAuthRepository,
  type AuthRecordKind, type AuthRecords, type AuthRepository,
} from "../src/auth.js";

class MemoryRepository implements AuthRepository {
  readonly values = new Map<string, Buffer>();
  private readonly tails = new Map<string, Promise<void>>();

  async hostId(): Promise<string> {
    return "urn:uuid:00000000-0000-4000-8000-000000000001";
  }

  async withUser<T>(userId: string, work: (records: AuthRecords) => Promise<T>): Promise<T> {
    const previous = this.tails.get(userId) ?? Promise.resolve();
    let release!: () => void;
    const current = new Promise<void>((resolve) => { release = resolve; });
    this.tails.set(userId, current);
    await previous;
    const key = (kind: AuthRecordKind) => userId + ":" + kind;
    try {
      return await work({
        read: async (kind) => this.values.get(key(kind)) ?? null,
        write: async (kind, value) => { this.values.set(key(kind), value); },
        delete: async (kind) => { this.values.delete(key(kind)); },
      });
    } finally {
      release();
      if (this.tails.get(userId) === current) this.tails.delete(userId);
    }
  }
}

let signing: Awaited<ReturnType<typeof generateKeyPair>>;
let incorrectSigning: Awaited<ReturnType<typeof generateKeyPair>>;
let publicJwk: JWK;

beforeAll(async () => {
  signing = await generateKeyPair("RS256");
  incorrectSigning = await generateKeyPair("RS256");
  publicJwk = await exportJWK(signing.publicKey);
});

class Provider {
  readonly calls: string[] = [];
  readonly forms: Record<string, string>[] = [];
  nonce = "";
  clientId = "oaiapp_test_registration";
  subject = "test_subject";
  expiresIn = 3600;
  refreshCount = 0;
  revokeStatus = 200;
  refreshDelay = 0;
  exchangeError: string | null = null;
  claimOverrides: Record<string, unknown> = {};
  tokenOverrides: Record<string, unknown> = {};
  discoveryOverride: Record<string, unknown> = {};
  wrongSignature = false;
  kid = "test-key";

  fetch: typeof globalThis.fetch = async (input, init) => {
    const request = new Request(input, init);
    this.calls.push(request.url);
    expect(request.redirect).toBe("error");
    expect(request.signal).toBeDefined();
    if (request.url === DISCOVERY) {
      return Response.json({
        issuer: ISSUER, authorization_endpoint: AUTHORIZE, token_endpoint: TOKEN,
        jwks_uri: JWKS, revocation_endpoint: REVOKE,
        id_token_signing_alg_values_supported: ["RS256"],
        ...this.discoveryOverride,
      });
    }
    if (request.url === JWKS) return Response.json({ keys: [{ ...publicJwk, kid: this.kid, use: "sig" }] });
    if (request.url === REVOKE) return new Response(null, { status: this.revokeStatus });
    expect(request.url).toBe(TOKEN);
    const form = Object.fromEntries(new URLSearchParams(await request.text()));
    this.forms.push(form);
    if (this.exchangeError) {
      return Response.json({ error: this.exchangeError, detail: "DO_NOT_EXPOSE_PROVIDER_DETAIL" }, { status: 400 });
    }
    if (form.grant_type === "refresh_token") {
      this.refreshCount++;
      await delay(this.refreshDelay);
      return Response.json({
        access_token: "refreshed-access-" + this.refreshCount,
        refresh_token: "replacement-refresh-" + this.refreshCount,
        token_type: "Bearer",
        expires_in: 3600,
        ...this.tokenOverrides,
      });
    }
    const now = Math.floor(Date.now() / 1000);
    const token = await new SignJWT({
      iss: ISSUER, aud: this.clientId, sub: this.subject,
      iat: now, exp: now + 3600, nonce: this.nonce,
      email: "test@example.invalid", ...this.claimOverrides,
    }).setProtectedHeader({ alg: "RS256", kid: this.kid })
      .sign(this.wrongSignature ? incorrectSigning.privateKey : signing.privateKey);
    return Response.json({
      access_token: "initial-access",
      refresh_token: "initial-refresh",
      id_token: token,
      token_type: "Bearer",
      expires_in: this.expiresIn,
      scope: "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct",
      ...this.tokenOverrides,
    });
  };
}

let repo: MemoryRepository;
let provider: Provider;
let service: AuthService;
let encryptionKey: string;
let now: number;

beforeEach(() => {
  repo = new MemoryRepository();
  provider = new Provider();
  encryptionKey = randomBytes(32).toString("base64");
  now = Date.now() / 1000;
  service = new AuthService({ repo, encryptionKey, fetch: provider.fetch, now: () => now });
});

async function start(auth = service, user = "alice", browser = "browser-alice") {
  const query = new URL(await auth.begin(user, browser)).searchParams;
  provider.nonce = query.get("nonce")!;
  return query;
}

async function connect(auth = service, user = "alice", browser = "browser-alice") {
  const query = await start(auth, user, browser);
  const status = await auth.complete(user, browser, {
    state: query.get("state")!, code: "code-one", clientId: provider.clientId,
  });
  return { query, status };
}

describe("ChatGPT authorization", () => {
  it("uses dynamic registration, exact loopback, PKCE and an issued client for exchange", async () => {
    expect(provider.calls).toEqual([]);
    const { query, status } = await connect();
    expect(query.get("client_id")).toBe("dynamic_agent_client");
    expect(query.get("agent_name_hint")).toBe("taskmate-lab");
    expect(query.get("ext_agent_host_id")).toMatch(/^urn:uuid:/);
    expect(query.get("redirect_uri")).toBe("http://127.0.0.1:8765/auth/callback");
    const form = provider.forms[0]!;
    expect(query.get("code_challenge")).toBe(createHash("sha256").update(form.code_verifier!).digest("base64url"));
    expect(query.get("code_challenge_method")).toBe("S256");
    expect(form.client_id).toBe("oaiapp_test_registration");
    expect(form.redirect_uri).toBe(query.get("redirect_uri"));
    expect(form.resource).toBe("https://api.openai.com/v1");
    expect(form).not.toHaveProperty("client_secret");
    expect(status).toMatchObject({ connected: true, plan_enabled: true });
    expect(status).not.toHaveProperty("access_token");
    expect(status).not.toHaveProperty("id_token");
    expect(await service.accessToken("alice")).toBe("initial-access");
  });

  it("encrypts credentials, binds AAD to user and kind, and keeps keys out of inspection", async () => {
    await connect();
    const encrypted = repo.values.get("alice:credentials")!;
    expect(encrypted.includes(Buffer.from("initial-access"))).toBe(false);
    expect(encrypted.includes(Buffer.from("test@example.invalid"))).toBe(false);
    expect(inspect(service, { showHidden: true, depth: 5 })).not.toContain(encryptionKey);
    expect(inspect(service, { showHidden: true, depth: 5 })).not.toContain(Buffer.from(encryptionKey, "base64").toString("hex"));
    expect((await service.status("bob")).connected).toBe(false);
    repo.values.set("bob:credentials", encrypted);
    await expect(service.status("bob")).rejects.toMatchObject({ code: "invalid_stored_credentials" });
    repo.values.set("alice:pending", encrypted);
    await expect(service.complete("alice", "browser-alice", { state: "x", code: "x" }))
      .rejects.toMatchObject({ code: "invalid_stored_credentials" });
  });

  it("uses a fresh IV and rejects ciphertext tampering and another encryption key", async () => {
    await connect();
    const first = Buffer.from(repo.values.get("alice:credentials")!);
    const query = await start();
    await service.complete("alice", "browser-alice", { state: query.get("state")!, code: "second-code" });
    const second = repo.values.get("alice:credentials")!;
    expect(second.subarray(1, 13).equals(first.subarray(1, 13))).toBe(false);
    const wrongKey = new AuthService({
      repo, encryptionKey: randomBytes(32).toString("base64"), fetch: provider.fetch,
    });
    await expect(wrongKey.status("alice")).rejects.toMatchObject({ code: "invalid_stored_credentials" });
    const tampered = Buffer.from(second);
    tampered[29] = tampered[29]! ^ 1;
    repo.values.set("alice:credentials", tampered);
    await expect(service.status("alice")).rejects.toMatchObject({ code: "invalid_stored_credentials" });
  });

  it.each(["wrong-state", "wrong-browser", "expired", "denied", "missing-client", "dynamic-client"])(
    "rejects %s without exchanging a code and consumes the attempt", async (failure) => {
      const query = await start();
      if (failure === "expired") now += 601;
      const callback = {
        state: failure === "wrong-state" ? "wrong" : query.get("state")!,
        code: "code-one",
        ...(failure === "missing-client" ? {} : { clientId: failure === "dynamic-client" ? "dynamic_agent_client" : provider.clientId }),
        ...(failure === "denied" ? { error: "access_denied" } : {}),
      };
      await expect(service.complete(
        "alice", failure === "wrong-browser" ? "other-browser" : "browser-alice", callback,
      )).rejects.toBeInstanceOf(AuthError);
      await expect(service.complete("alice", "browser-alice", {
        state: query.get("state")!, code: "replay", clientId: provider.clientId,
      })).rejects.toMatchObject({ code: "missing_sign_in" });
      expect(provider.forms).toEqual([]);
    },
  );

  it("rejects successful callback replay", async () => {
    const { query } = await connect();
    await expect(service.complete("alice", "browser-alice", {
      state: query.get("state")!, code: "replay", clientId: provider.clientId,
    })).rejects.toMatchObject({ code: "missing_sign_in" });
    expect(provider.forms).toHaveLength(1);
  });

  it.each([
    { iss: "https://attacker.invalid" },
    { aud: "other-client" },
    { sub: "" },
    { exp: 1 },
    { iat: 9999999999 },
    { nonce: "wrong-nonce" },
    { aud: ["oaiapp_test_registration", "other-client"] },
    { azp: "other-client" },
  ])("verifies OIDC claims %j", async (claims) => {
    provider.claimOverrides = claims;
    await expect(connect()).rejects.toMatchObject({ code: "invalid_identity_token" });
    expect((await service.status("alice")).connected).toBe(false);
  });

  it("verifies the actual RS256 signature", async () => {
    provider.wrongSignature = true;
    await expect(connect()).rejects.toMatchObject({ code: "invalid_identity_token" });
  });

  it("rejects an HMAC-signed ID token even when its signature is valid", async () => {
    provider.tokenOverrides.id_token = await new SignJWT({ sub: "test_subject" })
      .setProtectedHeader({ alg: "HS256", kid: "test-key" }).sign(randomBytes(32));
    await expect(connect()).rejects.toMatchObject({ code: "invalid_identity_token" });
  });

  it("reloads JWKS when the issuer rotates to an unfamiliar kid", async () => {
    await connect();
    provider.kid = "replacement-key";
    const query = await start();
    await service.complete("alice", "browser-alice", { state: query.get("state")!, code: "new-code" });
    expect(provider.calls.filter((url) => url === JWKS)).toHaveLength(2);
  });

  it("reuses the client and host, omits token hints, and rejects another subject", async () => {
    const first = await connect();
    const query = await start();
    expect(query.get("client_id")).toBe(provider.clientId);
    expect(query.get("ext_agent_host_id")).toBe(first.query.get("ext_agent_host_id"));
    expect(query.has("agent_name_hint")).toBe(false);
    expect(query.has("id_token_hint")).toBe(false);
    expect(query.get("state")).not.toBe(first.query.get("state"));
    provider.subject = "another-subject";
    await expect(service.complete("alice", "browser-alice", {
      state: query.get("state")!, code: "second-code",
    })).rejects.toMatchObject({ code: "account_mismatch" });
    expect(await service.accessToken("alice")).toBe("initial-access");
  });

  it("rejects changed registration before exchanging", async () => {
    await connect();
    const query = await start();
    await expect(service.complete("alice", "browser-alice", {
      state: query.get("state")!, code: "second-code", clientId: "other-client",
    })).rejects.toMatchObject({ code: "registration_mismatch" });
    expect(provider.forms).toHaveLength(1);
  });

  it("retains an issued client after an expired first code", async () => {
    provider.exchangeError = "invalid_grant";
    await expect(connect()).rejects.toMatchObject({ code: "invalid_grant" });
    expect((await start()).get("client_id")).toBe(provider.clientId);
    expect((await service.status("alice")).connected).toBe(false);
  });

  it.each(["openid profile email", undefined])("requires explicitly granted direct-plan scope %s", async (scope) => {
    provider.tokenOverrides.scope = scope;
    const { status } = await connect();
    expect(status.connected).toBe(true);
    expect(status.plan_enabled).toBe(false);
    await expect(service.accessToken("alice")).rejects.toMatchObject({ code: "plan_permission_required" });
  });

  it("serializes refreshes across service instances and stores the replacement", async () => {
    provider.expiresIn = 30;
    provider.refreshDelay = 10;
    await connect();
    const other = new AuthService({ repo, encryptionKey, fetch: provider.fetch, now: () => now });
    expect(await Promise.all([service.accessToken("alice"), other.accessToken("alice")]))
      .toEqual(["refreshed-access-1", "refreshed-access-1"]);
    expect(provider.refreshCount).toBe(1);
    expect(provider.forms.at(-1)).toEqual({
      grant_type: "refresh_token", client_id: provider.clientId,
      refresh_token: "initial-refresh", resource: "https://api.openai.com/v1",
    });
    now += 3570;
    expect(await other.accessToken("alice")).toBe("refreshed-access-2");
    expect(provider.forms.at(-1)?.refresh_token).toBe("replacement-refresh-1");
    expect((await service.status("alice")).plan_enabled).toBe(true);
  });

  it("does not refresh before earliest_refresh_at", async () => {
    provider.expiresIn = 30;
    provider.tokenOverrides.earliest_refresh_at = now + 100;
    await connect();
    expect(await service.accessToken("alice")).toBe("initial-access");
    now += 31;
    await expect(service.accessToken("alice")).rejects.toMatchObject({ code: "refresh_not_due" });
    expect(provider.refreshCount).toBe(0);
  });

  it("does not replay a revoked refresh token", async () => {
    provider.expiresIn = 30;
    await connect();
    provider.exchangeError = "invalid_grant";
    await expect(service.accessToken("alice")).rejects.toMatchObject({ code: "invalid_grant" });
    const count = provider.forms.length;
    await expect(service.accessToken("alice")).rejects.toMatchObject({ code: "sign_in_required" });
    expect(provider.forms).toHaveLength(count);
    expect((await service.status("alice")).connected).toBe(false);
  });

  it.each([[200, true], [503, false]] as const)("clears tokens after revocation %s", async (status, confirmed) => {
    await connect();
    provider.revokeStatus = status;
    expect(await service.disconnect("alice")).toBe(confirmed);
    expect((await service.status("alice")).connected).toBe(false);
    expect((await start()).get("client_id")).toBe(provider.clientId);
  });

  it("pins discovery endpoints", async () => {
    provider.discoveryOverride.token_endpoint = "https://attacker.invalid/token";
    await expect(start()).rejects.toMatchObject({ code: "unexpected_provider_endpoints" });
    expect(provider.calls).toEqual([DISCOVERY]);
  });

  it("redacts provider errors", async () => {
    provider.exchangeError = "untrusted-error-value";
    await expect(connect()).rejects.toMatchObject({
      code: "token_exchange_failed",
      publicMessage: "ChatGPT sign-in could not be completed. Please sign in again.",
    });
  });

  it.each([
    "http://localhost:8765/auth/callback",
    "http://2130706433:8765/auth/callback",
    "http://127.0.0.1:0/auth/callback",
    "https://127.0.0.1:8765/auth/callback",
    "http://127.0.0.1:8765/other",
    "http://127.0.0.1:8765/auth/callback?next=elsewhere",
  ])("rejects callback URI %s", (redirectUri) => {
    expect(() => new AuthService({ repo, encryptionKey, redirectUri })).toThrow();
  });

  it.each(["", "not-base64", Buffer.alloc(16).toString("base64")])("rejects invalid encryption key %s", (key) => {
    expect(() => new AuthService({ repo, encryptionKey: key })).toThrow(/32 bytes/);
  });

  it.skipIf(!process.env.TEST_DATABASE_URL)("persists one-use state and synchronizes PostgreSQL refresh", async () => {
    const pool = new Pool({ connectionString: process.env.TEST_DATABASE_URL, connectionTimeoutMillis: 5000 });
    const postgres = new PostgresAuthRepository(pool);
    await postgres.setup();
    const user = "auth-test-" + randomUUID();
    const one = new AuthService({ repo: postgres, encryptionKey, fetch: provider.fetch });
    const two = new AuthService({ repo: postgres, encryptionKey, fetch: provider.fetch });
    try {
      provider.expiresIn = 30;
      provider.refreshDelay = 25;
      const { query } = await connect(one, user, "integration-browser");
      await expect(two.complete(user, "integration-browser", {
        state: query.get("state")!, code: "replayed",
      })).rejects.toMatchObject({ code: "missing_sign_in" });
      expect(await Promise.all([one.accessToken(user), two.accessToken(user)]))
        .toEqual(["refreshed-access-1", "refreshed-access-1"]);
      expect(provider.refreshCount).toBe(1);
      expect(await postgres.hostId()).toBe(await postgres.hostId());
    } finally {
      await postgres.withUser(user, async (records) => {
        await records.delete("pending");
        await records.delete("credentials");
      });
      await pool.end();
    }
  });
});
