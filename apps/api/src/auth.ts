import {
  createCipheriv,
  createDecipheriv,
  createHash,
  randomBytes,
  randomUUID,
  timingSafeEqual,
} from "node:crypto";
import { setTimeout as delay } from "node:timers/promises";
import { createLocalJWKSet, decodeProtectedHeader, jwtVerify, type JWK, type JWTPayload } from "jose";
import type { Pool, PoolClient } from "pg";

export const ISSUER = "https://auth.openai.com";
export const DISCOVERY = ISSUER + "/.well-known/openid-configuration";
export const AUTHORIZE = ISSUER + "/api/accounts/authorize";
export const TOKEN = ISSUER + "/api/accounts/oauth/token";
export const JWKS = ISSUER + "/.well-known/jwks.json";
export const REVOKE = ISSUER + "/api/accounts/oauth/revoke";
const RESOURCE = "https://api.openai.com/v1";
const SCOPES = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct";
const DYNAMIC_CLIENT = "dynamic_agent_client";

export class AuthError extends Error {
  constructor(
    readonly code: string,
    readonly publicMessage = "ChatGPT sign-in could not be completed. Please sign in again.",
  ) {
    super(publicMessage);
    this.name = "AuthError";
  }
}

export type AuthStatus = {
  connected: boolean;
  plan_enabled: boolean;
  email: string | null;
  client_id: string | null;
};

type PendingSignIn = {
  state: string;
  browserSessionId: string;
  verifier: string;
  nonce: string;
  expiresAt: number;
  redirectUri: string;
  clientId: string;
  subject: string | null;
};

type Tokens = {
  accessToken: string;
  refreshToken: string;
  idToken: string;
  scopes: string[];
  expiresAt: number;
  earliestRefreshAt: number;
};

type Registration = {
  clientId: string;
  subject: string | null;
  email: string | null;
  tokens: Tokens | null;
};

export type AuthRecordKind = "pending" | "credentials";

export interface AuthRecords {
  read(kind: AuthRecordKind): Promise<Buffer | null>;
  write(kind: AuthRecordKind, ciphertext: Buffer): Promise<void>;
  delete(kind: AuthRecordKind): Promise<void>;
}

export interface AuthRepository {
  hostId(): Promise<string>;
  withUser<T>(userId: string, work: (records: AuthRecords) => Promise<T>): Promise<T>;
}

const TABLES = {
  pending: "tm_auth_pending",
  credentials: "tm_auth_credentials",
} as const;

class PostgresRecords implements AuthRecords {
  constructor(private readonly client: PoolClient, private readonly userId: string) {}

  async read(kind: AuthRecordKind): Promise<Buffer | null> {
    const result = await this.client.query<{ ciphertext: Buffer }>(
      "SELECT ciphertext FROM " + TABLES[kind] + " WHERE user_id = $1",
      [this.userId],
    );
    return result.rows[0]?.ciphertext ?? null;
  }

  async write(kind: AuthRecordKind, ciphertext: Buffer): Promise<void> {
    await this.client.query(
      "INSERT INTO " + TABLES[kind] + " (user_id, ciphertext) VALUES ($1, $2) " +
      "ON CONFLICT (user_id) DO UPDATE SET ciphertext = EXCLUDED.ciphertext",
      [this.userId, ciphertext],
    );
  }

  async delete(kind: AuthRecordKind): Promise<void> {
    await this.client.query("DELETE FROM " + TABLES[kind] + " WHERE user_id = $1", [this.userId]);
  }
}

export class PostgresAuthRepository implements AuthRepository {
  readonly #pool: Pool;

  constructor(pool: Pool) { this.#pool = pool; }

  async setup(): Promise<void> {
    await this.#pool.query(
      "CREATE TABLE IF NOT EXISTS tm_auth_hosts " +
      "(singleton boolean PRIMARY KEY CHECK (singleton), host_id text NOT NULL)",
    );
    for (const table of Object.values(TABLES)) {
      await this.#pool.query(
        "CREATE TABLE IF NOT EXISTS " + table + " (user_id text PRIMARY KEY, ciphertext bytea NOT NULL)",
      );
    }
  }

  async hostId(): Promise<string> {
    const result = await this.#pool.query<{ host_id: string }>(
      "INSERT INTO tm_auth_hosts (singleton, host_id) VALUES (true, $1) " +
      "ON CONFLICT (singleton) DO UPDATE SET host_id = tm_auth_hosts.host_id RETURNING host_id",
      ["urn:uuid:" + randomUUID()],
    );
    const row = result.rows[0];
    if (!row) throw new AuthError("host_id_unavailable");
    return row.host_id;
  }

  async withUser<T>(userId: string, work: (records: AuthRecords) => Promise<T>): Promise<T> {
    const client = await this.#pool.connect();
    const lockKey = "taskmate-auth:" + userId;
    let destroy = false;
    try {
      const deadline = performance.now() + 10_000;
      while (true) {
        const result = await client.query<{ acquired: boolean }>(
          "SELECT pg_try_advisory_lock(hashtextextended($1, 0)) AS acquired", [lockKey],
        );
        if (result.rows[0]?.acquired) break;
        if (performance.now() >= deadline) {
          throw new AuthError("authentication_busy", "Another sign-in operation is running. Try again shortly.");
        }
        await delay(25);
      }
      return await work(new PostgresRecords(client, userId));
    } finally {
      try {
        await client.query("SELECT pg_advisory_unlock(hashtextextended($1, 0))", [lockKey]);
      } catch {
        destroy = true;
      }
      client.release(destroy);
    }
  }
}

type Callback = {
  state: string;
  code?: string;
  clientId?: string;
  error?: string;
};

type AuthConfig = {
  repo: AuthRepository;
  encryptionKey: string;
  redirectUri?: string;
  fetch?: typeof globalThis.fetch;
  now?: () => number;
};

function object(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new AuthError("invalid_provider_response");
  }
  return value as Record<string, unknown>;
}

function requiredString(value: unknown): string {
  if (typeof value !== "string" || !value) throw new AuthError("invalid_provider_response");
  return value;
}

function sameSecret(left: string, right: string): boolean {
  const a = Buffer.from(left);
  const b = Buffer.from(right);
  return a.length === b.length && timingSafeEqual(a, b);
}

function statusOf(registration: Registration | null): AuthStatus {
  return {
    connected: Boolean(registration?.subject && registration.tokens),
    plan_enabled: Boolean(registration?.tokens?.scopes.includes("chatgpt.tokens.use.direct")),
    email: registration?.email ?? null,
    client_id: registration?.clientId ?? null,
  };
}

export class AuthService {
  readonly #repo: AuthRepository;
  readonly #key: Buffer;
  private readonly redirectUri: string;
  readonly #fetcher: typeof globalThis.fetch;
  private readonly now: () => number;
  private discovered = false;
  private keySet: ReturnType<typeof createLocalJWKSet> | null = null;
  private keyIds = new Set<string>();
  private keysUntil = 0;

  constructor(config: AuthConfig) {
    this.#repo = config.repo;
    this.redirectUri = config.redirectUri ?? "http://127.0.0.1:8765/auth/callback";
    const uri = new URL(this.redirectUri);
    if (
      uri.protocol !== "http:" || uri.hostname !== "127.0.0.1" || Number(uri.port) < 1 ||
      uri.username || uri.password || uri.pathname !== "/auth/callback" || uri.search || uri.hash ||
      this.redirectUri !== uri.href
    ) throw new Error("Use http://127.0.0.1:<port>/auth/callback.");
    this.#key = Buffer.from(config.encryptionKey, "base64");
    if (this.#key.length !== 32 || this.#key.toString("base64") !== config.encryptionKey) {
      throw new Error("TASKMATE_ENCRYPTION_KEY must be canonical base64 for exactly 32 bytes.");
    }
    this.#fetcher = config.fetch ?? globalThis.fetch;
    this.now = config.now ?? (() => Date.now() / 1000);
  }

  private seal(userId: string, kind: AuthRecordKind, value: PendingSignIn | Registration): Buffer {
    const iv = randomBytes(12);
    const cipher = createCipheriv("aes-256-gcm", this.#key, iv);
    cipher.setAAD(Buffer.from(JSON.stringify([userId, kind])));
    const encrypted = Buffer.concat([cipher.update(JSON.stringify(value), "utf8"), cipher.final()]);
    return Buffer.concat([Buffer.from([1]), iv, cipher.getAuthTag(), encrypted]);
  }

  private open<T extends PendingSignIn | Registration>(userId: string, kind: AuthRecordKind, data: Buffer): T {
    try {
      if (data.length < 30 || data[0] !== 1) throw new Error();
      const decipher = createDecipheriv("aes-256-gcm", this.#key, data.subarray(1, 13));
      decipher.setAAD(Buffer.from(JSON.stringify([userId, kind])));
      decipher.setAuthTag(data.subarray(13, 29));
      const plaintext = Buffer.concat([decipher.update(data.subarray(29)), decipher.final()]);
      return JSON.parse(plaintext.toString("utf8")) as T;
    } catch {
      throw new AuthError("invalid_stored_credentials");
    }
  }

  private async request(method: "GET" | "POST", url: string, form?: Record<string, string>) {
    const allowed = method === "GET" ? [DISCOVERY, JWKS] : [TOKEN, REVOKE];
    if (!allowed.includes(url)) throw new AuthError("endpoint_not_allowed");
    try {
      const response = await this.#fetcher(url, {
        method,
        redirect: "error",
        signal: AbortSignal.timeout(10_000),
        ...(form ? {
          headers: { "Content-Type": "application/x-www-form-urlencoded" },
          body: new URLSearchParams(form),
        } : {}),
      });
      const body = await response.text();
      if (body.length > 1_048_576) throw new AuthError("invalid_provider_response");
      return { status: response.status, body };
    } catch (error) {
      if (error instanceof AuthError) throw error;
      throw new AuthError("provider_unavailable", "ChatGPT could not be reached. Try again shortly.");
    }
  }

  private json(body: string): Record<string, unknown> {
    try {
      return object(JSON.parse(body));
    } catch {
      throw new AuthError("invalid_provider_response");
    }
  }

  private async discover(): Promise<void> {
    if (this.discovered) return;
    const response = await this.request("GET", DISCOVERY);
    if (response.status !== 200) throw new AuthError("discovery_failed");
    const metadata = this.json(response.body);
    const expected = {
      issuer: ISSUER, authorization_endpoint: AUTHORIZE, token_endpoint: TOKEN,
      jwks_uri: JWKS, revocation_endpoint: REVOKE,
    };
    if (Object.entries(expected).some(([key, value]) => metadata[key] !== value)) {
      throw new AuthError("unexpected_provider_endpoints");
    }
    if (
      !Array.isArray(metadata.id_token_signing_alg_values_supported) ||
      !metadata.id_token_signing_alg_values_supported.includes("RS256")
    ) throw new AuthError("unsupported_signing_algorithm");
    this.discovered = true;
  }

  private async identity(token: string, clientId: string, nonce?: string): Promise<JWTPayload> {
    try {
      const header = decodeProtectedHeader(token);
      if (header.alg !== "RS256" || typeof header.kid !== "string" || !header.kid) throw new Error();
      if (!this.keySet || this.now() >= this.keysUntil || !this.keyIds.has(header.kid)) {
        const response = await this.request("GET", JWKS);
        if (response.status !== 200) throw new AuthError("jwks_failed");
        const keys = this.json(response.body).keys;
        if (!Array.isArray(keys)) throw new Error();
        const publicKeys: JWK[] = keys.map((value: unknown) => {
          const key = object(value);
          if (typeof key.kty !== "string") throw new Error();
          return key as JWK;
        });
        this.keySet = createLocalJWKSet({ keys: publicKeys });
        this.keyIds = new Set(publicKeys.flatMap((key) => typeof key.kid === "string" ? [key.kid] : []));
        this.keysUntil = this.now() + 300;
      }
      const { payload } = await jwtVerify(token, this.keySet, {
        issuer: ISSUER,
        audience: clientId,
        algorithms: ["RS256"],
        requiredClaims: ["iss", "aud", "sub", "exp", "iat"],
        clockTolerance: 5,
        currentDate: new Date(this.now() * 1000),
      });
      if (
        !payload.sub || typeof payload.sub !== "string" ||
        typeof payload.exp !== "number" || !Number.isInteger(payload.exp) ||
        typeof payload.iat !== "number" || !Number.isInteger(payload.iat) ||
        payload.iat > this.now() + 5
      ) throw new Error();
      if (
        ((Array.isArray(payload.aud) && payload.aud.length > 1) || payload.azp !== undefined) &&
        payload.azp !== clientId
      ) throw new Error();
      if (nonce !== undefined && (typeof payload.nonce !== "string" || !sameSecret(payload.nonce, nonce))) {
        throw new Error();
      }
      return payload;
    } catch (error) {
      if (error instanceof AuthError) throw error;
      throw new AuthError("invalid_identity_token");
    }
  }

  private async exchange(form: Record<string, string>): Promise<Record<string, unknown>> {
    const response = await this.request("POST", TOKEN, form);
    const body = this.json(response.body);
    if (response.status !== 200) {
      const safeCodes = ["invalid_grant", "invalid_client", "invalid_scope", "slow_down", "temporarily_unavailable"];
      const code = typeof body.error === "string" && safeCodes.includes(body.error)
        ? body.error : "token_exchange_failed";
      throw new AuthError(code);
    }
    return body;
  }

  private tokens(body: Record<string, unknown>, idToken: string, previous?: Tokens): Tokens {
    if (typeof body.token_type !== "string" || body.token_type.toLowerCase() !== "bearer") {
      throw new AuthError("invalid_token_response");
    }
    const lifetime = body.expires_in;
    if (typeof lifetime !== "number" || !Number.isFinite(lifetime) || lifetime <= 0 || lifetime > 86400) {
      throw new AuthError("invalid_token_response");
    }
    let scopes: string[];
    if (body.scope === undefined) scopes = previous?.scopes ?? [];
    else if (typeof body.scope === "string") scopes = [...new Set(body.scope.split(/\s+/).filter(Boolean))];
    else throw new AuthError("invalid_token_response");
    const rawEarliest = body.earliest_refresh_at ?? 0;
    const earliest = typeof rawEarliest === "string" && /(?:Z|[+-]\d\d:\d\d)$/.test(rawEarliest)
      ? Date.parse(rawEarliest) / 1000 : rawEarliest;
    if (typeof earliest !== "number" || !Number.isFinite(earliest) || earliest < 0) {
      throw new AuthError("invalid_token_response");
    }
    return {
      accessToken: requiredString(body.access_token),
      refreshToken: requiredString(body.refresh_token),
      idToken,
      scopes,
      expiresAt: this.now() + lifetime,
      earliestRefreshAt: earliest,
    };
  }

  async begin(userId: string, browserSessionId: string): Promise<string> {
    await this.discover();
    const hostId = await this.#repo.hostId();
    return this.#repo.withUser(userId, async (records) => {
      const saved = await records.read("credentials");
      const previous = saved ? this.open<Registration>(userId, "credentials", saved) : null;
      const pending: PendingSignIn = {
        state: randomBytes(32).toString("base64url"),
        browserSessionId,
        verifier: randomBytes(64).toString("base64url"),
        nonce: randomBytes(32).toString("base64url"),
        expiresAt: this.now() + 600,
        redirectUri: this.redirectUri,
        clientId: previous?.clientId ?? DYNAMIC_CLIENT,
        subject: previous?.subject ?? null,
      };
      await records.write("pending", this.seal(userId, "pending", pending));
      const query = new URLSearchParams({
        client_id: pending.clientId,
        ext_agent_host_id: hostId,
        response_type: "code",
        redirect_uri: pending.redirectUri,
        scope: SCOPES,
        resource: RESOURCE,
        state: pending.state,
        nonce: pending.nonce,
        code_challenge_method: "S256",
        code_challenge: createHash("sha256").update(pending.verifier).digest("base64url"),
      });
      if (!previous) query.set("agent_name_hint", "taskmate-lab");
      else if (previous.email) query.set("login_hint", previous.email);
      return AUTHORIZE + "?" + query.toString();
    });
  }

  async complete(userId: string, browserSessionId: string, callback: Callback): Promise<AuthStatus> {
    if (!callback || typeof callback.state !== "string" || !callback.state || callback.state.length > 512) {
      throw new AuthError("invalid_sign_in_state");
    }
    return this.#repo.withUser(userId, async (records) => {
      const saved = await records.read("pending");
      if (!saved) throw new AuthError("missing_sign_in");
      const pending = this.open<PendingSignIn>(userId, "pending", saved);
      await records.delete("pending");
      if (
        this.now() >= pending.expiresAt ||
        !sameSecret(pending.state, callback.state) ||
        !sameSecret(pending.browserSessionId, browserSessionId)
      ) throw new AuthError("invalid_sign_in_state");
      if (callback.error !== undefined) throw new AuthError("authorization_denied", "ChatGPT sign-in was cancelled or denied.");
      if (!callback.code || callback.code.length > 8192) throw new AuthError("missing_authorization_code");
      if (pending.clientId === DYNAMIC_CLIENT) {
        if (!callback.clientId || callback.clientId === DYNAMIC_CLIENT || callback.clientId.length > 512) {
          throw new AuthError("incomplete_registration");
        }
      } else if (callback.clientId !== undefined && callback.clientId !== pending.clientId) {
        throw new AuthError("registration_mismatch");
      }
      const issuedId = callback.clientId ?? pending.clientId;
      await this.discover();
      if (pending.clientId === DYNAMIC_CLIENT) {
        await records.write("credentials", this.seal(userId, "credentials", {
          clientId: issuedId, subject: null, email: null, tokens: null,
        }));
      }
      const body = await this.exchange({
        grant_type: "authorization_code",
        client_id: issuedId,
        code: callback.code,
        code_verifier: pending.verifier,
        redirect_uri: pending.redirectUri,
        resource: RESOURCE,
      });
      const idToken = requiredString(body.id_token);
      const claims = await this.identity(idToken, issuedId, pending.nonce);
      if (pending.subject !== null && claims.sub !== pending.subject) throw new AuthError("account_mismatch");
      const registration: Registration = {
        clientId: issuedId,
        subject: requiredString(claims.sub),
        email: typeof claims.email === "string" ? claims.email : null,
        tokens: this.tokens(body, idToken),
      };
      await records.write("credentials", this.seal(userId, "credentials", registration));
      return statusOf(registration);
    });
  }

  async status(userId: string): Promise<AuthStatus> {
    return this.#repo.withUser(userId, async (records) => {
      const saved = await records.read("credentials");
      return statusOf(saved ? this.open<Registration>(userId, "credentials", saved) : null);
    });
  }

  async accessToken(userId: string): Promise<string> {
    return this.#repo.withUser(userId, async (records) => {
      const saved = await records.read("credentials");
      const registration = saved ? this.open<Registration>(userId, "credentials", saved) : null;
      if (!registration?.subject || !registration.tokens) {
        throw new AuthError("sign_in_required", "Sign in with ChatGPT first.");
      }
      const previous = registration.tokens;
      if (!previous.scopes.includes("chatgpt.tokens.use.direct")) {
        throw new AuthError("plan_permission_required", "ChatGPT plan access has not been granted.");
      }
      const now = this.now();
      if (previous.expiresAt - now > 60) return previous.accessToken;
      if (now < previous.earliestRefreshAt) {
        if (now < previous.expiresAt) return previous.accessToken;
        throw new AuthError("refresh_not_due", "ChatGPT renewal is not available yet. Try again shortly.");
      }
      await this.discover();
      let body: Record<string, unknown>;
      try {
        body = await this.exchange({
          grant_type: "refresh_token",
          client_id: registration.clientId,
          refresh_token: previous.refreshToken,
          resource: RESOURCE,
        });
      } catch (error) {
        if (error instanceof AuthError && ["invalid_grant", "invalid_client"].includes(error.code)) {
          await records.write("credentials", this.seal(userId, "credentials", { ...registration, tokens: null }));
        }
        throw error;
      }
      let idToken = previous.idToken;
      if (body.id_token !== undefined) {
        idToken = requiredString(body.id_token);
        const claims = await this.identity(idToken, registration.clientId);
        if (claims.sub !== registration.subject) throw new AuthError("account_mismatch");
      }
      const updated = this.tokens(body, idToken, previous);
      await records.write("credentials", this.seal(userId, "credentials", { ...registration, tokens: updated }));
      if (!updated.scopes.includes("chatgpt.tokens.use.direct")) {
        throw new AuthError("plan_permission_required", "ChatGPT plan access has not been granted.");
      }
      return updated.accessToken;
    });
  }

  async disconnect(userId: string): Promise<boolean> {
    return this.#repo.withUser(userId, async (records) => {
      await records.delete("pending");
      const saved = await records.read("credentials");
      if (!saved) return true;
      const registration = this.open<Registration>(userId, "credentials", saved);
      let confirmed = !registration.tokens;
      try {
        if (registration.tokens) {
          await this.discover();
          for (let attempt = 0; attempt < 3; attempt++) {
            let retry = true;
            try {
              const response = await this.request("POST", REVOKE, {
                token: registration.tokens.refreshToken,
                token_type_hint: "refresh_token",
                client_id: registration.clientId,
              });
              confirmed = response.status === 200;
              retry = response.status >= 500;
            } catch {
              confirmed = false;
            }
            if (confirmed || !retry || attempt === 2) break;
            await delay(250 * 2 ** attempt);
          }
        }
      } catch {
        confirmed = false;
      } finally {
        await records.write("credentials", this.seal(userId, "credentials", { ...registration, tokens: null }));
      }
      return confirmed;
    });
  }
}
