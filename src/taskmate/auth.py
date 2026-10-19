"""Local OSS Sign in with ChatGPT. Credentials never cross the browser API boundary."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import math
import secrets
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from typing import Any, Literal, Protocol
from urllib.parse import urlencode, urlsplit

import httpx
import jwt
from cryptography.fernet import Fernet, InvalidToken
from psycopg import AsyncConnection
from psycopg.rows import tuple_row
from psycopg_pool import AsyncConnectionPool

ISSUER = "https://auth.openai.com"
DISCOVERY = ISSUER + "/.well-known/openid-configuration"
AUTHORIZE = ISSUER + "/api/accounts/authorize"
TOKEN = ISSUER + "/api/accounts/oauth/token"
JWKS = ISSUER + "/.well-known/jwks.json"
REVOKE = ISSUER + "/api/accounts/oauth/revoke"
RESOURCE = "https://api.openai.com/v1"
SCOPES = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
DYNAMIC_CLIENT = "dynamic_agent_client"
Kind = Literal["pending", "credentials"]


class AuthError(Exception):
    def __init__(self, code: str, public_message: str = "ChatGPT sign-in could not be completed. Please try signing in again."):
        self.code = code
        self.public_message = public_message
        super().__init__(public_message)


@dataclass(frozen=True, repr=False)
class PendingSignIn:
    state: str
    browser_session_id: str
    verifier: str
    nonce: str
    expires_at: float
    redirect_uri: str
    client_id: str
    subject: str | None


@dataclass(frozen=True, repr=False)
class Credential:
    client_id: str
    subject: str | None = None
    email: str | None = None
    access_token: str | None = None
    refresh_token: str | None = None
    id_token: str | None = None
    scopes: list[str] = field(default_factory=list)
    expires_at: float = 0
    earliest_refresh_at: float = 0

    def public(self) -> dict[str, Any]:
        return {
            "connected": bool(self.subject and self.access_token),
            "plan_enabled": bool(self.access_token and "chatgpt.tokens.use.direct" in self.scopes),
            "email": self.email,
            "client_id": self.client_id,
        }

    def cleared(self) -> Credential:
        return Credential(self.client_id, self.subject, self.email)


class AuthRecords(Protocol):
    async def read(self, kind: Kind) -> bytes | None: ...
    async def write(self, kind: Kind, value: bytes) -> None: ...
    async def delete(self, kind: Kind) -> None: ...


class AuthRepository(Protocol):
    def locked(self, user_id: str) -> AbstractAsyncContextManager[AuthRecords]: ...
    async def host_id(self) -> str: ...


class _PostgresRecords:
    def __init__(self, connection: AsyncConnection, user_id: str):
        self.connection = connection
        self.user_id = user_id

    @staticmethod
    def _table(kind: Kind) -> str:
        return {"pending": "tm_auth_pending", "credentials": "tm_auth_credentials"}[kind]

    async def read(self, kind: Kind) -> bytes | None:
        async with self.connection.cursor(row_factory=tuple_row) as cursor:
            await cursor.execute(
                f"SELECT ciphertext FROM {self._table(kind)} WHERE user_id = %s",
                (self.user_id,),
            )
            row = await cursor.fetchone()
        await self.connection.commit()
        return bytes(row[0]) if row else None

    async def write(self, kind: Kind, value: bytes) -> None:
        await self.connection.execute(
            f"INSERT INTO {self._table(kind)} (user_id, ciphertext) VALUES (%s, %s) "
            "ON CONFLICT (user_id) DO UPDATE SET ciphertext = EXCLUDED.ciphertext",
            (self.user_id, value),
        )
        await self.connection.commit()

    async def delete(self, kind: Kind) -> None:
        await self.connection.execute(
            f"DELETE FROM {self._table(kind)} WHERE user_id = %s", (self.user_id,)
        )
        await self.connection.commit()


class PostgresAuthRepository:
    def __init__(self, pool: AsyncConnectionPool):
        self.pool = pool

    async def setup(self) -> None:
        async with self.pool.connection() as connection:
            await connection.execute(
                "CREATE TABLE IF NOT EXISTS tm_auth_hosts "
                "(singleton boolean PRIMARY KEY CHECK (singleton), host_id text NOT NULL)"
            )
            for table in ("tm_auth_pending", "tm_auth_credentials"):
                await connection.execute(
                    f"CREATE TABLE IF NOT EXISTS {table} "
                    "(user_id text PRIMARY KEY, ciphertext bytea NOT NULL)"
                )

    async def host_id(self) -> str:
        async with self.pool.connection() as connection:
            await connection.execute(
                "INSERT INTO tm_auth_hosts (singleton, host_id) VALUES (true, %s) "
                "ON CONFLICT (singleton) DO NOTHING",
                ("urn:uuid:" + str(uuid.uuid4()),),
            )
            async with connection.cursor(row_factory=tuple_row) as cursor:
                await cursor.execute("SELECT host_id FROM tm_auth_hosts WHERE singleton")
                row = await cursor.fetchone()
                assert row is not None
                return row[0]

    @asynccontextmanager
    async def locked(self, user_id: str) -> AsyncIterator[AuthRecords]:
        async with self.pool.connection() as connection:
            key = "taskmate-auth:" + user_id
            try:
                deadline = asyncio.get_running_loop().time() + 10
                while True:
                    async with connection.cursor(row_factory=tuple_row) as cursor:
                        await cursor.execute(
                            "SELECT pg_try_advisory_lock(hashtextextended(%s, 0))", (key,)
                        )
                        row = await cursor.fetchone()
                    await connection.commit()
                    if row and row[0]:
                        break
                    if asyncio.get_running_loop().time() >= deadline:
                        raise AuthError("authentication_busy", "Another sign-in operation is running. Try again shortly.")
                    await asyncio.sleep(0.05)
                yield _PostgresRecords(connection, user_id)
            finally:
                async def release() -> None:
                    try:
                        await connection.rollback()
                        await connection.execute(
                            "SELECT pg_advisory_unlock(hashtextextended(%s, 0))", (key,)
                        )
                        await connection.commit()
                    except BaseException:
                        await connection.close()
                        raise

                cleanup = asyncio.create_task(release())
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    await cleanup
                    raise


class AuthService:
    def __init__(
        self,
        repository: AuthRepository,
        *,
        encryption_key: str | bytes,
        redirect_uri: str = "http://127.0.0.1:8765/auth/callback",
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Callable[[], float] = time.time,
    ):
        parsed = urlsplit(redirect_uri)
        if (
            parsed.scheme != "http"
            or parsed.hostname != "127.0.0.1"
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port is None
            or parsed.path != "/auth/callback"
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Use http://127.0.0.1:<port>/auth/callback.")
        self.repository = repository
        self.cipher = Fernet(encryption_key)
        self.redirect_uri = redirect_uri
        self.clock = clock
        self.http = httpx.AsyncClient(
            transport=transport,
            timeout=httpx.Timeout(10, connect=5),
            follow_redirects=False,
            trust_env=False,
        )
        self._discovered = False
        self._keys: dict[str, Any] = {}
        self._keys_until = 0.0

    async def aclose(self) -> None:
        await self.http.aclose()

    def _seal(self, user_id: str, kind: Kind, value: PendingSignIn | Credential) -> bytes:
        raw = {"user_id": user_id, "kind": kind, "value": asdict(value)}
        return self.cipher.encrypt(json.dumps(raw, separators=(",", ":")).encode())

    def _open(self, user_id: str, kind: Kind, value: bytes) -> PendingSignIn | Credential:
        try:
            raw = json.loads(self.cipher.decrypt(value))
            if raw["user_id"] != user_id or raw["kind"] != kind:
                raise ValueError
            cls = PendingSignIn if kind == "pending" else Credential
            return cls(**raw["value"])
        except (InvalidToken, ValueError, TypeError, KeyError):
            raise AuthError("invalid_stored_credentials") from None

    async def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        allowed = {("GET", DISCOVERY), ("GET", JWKS), ("POST", TOKEN), ("POST", REVOKE)}
        if (method, url) not in allowed:
            raise AuthError("endpoint_not_allowed")
        try:
            async with asyncio.timeout(15):
                return await self.http.request(method, url, **kwargs)
        except (httpx.HTTPError, TimeoutError):
            raise AuthError("provider_unavailable", "ChatGPT could not be reached. Try again shortly.") from None

    @staticmethod
    def _json(response: httpx.Response) -> dict[str, Any]:
        try:
            value = response.json()
            if not isinstance(value, dict):
                raise ValueError
            return value
        except (ValueError, UnicodeDecodeError):
            raise AuthError("invalid_provider_response") from None

    async def _discover(self) -> None:
        if self._discovered:
            return
        response = await self._request("GET", DISCOVERY)
        if response.status_code != 200:
            raise AuthError("discovery_failed")
        metadata = self._json(response)
        required = {
            "issuer": ISSUER,
            "authorization_endpoint": AUTHORIZE,
            "token_endpoint": TOKEN,
            "jwks_uri": JWKS,
            "revocation_endpoint": REVOKE,
        }
        if any(metadata.get(key) != value for key, value in required.items()):
            raise AuthError("unexpected_provider_endpoints")
        algorithms = metadata.get("id_token_signing_alg_values_supported")
        if not isinstance(algorithms, list) or "RS256" not in algorithms:
            raise AuthError("unsupported_signing_algorithm")
        self._discovered = True

    async def _identity(self, token: str, client_id: str, nonce: str | None) -> dict[str, Any]:
        try:
            header = jwt.get_unverified_header(token)
            kid = header.get("kid")
            if header.get("alg") != "RS256" or not isinstance(kid, str) or not kid:
                raise ValueError
            if self.clock() >= self._keys_until or kid not in self._keys:
                response = await self._request("GET", JWKS)
                if response.status_code != 200:
                    raise AuthError("jwks_failed")
                keys = self._json(response).get("keys")
                if not isinstance(keys, list):
                    raise ValueError
                self._keys = {
                    key["kid"]: key for key in keys
                    if isinstance(key, dict) and isinstance(key.get("kid"), str)
                }
                self._keys_until = self.clock() + 300
            key = self._keys[kid]
            if (
                key.get("kty") != "RSA"
                or key.get("use", "sig") != "sig"
                or key.get("alg", "RS256") != "RS256"
                or "verify" not in key.get("key_ops", ["verify"])
            ):
                raise ValueError
            claims = jwt.decode(
                token,
                jwt.PyJWK.from_dict(key, algorithm="RS256").key,
                algorithms=["RS256"],
                issuer=ISSUER,
                audience=client_id,
                leeway=5,
                options={"require": ["iss", "aud", "sub", "exp", "iat"]},
            )
            if not isinstance(claims["sub"], str) or not claims["sub"]:
                raise ValueError
            if type(claims["exp"]) is not int or type(claims["iat"]) is not int:
                raise ValueError
            audience = claims["aud"]
            if (
                (isinstance(audience, list) and len(audience) > 1)
                or "azp" in claims
            ) and claims.get("azp") != client_id:
                raise ValueError
            if nonce is not None:
                returned = claims.get("nonce")
                if not isinstance(returned, str) or not secrets.compare_digest(returned.encode(), nonce.encode()):
                    raise ValueError
            return claims
        except AuthError:
            raise
        except (jwt.PyJWTError, ValueError, TypeError, KeyError):
            raise AuthError("invalid_identity_token") from None

    async def _exchange(self, form: dict[str, str]) -> dict[str, Any]:
        response = await self._request("POST", TOKEN, data=form)
        body = self._json(response)
        if response.status_code != 200:
            code = body.get("error")
            if not isinstance(code, str) or code not in {
                "invalid_grant", "invalid_client", "invalid_scope", "slow_down", "temporarily_unavailable"
            }:
                code = "token_exchange_failed"
            raise AuthError(code)
        return body

    def _tokens(self, body: dict[str, Any], previous: Credential, id_token: str) -> Credential:
        token_type = body.get("token_type")
        if not isinstance(token_type, str) or token_type.lower() != "bearer":
            raise AuthError("invalid_token_response")
        for key in ("access_token", "refresh_token"):
            if not isinstance(body.get(key), str) or not body[key]:
                raise AuthError("invalid_token_response")
        lifetime = body.get("expires_in")
        if type(lifetime) not in (int, float) or not math.isfinite(lifetime) or not 0 < lifetime <= 86400:
            raise AuthError("invalid_token_response")
        scope = body.get("scope")
        if scope is None:
            scopes = previous.scopes
        elif isinstance(scope, str):
            scopes = sorted(set(scope.split()))
        else:
            raise AuthError("invalid_token_response")
        earliest = body.get("earliest_refresh_at", 0)
        if isinstance(earliest, str):
            try:
                timestamp = datetime.fromisoformat(earliest.replace("Z", "+00:00"))
                if timestamp.tzinfo is None:
                    raise ValueError
                earliest = timestamp.timestamp()
            except ValueError:
                raise AuthError("invalid_token_response") from None
        if type(earliest) not in (int, float) or earliest < 0 or not math.isfinite(earliest):
            raise AuthError("invalid_token_response")
        return replace(
            previous,
            access_token=body["access_token"],
            refresh_token=body["refresh_token"],
            id_token=id_token,
            scopes=scopes,
            expires_at=self.clock() + lifetime,
            earliest_refresh_at=earliest,
        )

    async def begin(self, user_id: str, browser_session_id: str) -> str:
        await self._discover()
        host_id = await self.repository.host_id()
        async with self.repository.locked(user_id) as records:
            saved = await records.read("credentials")
            previous = self._open(user_id, "credentials", saved) if saved else None
            pending = PendingSignIn(
                state=secrets.token_urlsafe(32),
                browser_session_id=browser_session_id,
                verifier=secrets.token_urlsafe(64),
                nonce=secrets.token_urlsafe(32),
                expires_at=self.clock() + 600,
                redirect_uri=self.redirect_uri,
                client_id=previous.client_id if previous else DYNAMIC_CLIENT,
                subject=previous.subject if previous else None,
            )
            await records.write("pending", self._seal(user_id, "pending", pending))
            challenge = base64.urlsafe_b64encode(
                hashlib.sha256(pending.verifier.encode()).digest()
            ).rstrip(b"=").decode()
            query = {
                "client_id": pending.client_id,
                "ext_agent_host_id": host_id,
                "response_type": "code",
                "redirect_uri": pending.redirect_uri,
                "scope": SCOPES,
                "resource": RESOURCE,
                "state": pending.state,
                "nonce": pending.nonce,
                "code_challenge_method": "S256",
                "code_challenge": challenge,
            }
            if previous:
                if previous.email:
                    query["login_hint"] = previous.email
            else:
                query["agent_name_hint"] = "taskmate-lab"
            return AUTHORIZE + "?" + urlencode(query)

    async def complete(
        self,
        user_id: str,
        browser_session_id: str,
        *,
        state: str,
        code: str | None = None,
        client_id: str | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        async with self.repository.locked(user_id) as records:
            saved = await records.read("pending")
            if not saved:
                raise AuthError("missing_sign_in")
            pending = self._open(user_id, "pending", saved)
            await records.delete("pending")
            if (
                pending.expires_at <= self.clock()
                or not secrets.compare_digest(pending.state.encode(), state.encode())
                or not secrets.compare_digest(pending.browser_session_id.encode(), browser_session_id.encode())
            ):
                raise AuthError("invalid_sign_in_state")
            if error is not None:
                raise AuthError("authorization_denied", "ChatGPT sign-in was cancelled or denied.")
            if not code or len(code) > 8192:
                raise AuthError("missing_authorization_code")
            if pending.client_id == DYNAMIC_CLIENT:
                if not client_id or client_id == DYNAMIC_CLIENT or len(client_id) > 512:
                    raise AuthError("incomplete_registration")
            elif client_id is not None and client_id != pending.client_id:
                raise AuthError("registration_mismatch")
            issued_id = client_id or pending.client_id
            await self._discover()
            # A committed mapping lets an expired first code retry without creating another client.
            if pending.client_id == DYNAMIC_CLIENT:
                await records.write(
                    "credentials", self._seal(user_id, "credentials", Credential(issued_id))
                )
            body = await self._exchange({
                "grant_type": "authorization_code",
                "client_id": issued_id,
                "code": code,
                "code_verifier": pending.verifier,
                "redirect_uri": pending.redirect_uri,
                "resource": RESOURCE,
            })
            id_token = body.get("id_token")
            if not isinstance(id_token, str) or not id_token:
                raise AuthError("missing_identity_token")
            claims = await self._identity(id_token, issued_id, pending.nonce)
            if pending.subject is not None and claims["sub"] != pending.subject:
                raise AuthError("account_mismatch")
            email = claims.get("email")
            identity = Credential(
                client_id=issued_id,
                subject=claims["sub"],
                email=email if isinstance(email, str) else None,
            )
            credential = self._tokens(body, identity, id_token)
            await records.write("credentials", self._seal(user_id, "credentials", credential))
            return credential.public()

    async def status(self, user_id: str) -> dict[str, Any]:
        async with self.repository.locked(user_id) as records:
            saved = await records.read("credentials")
            if not saved:
                return {"connected": False, "plan_enabled": False, "email": None, "client_id": None}
            return self._open(user_id, "credentials", saved).public()

    async def access_token(self, user_id: str) -> str:
        async with self.repository.locked(user_id) as records:
            saved = await records.read("credentials")
            if not saved:
                raise AuthError("sign_in_required", "Sign in with ChatGPT first.")
            credential = self._open(user_id, "credentials", saved)
            if not credential.access_token or not credential.subject:
                raise AuthError("sign_in_required", "Sign in with ChatGPT first.")
            if "chatgpt.tokens.use.direct" not in credential.scopes:
                raise AuthError("plan_permission_required", "ChatGPT plan access has not been granted.")
            now = self.clock()
            if credential.expires_at - now > 60:
                return credential.access_token
            if now < credential.earliest_refresh_at:
                if now < credential.expires_at:
                    return credential.access_token
                raise AuthError("refresh_not_due", "ChatGPT renewal is not available yet. Try again shortly.")
            if not credential.refresh_token:
                raise AuthError("sign_in_required", "Sign in with ChatGPT again.")
            await self._discover()
            try:
                body = await self._exchange({
                    "grant_type": "refresh_token",
                    "client_id": credential.client_id,
                    "refresh_token": credential.refresh_token,
                    "resource": RESOURCE,
                })
            except AuthError as exc:
                if exc.code in {"invalid_grant", "invalid_client"}:
                    await records.write(
                        "credentials", self._seal(user_id, "credentials", credential.cleared())
                    )
                raise
            new_id_token = body.get("id_token", credential.id_token)
            if "id_token" in body:
                if not isinstance(new_id_token, str):
                    raise AuthError("invalid_identity_token")
                claims = await self._identity(new_id_token, credential.client_id, None)
                if claims["sub"] != credential.subject:
                    raise AuthError("account_mismatch")
            updated = self._tokens(body, credential, new_id_token)
            await records.write("credentials", self._seal(user_id, "credentials", updated))
            if "chatgpt.tokens.use.direct" not in updated.scopes:
                raise AuthError("plan_permission_required", "ChatGPT plan access has not been granted.")
            return updated.access_token

    async def disconnect(self, user_id: str) -> bool:
        async with self.repository.locked(user_id) as records:
            await records.delete("pending")
            saved = await records.read("credentials")
            if not saved:
                return True
            credential = self._open(user_id, "credentials", saved)
            confirmed = not bool(credential.refresh_token)
            try:
                if credential.refresh_token:
                    await self._discover()
                    for attempt in range(3):
                        try:
                            response = await self._request("POST", REVOKE, data={
                                "token": credential.refresh_token,
                                "token_type_hint": "refresh_token",
                                "client_id": credential.client_id,
                            })
                            confirmed = response.status_code == 200
                            retry = response.status_code >= 500
                        except AuthError:
                            retry = True
                        if confirmed or not retry or attempt == 2:
                            break
                        await asyncio.sleep(0.25 * (2 ** attempt))
            except AuthError:
                confirmed = False
            finally:
                await records.write(
                    "credentials", self._seal(user_id, "credentials", credential.cleared())
                )
            return confirmed
