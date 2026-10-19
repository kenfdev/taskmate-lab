import asyncio
import base64
import hashlib
import json
import os
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from urllib.parse import parse_qs, urlsplit

import httpx
import jwt
import pytest
import pytest_asyncio
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.asymmetric import rsa
from psycopg_pool import AsyncConnectionPool

from taskmate.auth import (
    AUTHORIZE,
    DISCOVERY,
    ISSUER,
    JWKS,
    REVOKE,
    TOKEN,
    AuthError,
    AuthService,
    PostgresAuthRepository,
)

pytestmark = pytest.mark.asyncio


class MemoryRecords:
    def __init__(self, repository, user_id):
        self.repository = repository
        self.user_id = user_id

    async def read(self, kind):
        return self.repository.values.get((self.user_id, kind))

    async def write(self, kind, value):
        self.repository.values[self.user_id, kind] = value

    async def delete(self, kind):
        self.repository.values.pop((self.user_id, kind), None)


class MemoryRepository:
    def __init__(self):
        self.values = {}
        self.locks = defaultdict(asyncio.Lock)

    @asynccontextmanager
    async def locked(self, user_id):
        async with self.locks[user_id]:
            yield MemoryRecords(self, user_id)

    async def host_id(self):
        return "urn:uuid:00000000-0000-4000-8000-000000000001"


class Provider:
    def __init__(self):
        self.private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.other_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.calls = []
        self.forms = []
        self.claim_overrides = {}
        self.token_overrides = {}
        self.nonce = ""
        self.client_id = "oaiapp_test_registration"
        self.subject = "test_subject"
        self.expires_in = 3600
        self.refresh_count = 0
        self.revoke_status = 200
        self.exchange_error = None
        self.discovery_override = {}
        self.sign_with_wrong_key = False
        self.kid = "test-key"
        self.refresh_delay = 0

    async def handle(self, request):
        self.calls.append(str(request.url))
        if str(request.url) == DISCOVERY:
            return httpx.Response(200, json={
                "issuer": ISSUER,
                "authorization_endpoint": AUTHORIZE,
                "token_endpoint": TOKEN,
                "jwks_uri": JWKS,
                "revocation_endpoint": REVOKE,
                "id_token_signing_alg_values_supported": ["RS256"],
                **self.discovery_override,
            })
        if str(request.url) == JWKS:
            key = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(self.private_key.public_key()))
            return httpx.Response(200, json={"keys": [{**key, "kid": self.kid, "use": "sig"}]})
        if str(request.url) == REVOKE:
            return httpx.Response(self.revoke_status)
        assert str(request.url) == TOKEN
        form = {key: value[0] for key, value in parse_qs(request.content.decode()).items()}
        self.forms.append(form)
        if self.exchange_error:
            return httpx.Response(400, json={"error": self.exchange_error, "detail": "never expose this"})
        if form["grant_type"] == "refresh_token":
            self.refresh_count += 1
            await asyncio.sleep(self.refresh_delay)
            return httpx.Response(200, json={
                "access_token": "refreshed-access",
                "refresh_token": "replacement-refresh",
                "token_type": "Bearer",
                "expires_in": 3600,
                **self.token_overrides,
            })
        now = int(time.time())
        claims = {
            "iss": ISSUER,
            "aud": self.client_id,
            "sub": self.subject,
            "iat": now,
            "exp": now + 3600,
            "nonce": self.nonce,
            "email": "test@example.invalid",
            **self.claim_overrides,
        }
        token = jwt.encode(
            claims,
            self.other_key if self.sign_with_wrong_key else self.private_key,
            algorithm="RS256",
            headers={"kid": self.kid},
        )
        return httpx.Response(200, json={
            "access_token": "initial-access",
            "refresh_token": "initial-refresh",
            "id_token": token,
            "token_type": "Bearer",
            "expires_in": self.expires_in,
            "scope": "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct",
            **self.token_overrides,
        })


@pytest_asyncio.fixture
async def harness():
    repository = MemoryRepository()
    provider = Provider()
    key = Fernet.generate_key()
    now = [time.time()]
    service = AuthService(
        repository,
        encryption_key=key,
        transport=httpx.MockTransport(provider.handle),
        clock=lambda: now[0],
    )
    yield service, repository, provider, key, now
    await service.aclose()


async def start(service, provider, user="alice", browser="browser-alice"):
    url = await service.begin(user, browser)
    query = {key: values[0] for key, values in parse_qs(urlsplit(url).query).items()}
    provider.nonce = query["nonce"]
    return query


async def connect(service, provider, user="alice", browser="browser-alice"):
    query = await start(service, provider, user, browser)
    result = await service.complete(
        user, browser, state=query["state"], code="code-one", client_id=provider.client_id
    )
    return query, result


async def test_dynamic_registration_pkce_and_encrypted_binding(harness):
    service, repository, provider, _, _ = harness
    assert provider.calls == []
    query, result = await connect(service, provider)
    assert query["client_id"] == "dynamic_agent_client"
    assert query["agent_name_hint"] == "taskmate-lab"
    assert query["ext_agent_host_id"].startswith("urn:uuid:")
    assert query["redirect_uri"] == "http://127.0.0.1:8765/auth/callback"
    form = provider.forms[0]
    expected = base64.urlsafe_b64encode(hashlib.sha256(form["code_verifier"].encode()).digest())
    assert query["code_challenge"] == expected.rstrip(b"=").decode()
    assert query["code_challenge_method"] == "S256"
    assert form["client_id"] == "oaiapp_test_registration"
    assert form["redirect_uri"] == query["redirect_uri"]
    assert form["resource"] == "https://api.openai.com/v1"
    assert "client_secret" not in form
    assert result["connected"] is True
    assert result["plan_enabled"] is True
    assert "access_token" not in result
    assert "id_token" not in result
    ciphertext = repository.values["alice", "credentials"]
    assert b"initial-access" not in ciphertext
    assert b"test@example.invalid" not in ciphertext
    assert await service.access_token("alice") == "initial-access"
    assert (await service.status("bob"))["connected"] is False
    repository.values["bob", "credentials"] = ciphertext
    with pytest.raises(AuthError) as caught:
        await service.status("bob")
    assert caught.value.code == "invalid_stored_credentials"


@pytest.mark.parametrize("failure", ["wrong_state", "wrong_browser", "expired", "denied", "missing_client", "dynamic_client"])
async def test_bad_callbacks_never_exchange_and_are_one_use(harness, failure):
    service, _, provider, _, now = harness
    query = await start(service, provider)
    state = "wrong" if failure == "wrong_state" else query["state"]
    browser = "other-browser" if failure == "wrong_browser" else "browser-alice"
    if failure == "expired":
        now[0] += 601
    client = None if failure == "missing_client" else provider.client_id
    if failure == "dynamic_client":
        client = "dynamic_agent_client"
    with pytest.raises(AuthError):
        await service.complete(
            "alice", browser, state=state, code="code-one", client_id=client,
            error="access_denied" if failure == "denied" else None,
        )
    with pytest.raises(AuthError) as caught:
        await service.complete(
            "alice", "browser-alice", state=query["state"], code="code-one", client_id=provider.client_id
        )
    assert caught.value.code == "missing_sign_in"
    assert provider.forms == []


async def test_successful_callback_cannot_be_replayed(harness):
    service, _, provider, _, _ = harness
    query, _ = await connect(service, provider)
    with pytest.raises(AuthError):
        await service.complete(
            "alice", "browser-alice", state=query["state"], code="code-one", client_id=provider.client_id
        )
    assert len(provider.forms) == 1


@pytest.mark.parametrize("claims", [
    {"iss": "https://attacker.invalid"},
    {"aud": "other-client"},
    {"sub": ""},
    {"exp": 1},
    {"iat": 9999999999},
    {"nonce": "wrong-nonce"},
    {"aud": ["oaiapp_test_registration", "other-client"]},
    {"azp": "other-client"},
])
async def test_oidc_claim_validation(harness, claims):
    service, _, provider, _, _ = harness
    provider.claim_overrides = claims
    with pytest.raises(AuthError) as caught:
        await connect(service, provider)
    assert caught.value.code == "invalid_identity_token"
    assert (await service.status("alice"))["connected"] is False


async def test_signature_verification_is_not_mocked_out(harness):
    service, _, provider, _, _ = harness
    provider.sign_with_wrong_key = True
    with pytest.raises(AuthError) as caught:
        await connect(service, provider)
    assert caught.value.code == "invalid_identity_token"


async def test_reauthorization_reuses_client_and_rejects_identity_change(harness):
    service, _, provider, _, _ = harness
    first, _ = await connect(service, provider)
    query = await start(service, provider)
    assert query["client_id"] == provider.client_id
    assert query["ext_agent_host_id"] == first["ext_agent_host_id"]
    assert "agent_name_hint" not in query
    assert "id_token_hint" not in query
    assert query["state"] != first["state"]
    provider.subject = "another-subject"
    with pytest.raises(AuthError) as caught:
        await service.complete("alice", "browser-alice", state=query["state"], code="second-code")
    assert caught.value.code == "account_mismatch"
    assert await service.access_token("alice") == "initial-access"


async def test_reauthorization_rejects_changed_client_before_exchange(harness):
    service, _, provider, _, _ = harness
    await connect(service, provider)
    query = await start(service, provider)
    with pytest.raises(AuthError) as caught:
        await service.complete(
            "alice", "browser-alice", state=query["state"], code="second-code", client_id="other-client"
        )
    assert caught.value.code == "registration_mismatch"
    assert len(provider.forms) == 1


async def test_invalid_first_code_keeps_issued_client_for_retry(harness):
    service, _, provider, _, _ = harness
    provider.exchange_error = "invalid_grant"
    with pytest.raises(AuthError):
        await connect(service, provider)
    query = await start(service, provider)
    assert query["client_id"] == provider.client_id
    assert "agent_name_hint" not in query
    assert (await service.status("alice"))["connected"] is False


async def test_granted_scopes_gate_plan_usage(harness):
    service, _, provider, _, _ = harness
    provider.token_overrides["scope"] = "openid profile email"
    _, result = await connect(service, provider)
    assert result["connected"] is True
    assert result["plan_enabled"] is False
    with pytest.raises(AuthError) as caught:
        await service.access_token("alice")
    assert caught.value.code == "plan_permission_required"


async def test_refresh_serializes_across_service_instances_and_rotates(harness):
    service, repository, provider, key, _ = harness
    provider.expires_in = 30
    provider.refresh_delay = 0.01
    await connect(service, provider)
    other = AuthService(repository, encryption_key=key, transport=httpx.MockTransport(provider.handle))
    try:
        values = await asyncio.gather(service.access_token("alice"), other.access_token("alice"))
        assert values == ["refreshed-access", "refreshed-access"]
        assert provider.refresh_count == 1
        form = provider.forms[-1]
        assert form == {
            "grant_type": "refresh_token",
            "client_id": "oaiapp_test_registration",
            "refresh_token": "initial-refresh",
            "resource": "https://api.openai.com/v1",
        }
        assert (await service.status("alice"))["plan_enabled"] is True
    finally:
        await other.aclose()


async def test_earliest_refresh_time_is_respected(harness):
    service, _, provider, _, now = harness
    provider.expires_in = 30
    provider.token_overrides["earliest_refresh_at"] = now[0] + 100
    await connect(service, provider)
    assert await service.access_token("alice") == "initial-access"
    now[0] += 31
    with pytest.raises(AuthError) as caught:
        await service.access_token("alice")
    assert caught.value.code == "refresh_not_due"
    assert provider.refresh_count == 0


async def test_revoked_refresh_requires_reauth_without_replay(harness):
    service, _, provider, _, _ = harness
    provider.expires_in = 30
    await connect(service, provider)
    provider.exchange_error = "invalid_grant"
    with pytest.raises(AuthError):
        await service.access_token("alice")
    count = len(provider.forms)
    with pytest.raises(AuthError) as caught:
        await service.access_token("alice")
    assert caught.value.code == "sign_in_required"
    assert len(provider.forms) == count
    assert (await service.status("alice"))["connected"] is False


@pytest.mark.parametrize("revoke_status,confirmed", [(200, True), (503, False)])
async def test_disconnect_clears_tokens_and_preserves_registration(harness, revoke_status, confirmed):
    service, _, provider, _, _ = harness
    await connect(service, provider)
    provider.revoke_status = revoke_status
    assert await service.disconnect("alice") is confirmed
    assert (await service.status("alice"))["connected"] is False
    query = await start(service, provider)
    assert query["client_id"] == provider.client_id
    assert "id_token_hint" not in query


async def test_discovery_cannot_redirect_credentials_to_other_hosts(harness):
    service, _, provider, _, _ = harness
    provider.discovery_override["token_endpoint"] = "https://attacker.invalid/tokens"
    with pytest.raises(AuthError) as caught:
        await start(service, provider)
    assert caught.value.code == "unexpected_provider_endpoints"
    assert provider.calls == [DISCOVERY]


@pytest.mark.parametrize("uri", [
    "http://localhost:8765/auth/callback",
    "https://127.0.0.1:8765/auth/callback",
    "http://127.0.0.1:8765/other",
    "http://127.0.0.1:8765/auth/callback?next=elsewhere",
])
async def test_callback_uri_is_fixed_loopback(uri):
    with pytest.raises(ValueError):
        AuthService(MemoryRepository(), encryption_key=Fernet.generate_key(), redirect_uri=uri)


@pytest.mark.skipif(not os.environ.get("TEST_DATABASE_URL"), reason="PostgreSQL integration requires TEST_DATABASE_URL")
async def test_postgres_state_consumption_and_cross_instance_refresh():
    pool = AsyncConnectionPool(os.environ["TEST_DATABASE_URL"], open=False)
    await pool.open()
    repository = PostgresAuthRepository(pool)
    await repository.setup()
    provider = Provider()
    key = Fernet.generate_key()
    user = "auth-test-" + str(uuid.uuid4())
    one = AuthService(repository, encryption_key=key, transport=httpx.MockTransport(provider.handle))
    two = AuthService(repository, encryption_key=key, transport=httpx.MockTransport(provider.handle))
    try:
        provider.expires_in = 30
        provider.refresh_delay = 0.05
        query, _ = await connect(one, provider, user=user, browser="integration-browser")
        with pytest.raises(AuthError):
            await two.complete(user, "integration-browser", state=query["state"], code="replayed")
        assert await asyncio.gather(one.access_token(user), two.access_token(user)) == [
            "refreshed-access", "refreshed-access"
        ]
        assert provider.refresh_count == 1
        assert await repository.host_id() == await repository.host_id()
    finally:
        async with repository.locked(user) as records:
            await records.delete("pending")
            await records.delete("credentials")
        await one.aclose()
        await two.aclose()
        await pool.close()
