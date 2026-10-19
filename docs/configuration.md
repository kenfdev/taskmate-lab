# Configuration

All settings belong in the root .env, which is loaded by the start and worker scripts. Do not commit this file.

| Setting | Purpose |
| --- | --- |
| DATABASE_URL | PostgreSQL connection string. The supplied Compose service is loopback-only and uses demo credentials. |
| TASKMATE_ENCRYPTION_KEY | Canonical base64 encoding of 32 random bytes, created locally by the operator. |
| TASKMATE_MODEL | Exact available model ID from the signed-in account catalog. Blank disables model operations. |
| LANGSMITH_TRACING | Keep false for this local PoC. |

The API and OAuth callback use http://127.0.0.1:8765. The app intentionally has no public host option. Use this exact origin, rather than localhost aliases.

OAuth tokens are encrypted with AES-256-GCM before PostgreSQL storage. Encryption authenticates the demo user ID and record kind. Browser state contains only a local profile's current thread ID, never OAuth tokens. The key is not recoverable from the database. Changing it without a migration makes existing encrypted records unreadable.

Each demo profile has its own ChatGPT connection. Disconnect clears local access and attempts provider revocation; the UI reports if remote revocation was not confirmed. A retained registration is bound to its original ChatGPT subject. Account switching within a profile is intentionally unsupported in this PoC.

Preferences include tone, notification hour (0–23), and an IANA timezone, defaulting to Asia/Tokyo. A notification is generated at most once per profile and local date. The API and worker share PostgreSQL, while model calls use the corresponding profile's token.

Do not put real work data into this learning app until you have reviewed its boundaries. A production service requires actual application authentication, account linking policy, TLS, operational secret management, rate limits, backups, retention, and hosted-product eligibility.

If you tried the earlier draft's Python runtime, use a fresh disposable database for this TypeScript version. Python/JS checkpoints and encrypted OAuth formats are not a supported migration path. Preserve any old database you need; do not delete it blindly.
