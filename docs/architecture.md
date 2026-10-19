# Architecture

## Processes and state

The browser loads a React build from Hono on the same loopback origin. Hono supplies a random HttpOnly session cookie and a CSRF token. Alice/Bob selection rotates the session; these profiles are demonstrations rather than authentication. Scoped reads and writes require the current tab's CSRF token, preventing a stale tab from displaying another profile's data after the shared cookie changes.

PostgreSQL holds domain tables, OAuth records, the LangGraph checkpointer, and the LangGraph store. The checkpointer preserves a conversation's graph state. The store holds user-scoped preferences across conversations. Tasks and notifications remain application data, not model memory.

The notification worker is a separate entrypoint in the same API package. It shares the application services and database; it is not an autonomous scheduler inside DeepAgents.

## Trusted context

The server resolves a profile from the session, checks ownership of each task/thread, and supplies userId/threadId as trusted graph context. Models cannot choose another profile by adding tool arguments. Tool effects use a request/tool ledger to avoid duplicate creates on retries. A pending turn is resumable with its original request ID.

DeepAgents uses the official JS PostgreSQL checkpointer/store. Only explicit task/preference tools are exposed. Filesystem and other general-purpose agent capabilities must not become a route around these domain boundaries.

## ChatGPT connection

The app uses official OAuth authorization-code + PKCE with one-use state/nonce, fixed issuer endpoints, validated ID tokens, granted-scope checks, and profile-specific token storage. Refresh is coordinated with a database lock. There is no API-key fallback and no Codex process.

A custom LangChain chat model translates messages and tools into the official Responses token-sharing contract: streaming requests, store:false, full conversation input, developer instructions, namespaced tools, and encrypted reasoning replay. The adapter preserves response items through checkpoint serialization and rejects unsupported shapes/options. Incomplete streams cannot become successful responses.

## Local security boundary

Host/origin checks, CSRF, loopback binding, restricted tool lists, ownership checks, encrypted credentials, and sanitized provider errors reduce accidental exposure. They do not turn a demo profile switcher into SaaS authentication. Any process or person already controlling the user's computer remains inside the trust boundary.

Future multi-instance deployment needs real authentication, server-controlled identity mapping, shared storage, request idempotency, coordinated scheduling, and product eligibility review. Persistent conversation state alone is not tenant isolation.
