# Architecture

The app is a local experiment in building a service around an agent. Its modules have separate responsibilities.

- FastAPI serves the browser and validates the selected demo profile, session cookie, CSRF token, and resource ownership.
- The task database is the source of truth for task state.
- DeepAgents decides when to call the allowed task and preference tools.
- The ChatGPT model adapter converts LangChain messages and tool calls to the official Responses wire format.
- The OAuth service manages browser-bound authorization attempts, token validation, encryption, refresh, and disconnect.
- PostgreSQL checkpoints persist a conversation's agent state. PostgreSQL Store persists profile preferences across conversations.
- A separate worker decides when a daily in-app reminder is due. The model writes the reminder text.

## Profile and conversation ownership

The server supplies the profile ID to tools. A model argument cannot choose another profile. Every task and conversation lookup includes its owner. Conversation identifiers are not credentials.

Alice and Bob demonstrate data scoping, not account security. The profile selector can impersonate either demo profile by design. A hosted service needs real user authentication and authorization before this app can be reused.

A conversation checkpoint and a profile preference serve different purposes. Checkpoints resume one conversation. Preferences apply to multiple conversations for the same profile. Neither replaces the task database.

## Retries and concurrent requests

Chat commands carry a request ID. Database coordination prevents concurrent turns from changing the same conversation and preserves the identity of a retried command. Task tool effects also have an idempotency key.

OAuth refresh holds a PostgreSQL session advisory lock. This prevents two app processes from rotating the same refresh token concurrently. Pending sign-in state is consumed before the token exchange.

A unique daily notification key prevents the worker from delivering the same daily reminder twice. This is an in-app database record, not an email or mobile notification.

## Local security boundary

The application binds to 127.0.0.1:8765. It rejects unexpected hosts and cross-origin mutation requests. OAuth callback handling validates one-use state, PKCE, nonce, and the ID token before saving credentials.

Tokens stay encrypted in PostgreSQL and never enter browser storage. Do not enable request-body logging or record OAuth screens and network payloads during verification.

DeepAgents receives no local shell backend, arbitrary HTTP tool, or SQL tool. Its virtual file operations do not grant host filesystem access.

The [Sign in with ChatGPT open-source preview](https://developers.openai.com/siwc/token-sharing-open-source) is the access contract for this local app. Hosting a paid service requires a separate eligibility review. Do not assume local preview access authorizes a public SaaS deployment.
