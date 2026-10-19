# Taskmate Lab

A local-first learning app for exploring a personalized task assistant with DeepAgents, durable PostgreSQL state, and Sign in with ChatGPT. Version 0.1.0.

## Stack

- TypeScript throughout, Node.js 24
- React browser UI, built with Vite+
- Hono HTTP API, served on http://127.0.0.1:8765
- DeepAgents / LangGraph JS with the official PostgreSQL checkpointer and store
- PostgreSQL 17 for tasks, sessions, notifications, memory, and encrypted OAuth records
- Vite+ for build, Vitest tests, formatting, linting, and type checks; Playwright for browser evidence

Alice and Bob are local demo profiles. They are not authenticated SaaS accounts. Anyone with access to this local app can switch profiles. Do not expose it through a proxy, tunnel, or public interface.

## Start locally

Install Node.js 24 and pnpm 12.14.0. Vite+ is a project dependency; a global installation is optional.

```sh
pnpm install
cp .env.example .env
docker compose up -d
```

Generate your own encryption key locally and put the output in TASKMATE_ENCRYPTION_KEY in .env:

```sh
node -e "console.log(require('node:crypto').randomBytes(32).toString('base64'))"
```

Keep .env private and retain this key while using the same database. Never paste it into a chat or commit it.

```sh
pnpm build
pnpm start
```

Open http://127.0.0.1:8765, select a demo profile, and use “ChatGPTでログイン”. You complete consent directly with OpenAI. No API key is required.

After connecting, load the available model catalog in the UI, copy an available model ID into TASKMATE_MODEL, then restart the API and worker. Chat and generated notification previews stay disabled until a model is configured. Direct task CRUD remains available.

For scheduled in-app notifications, start another terminal:

```sh
pnpm worker
```

The worker checks once a minute. Notifications appear in this app; this PoC does not send email or push notifications.

## Checks

```sh
pnpm exec vp check
pnpm typecheck
pnpm test
pnpm build
pnpm --filter @taskmate/web exec playwright install chromium
pnpm test:ui
```

Set TEST_DATABASE_URL to a disposable PostgreSQL database when running the API integration suite. Tests use mock OAuth and model responses; they do not sign in to OpenAI or spend model credits. See [verification instructions](.cursor/skills/verify-taskmate-lab/SKILL.md).

Vite+ development UI is available with `pnpm --filter @taskmate/web dev`. It is for UI development and mocked browser tests. Use the built same-origin app on port 8765 for actual OAuth and API operations.

## Boundaries

The model can list/create/complete tasks and read/update preferences. It cannot execute a shell, make arbitrary HTTP requests, or run arbitrary SQL. User and thread identity are supplied by the server, not model arguments.

ChatGPT login is OpenAI's official [token-sharing preview](https://developers.openai.com/siwc/token-sharing-open-source). Availability depends on the account and current preview eligibility. This localhost OSS app does not establish eligibility for a hosted paid SaaS. Real OAuth and real inference remain a manual local verification step.

- [Configuration](docs/configuration.md)
- [Architecture and security boundaries](docs/architecture.md)
- [Verification status](docs/verification-status.md)
