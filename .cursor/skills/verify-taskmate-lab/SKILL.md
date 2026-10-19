---
name: verify-taskmate-lab
description: Verify Taskmate Lab's local TypeScript app, OAuth boundaries, persistence, and browser behavior with synthetic data.
---

# Verify Taskmate Lab

Use a disposable PostgreSQL database and synthetic Alice/Bob data. Do not collect real tokens or perform OAuth consent for the operator.

1. Read AGENTS.md and the current API/UI contracts. Check that source, docs, and this skill agree.
2. Install from the committed pnpm lockfile. Run pnpm exec vp check, pnpm typecheck, pnpm test, and pnpm build.
3. After the build, run node scripts/smoke.ts with TEST_DATABASE_URL to verify the real localhost entrypoint, static assets, request guards, demo profiles, and worker persistence. Set TEST_DATABASE_URL for PostgreSQL integration. Verify ownership isolation, durable preferences, checkpoints, retries, and notification deduplication.
4. Verify mocked OAuth PKCE/state/nonce, JWT issuer/audience/algorithm/nonce, scopes, encrypted records, refresh rotation, and disconnect behavior.
5. Verify the Responses adapter against mocked SSE: full input, developer role, namespace tools, encrypted reasoning replay, malformed/truncated streams, and sanitized errors.
6. Run pnpm test:ui after installing Playwright Chromium. Capture synthetic screenshots and videos in artifacts; include the artifact link in the PR.
7. Check profile switching drops stale responses, missing model configuration is visible, direct CRUD works without a model, timezone persists, and pending chat can be retried without duplicate effects.
8. Record actual command results, commit SHA, and evidence links in docs/verification-status.md and the PR. Do not call generated screenshots visually reviewed unless inspected.
9. Keep the app on 127.0.0.1:8765. Do not deploy, expose a tunnel, or perform real OAuth in automated checks.

## Optional operator-only live check

The operator starts the built app locally, completes ChatGPT consent, loads their available model catalog, configures a returned model ID, and tests one synthetic chat/tool turn. They can restart the app to inspect persistence and disconnect afterward. Never record or share authorization codes, tokens, the encryption key, or raw provider payloads.
