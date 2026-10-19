# Verification status

This file records evidence, not intended coverage.

## Current status

- Source review and test preparation are in progress.
- No live ChatGPT consent or inference has been performed.
- No browser screenshot or recording has been captured.
- No local or cloud coding environment has been used to execute the application.
- GitHub Actions results must be linked here after the workflow runs.

## Required evidence before claiming end-to-end support

1. The mocked unit and PostgreSQL integration tests pass in CI.
2. A local operator completes ChatGPT sign-in using an eligible account.
3. The account model catalog is read, and an allowed model answers a task request.
4. A tool call changes a task exactly once and its result appears in the reply.
5. Profile isolation and persistence checks pass in the browser.
6. A screenshot and short recording show the app only, with no credentials, callback query strings, or private tasks.
