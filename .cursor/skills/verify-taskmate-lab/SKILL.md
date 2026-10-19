---
name: verify-taskmate-lab
description: Verify local taskmate-lab behavior, OAuth boundaries, profile isolation, and persistence. Use after code or schema changes and before opening a PR.
---

# Verify taskmate-lab

This verification procedure is a draft until an operator completes the browser checks. Keep the report explicit about what ran.

## Prepare

1. Read README.md and docs/configuration.md.
2. Use a disposable local PostgreSQL database. Do not use production data.
3. Install the project test dependencies.
4. Set TEST_DATABASE_URL and run python -m pytest -q.
5. Start the app with python -m taskmate and open http://127.0.0.1:8765.

Expected result: the UI loads, identifies Alice and Bob as demo profiles, and shows the ChatGPT connection state. The app listens only on 127.0.0.1.

## Check task ownership

1. In Alice, create a task named "Alice verification task".
2. Complete the task and confirm its state changes.
3. Switch to Bob. Confirm Alice's task is absent.
4. Create "Bob verification task" and switch back.
5. Confirm Bob's task is absent from Alice.
6. Verify the API test suite rejects access to a task or thread owned by another profile.

## Check ChatGPT sign-in

The operator performs real consent. Never enter credentials, grant access, or collect tokens on the operator's behalf.

1. Select Alice and start ChatGPT sign-in.
2. Have the operator review and complete the OpenAI consent screen.
3. Confirm the app returns to the same profile with a connected status.
4. Read the account-supported model IDs. Configure TASKMATE_MODEL and restart.
5. Ask the assistant to add a task. Confirm both the task list and reply show the result.
6. Switch to Bob. Confirm Alice's connection is not reused.
7. Check the mocked OAuth tests cover expired or replayed state, mismatched nonce or audience, refresh coordination, and disconnect.
8. Have the operator disconnect Alice when finished. Distinguish local removal from confirmed remote revocation.

Do not record the consent page, browser network panel, callback query string, environment file, or database credentials.

## Check durable preferences and chat

1. Set Alice's tone preference.
2. Start a new conversation. Ask the assistant for a short task summary.
3. Confirm the preference remains across threads.
4. Restart the application. Confirm tasks and preferences remain.
5. Review the integration tests for duplicate chat request IDs and tool-call idempotency.
6. Check that a failed or retried request does not create a second task.

## Check in-app reminders

1. Preview Alice's reminder and confirm it reflects Alice's tasks.
2. Confirm the preview does not send an email or push notification.
3. Run the separate worker with the same database.
4. Use a controlled test clock in the tests to check one daily record per profile.
5. Confirm repeated worker runs do not create duplicate daily records.

## Capture and report

Capture the application task list, model-connected status, and profile switch using synthetic tasks. Record a short interaction only after OAuth has finished and the callback URL has cleared.

Attach screenshots and recordings to the PR when the available tools support it. Otherwise state that visual evidence is missing. Never describe mocked model output as real ChatGPT inference.

The PR report must include:
- CI run link and test results.
- Browser checks completed.
- Actual ChatGPT account flow completed or not tested.
- Screenshots and recording, or the reason they are missing.
- Any failed checks and their reproduction steps.

## Clean up

Delete synthetic tasks through supported app behavior if available, or retain them in the disposable verification database. Stop the worker and web process. Do not delete an operator's existing data or revoke unrelated grants.
