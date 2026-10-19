# taskmate-lab

A local learning app for a task assistant. ChatGPT sign-in supplies model access without an API key. DeepAgents connects the model to task operations. PostgreSQL keeps tasks, conversations, profile preferences, and in-app reminders across restarts.

Alice and Bob are demo profiles. Anyone who can use this local app can switch profiles. Do not expose it to a network or treat the profile switch as authentication.

## Start locally

You need Python 3.12+, Docker with Compose, and a ChatGPT account eligible for the [Sign in with ChatGPT open-source preview](https://developers.openai.com/siwc/token-sharing-open-source). Preview availability and model access depend on the account.

1. Clone this repository and enter its directory.
2. Create a virtual environment and install the project.

   ```sh
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install -e '.[test]'
   ```

3. Copy `.env.example` to `.env`.
4. Generate an encryption key locally. Put the result in `TASKMATE_ENCRYPTION_KEY` in `.env`. Keep that file private.

   ```sh
   python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'
   ```

5. Start PostgreSQL and the application.

   ```sh
   docker compose up -d postgres
   python -m taskmate
   ```

6. Open http://127.0.0.1:8765. Choose a demo profile, then select the ChatGPT sign-in button. Review and complete OpenAI's consent screen yourself.
7. Use the model catalog button to read account-supported model IDs. Put one in `TASKMATE_MODEL` in `.env`, then restart the app.
8. Add a task directly or ask the assistant to add one.

Keep the encryption key across restarts. Losing it prevents the app from decrypting stored ChatGPT credentials. Back up the key separately from the database. Do not commit either.

The callback URL is http://127.0.0.1:8765/auth/callback. Use that exact host and port. This app does not require a Codex process or an API key.

## Try the learning scenarios

- Add a task in Alice's profile. Ask the assistant to list or complete it.
- Set Alice's response tone. Start another conversation and check that the tone remains.
- Switch to Bob. Check that Alice's tasks and conversation are absent.
- Preview an in-app reminder. No email or push message is sent.
- Restart the app and check that tasks and preferences remain.

To generate scheduled in-app reminders, run the worker in another terminal with the same environment:

```sh
source .venv/bin/activate
set -a
source .env
set +a
python -m taskmate.worker
```

## Verify changes

```sh
python -m pytest -q
```

Set `TEST_DATABASE_URL` to a disposable PostgreSQL database to include integration tests. Tests use mocked OAuth and model responses. They do not establish that a real ChatGPT account can complete consent or inference.

Follow the [verification skill](.cursor/skills/verify-taskmate-lab/SKILL.md) for browser checks and evidence. See [architecture](docs/architecture.md) for data ownership and [verification status](docs/verification-status.md) for checks that have actually run.
