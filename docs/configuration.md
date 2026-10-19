# Configuration

| Variable | Meaning |
| --- | --- |
| DATABASE_URL | PostgreSQL connection string. Use a direct connection or session pooling. Transaction-pooling proxies cannot preserve the OAuth advisory lock. |
| TASKMATE_ENCRYPTION_KEY | A local Fernet key that encrypts OAuth state and credentials at rest. Required to run the app. |
| TASKMATE_MODEL | A model ID returned by the signed-in account's model catalog. An empty value leaves model operations unavailable. |
| TEST_DATABASE_URL | Disposable PostgreSQL database for integration tests. Tests can create and remove their own data. |
| LANGSMITH_TRACING | Keep false for this PoC. Enabling external traces can transmit prompts and private task data. |

The web process loads `.env` from the working directory. The worker uses exported environment variables.

PostgreSQL in `compose.yaml` binds to 127.0.0.1. Its fixed development password is for this local disposable database. Do not reuse this configuration for a hosted database.

ChatGPT sign-in is separate for Alice and Bob. A profile's credential is encrypted and associated with that profile on the server. The browser receives only connection status and a CSRF token.

The model catalog comes from OpenAI using the profile's OAuth credential. The app does not assume that any named model is available to every account. If the selected model is absent, choose a model that the account supports.

Each demo profile retains its original ChatGPT registration and account identity after disconnect. Reconnecting uses that same identity. Switching the connected ChatGPT account within a profile is outside this PoC.

Disconnect clears the local connection and attempts remote revocation. Review the app's result rather than assuming a failed network request revoked access.

The [preview limitations](https://developers.openai.com/siwc/token-sharing-open-source/preview-limitations) govern model requests. This project sends streamed, stateless Responses requests with the full conversation history. It does not use server-side Responses conversation storage.
