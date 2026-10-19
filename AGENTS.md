# Taskmate development

Read the current code and tests before changing behavior. Use the upstream pstack poteto-mode workflow for development: https://github.com/cursor/plugins/tree/main/pstack/skills/poteto-mode.

- Keep the application bound to 127.0.0.1. Alice and Bob are demo profiles, not production authentication.
- Keep ChatGPT OAuth credentials on the server. Never commit, print, record, or return tokens to the browser.
- Preserve per-user ownership checks for tasks, threads, memory, credentials, and notifications.
- Keep the model tool list limited to task and preference operations. Do not add shell, arbitrary SQL, or arbitrary HTTP tools.
- Verify behavior with mocked OAuth and model responses. Real ChatGPT consent is a manual local check by the operator.
- Update the verification skill and relevant docs whenever behavior, configuration, or schema changes.
- Include a version update in the same PR as a functional change. Do not create a version-only PR.
- Use .cursor/skills/verify-taskmate-lab/SKILL.md for verification. Record what actually ran and what remains unverified.
- Keep CLAUDE.md limited to @AGENTS.md.
