# Copilot workspace instructions

- Treat `config/settings.yaml` as a local secret-bearing file. Never display, quote,
  copy, commit, or transmit any `api_key` value from it.
- When configuration context is needed, inspect only non-secret fields such as
  provider, model, base URL presence, dimensions, and feature flags. Redact all
  credential values as `***`.
- Activate `.venv` before running Python or pytest commands in this Windows project.
- Follow `DEV_SPEC.md` for architecture and acceptance criteria, but verify the
  current implementation before claiming that a specified feature exists.
- Explain planned changes to the learner before editing files, and summarize test
  evidence afterward.
