## Contributing

### Development setup
1. Create a virtualenv and install deps:

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows
pip install -r requirements.txt
```

2. Configure environment:
- Copy `.env.example` to `.env`
- Set `TELEGRAM_BOT_TOKEN` at minimum for runtime
- Ensure Postgres/Redis are reachable for full functionality

3. Run:

```bash
python tools/dev.py run
```

### Quality gates
Before opening a PR, run:

```bash
python tools/dev.py ci
```

This runs:
- Ruff lint
- Mypy (current configured scope in `pyproject.toml`)
- Pytest

### Tests
Add tests for:
- new parsing behavior (NLU)
- external adapter behavior (use mocks/fixtures)
- error paths (timeouts/upstream failures)

### Security
- Never commit `.env` or secrets.
- Avoid logging secrets; use structured `extra={...}` fields.
- Keep endpoints that trigger work (`/tasks/*`, `/metrics`, `/admin/*`) protected by `CRON_SECRET`.

### PR guidelines
- Keep changes small and reviewable.
- Prefer incremental refactors with tests.
- If you add a new dependency, explain why in the PR description.

