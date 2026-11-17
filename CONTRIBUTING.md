# Contributing

## Development setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## Workflow

1. Create a branch for your change.
2. Add or update tests for the behavior you change.
3. Run `ruff check .` and `pytest` locally.
4. Open a pull request; CI runs the same checks on Linux, Windows and macOS.

## Release policy

- `0.x` releases may introduce small breaking changes; document them in
  `CHANGELOG.md`.
- `1.0` and later follow semantic versioning strictly.
