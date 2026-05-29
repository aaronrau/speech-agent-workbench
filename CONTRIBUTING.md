# Contributing

Thanks for taking the time to improve `speech-agent-workbench`.

## Development Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp config.example.json config.json
```

Keep machine-specific settings in `config.json`; it is intentionally ignored by
git.

## Tests

```bash
make test
```

The test suite uses `unittest` and lives in `tests/`.

## Pull Requests

- Keep changes focused and explain behavior changes in the PR description.
- Add or update tests for logic changes.
- Do not commit local model files, generated audio, virtual environments, or
  private device configuration.
