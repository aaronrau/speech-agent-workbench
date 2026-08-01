# Contributing

Thanks for taking the time to improve `speech-agent-workbench`.

## Development Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
WORKBENCH_CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/speech-agent-workbench"
mkdir -p "$WORKBENCH_CONFIG_DIR"
cp config.example.json "$WORKBENCH_CONFIG_DIR/config.json"
```

Keep machine-specific settings in the user config outside the repository.

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
