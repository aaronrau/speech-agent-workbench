# Repository Guidelines

## Project Structure & Module Organization

`app.py` is the main Python entry point for voice capture, transcription, and agent routing. Shell launchers live at the repository root: `run.sh` starts the voice app, `run-auto.sh` starts the tmux workbench flow, and `start-agent-workbench.sh` manages panes and agent sessions. Install/setup helpers are `install.sh` and `setup.sh`. Tests live in `tests/` and use Python `unittest`. Runtime assets include `models/silero_vad.onnx` and local Parakeet/Sherpa model files; private runtime state belongs in `config.json`, `.venv/`, caches, or ignored generated files.

## Build, Test, and Development Commands

- `./install.sh`: creates/repairs the local `.venv` and installs runtime dependencies.
- `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`: manual dependency setup.
- `cp config.example.json config.json`: creates a local editable config file.
- `./run.sh`: runs the voice command app.
- `./run-auto.sh`: starts the agent workbench and performs model preflight/download work.
- `make test`: runs `python -m unittest discover -s tests -t .`.
- `PYTHON=.venv/bin/python make test`: runs tests against the project virtualenv.
- `bash -n run.sh run-auto.sh start-agent-workbench.sh`: syntax-check shell launchers.

## Coding Style & Naming Conventions

Use 4-space indentation for Python and keep names in `snake_case`. Prefer small functions that isolate audio, transcription, tmux, and configuration behavior. Match existing root-level shell style: Bash scripts should be explicit, quote paths, and use clear status prefixes such as `[run]` or `[auto]`. Keep environment variables uppercase, for example `VOICE_BACKEND` or `VOICE_AUTO_STT`.

## Testing Guidelines

Add or update `tests/test_*.py` files for behavior changes. Use `unittest` assertions and favor deterministic unit tests over microphone, tmux, or network-dependent integration tests. When changing launch scripts, pair `bash -n` with focused Python tests for any parsing or config logic moved into `app.py`.

## Commit & Pull Request Guidelines

Recent commits use short imperative summaries, for example `Default voice backend to Parakeet ONNX` or `Repair incomplete voice runtime venvs`. Keep commits focused and describe user-visible behavior changes in the body when needed. Pull requests should explain the problem, summarize the fix, list tests run, and include terminal output or screenshots when UI, tmux layout, or startup logs change.

## Security & Configuration Tips

Do not commit `config.json`, virtualenvs, generated audio, local model caches, private paths, or credentials. Prefer `config.example.json` for shareable defaults. If a change downloads models, log the target cache directory and keep downloads resumable through the normal Hugging Face or ONNX tooling.

## Agent Audio Pipe Integration

- Agent Audio Pipe uses this repository's `POST /messages` API; the
  `linux-voice-codex` dashboard is not a compatible substitute.
- For glasses-provided ASR, start this workbench with
  `./run-auto.sh --disable-stt` and keep the API token, API port, agent names,
  summary webhook URL, and summary token aligned with Agent Audio Pipe.
- Prefer copying `.env.agent-audio-pipe.example` to the ignored `.env` for a
  persistent local integration. The template keeps STT disabled, pins the tmux
  session, and keeps the auto/agent console logs under the runtime directory.
- Keep `auto_enable_terminate_commands` disabled for a persistent API. An
  enabled terminate phrase kills the tmux session and the API listener on port
  `8787`, causing callers to receive connection failures until restart.
