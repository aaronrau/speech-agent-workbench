# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

See also `AGENTS.md` for the canonical repository guidelines (project structure, build/test commands, coding style, testing, commit/PR conventions, and security/configuration tips). The notes here supplement that file with architecture context; if the two ever disagree, `AGENTS.md` wins.

## Project Status

Experimental local testbed for voice-driven human-agent interaction. Not production-hardened. Treat changes as personal-script ergonomics, not platform engineering.

## Commands

- `./install.sh` — create `.venv`, install Python deps from `requirements.txt`, install apt packages, and seed `config.json` from `config.example.json`. Use `INSTALL_SYSTEM_DEPS=0 ./install.sh` to skip apt.
- `./run.sh` — plain dictation mode. Self-heals `.venv` and installs the default Parakeet ONNX runtime if packages are missing. Set `VOICE_INSTALL_FULL_REQUIREMENTS=1` to install every backend in `requirements.txt`. Set `VOICE_PREFETCH_ONLY=1` to download models and exit.
- `./run-auto.sh` — agent workbench mode. Prefetches voice models in the foreground, then `exec`s `start-agent-workbench.sh` (which in turn launches `run.sh` inside a tmux pane). Set `VOICE_AUTO_PREFETCH_MODELS=off` to skip the foreground download check.
- `./start-agent-workbench.sh` — builds the tmux session directly (3 agent panes + 1 voice pane). Used by `run-auto.sh`; rarely invoked alone.
- `make test` — runs `python -m unittest discover -s tests -t .`. Use `PYTHON=.venv/bin/python make test` to run against the project venv.
- Single test: `.venv/bin/python -m unittest tests.test_workbench_config` (or any module under `tests/`).
- `bash -n run.sh run-auto.sh start-agent-workbench.sh` — syntax-check the shell launchers after editing them.

## Architecture

### Single-file Python core

`app.py` (~5900 lines, ~180 top-level functions, one `Recorder` class) is the entire runtime. `main()` at the bottom orchestrates: config load → device/hotkey prompts → transcriber resolution → audio loop. There is intentionally no package structure; new functionality is added as module-level functions grouped by concern (audio, transcription backend X, paste, focus, tmux send, auto trigger). The `DEFAULT_CONFIG` dict at the top is the source of truth for every config key — keep it in sync when adding settings.

### Two run modes

Selected by `run_mode` config / `VOICE_RUN_MODE`:

- `hotkey` — push-to-talk. Recorder runs while the configured key is held; on release, the buffer is transcribed and pasted.
- `auto` — continuous listening with a trigger word (default `agent`). VAD segments speech; transcripts matching trigger/alias prefixes are routed. `auto_*` config keys all belong to this mode.

### Transcription backends (pluggable)

Backends share a `transcribe_audio(samples)` callable produced by `build_transcriber()` / `resolve_initial_transcriber()`. Supported backends: `parakeet-onnx` (default, via `onnx-asr`), `sherpa` (sherpa-onnx with the bundled `sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/` model dir), `whisper`, `faster-whisper`, `vosk`, `nemo` / `nemo-canary`, and `remote` (HTTP `remote_url`). Each backend has its own `load_*`, `get_*`, and `transcribe_*` family of functions and a global `*_MODEL_CACHE` dict. `fallback_backend` causes failed primaries to retry against a second backend (with `backend_retry_cooldown`).

`nemo_subprocess.py` runs NeMo Canary in an isolated Python process — CUDA failures in NeMo crash the whole interpreter, so the subprocess wrapper catches CUDA errors and retries on CPU without taking down the listener.

### VAD

`auto_vad_backend` is `sherpa` (Silero VAD ONNX at `models/silero_vad.onnx`, auto-downloaded by the launchers) or `rms` (energy threshold). `build_sherpa_vad()` returns a streaming detector used inside the auto-mode loop.

### Voice routing → tmux

`run-auto.sh` derives a switch map from `agent_workbench.agents` names in the config (spoken pane name → tmux target). Digits in pane names get spoken-number aliases (`2` → `two`) via `spoken_digit_alias()`. The map is exported as `VOICE_AUTO_TMUX_SWITCHES` and consumed by `build_auto_tmux_switch_commands()`. When a routed transcript starts with a pane name, `send_text_to_tmux_target()` writes directly to that pane via `tmux send-keys` — bypassing desktop typing and focus. `VOICE_AUTO_TMUX_DIRECT_SEND=0` reverts to keyboard typing.

### Paste / focus

`paste_mode` (`type`, `clipboard`, `hotkey`, `auto`) routes through `paste_text()`. Typing uses `xdotool` (X11), `wtype` (Wayland), `ydotool` (Wayland fallback, requires `ydotoold` and `/dev/uinput`). GNOME Wayland needs special handling: `focus_gnome_terminal_*` functions use the GNOME Terminal favorite hotkey or launch a focused terminal. All focus and direct-send attempts are logged to `VOICE_AUTO_FOCUS_LOG` (default `${XDG_RUNTIME_DIR:-/tmp}/speech-agent-workbench-focus.log`) — check this when debugging input failures.

### Config

`config.json` is git-ignored and the only file the app reads at runtime. `config.example.json` is the shareable defaults. `agent_workbench` is the current key; `codex_agents` is the legacy key and is read as a fallback in both shell launchers and is documented in `tests/test_workbench_config.py`. Most settings can be overridden by `VOICE_*` environment variables of the same name (uppercased) — see `get_config_string/float/int` and the `VOICE_*` blocks in `run.sh` / `run-auto.sh`.

## Conventions

- Python: 4-space indent, `snake_case`, prefer small focused functions over classes. Avoid introducing new modules unless the addition is genuinely independent.
- Bash launchers: explicit, quote paths, prefix status output with `[run]`, `[auto]`, or `[hotkey]`.
- Environment variables: uppercase, `VOICE_*` prefix (`VOICE_TRANSCRIBE_BACKEND`, `VOICE_RUN_MODE`, etc.).
- Tests: stdlib `unittest`, deterministic, no real microphone/tmux/network. New backend or parsing logic should land with a unit test in `tests/test_*.py`.
- Do not commit `config.json`, model files, `.wav` captures, `.venv/`, or `transcript-history*.txt` (already in `.gitignore`).
- When a change downloads a model, log the target cache directory and keep downloads resumable through normal Hugging Face / ONNX tooling.
