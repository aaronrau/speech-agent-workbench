# speech-agent-workbench

Voice control for Linux and macOS agent workspaces: speak a command, route it to the right tmux agent, and submit it without touching the keyboard.

## Project Status

This is an experimental testbed for exploring voice-driven human-agent
interaction. It is not production-ready, hardened, or designed to scale; treat
it as a fun local script for trying ideas quickly.

## Features

- Plain dictation mode that types text into the focused app after a trigger word.
- Agent workbench mode that starts a tmux workspace with three configurable agent panes and one voice listener pane.
- Voice routing by pane/window name: say `<agent name>` to switch targets or `<agent name> <message>` to send text directly to that tmux target and press Enter.
- Per-agent clear command: say `<agent name> clear terminal` to send `/clear` to that agent's tmux terminal.
- Configurable agent names, working directories, pane/window layout, voice pane name, trigger word, aliases, paste mode, and input devices.
- Numeric pane names get spoken-number aliases, so a name containing `2` can be addressed as `two`.
- Optional voice session shutdown commands, disabled by default.
- Default local STT uses Parakeet ONNX through `onnx-asr`.
- First-run startup creates `.venv`, installs Python requirements, checks model downloads, logs the Hugging Face cache path, and preloads Parakeet ONNX before listening starts.
- The workbench launcher waits for STT/VAD models to finish loading before reporting the voice listener ready.
- Optional remote STT auto-detection with local Parakeet ONNX fallback.
- Sherpa/Silero VAD support for automatic speech detection.
- X11 and Wayland typing support through `xdotool`/`xclip`, `wtype`, `wl-clipboard`, or `ydotool`.
- Native macOS clipboard, Command-V, Return-key, mouse-click, and terminal-focus support through `pbcopy`, AppleScript, and `pynput`.
- Direct tmux send mode for agent-prefixed messages, so routed commands do not depend on desktop focus.
- Optional Gemma GGUF transcript correction through llama.cpp, with fast built-in cleanup for common Codex/tmux/GitHub STT mistakes.
- Focus and tmux-send diagnostics are written to a configurable log file.

## Install

```bash
./install.sh
```

The installer detects Linux or macOS, creates `.venv`, installs the matching
system and Python dependencies (including `llama-cli` and `llama-server`),
creates `config.json` from `config.example.json`, and downloads and validates
the default Parakeet ONNX STT model. The public launch commands perform the
same OS detection and preserve their existing arguments and environment
overrides. Installation fails with a platform-specific recovery command if
`tmux` is not executable after the system-package stage.

The Parakeet files are stored in the normal Hugging Face cache. Set
`VOICE_INSTALL_STT_MODEL=0` to skip the model prefetch, or
`VOICE_INSTALL_LLAMA_CPP=0` to skip llama.cpp when those assets are managed
separately.

### Linux

On Debian/Ubuntu, `install.sh` installs packages with `apt-get`. The existing
X11 and Wayland integrations remain the default Linux implementations. When a
system llama.cpp package is not already available, the installer builds
`llama-cli` and `llama-server` under the ignored `models/llama.cpp` directory
and exposes them through `.venv/bin`.

### macOS

Install [Homebrew](https://brew.sh/) first, then run the normal installer:

```bash
./install.sh
```

The macOS installer supports Apple Silicon and Intel Homebrew locations and
installs/checks Python 3.10+, tmux, ffmpeg, PortAudio, and llama.cpp. It installs
the minimal Parakeet ONNX/Sherpa runtime and prefetches Parakeet by default. To
install every optional STT backend, use:

```bash
VOICE_INSTALL_FULL_REQUIREMENTS=1 ./install.sh
```

On first use, macOS may ask for privacy permissions for the terminal app that
launched the workbench. Enable these under **System Settings → Privacy &
Security**:

- **Microphone** for audio capture.
- **Accessibility** for paste, Return, focus, and mouse automation.
- **Input Monitoring** for global hotkeys when macOS requires it.

The launcher detects Terminal, iTerm, Warp, WezTerm, Ghostty, and VS Code from
`TERM_PROGRAM`. Override the focus target when needed with
`VOICE_AUTO_MACOS_TERMINAL_APP`, for example:

```bash
VOICE_AUTO_MACOS_TERMINAL_APP=Ghostty ./run-auto.sh
```

To skip system packages:

```bash
INSTALL_SYSTEM_DEPS=0 ./install.sh
```

If `.venv` is missing or missing core runtime packages, `run.sh` creates it
and installs the default Parakeet runtime dependencies before checking model
downloads. Set `VOICE_INSTALL_FULL_REQUIREMENTS=1` to install every backend in
`requirements.txt`, or `VOICE_CREATE_VENV=off` to require manual venv creation.

## Configure

Edit `config.json`.

Important fields:

- `transcribe_backend`: default is `parakeet-onnx`
- `sherpa_model_dir`: included Parakeet ONNX model directory
- `auto_trigger_word`: default is `agent`
- `auto_trigger_aliases`: defaults include `codex`, `code x`, and `condex`
- `auto_enable_terminate_commands`: default is `false`
- `transcript_correction_backend`: set to `llama-cpp` for model cleanup
- `transcript_correction_llama_cpp_model`: GGUF model path for llama.cpp cleanup
- `paste_mode`: `type`, `clipboard`, `hotkey`, or `auto`
- `agent_workbench.agent_command`: command to start in each agent pane; default example is `codex --sandbox danger-full-access --ask-for-approval never`
- `agent_workbench.agents`: pane names and working directories

### llama.cpp STT Cleanup

The built-in fast cleanup always fixes common coding-agent ASR mistakes such as
`condex`/`code x` to `Codex`, `tea mux` to `tmux`, and `git hub` to `GitHub`.
It also treats `length view`, `lang fuse`, and similar phonetic variants as
`Langfuse`, and `yaws`, `evalues`, `e values`, and `e vals` as `EVALS`. It
adds command aliases like `agent to`, `agent too`, and `agent 2` for a
configured `agent two` pane.

For model-based cleanup with llama.cpp, add values like these to your local
`config.json`. These are examples; keep your real local paths and runtime
settings in `config.json`, which is git-ignored.
The default correction prompt is also configured in `config.example.json` under
`transcript_correction_prompt`; copy or edit that value in your local
`config.json` when changing correction behavior.

```json
{
  "transcript_correction_backend": "llama-cpp",
  "transcript_correction_llama_cpp_path": "models/llama.cpp-rocm/build-rocm/bin/llama-cli",
  "transcript_correction_llama_cpp_server_path": "llama-server",
  "transcript_correction_llama_cpp_server_url": "http://127.0.0.1:18087",
  "transcript_correction_llama_cpp_server_autostart": true,
  "transcript_correction_llama_cpp_model": "models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q8_0.gguf",
  "transcript_correction_llama_cpp_gpu_layers": 99,
  "transcript_correction_max_new_tokens": 256
}
```

The llama.cpp backend prefers a persistent `llama-server` process using the
OpenAI-compatible chat endpoint, so the GGUF stays loaded between utterances.
If no server can be reached or started, it falls back to one-shot `llama-cli`.
Use a `llama-cli`/`llama-server` pair built from the same llama.cpp build, for
example a local ROCm/HIP build plus a compatible GGUF model.

Successful pasted or tmux-sent utterances are appended to
`transcript-history.txt`. When transcript correction details are available, the
history payload is JSON and includes `raw_transcript`, `pre_llm_transcript`,
and `corrected_transcript` alongside the final `text` that was sent.
The console also logs the raw STT text, pre-LLM cleanup text, llama.cpp output,
and final accepted text by default. Set
`VOICE_TRANSCRIPT_CORRECTION_CONSOLE_LOG=0` or
`"transcript_correction_console_log": false` to disable that console output.

### Clear Terminal Command

Every configured tmux switch target also gets an exact clear command. Say
`<agent name> clear terminal` and the workbench focuses that target, sends
`/clear`, and presses Enter. The clear phrase is not treated as a normal
agent-prefixed prompt message.

### Optional Terminate Command

Voice commands that kill the tmux workbench are disabled by default because STT
can mishear short greetings or filler words. To opt in, use an explicit phrase
in your local `config.json`:

```json
{
  "auto_enable_terminate_commands": true,
  "auto_tmux_terminate_words": [
    "voice confirm terminate session"
  ]
}
```

Terminate commands require an exact phrase match and are not used for
agent-prefixed message routing. `terminate session`, `terminates session`,
`terminate sessions`, and `terminates sessions` variants are expanded from one
configured terminate phrase. They also require the raw ASR transcript to contain
the exact phrase; transcript correction cannot invent a terminate command from
empty or unclear audio.

### Voice-Friendly Names

Use short names with hard consonants and distinct vowel sounds so STT does not
confuse pane names with each other or with command words like `yes`, `no`,
`send`, `save`, or `stop`.

Good examples:

- `Flux`
- `Brock`
- `Knox`
- `Pike`
- `Slate`
- `Brock`
- `Vance`
- `Rook`
- `Wolf`

Avoid names that sound like common confirmations or commands, such as `Jazz`
near `yes`, `Bo` near `no`, or `Sage` near `save`.

## Run Plain Dictation

```bash
./run.sh
```

The run script defaults the local recognizer to `parakeet-onnx`. Set
`VOICE_TRANSCRIBE_BACKEND` to choose another backend, or set
`VOICE_DEFAULT_TRANSCRIBE_BACKEND=off` to use the value from `config.json`.
When `parakeet-onnx` is selected, startup preloads the model so a missing model
downloads before listening starts. Set `VOICE_PARAKEET_ONNX_DOWNLOAD=off` to
skip that preload.

Say the trigger word, then the text to type:

```text
agent write a reply about payments
```

The app clicks the current mouse position, types the words after the trigger, and presses Enter after silence.

## Run Agent Workbench

```bash
./run-auto.sh
```

To run the workbench listener services without microphone/STT capture, use:

```bash
./run-auto.sh --disable-stt
```

This starts the API, tmux summary tailer, completion tailer, and webhook
delivery, but audio listening stays permanently paused and Ctrl cannot resume
STT. `AUTO_STT=0` is different: it skips starting the voice listener process
entirely.

When Agent Audio Pipe supplies ASR, `--disable-stt` is the intended mode. Keep
`auto_enable_terminate_commands` set to `false` when the `/messages` API should
remain continuously available. If termination commands are enabled,
`Wolf terminate session` kills the workbench and the API listener on port
`8787`; the terminating request may receive an empty HTTP response because the
server stops before it can reply. Later callers will fail until `run-auto.sh` is
started again.

For a persistent Agent Audio Pipe setup, copy the tracked environment template
once and then use the normal launcher:

```bash
cp .env.agent-audio-pipe.example .env
./run-auto.sh
```

The template disables duplicate STT probing, fixes the tmux target to
`speech-workbench`, keeps Wolf in the same pane-based session, and sends the
Wolf and agent-pane output to `speech-agent-workbench-auto.log` and
`speech-agent-workbench-console.log` under `${XDG_RUNTIME_DIR:-/tmp}`. Update
the two example tokens if the Agent Audio Pipe configuration uses different
values.

On interactive launch, the workbench script shows the saved agent command, pane names, and paths. Accept the defaults or update them. The values are saved in `config.json`.
When launched through `./run-auto.sh`, model downloads and enabled
transcript-correction assets are checked in the foreground before tmux starts so
download/cache logs are visible. Set `VOICE_AUTO_PREFETCH_MODELS=off` to skip
that foreground check. If `tmux` is missing, `run-auto.sh` installs it with
Homebrew on macOS or apt on Debian/Ubuntu before starting the workbench. Set
`VOICE_AUTO_INSTALL_TMUX=0` to disable that automatic system-package change.
The launcher does not block on the voice pane by default after startup. Set `AUTO_READY_TIMEOUT=300` to wait for a ready signal from the listener.

### Example Run

With `agent_workbench.agent_command` set to `codex --sandbox danger-full-access --ask-for-approval never`, the launcher creates
three Codex panes and one voice orchestrator pane. A tiled tmux workbench might
look like this:

```text
+------------------------------+------------------------------+
| Flux                         | Brock                        |
| $ codex --sandbox ...        | $ codex --sandbox ...        |
| > Review phone verification  | > Add backend validation     |
|                              |                              |
+------------------------------+------------------------------+
| Pike                         | Wolf                         |
| $ codex --sandbox ...        | $ ./run-auto.sh              |
| > Update Flutter states      | [auto] parakeet-onnx ready   |
|                              | routes: "brock add tests"    |
+------------------------------+------------------------------+
```

Say `brock add tests for phone verification` to send that prompt to the
`Brock` pane. If you opt into terminate commands, use the exact phrase you
configured in `auto_tmux_terminate_words`.

### Local Message API

Enable the local API to route text into known tmux agent targets without using
the microphone:

```bash
VOICE_API_ENABLED=1 VOICE_API_TOKEN=local-secret ./run-auto.sh
```

Once the Wolf process binds successfully, its console prints both
`[api] server listening on http://127.0.0.1:8787` and the full
`POST /messages` endpoint. If those lines are absent, the workbench API is not
running and callers will receive a connection failure.

`run-auto.sh` also loads local defaults from `.env` when that file exists.
Values supplied explicitly in the shell take precedence over `.env` values.
For Agent Audio Pipe, start from `.env.agent-audio-pipe.example` so a bare
`./run-auto.sh` does not probe the unused remote STT endpoint on port `8765`.

Send an agent-prefixed message with JSON:

```bash
curl -sS http://127.0.0.1:8787/messages \
  -H 'Authorization: Bearer local-secret' \
  -H 'Content-Type: application/json' \
  -d '{"message":"flux: pull the latest"}'
```

You can also pass the agent separately:

```json
{"agent":"Flux","message":"pull the latest"}
```

To request an on-demand local summary of an agent's latest tmux output without
sending text to that agent, pass `"type": "local"`:

```json
{"type":"local","agent":"Flux","message":"progress_summary"}
```

Local messages read the configured agent tmux pane with a read-only
`capture-pane` call, falling back to the configured tmux console log when live
capture is unavailable. They run the same local llama.cpp summary flow used by
idle tmux summaries. The JSON response includes `summary`, `detail`, and
`detail_lines`; `detail` is the cleaned current tail of that agent's tmux
output. Local messages return this payload directly and do not emit the summary
webhook.

Only configured agent targets are accepted. Unknown names return a JSON error
with the available agents.

Set `VOICE_TMUX_SUMMARY_WEBHOOK_URL` to receive each tmux summary as JSON:

```json
{
  "agent": "Flux",
  "command": "pull the latest",
  "detail": "git pull --ff-only\nAlready up to date.",
  "detail_line_count": 2,
  "detail_lines": ["git pull --ff-only", "Already up to date."],
  "is_final": false,
  "phase": "in_progress",
  "summary": "Flux pulled the latest changes successfully.",
  "timestamp": 1781852824.8
}
```

Idle tmux summaries use `"phase": "in_progress"`. When an agent calls
`$VOICE_AGENT_SIGNAL_COMMAND done "tests passed"`, the webhook sends a final
event with `"phase": "final"`, `"is_final": true`, and the latest captured
`detail_lines` for that agent.
Webhook detail lines are incremental per agent: lines already delivered in the
previous webhook payload for that same agent are omitted from the next
`detail`/`detail_lines` value.
The voice listener pane is excluded from summary webhook delivery, including
the default `Voice` name and local `Wolf` voice pane name.

Use `VOICE_TMUX_SUMMARY_WEBHOOK_TOKEN` to send an `Authorization: Bearer ...`
header and `VOICE_TMUX_SUMMARY_WEBHOOK_TIMEOUT` to tune the POST timeout.

## Notes

- macOS clipboard/paste uses `pbcopy` and Command-V through AppleScript.
- macOS desktop automation requires Accessibility permission for the terminal
  application running the workbench.
- X11 typing uses `xdotool`/`xclip`.
- Wayland typing uses `wtype`, `wl-clipboard`, or `ydotool`.
- If direct typing fails, set `VOICE_PASTE_MODE=clipboard`.
- Focus changes are best effort. X11 or XWayland terminals can be targeted with
  `VOICE_AUTO_TERMINAL_WINDOW_TITLE` or `VOICE_AUTO_TERMINAL_WINDOW_ID`.
- GNOME Wayland can block specific-window focus. Set
  `VOICE_AUTO_GNOME_TERMINAL_FOCUS_MODE=hotkey` to use the Terminal favorite
  shortcut, or `VOICE_AUTO_GNOME_TERMINAL_FOCUS_MODE=launch` to open a focused
  GNOME Terminal attached to the configured tmux session.
- Set `VOICE_AUTO_TMUX_DIRECT_SEND=0` to fall back to desktop typing for
  prefixed tmux messages.
- Focus attempts and tmux direct-send attempts are logged to
  `${XDG_RUNTIME_DIR:-/tmp}/speech-agent-workbench-focus.log` by default.
  Override with `VOICE_AUTO_FOCUS_LOG=/path/to/focus.log` or disable with
  `VOICE_AUTO_FOCUS_LOG=0`.
- Auto listener console output is appended to
  `${XDG_RUNTIME_DIR:-/tmp}/speech-agent-workbench-auto.log` when the workbench
  starts the listener. Override with `AUTO_LOG=/path/to/auto.log`.
- Agent pane output is piped through
  `${XDG_RUNTIME_DIR:-/tmp}/speech-agent-workbench-console.log` by default. After
  a pane is idle, the listener summarizes the last 50 lines with the configured
  Gemma/llama.cpp model and prints the original routed command plus a one-line
  summary. Tune retention with `VOICE_AUTO_TMUX_CONSOLE_RETENTION_SECONDS`, cap
  size with `VOICE_AUTO_TMUX_CONSOLE_MAX_BYTES`, poll for changes with
  `VOICE_AUTO_TMUX_CONSOLE_POLL_SECONDS`. Raw pane output is not printed in
  summary mode; set `VOICE_AUTO_TMUX_SUMMARY_ENABLED=0` only when debugging the
  pipe itself.
- Agent panes receive `VOICE_AGENT_SIGNAL_COMMAND`. A running agent can signal
  completion with `$VOICE_AGENT_SIGNAL_COMMAND done "tests passed"`; the voice
  listener prints that event from
  `${XDG_RUNTIME_DIR:-/tmp}/speech-agent-workbench-completions.log`.
- On Wayland, `ydotoold` may need to be running:

```bash
sudo ydotoold --socket-path=/tmp/.ydotool_socket
```

## Tests

```bash
make test
```
