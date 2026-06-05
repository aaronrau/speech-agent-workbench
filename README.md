# speech-agent-workbench

Voice control for Linux agent workspaces: speak a command, route it to the right tmux agent, and submit it without touching the keyboard.

## Project Status

This is an experimental testbed for exploring voice-driven human-agent
interaction. It is not production-ready, hardened, or designed to scale; treat
it as a fun local script for trying ideas quickly.

## Features

- Plain dictation mode that types text into the focused app after a trigger word.
- Agent workbench mode that starts a tmux workspace with three configurable agent panes and one voice listener pane.
- Voice routing by pane/window name: say `<agent name>` to switch targets or `<agent name> <message>` to send text directly to that tmux target and press Enter.
- Configurable agent names, working directories, pane/window layout, voice pane name, trigger word, aliases, paste mode, and input devices.
- Numeric pane names get spoken-number aliases, so a name containing `2` can be addressed as `two`.
- Optional voice session shutdown commands, disabled by default.
- Default local STT uses Parakeet ONNX through `onnx-asr`.
- First-run startup creates `.venv`, installs Python requirements, checks model downloads, logs the Hugging Face cache path, and preloads Parakeet ONNX before listening starts.
- The workbench launcher waits for STT/VAD models to finish loading before reporting the voice listener ready.
- Optional remote STT auto-detection with local Parakeet ONNX fallback.
- Sherpa/Silero VAD support for automatic speech detection.
- X11 and Wayland typing support through `xdotool`/`xclip`, `wtype`, `wl-clipboard`, or `ydotool`.
- Direct tmux send mode for agent-prefixed messages, so routed commands do not depend on desktop focus.
- Optional Gemma GGUF transcript correction through llama.cpp, with fast built-in cleanup for common Codex/tmux/GitHub STT mistakes.
- Focus and tmux-send diagnostics are written to a configurable log file.

## Install

```bash
./install.sh
```

The installer creates `.venv`, installs Python dependencies, installs common Ubuntu packages when `apt-get` is available, and creates `config.json` from `config.example.json`.

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
- `agent_workbench.agent_command`: command to start in each agent pane
- `agent_workbench.agents`: pane names and working directories

### llama.cpp STT Cleanup

The built-in fast cleanup always fixes common coding-agent ASR mistakes such as
`condex`/`code x` to `Codex`, `tea mux` to `tmux`, and `git hub` to `GitHub`.
It also treats `length view`, `lang fuse`, and similar phonetic variants as
`Langfuse`. It adds command aliases like `agent to`, `agent too`, and `agent 2`
for a configured `agent two` pane.

For model-based cleanup with llama.cpp, add values like these to your local
`config.json`. These are examples; keep your real local paths and runtime
settings in `config.json`, which is git-ignored.

```json
{
  "transcript_correction_backend": "llama-cpp",
  "transcript_correction_llama_cpp_path": "models/llama.cpp-rocm/build-rocm/bin/llama-cli",
  "transcript_correction_llama_cpp_server_path": "llama-server",
  "transcript_correction_llama_cpp_server_url": "http://127.0.0.1:18087",
  "transcript_correction_llama_cpp_server_autostart": true,
  "transcript_correction_llama_cpp_model": "models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q8_0.gguf",
  "transcript_correction_llama_cpp_gpu_layers": 99,
  "transcript_correction_max_new_tokens": 32
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
agent-prefixed message routing.

### Voice-Friendly Names

Use short names with hard consonants and distinct vowel sounds so STT does not
confuse pane names with each other or with command words like `yes`, `no`,
`send`, `save`, or `stop`.

Good examples:

- `Flux`
- `Forge`
- `Niles`
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

On interactive launch, the workbench script shows the saved agent command, pane names, and paths. Accept the defaults or update them. The values are saved in `config.json`.
When launched through `./run-auto.sh`, model downloads and enabled
transcript-correction assets are checked in the foreground before tmux starts so
download/cache logs are visible. Set `VOICE_AUTO_PREFETCH_MODELS=off` to skip
that foreground check.
The launcher does not block on the voice pane by default after startup. Set `AUTO_READY_TIMEOUT=300` to wait for a ready signal from the listener.

### Example Run

With `agent_workbench.agent_command` set to `codex`, the launcher creates
three Codex panes and one voice orchestrator pane. A tiled tmux workbench might
look like this:

```text
+------------------------------+------------------------------+
| Flux                         | Forge                        |
| $ codex                      | $ codex                      |
| > Review phone verification  | > Add backend validation     |
|                              |                              |
+------------------------------+------------------------------+
| Niles                        | Wolf                         |
| $ codex                      | $ ./run-auto.sh              |
| > Update Flutter states      | [auto] parakeet-onnx ready   |
|                              | routes: "forge add tests"    |
+------------------------------+------------------------------+
```

Say `forge add tests for phone verification` to send that prompt to the
`Forge` pane. If you opt into terminate commands, use the exact phrase you
configured in `auto_tmux_terminate_words`.

## Notes

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
- On Wayland, `ydotoold` may need to be running:

```bash
sudo ydotoold --socket-path=/tmp/.ydotool_socket
```

## Tests

```bash
make test
```
