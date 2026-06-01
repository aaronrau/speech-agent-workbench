# speech-agent-workbench

Voice control for Linux agent workspaces: speak a command, route it to the right tmux agent, and submit it without touching the keyboard.

## Features

- Plain dictation mode that types text into the focused app after a trigger word.
- Agent workbench mode that starts a tmux workspace with three configurable agent panes and one voice listener pane.
- Voice routing by pane/window name: say `<agent name>` to switch targets or `<agent name> <message>` to send text directly to that tmux target and press Enter.
- Configurable agent names, working directories, pane/window layout, voice pane name, trigger word, aliases, paste mode, and input devices.
- Numeric pane names get spoken-number aliases, so a name containing `2` can be addressed as `two`.
- Voice session shutdown with phrases like `<voice pane name> terminate session`.
- Default local STT uses Parakeet ONNX through `onnx-asr`.
- First-run startup creates `.venv`, installs Python requirements, checks model downloads, logs the Hugging Face cache path, and preloads Parakeet ONNX before listening starts.
- The workbench launcher waits for STT/VAD models to finish loading before reporting the voice listener ready.
- Optional remote STT auto-detection with local Parakeet ONNX fallback.
- Sherpa/Silero VAD support for automatic speech detection.
- X11 and Wayland typing support through `xdotool`/`xclip`, `wtype`, `wl-clipboard`, or `ydotool`.
- Direct tmux send mode for agent-prefixed messages, so routed commands do not depend on desktop focus.
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
- `paste_mode`: `type`, `clipboard`, `hotkey`, or `auto`
- `agent_workbench.agent_command`: command to start in each agent pane
- `agent_workbench.agents`: pane names and working directories

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
When launched through `./run-auto.sh`, model downloads are checked in the foreground before tmux starts so download/cache logs are visible. Set `VOICE_AUTO_PREFETCH_MODELS=off` to skip that foreground check.
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
`Forge` pane, or `wolf terminate session` to stop the voice workbench.

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
