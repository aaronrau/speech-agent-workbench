# speech-agent-workbench

Speech Agent Workbench is a Linux voice input tool for dictation and voice-routed agent terminals. It can run as a plain speech-to-text listener or start a tmux workbench with three configurable agent panes and one voice listener pane.

The repo includes the Silero VAD ONNX file and a Sherpa ONNX Parakeet model so the default local setup can run without downloading a speech model.

## Install

```bash
./install.sh
```

The installer creates `.venv`, installs Python dependencies, installs common Ubuntu packages when `apt-get` is available, and creates `config.json` from `config.example.json`.

To skip system packages:

```bash
INSTALL_SYSTEM_DEPS=0 ./install.sh
```

## Configure

Edit `config.json`.

Important fields:

- `transcribe_backend`: default is `sherpa`
- `sherpa_model_dir`: included Parakeet ONNX model directory
- `auto_trigger_word`: default is `agent`
- `paste_mode`: `type`, `clipboard`, `hotkey`, or `auto`
- `agent_workbench.agent_command`: command to start in each agent pane
- `agent_workbench.agents`: pane names and working directories

## Run Plain Dictation

```bash
./run.sh
```

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

### Workbench Voice Functionality

- Say `<agent name>` to select that configured tmux pane or window.
- Say `<agent name> <message>` to select that tmux target, paste the message
  directly into tmux, and press Enter. This does not depend on the desktop's
  focused window.
- Names are derived from `agent_workbench.agents` and
  `agent_workbench.voice`. Numeric names also get spoken-number aliases, so a
  configured name containing `2` can be addressed as `two`.
- Say `<voice pane name> terminate session` to kill the configured tmux session
  and exit the voice listener. The listener also accepts
  `<voice pane name> terminates session`,
  `<voice pane name> terminate sessions`, and
  `<voice pane name> terminates sessions`.
- Focus changes are best effort. X11 or XWayland terminals can be targeted with
  `VOICE_AUTO_TERMINAL_WINDOW_TITLE` or `VOICE_AUTO_TERMINAL_WINDOW_ID`.
  GNOME Wayland can block specific-window focus, so opening a new GNOME
  Terminal is off by default.
- Set `VOICE_AUTO_GNOME_TERMINAL_FOCUS_MODE=hotkey` to use the Terminal
  favorite shortcut, or `VOICE_AUTO_GNOME_TERMINAL_FOCUS_MODE=launch` to open a
  focused GNOME Terminal attached to the configured tmux session.
- Set `VOICE_AUTO_TMUX_DIRECT_SEND=0` to fall back to desktop typing for
  prefixed tmux messages.
- Focus attempts and tmux direct-send attempts are logged to
  `${XDG_RUNTIME_DIR:-/tmp}/speech-agent-workbench-focus.log` by default.
  Override with `VOICE_AUTO_FOCUS_LOG=/path/to/focus.log` or disable with
  `VOICE_AUTO_FOCUS_LOG=0`.

## Notes

- X11 typing uses `xdotool`/`xclip`.
- Wayland typing uses `wtype`, `wl-clipboard`, or `ydotool`.
- If direct typing fails, set `VOICE_PASTE_MODE=clipboard`.
- On Wayland, `ydotoold` may need to be running:

```bash
sudo ydotoold --socket-path=/tmp/.ydotool_socket
```

## Tests

```bash
make test
```
