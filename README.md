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

Say an agent pane name to focus it. Say a pane name followed by a message to switch there and submit the message.

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
