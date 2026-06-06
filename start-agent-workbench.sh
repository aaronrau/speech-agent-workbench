#!/bin/bash
set -euo pipefail

export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin${PATH:+:$PATH}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${VOICE_HOTKEY_CONFIG:-$ROOT/config.json}"

SESSION_NAME="${SESSION_NAME:-}"
AGENT_COMMAND="${AGENT_COMMAND:-}"
AGENT_LAYOUT="${AGENT_LAYOUT:-}"
PANES_WINDOW="${PANES_WINDOW:-}"
AGENT1_STYLE="${AGENT1_STYLE:-bg=colour0,fg=colour34}"
AGENT2_STYLE="${AGENT2_STYLE:-bg=colour0,fg=colour39}"
AGENT3_STYLE="${AGENT3_STYLE:-bg=colour0,fg=colour208}"
VOICE_STYLE="${VOICE_STYLE:-bg=colour0,fg=colour244}"
AUTO_STT="${AUTO_STT:-1}"
AUTO_STT_MODE="${AUTO_STT_MODE:-auto}"
VOICE_SESSION="${VOICE_SESSION:-}"
VOICE_WINDOW="${VOICE_WINDOW:-}"
ATTACH="${ATTACH:-1}"
AUTO_LOG="${AUTO_LOG:-${XDG_RUNTIME_DIR:-/tmp}/speech-agent-workbench-auto.log}"
AUTO_PID_FILE="${AUTO_PID_FILE:-${XDG_RUNTIME_DIR:-/tmp}/speech-agent-workbench-auto.pid}"
AUTO_FOCUS_LOG="${AUTO_FOCUS_LOG:-${VOICE_AUTO_FOCUS_LOG:-${XDG_RUNTIME_DIR:-/tmp}/speech-agent-workbench-focus.log}}"
AUTO_READY_FILE="${AUTO_READY_FILE:-${VOICE_READY_FILE:-}}"
AUTO_READY_TIMEOUT="${AUTO_READY_TIMEOUT:-${VOICE_READY_TIMEOUT:-off}}"

usage() {
  cat <<'EOF'
Usage:
  ./start-agent-workbench.sh [AGENT1_DIR] [AGENT2_DIR] [AGENT3_DIR]

Starts a tmux workbench with three agent panes and one speech listener pane.
On first launch, review the saved command, pane names, and paths.

Examples:
  ./start-agent-workbench.sh
  ./start-agent-workbench.sh ~/Code/main ~/Code/api ~/Code/web

Optional environment variables:
  SESSION_NAME=speech-agent-workbench
  AGENT_COMMAND=codex
  AGENT_LAYOUT=panes
  AUTO_STT=1
  AUTO_STT_MODE=auto
  ATTACH=1
  AUTO_FOCUS_LOG=/tmp/speech-agent-workbench-focus.log
  AUTO_READY_TIMEOUT=off
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed. Install it with: sudo apt-get install -y tmux" >&2
  exit 1
fi

load_workbench_config() {
  python3 - "$CONFIG_PATH" "$HOME" "$ROOT" <<'PY'
import json
import os
import shlex
import sys

path, home, root = sys.argv[1:4]
defaults = {
    "session_name": "speech-agent-workbench",
    "layout": "panes",
    "panes_window": "Workbench",
    "agent_command": "codex",
    "agents": [
        {"name": "Agent 1", "path": home},
        {"name": "Agent 2", "path": home},
        {"name": "Agent 3", "path": home},
    ],
    "voice": {"name": "Voice", "path": root},
}
try:
    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
except Exception:
    config = {}
settings = config.get("agent_workbench") or {}
legacy_settings = config.get("codex_agents") or {}
if not isinstance(settings, dict):
    settings = {}
if not isinstance(legacy_settings, dict):
    legacy_settings = {}

def get_setting(name, default=None, legacy_name=None):
    value = settings.get(name)
    if value in (None, ""):
        value = legacy_settings.get(legacy_name or name)
    return default if value in (None, "") else value

agents = settings.get("agents") or legacy_settings.get("agents") or []
voice = settings.get("voice") or legacy_settings.get("voice") or {}

def agent(index):
    item = agents[index] if len(agents) > index and isinstance(agents[index], dict) else {}
    fallback = defaults["agents"][index]
    return {
        "name": item.get("name") or fallback["name"],
        "path": os.path.expanduser(str(item.get("path") or fallback["path"])),
    }

values = {
    "CONFIG_SESSION_NAME": get_setting("session_name", defaults["session_name"]),
    "CONFIG_AGENT_LAYOUT": get_setting("layout", defaults["layout"]),
    "CONFIG_PANES_WINDOW": get_setting("panes_window", defaults["panes_window"]),
    "CONFIG_AGENT_COMMAND": get_setting(
        "agent_command", defaults["agent_command"], legacy_name="codex_command"
    ),
    "CONFIG_AGENT1_NAME": agent(0)["name"],
    "CONFIG_AGENT1_DIR": agent(0)["path"],
    "CONFIG_AGENT2_NAME": agent(1)["name"],
    "CONFIG_AGENT2_DIR": agent(1)["path"],
    "CONFIG_AGENT3_NAME": agent(2)["name"],
    "CONFIG_AGENT3_DIR": agent(2)["path"],
    "CONFIG_VOICE_NAME": voice.get("name") or defaults["voice"]["name"],
    "CONFIG_VOICE_DIR": os.path.expanduser(str(voice.get("path") or defaults["voice"]["path"])),
}
for key, value in values.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
}

save_workbench_config() {
  WORKBENCH_SESSION_NAME="$SESSION_NAME" \
  WORKBENCH_LAYOUT="$AGENT_LAYOUT" \
  WORKBENCH_PANES_WINDOW="$PANES_WINDOW" \
  WORKBENCH_AGENT_COMMAND="$AGENT_COMMAND" \
  WORKBENCH_AGENT1_NAME="$AGENT1_NAME" \
  WORKBENCH_AGENT1_DIR="$AGENT1_DIR" \
  WORKBENCH_AGENT2_NAME="$AGENT2_NAME" \
  WORKBENCH_AGENT2_DIR="$AGENT2_DIR" \
  WORKBENCH_AGENT3_NAME="$AGENT3_NAME" \
  WORKBENCH_AGENT3_DIR="$AGENT3_DIR" \
  WORKBENCH_VOICE_NAME="$VOICE_NAME" \
  WORKBENCH_VOICE_DIR="$VOICE_DIR" \
  python3 - "$CONFIG_PATH" <<'PY'
import json
import os
import sys

path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
except Exception:
    config = {}

config["agent_workbench"] = {
    "session_name": os.environ["WORKBENCH_SESSION_NAME"],
    "layout": os.environ["WORKBENCH_LAYOUT"],
    "panes_window": os.environ["WORKBENCH_PANES_WINDOW"],
    "agent_command": os.environ["WORKBENCH_AGENT_COMMAND"],
    "agents": [
        {"name": os.environ["WORKBENCH_AGENT1_NAME"], "path": os.environ["WORKBENCH_AGENT1_DIR"]},
        {"name": os.environ["WORKBENCH_AGENT2_NAME"], "path": os.environ["WORKBENCH_AGENT2_DIR"]},
        {"name": os.environ["WORKBENCH_AGENT3_NAME"], "path": os.environ["WORKBENCH_AGENT3_DIR"]},
    ],
    "voice": {
        "name": os.environ["WORKBENCH_VOICE_NAME"],
        "path": os.environ["WORKBENCH_VOICE_DIR"],
    },
}

with open(path, "w", encoding="utf-8") as handle:
    json.dump(config, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
}

prompt_change() {
  local prompt="$1"
  local answer
  read -r -p "$prompt [y/N] " answer
  [[ "$answer" == "y" || "$answer" == "Y" || "$answer" == "yes" || "$answer" == "YES" ]]
}

prompt_value() {
  local prompt="$1"
  local current="$2"
  local value
  read -r -p "$prompt [$current]: " value
  printf '%s' "${value:-$current}"
}

normalize_spoken_name() {
  local name="$1"
  name="${name,,}"
  name="${name//[^a-z0-9]/ }"
  name="${name#"${name%%[![:space:]]*}"}"
  name="${name%"${name##*[![:space:]]}"}"
  printf '%s' "$name"
}

ensure_dir() {
  local label="$1"
  local dir="$2"
  if [[ ! -d "$dir" ]]; then
    echo "$label directory does not exist: $dir" >&2
    exit 1
  fi
}

get_pane_by_title() {
  local title="$1"
  tmux list-panes -t "$SESSION_NAME:$PANES_WINDOW" -F '#{@agent_name} #{pane_id}' 2>/dev/null |
    while read -r agent_name pane_id; do
      if [[ "$agent_name" == "$title" ]]; then
        echo "$pane_id"
        return
      fi
    done
}

start_agent_window() {
  local window_name="$1"
  local dir="$2"
  if tmux list-windows -t "$SESSION_NAME" -F '#W' 2>/dev/null | grep -Fxq "$window_name"; then
    return
  fi
  tmux new-window -t "$SESSION_NAME" -n "$window_name" -c "$dir"
  tmux send-keys -t "$SESSION_NAME:$window_name" "$AGENT_COMMAND" C-m
}

style_window() {
  local window="$1"
  local name="$2"
  local style="$3"
  tmux set-option -pt "$SESSION_NAME:$window.0" @agent_name "$name"
  tmux select-pane -t "$SESSION_NAME:$window.0" -T "$name" -P "$style"
}

build_switch_map() {
  local entries=()
  while [[ "$#" -gt 0 ]]; do
    local word="$1"
    local target="$2"
    local alias
    entries+=("$word=$target")
    alias="$(spoken_digit_alias "$word")"
    if [[ -n "$alias" && "$alias" != "$word" ]]; then
      entries+=("$alias=$target")
    fi
    shift 2
  done
  local IFS=,
  printf '%s' "${entries[*]}"
}

spoken_digit_alias() {
  python3 - "$1" <<'PY'
import sys

words = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}
tokens = sys.argv[1].split()
print(" ".join(words.get(token, token) for token in tokens))
PY
}

display_words() {
  printf '%s=32,%s=94,%s=38;5;208,%s=90' \
    "$(normalize_spoken_name "$AGENT1_NAME")" \
    "$(normalize_spoken_name "$AGENT2_NAME")" \
    "$(normalize_spoken_name "$AGENT3_NAME")" \
    "$(normalize_spoken_name "$VOICE_NAME")"
}

start_agent_panes() {
  local agent1_pane
  local agent2_pane
  local agent3_pane
  local voice_pane

  if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    tmux new-session -d -s "$SESSION_NAME" -n "$PANES_WINDOW" -c "$AGENT1_DIR"
  elif ! tmux list-windows -t "$SESSION_NAME" -F '#W' | grep -Fxq "$PANES_WINDOW"; then
    tmux new-window -t "$SESSION_NAME" -n "$PANES_WINDOW" -c "$AGENT1_DIR"
  fi

  agent1_pane="$(get_pane_by_title "$AGENT1_NAME")"
  if [[ -z "$agent1_pane" ]]; then
    agent1_pane="$(tmux display-message -p -t "$SESSION_NAME:$PANES_WINDOW.0" '#{pane_id}')"
    tmux select-pane -t "$agent1_pane" -T "$AGENT1_NAME"
    tmux set-option -pt "$agent1_pane" @agent_name "$AGENT1_NAME"
    if [[ "$(tmux display-message -p -t "$agent1_pane" '#{pane_current_command}')" == "bash" ]]; then
      tmux send-keys -t "$agent1_pane" "$AGENT_COMMAND" C-m
    fi
  fi

  agent2_pane="$(get_pane_by_title "$AGENT2_NAME")"
  if [[ -z "$agent2_pane" ]]; then
    agent2_pane="$(tmux split-window -h -t "$agent1_pane" -c "$AGENT2_DIR" -P -F '#{pane_id}')"
    tmux select-pane -t "$agent2_pane" -T "$AGENT2_NAME"
    tmux set-option -pt "$agent2_pane" @agent_name "$AGENT2_NAME"
    tmux send-keys -t "$agent2_pane" "$AGENT_COMMAND" C-m
  fi

  agent3_pane="$(get_pane_by_title "$AGENT3_NAME")"
  if [[ -z "$agent3_pane" ]]; then
    agent3_pane="$(tmux split-window -v -t "$agent2_pane" -c "$AGENT3_DIR" -P -F '#{pane_id}')"
    tmux select-pane -t "$agent3_pane" -T "$AGENT3_NAME"
    tmux set-option -pt "$agent3_pane" @agent_name "$AGENT3_NAME"
    tmux send-keys -t "$agent3_pane" "$AGENT_COMMAND" C-m
  fi

  voice_pane="$(get_pane_by_title "$VOICE_NAME")"
  if [[ -z "$voice_pane" ]]; then
    voice_pane="$(tmux split-window -v -t "$agent1_pane" -c "$VOICE_DIR" -P -F '#{pane_id}')"
    tmux select-pane -t "$voice_pane" -T "$VOICE_NAME"
    tmux set-option -pt "$voice_pane" @agent_name "$VOICE_NAME"
  fi

  tmux select-pane -t "$agent1_pane" -T "$AGENT1_NAME" -P "$AGENT1_STYLE"
  tmux set-option -pt "$agent1_pane" @agent_name "$AGENT1_NAME"
  tmux select-pane -t "$agent2_pane" -T "$AGENT2_NAME" -P "$AGENT2_STYLE"
  tmux set-option -pt "$agent2_pane" @agent_name "$AGENT2_NAME"
  tmux select-pane -t "$agent3_pane" -T "$AGENT3_NAME" -P "$AGENT3_STYLE"
  tmux set-option -pt "$agent3_pane" @agent_name "$AGENT3_NAME"
  tmux select-pane -t "$voice_pane" -T "$VOICE_NAME" -P "$VOICE_STYLE"
  tmux set-option -pt "$voice_pane" @agent_name "$VOICE_NAME"
  tmux select-layout -t "$SESSION_NAME:$PANES_WINDOW" tiled >/dev/null
}

default_switches() {
  local agent1_word
  local agent2_word
  local agent3_word
  local voice_word
  agent1_word="$(normalize_spoken_name "$AGENT1_NAME")"
  agent2_word="$(normalize_spoken_name "$AGENT2_NAME")"
  agent3_word="$(normalize_spoken_name "$AGENT3_NAME")"
  voice_word="$(normalize_spoken_name "$VOICE_NAME")"

  case "$AGENT_LAYOUT" in
    panes)
      build_switch_map \
        "$agent1_word" "pane:$(get_pane_by_title "$AGENT1_NAME")" \
        "$agent2_word" "pane:$(get_pane_by_title "$AGENT2_NAME")" \
        "$agent3_word" "pane:$(get_pane_by_title "$AGENT3_NAME")" \
        "$voice_word" "pane:$(get_pane_by_title "$VOICE_NAME")"
      ;;
    windows)
      build_switch_map \
        "$agent1_word" "$AGENT1_NAME" \
        "$agent2_word" "$AGENT2_NAME" \
        "$agent3_word" "$AGENT3_NAME" \
        "$voice_word" "$VOICE_WINDOW"
      ;;
    *)
      build_switch_map \
        "$agent1_word" "$AGENT1_NAME" \
        "$agent2_word" "$AGENT2_NAME" \
        "$agent3_word" "$AGENT3_NAME"
      ;;
  esac
}

auto_stt_command() {
  printf 'VOICE_HOTKEY_CONFIG=%q VOICE_READY_FILE=%q VOICE_CONFIG_PROMPT=0 VOICE_AUTO_START_AGENT_WORKBENCH=0 VOICE_AUTO_TMUX_SESSION=%q VOICE_AUTO_FOCUS_LOG=%q /bin/bash %q' \
    "$CONFIG_PATH" "$AUTO_READY_FILE" "$SESSION_NAME" "$AUTO_FOCUS_LOG" "$ROOT/run-auto.sh"
}

start_auto_stt_pane_log() {
  local voice_pane="$1"
  local log_target
  mkdir -p "$(dirname "$AUTO_LOG")"
  : >>"$AUTO_LOG"
  printf -v log_target %q "$AUTO_LOG"
  if ! tmux pipe-pane -t "$voice_pane" "cat >> $log_target"; then
    echo "[agents] unable to attach auto STT pane output log: $AUTO_LOG" >&2
    return 0
  fi
  echo "[agents] auto STT pane output log: $AUTO_LOG"
}

process_is_running() {
  local pid="$1"
  [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1
}

auto_ready_timeout_seconds() {
  local value="${AUTO_READY_TIMEOUT,,}"
  case "$value" in
    ""|0|false|no|none|null|off)
      echo 0
      return
      ;;
  esac
  if [[ "$value" =~ ^[0-9]+$ ]]; then
    echo "$value"
  else
    echo 300
  fi
}

clear_auto_stt_ready() {
  if [[ -z "$AUTO_READY_FILE" ]]; then
    return
  fi
  mkdir -p "$(dirname "$AUTO_READY_FILE")"
  rm -f "$AUTO_READY_FILE"
}

wait_for_auto_stt_ready() {
  local pid="${1:-}"
  local timeout
  timeout="$(auto_ready_timeout_seconds)"
  if [[ -z "$AUTO_READY_FILE" || "$timeout" == "0" ]]; then
    return 0
  fi

  echo "[agents] waiting for voice models to load..."
  local start
  start="$(date +%s)"
  while true; do
    if [[ -s "$AUTO_READY_FILE" ]]; then
      echo "[agents] voice models ready: $AUTO_READY_FILE"
      return 0
    fi
    if [[ -n "$pid" ]] && ! process_is_running "$pid"; then
      echo "[agents] auto STT exited before voice models were ready." >&2
      return 1
    fi
    if (( $(date +%s) - start >= timeout )); then
      echo "[agents] voice models were not ready after ${timeout}s." >&2
      return 1
    fi
    sleep 1
  done
}

start_auto_stt() {
  if [[ "$AUTO_STT" == "0" || "$AUTO_STT" == "off" || "$AUTO_STT" == "false" ]]; then
    return
  fi

  local switches
  switches="${VOICE_AUTO_TMUX_SWITCHES:-$(default_switches)}"
  local words
  words="${VOICE_AUTO_DISPLAY_WORDS:-$(display_words)}"
  local command
  command="$(auto_stt_command)"
  local mode
  mode="$AUTO_STT_MODE"
  if [[ "$mode" == "auto" ]]; then
    if [[ "$AGENT_LAYOUT" == "panes" ]]; then
      mode="pane"
    else
      mode="tmux"
    fi
  fi

  if [[ "$mode" == "pane" ]]; then
    local voice_pane
    voice_pane="$(get_pane_by_title "$VOICE_NAME")"
    if [[ -z "$voice_pane" ]]; then
      echo "[agents] voice pane '$VOICE_NAME' not found; skipping auto STT." >&2
      return
    fi
    local current_pane
    current_pane="$(tmux display-message -p '#{pane_id}' 2>/dev/null || true)"
    if [[ -n "$voice_pane" && "$voice_pane" == "$current_pane" ]]; then
      local delayed_command
      clear_auto_stt_ready
      start_auto_stt_pane_log "$voice_pane"
      printf -v delayed_command 'sleep 0.5; tmux send-keys -t %q C-c; tmux send-keys -t %q %q C-m' \
        "$voice_pane" "$voice_pane" "$command"
      if tmux run-shell -b "$delayed_command"; then
        echo "[agents] auto STT scheduled in current voice pane: $voice_pane"
      else
        echo "[agents] unable to schedule auto STT in current voice pane: $voice_pane" >&2
      fi
      return
    fi
    if [[ "$(tmux display-message -p -t "$voice_pane" '#{pane_current_command}' 2>/dev/null || true)" == "python" ]]; then
      echo "[agents] auto STT already running in pane: $voice_pane"
      start_auto_stt_pane_log "$voice_pane"
      wait_for_auto_stt_ready
      return
    fi
    clear_auto_stt_ready
    start_auto_stt_pane_log "$voice_pane"
    tmux send-keys -t "$voice_pane" C-c
    tmux send-keys -t "$voice_pane" "$command" C-m
    echo "[agents] auto STT running in pane: $voice_pane"
    wait_for_auto_stt_ready
    return
  fi

  if [[ "$mode" == "session" ]]; then
    if tmux has-session -t "$VOICE_SESSION" 2>/dev/null; then
      echo "[agents] auto STT tmux session already exists: $VOICE_SESSION"
      wait_for_auto_stt_ready
      return
    fi
    clear_auto_stt_ready
    tmux new-session -d -s "$VOICE_SESSION" -n "$VOICE_WINDOW" -c "$VOICE_DIR"
    tmux select-pane -t "$VOICE_SESSION:$VOICE_WINDOW.0" -T "$VOICE_NAME" -P "$VOICE_STYLE"
    tmux send-keys -t "$VOICE_SESSION:$VOICE_WINDOW" "$command" C-m
    echo "[agents] auto STT running in tmux session: $VOICE_SESSION"
    wait_for_auto_stt_ready
    return
  fi

  if [[ "$mode" == "tmux" ]]; then
    if tmux list-windows -t "$SESSION_NAME" -F '#W' 2>/dev/null | grep -Fxq "$VOICE_WINDOW"; then
      echo "[agents] auto STT tmux window already exists: $VOICE_WINDOW"
      wait_for_auto_stt_ready
      return
    fi
    clear_auto_stt_ready
    tmux new-window -d -t "$SESSION_NAME" -n "$VOICE_WINDOW" -c "$VOICE_DIR"
    tmux select-pane -t "$SESSION_NAME:$VOICE_WINDOW.0" -T "$VOICE_NAME" -P "$VOICE_STYLE"
    tmux set-option -pt "$SESSION_NAME:$VOICE_WINDOW.0" @agent_name "$VOICE_NAME"
    tmux send-keys -t "$SESSION_NAME:$VOICE_WINDOW" "$command" C-m
    echo "[agents] auto STT running in tmux window: $VOICE_WINDOW"
    wait_for_auto_stt_ready
    return
  fi

  if [[ -f "$AUTO_PID_FILE" ]]; then
    local existing_pid
    existing_pid="$(<"$AUTO_PID_FILE")"
    if process_is_running "$existing_pid"; then
      echo "[agents] auto STT already running: pid $existing_pid"
      wait_for_auto_stt_ready "$existing_pid"
      return
    fi
    rm -f "$AUTO_PID_FILE"
  fi

  mkdir -p "$(dirname "$AUTO_LOG")" "$(dirname "$AUTO_PID_FILE")"
  echo "[agents] starting auto STT in background; log: $AUTO_LOG"
  clear_auto_stt_ready
  nohup env \
    VOICE_READY_FILE="$AUTO_READY_FILE" \
    VOICE_CONFIG_PROMPT=0 \
    VOICE_AUTO_START_AGENT_WORKBENCH=0 \
    VOICE_AUTO_TMUX_SESSION="$SESSION_NAME" \
    VOICE_AUTO_TMUX_SWITCHES="$switches" \
    VOICE_AUTO_DISPLAY_WORDS="$words" \
    VOICE_AUTO_TRIGGER_WORD="$(normalize_spoken_name "$AGENT1_NAME")" \
    VOICE_AUTO_FOCUS_LOG="$AUTO_FOCUS_LOG" \
    "$ROOT/run-auto.sh" >"$AUTO_LOG" 2>&1 &
  echo "$!" >"$AUTO_PID_FILE"
  wait_for_auto_stt_ready "$!"
}

eval "$(load_workbench_config)"

SESSION_NAME="${SESSION_NAME:-$CONFIG_SESSION_NAME}"
AGENT_COMMAND="${AGENT_COMMAND:-$CONFIG_AGENT_COMMAND}"
AGENT_LAYOUT="${AGENT_LAYOUT:-$CONFIG_AGENT_LAYOUT}"
PANES_WINDOW="${PANES_WINDOW:-$CONFIG_PANES_WINDOW}"
AGENT1_NAME="${AGENT1_NAME:-$CONFIG_AGENT1_NAME}"
AGENT1_DIR="${1:-${AGENT1_DIR:-$CONFIG_AGENT1_DIR}}"
AGENT2_NAME="${AGENT2_NAME:-$CONFIG_AGENT2_NAME}"
AGENT2_DIR="${2:-${AGENT2_DIR:-$CONFIG_AGENT2_DIR}}"
AGENT3_NAME="${AGENT3_NAME:-$CONFIG_AGENT3_NAME}"
AGENT3_DIR="${3:-${AGENT3_DIR:-$CONFIG_AGENT3_DIR}}"
VOICE_NAME="${VOICE_NAME:-$CONFIG_VOICE_NAME}"
VOICE_DIR="${VOICE_DIR:-$CONFIG_VOICE_DIR}"
VOICE_SESSION="${VOICE_SESSION:-${SESSION_NAME}-voice}"
VOICE_WINDOW="${VOICE_WINDOW:-$VOICE_NAME}"
READY_SESSION_NAME="${SESSION_NAME//[^a-zA-Z0-9_.-]/_}"
AUTO_READY_FILE="${AUTO_READY_FILE:-${XDG_RUNTIME_DIR:-/tmp}/speech-agent-workbench-${READY_SESSION_NAME}-auto.ready}"

if [[ -t 0 && "${AGENTS_CONFIG_PROMPT:-1}" != "0" && "${AGENTS_CONFIG_PROMPT:-1}" != "off" ]]; then
  echo "Saved agent workbench config: $CONFIG_PATH"
  echo "  Agent command: $AGENT_COMMAND"
  if prompt_change "Change saved agent command?"; then
    AGENT_COMMAND="$(prompt_value "Agent command" "$AGENT_COMMAND")"
  fi
  echo "  $AGENT1_NAME path: $AGENT1_DIR"
  if prompt_change "Change first agent name/path?"; then
    AGENT1_NAME="$(prompt_value "First agent name" "$AGENT1_NAME")"
    AGENT1_DIR="$(prompt_value "First agent path" "$AGENT1_DIR")"
  fi
  echo "  $AGENT2_NAME path: $AGENT2_DIR"
  if prompt_change "Change second agent name/path?"; then
    AGENT2_NAME="$(prompt_value "Second agent name" "$AGENT2_NAME")"
    AGENT2_DIR="$(prompt_value "Second agent path" "$AGENT2_DIR")"
  fi
  echo "  $AGENT3_NAME path: $AGENT3_DIR"
  if prompt_change "Change third agent name/path?"; then
    AGENT3_NAME="$(prompt_value "Third agent name" "$AGENT3_NAME")"
    AGENT3_DIR="$(prompt_value "Third agent path" "$AGENT3_DIR")"
  fi
  echo "  $VOICE_NAME path: $VOICE_DIR"
  if prompt_change "Change voice pane name/path?"; then
    VOICE_NAME="$(prompt_value "Voice pane name" "$VOICE_NAME")"
    VOICE_DIR="$(prompt_value "Voice pane path" "$VOICE_DIR")"
  fi
fi
VOICE_WINDOW="${VOICE_WINDOW:-$VOICE_NAME}"
save_workbench_config

ensure_dir "$AGENT1_NAME" "$AGENT1_DIR"
ensure_dir "$AGENT2_NAME" "$AGENT2_DIR"
ensure_dir "$AGENT3_NAME" "$AGENT3_DIR"
ensure_dir "$VOICE_NAME" "$VOICE_DIR"

if ! command -v "${AGENT_COMMAND%% *}" >/dev/null 2>&1; then
  echo "Warning: '$AGENT_COMMAND' was not found on PATH. The tmux panes will open, but the agent command may not start." >&2
fi

case "$AGENT_LAYOUT" in
  panes)
    start_agent_panes
    ;;
  windows)
    if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
      tmux new-session -d -s "$SESSION_NAME" -n "$AGENT1_NAME" -c "$AGENT1_DIR"
      tmux send-keys -t "$SESSION_NAME:$AGENT1_NAME" "$AGENT_COMMAND" C-m
    else
      start_agent_window "$AGENT1_NAME" "$AGENT1_DIR"
    fi
    start_agent_window "$AGENT2_NAME" "$AGENT2_DIR"
    start_agent_window "$AGENT3_NAME" "$AGENT3_DIR"
    style_window "$AGENT1_NAME" "$AGENT1_NAME" "$AGENT1_STYLE"
    style_window "$AGENT2_NAME" "$AGENT2_NAME" "$AGENT2_STYLE"
    style_window "$AGENT3_NAME" "$AGENT3_NAME" "$AGENT3_STYLE"
    ;;
  *)
    echo "Unknown AGENT_LAYOUT '$AGENT_LAYOUT'. Use 'panes' or 'windows'." >&2
    exit 1
    ;;
esac
start_auto_stt

case "$ATTACH" in
  0|off|false|no)
    echo "[agents] session ready: $SESSION_NAME"
    exit 0
    ;;
esac

if [[ "$AGENT_LAYOUT" == "panes" ]]; then
  agent1_pane="$(get_pane_by_title "$AGENT1_NAME")"
  voice_pane="$(get_pane_by_title "$VOICE_NAME")"
  tmux select-window -t "$SESSION_NAME:$PANES_WINDOW"
  if [[ -n "$agent1_pane" ]]; then
    tmux select-pane -t "$agent1_pane"
  else
    tmux select-pane -t "$SESSION_NAME:$PANES_WINDOW.0"
  fi
else
  if tmux list-windows -t "$SESSION_NAME" -F '#W' 2>/dev/null | grep -Fxq "$VOICE_WINDOW"; then
    tmux select-window -t "$SESSION_NAME:$VOICE_WINDOW"
  else
    tmux select-window -t "$SESSION_NAME:$AGENT1_NAME"
  fi
fi
if [[ -n "${TMUX:-}" ]]; then
  tmux switch-client -t "$SESSION_NAME"
else
  tmux attach-session -t "$SESSION_NAME"
fi
