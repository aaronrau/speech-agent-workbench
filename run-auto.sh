#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${VOICE_RUN_AUTO_PLATFORM_DISPATCHED:-0}" != "1" ]]; then
  case "${VOICE_PLATFORM_OVERRIDE:-$(uname -s)}" in
    Linux)
      exec "$ROOT/scripts/linux/run-auto.sh" "$@"
      ;;
    Darwin)
      exec "$ROOT/scripts/macos/run-auto.sh" "$@"
      ;;
    *)
      echo "Unsupported operating system: ${VOICE_PLATFORM_OVERRIDE:-$(uname -s)}" >&2
      exit 1
      ;;
  esac
fi

export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin${PATH:+:$PATH}"

load_local_env() {
  local env_file="$ROOT/.env"
  [[ -f "$env_file" ]] || return 0

  local env_snapshot
  env_snapshot="$(mktemp "${TMPDIR:-/tmp}/speech-agent-workbench-env.XXXXXX")"
  export -p >"$env_snapshot"

  set -a
  # shellcheck disable=SC1091
  if ! source "$env_file"; then
    set +a
    rm -f "$env_snapshot"
    return 1
  fi
  set +a

  # Restore variables inherited from the caller so explicit shell values keep
  # precedence while values introduced only by .env remain available.
  # shellcheck disable=SC1090
  source "$env_snapshot"
  rm -f "$env_snapshot"
}

load_local_env

"$ROOT/scripts/ensure-tmux.sh"

CONFIG_PATH="${VOICE_HOTKEY_CONFIG:-$ROOT/config.json}"
export VOICE_HOTKEY_CONFIG="$CONFIG_PATH"
VAD_MODEL="${VOICE_AUTO_SHERPA_VAD_MODEL:-$ROOT/models/silero_vad.onnx}"
VAD_URL="${VOICE_AUTO_SHERPA_VAD_URL:-https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx}"

args=()
for arg in "$@"; do
  case "$arg" in
    --)
      ;;
    --disable-stt|--stt-disable|disable-stt|stt-disable)
      export VOICE_DISABLE_STT=1
      ;;
    *)
      args+=("$arg")
      ;;
  esac
done
if (( ${#args[@]} )); then
  set -- "${args[@]}"
else
  set --
fi

to_lower() {
  printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]'
}

load_auto_config_defaults() {
  python3 - "$CONFIG_PATH" "$ROOT" <<'PY'
import json
import os
import shlex
import sys

path, root = sys.argv[1:3]
config = {}
for candidate in (path, os.path.join(root, "config.example.json")):
    try:
        with open(candidate, "r", encoding="utf-8") as handle:
            config = json.load(handle)
        break
    except Exception:
        continue

settings = config.get("agent_workbench") or {}
legacy_settings = config.get("codex_agents") or {}
if not isinstance(settings, dict):
    settings = {}
if not isinstance(legacy_settings, dict):
    legacy_settings = {}

def get_setting(name):
    value = settings.get(name)
    if value in (None, ""):
        value = legacy_settings.get(name)
    return value or ""

agents = settings.get("agents") or legacy_settings.get("agents") or []
voice = settings.get("voice") or legacy_settings.get("voice") or {}

def item_name(items, index):
    if len(items) > index and isinstance(items[index], dict):
        return items[index].get("name") or ""
    return ""

def list_value(value):
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value if str(item).strip())
    return value or ""

values = {
    "CONFIG_SESSION_NAME": get_setting("session_name"),
    "CONFIG_AGENT_LAYOUT": get_setting("layout"),
    "CONFIG_PANES_WINDOW": get_setting("panes_window"),
    "CONFIG_AGENT1_NAME": item_name(agents, 0),
    "CONFIG_AGENT2_NAME": item_name(agents, 1),
    "CONFIG_AGENT3_NAME": item_name(agents, 2),
    "CONFIG_VOICE_NAME": voice.get("name") or "",
    "CONFIG_TRIGGER_ALIASES": list_value(config.get("auto_trigger_aliases")),
    "CONFIG_ENABLE_TERMINATE_COMMANDS": config.get(
        "auto_enable_terminate_commands", ""
    ),
}
for key, value in values.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
}

eval "$(load_auto_config_defaults)"

normalize_spoken_name() {
  local name="$1"
  name="$(to_lower "$name")"
  name="${name//[^a-z0-9]/ }"
  name="${name#"${name%%[![:space:]]*}"}"
  name="${name%"${name##*[![:space:]]}"}"
  printf '%s' "$name"
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

get_pane_by_title() {
  local title="$1"
  tmux list-panes -t "$AUTO_TMUX_SESSION:$AUTO_PANES_WINDOW" -F '#{@agent_name} #{pane_id}' 2>/dev/null |
    while read -r agent_name pane_id; do
      if [[ "$agent_name" == "$title" ]]; then
        echo "$pane_id"
        return
      fi
    done
}

display_words() {
  printf '%s=32,%s=94,%s=38;5;208,%s=90' \
    "$(normalize_spoken_name "$AUTO_AGENT1_NAME")" \
    "$(normalize_spoken_name "$AUTO_AGENT2_NAME")" \
    "$(normalize_spoken_name "$AUTO_AGENT3_NAME")" \
    "$(normalize_spoken_name "$AUTO_VOICE_NAME")"
}

terminate_words_for_voice() {
  local voice_word="$1"
  printf '%s terminate session,%s terminates session,%s terminate sessions,%s terminates sessions' \
    "$voice_word" "$voice_word" "$voice_word" "$voice_word"
}

auto_terminate_enabled() {
  local value="${VOICE_AUTO_ENABLE_TERMINATE_COMMANDS:-${CONFIG_ENABLE_TERMINATE_COMMANDS:-0}}"
  case "$(to_lower "$value")" in
    1|true|yes|on)
      return 0
      ;;
  esac
  return 1
}

prefetch_voice_models() {
  case "${VOICE_AUTO_PREFETCH_MODELS:-1}" in
    0|false|no|none|null|off)
      return
      ;;
  esac

  if [[ "${VOICE_DISABLE_STT:-0}" == "1" ]]; then
    echo "[auto] STT disabled; checking non-STT runtime assets before starting workbench..."
  else
    echo "[auto] checking voice model downloads before starting workbench..."
  fi
  VOICE_AUTO_START_AGENT_WORKBENCH=0 VOICE_PREFETCH_ONLY=1 "$ROOT/run.sh"
}

default_switches() {
  local agent1_word
  local agent2_word
  local agent3_word
  local voice_word
  agent1_word="$(normalize_spoken_name "$AUTO_AGENT1_NAME")"
  agent2_word="$(normalize_spoken_name "$AUTO_AGENT2_NAME")"
  agent3_word="$(normalize_spoken_name "$AUTO_AGENT3_NAME")"
  voice_word="$(normalize_spoken_name "$AUTO_VOICE_NAME")"

  if [[ "$AUTO_LAYOUT" == "panes" ]]; then
    local agent1_pane
    local agent2_pane
    local agent3_pane
    local voice_pane
    agent1_pane="$(get_pane_by_title "$AUTO_AGENT1_NAME")"
    agent2_pane="$(get_pane_by_title "$AUTO_AGENT2_NAME")"
    agent3_pane="$(get_pane_by_title "$AUTO_AGENT3_NAME")"
    voice_pane="$(get_pane_by_title "$AUTO_VOICE_NAME")"
    if [[ -n "$agent1_pane" && -n "$agent2_pane" && -n "$agent3_pane" && -n "$voice_pane" ]]; then
      build_switch_map \
        "$agent1_word" "pane:$agent1_pane" \
        "$agent2_word" "pane:$agent2_pane" \
        "$agent3_word" "pane:$agent3_pane" \
        "$voice_word" "pane:$voice_pane"
      return
    fi
  fi

  build_switch_map \
    "$agent1_word" "$AUTO_AGENT1_NAME" \
    "$agent2_word" "$AUTO_AGENT2_NAME" \
    "$agent3_word" "$AUTO_AGENT3_NAME" \
    "$voice_word" "$AUTO_VOICE_WINDOW"
}

AUTO_AGENT1_NAME="${AGENT1_NAME:-$CONFIG_AGENT1_NAME}"
AUTO_AGENT2_NAME="${AGENT2_NAME:-$CONFIG_AGENT2_NAME}"
AUTO_AGENT3_NAME="${AGENT3_NAME:-$CONFIG_AGENT3_NAME}"
AUTO_VOICE_NAME="${VOICE_NAME:-$CONFIG_VOICE_NAME}"
AUTO_LAYOUT="${AGENT_LAYOUT:-$CONFIG_AGENT_LAYOUT}"
AUTO_PANES_WINDOW="${PANES_WINDOW:-$CONFIG_PANES_WINDOW}"
AUTO_VOICE_WINDOW="${VOICE_WINDOW:-$AUTO_VOICE_NAME}"
AUTO_TMUX_SESSION="${VOICE_AUTO_TMUX_SESSION:-${SESSION_NAME:-$CONFIG_SESSION_NAME}}"
AUTO_READY_SESSION_NAME="${AUTO_TMUX_SESSION//[^a-zA-Z0-9_.-]/_}"

export VOICE_RUN_MODE=auto
export VOICE_CONFIG_PROMPT="${VOICE_CONFIG_PROMPT:-0}"
export VOICE_AUTO_TRIGGER_WORD="${VOICE_AUTO_TRIGGER_WORD:-$(normalize_spoken_name "$AUTO_AGENT1_NAME")}"
export VOICE_AUTO_TRIGGER_ALIASES="${VOICE_AUTO_TRIGGER_ALIASES-$CONFIG_TRIGGER_ALIASES}"
export VOICE_AUTO_TRIGGER_PROBE_SECONDS="${VOICE_AUTO_TRIGGER_PROBE_SECONDS:-0.5}"
export VOICE_AUTO_TRIGGER_MIN_PROBE_SECONDS="${VOICE_AUTO_TRIGGER_MIN_PROBE_SECONDS:-1}"
export VOICE_AUTO_TRIGGER_PROBE_WINDOW_SECONDS="${VOICE_AUTO_TRIGGER_PROBE_WINDOW_SECONDS:-1.5}"
export VOICE_AUTO_TRIGGER_SILENCE_SECONDS="${VOICE_AUTO_TRIGGER_SILENCE_SECONDS:-2}"
export VOICE_AUTO_START_SPEECH_MS="${VOICE_AUTO_START_SPEECH_MS:-60}"
export VOICE_AUTO_PRE_ROLL_SECONDS="${VOICE_AUTO_PRE_ROLL_SECONDS:-1.5}"
export VOICE_AUTO_VAD_BACKEND="${VOICE_AUTO_VAD_BACKEND:-sherpa}"
export VOICE_AUTO_SHERPA_VAD_MODEL="$VAD_MODEL"
export VOICE_AUTO_TMUX_SESSION="$AUTO_TMUX_SESSION"
export VOICE_AUTO_FOCUS_LOG="${VOICE_AUTO_FOCUS_LOG:-${XDG_RUNTIME_DIR:-/tmp}/speech-agent-workbench-focus.log}"
export VOICE_READY_FILE="${VOICE_READY_FILE:-${XDG_RUNTIME_DIR:-/tmp}/speech-agent-workbench-${AUTO_READY_SESSION_NAME:-auto}-auto.ready}"
export VOICE_AUTO_ENABLE_TERMINATE_COMMANDS="${VOICE_AUTO_ENABLE_TERMINATE_COMMANDS:-${CONFIG_ENABLE_TERMINATE_COMMANDS:-}}"

case "${VOICE_AUTO_START_AGENT_WORKBENCH:-1}" in
  1|true|yes|on)
    if [[ -x "$ROOT/start-agent-workbench.sh" ]]; then
      prefetch_voice_models
      exec env SESSION_NAME="$AUTO_TMUX_SESSION" "$ROOT/start-agent-workbench.sh" "$@"
    else
      echo "[auto] start-agent-workbench.sh is missing or not executable; skipping agent startup." >&2
    fi
    ;;
esac

export VOICE_AUTO_TMUX_SWITCHES="${VOICE_AUTO_TMUX_SWITCHES:-$(default_switches)}"
export VOICE_AUTO_DISPLAY_WORDS="${VOICE_AUTO_DISPLAY_WORDS:-$(display_words)}"
if auto_terminate_enabled; then
  export VOICE_AUTO_TMUX_TERMINATE_WORDS="${VOICE_AUTO_TMUX_TERMINATE_WORDS:-$(terminate_words_for_voice "$(normalize_spoken_name "$AUTO_VOICE_NAME")")}"
else
  unset VOICE_AUTO_TMUX_TERMINATE_WORDS
fi

case "${VOICE_AUTO_VAD_BACKEND}" in
  sherpa|silero|silero-vad|sherpa-vad)
    if [[ ! -f "$VAD_MODEL" && "${VOICE_AUTO_VAD_DOWNLOAD:-1}" != "0" && "${VOICE_AUTO_VAD_DOWNLOAD:-1}" != "off" ]]; then
      mkdir -p "$(dirname "$VAD_MODEL")"
      tmp_model="${VAD_MODEL}.tmp.$$"
      if command -v curl >/dev/null 2>&1; then
        if ! curl -L --fail -o "$tmp_model" "$VAD_URL"; then
          rm -f "$tmp_model"
          echo "[auto] unable to download Sherpa VAD; falling back in app." >&2
        fi
      elif command -v wget >/dev/null 2>&1; then
        if ! wget -O "$tmp_model" "$VAD_URL"; then
          rm -f "$tmp_model"
          echo "[auto] unable to download Sherpa VAD; falling back in app." >&2
        fi
      else
        echo "[auto] curl or wget is required to download Sherpa VAD; falling back in app." >&2
      fi
      if [[ -f "$tmp_model" ]]; then
        mv "$tmp_model" "$VAD_MODEL"
      fi
    fi
    ;;
esac

exec "$ROOT/run.sh"
