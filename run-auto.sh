#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${VOICE_HOTKEY_CONFIG:-$ROOT/config.json}"
VAD_MODEL="${VOICE_AUTO_SHERPA_VAD_MODEL:-$ROOT/models/silero_vad.onnx}"
VAD_URL="${VOICE_AUTO_SHERPA_VAD_URL:-https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx}"

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
agents = settings.get("agents") or []
voice = settings.get("voice") or {}

def item_name(items, index):
    if len(items) > index and isinstance(items[index], dict):
        return items[index].get("name") or ""
    return ""

values = {
    "CONFIG_SESSION_NAME": settings.get("session_name") or "",
    "CONFIG_AGENT_LAYOUT": settings.get("layout") or "",
    "CONFIG_PANES_WINDOW": settings.get("panes_window") or "",
    "CONFIG_AGENT1_NAME": item_name(agents, 0),
    "CONFIG_AGENT2_NAME": item_name(agents, 1),
    "CONFIG_AGENT3_NAME": item_name(agents, 2),
    "CONFIG_VOICE_NAME": voice.get("name") or "",
}
for key, value in values.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
}

eval "$(load_auto_config_defaults)"

normalize_spoken_name() {
  local name="$1"
  name="${name,,}"
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

prefetch_voice_models() {
  case "${VOICE_AUTO_PREFETCH_MODELS:-1}" in
    0|false|no|none|null|off)
      return
      ;;
  esac

  echo "[auto] checking voice model downloads before starting workbench..."
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

export VOICE_RUN_MODE=auto
export VOICE_AUTO_TRIGGER_WORD="${VOICE_AUTO_TRIGGER_WORD:-$(normalize_spoken_name "$AUTO_AGENT1_NAME")}"
export VOICE_AUTO_TRIGGER_ALIASES="${VOICE_AUTO_TRIGGER_ALIASES:-}"
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
export VOICE_AUTO_TMUX_TERMINATE_WORDS="${VOICE_AUTO_TMUX_TERMINATE_WORDS:-$(terminate_words_for_voice "$(normalize_spoken_name "$AUTO_VOICE_NAME")")}"

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
