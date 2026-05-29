#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VAD_MODEL="${VOICE_AUTO_SHERPA_VAD_MODEL:-$ROOT/models/silero_vad.onnx}"
VAD_URL="${VOICE_AUTO_SHERPA_VAD_URL:-https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx}"
AUTO_TMUX_SESSION="${VOICE_AUTO_TMUX_SESSION:-${SESSION_NAME:-speech-agent-workbench}}"

export VOICE_RUN_MODE=auto
export VOICE_AUTO_TRIGGER_WORD="${VOICE_AUTO_TRIGGER_WORD:-agent}"
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

case "${VOICE_AUTO_START_AGENT_WORKBENCH:-1}" in
  1|true|yes|on)
    if [[ -x "$ROOT/start-agent-workbench.sh" ]]; then
      exec env SESSION_NAME="$AUTO_TMUX_SESSION" "$ROOT/start-agent-workbench.sh" "$@"
    else
      echo "[auto] start-agent-workbench.sh is missing or not executable; skipping agent startup." >&2
    fi
    ;;
esac

export VOICE_AUTO_TMUX_SWITCHES="${VOICE_AUTO_TMUX_SWITCHES:-agent=Agent 1,agent one=Agent 1,agent two=Agent 2,agent three=Agent 3,voice=Voice}"

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
