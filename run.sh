#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${VOICE_RUN_PLATFORM_DISPATCHED:-0}" != "1" ]]; then
  case "${VOICE_PLATFORM_OVERRIDE:-$(uname -s)}" in
    Linux)
      exec "$ROOT/scripts/linux/run.sh" "$@"
      ;;
    Darwin)
      exec "$ROOT/scripts/macos/run.sh" "$@"
      ;;
    *)
      echo "Unsupported operating system: ${VOICE_PLATFORM_OVERRIDE:-$(uname -s)}" >&2
      exit 1
      ;;
  esac
fi

VENV_ROOT="${VOICE_VENV:-$ROOT/.venv}"
export PATH="$VENV_ROOT/bin:/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin${PATH:+:$PATH}"
PYTHON_BIN="$VENV_ROOT/bin/python"
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

export PYTHONUNBUFFERED=1
export VOICE_HOTKEY_CONFIG="${VOICE_HOTKEY_CONFIG:-$ROOT/config.json}"
export VOICE_REMOTE_URL="${VOICE_REMOTE_URL:-http://127.0.0.1:8765/transcribe}"
export VOICE_DEFAULT_TRANSCRIBE_BACKEND="${VOICE_DEFAULT_TRANSCRIBE_BACKEND:-parakeet-onnx}"
export VOICE_FALLBACK_BACKEND="${VOICE_FALLBACK_BACKEND:-parakeet-onnx}"
export VOICE_RUN_MODE=auto
export VOICE_AUTO_TRIGGER_WORD="${VOICE_AUTO_TRIGGER_WORD:-agent}"
export VOICE_AUTO_TRIGGER_PROBE_SECONDS="${VOICE_AUTO_TRIGGER_PROBE_SECONDS:-0.5}"
export VOICE_AUTO_TRIGGER_MIN_PROBE_SECONDS="${VOICE_AUTO_TRIGGER_MIN_PROBE_SECONDS:-1}"
export VOICE_AUTO_TRIGGER_PROBE_WINDOW_SECONDS="${VOICE_AUTO_TRIGGER_PROBE_WINDOW_SECONDS:-1.5}"
export VOICE_AUTO_TRIGGER_SILENCE_SECONDS="${VOICE_AUTO_TRIGGER_SILENCE_SECONDS:-1}"
export VOICE_AUTO_START_SPEECH_MS="${VOICE_AUTO_START_SPEECH_MS:-60}"
export VOICE_AUTO_PRE_ROLL_SECONDS="${VOICE_AUTO_PRE_ROLL_SECONDS:-1.5}"
export VOICE_AUTO_VAD_BACKEND="${VOICE_AUTO_VAD_BACKEND:-sherpa}"
export VOICE_AUTO_SHERPA_VAD_MODEL="$VAD_MODEL"

is_wayland() {
  [ -n "${WAYLAND_DISPLAY:-}" ] || [ "${XDG_SESSION_TYPE:-}" = "wayland" ]
}

to_lower() {
  printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]'
}

check_ydotoold() {
  if ! command -v ydotool >/dev/null 2>&1; then
    return 0
  fi

  if ! command -v ydotoold >/dev/null 2>&1; then
    echo "[run] ydotoold not installed (needed for Wayland auto-typing)." >&2
    echo "[run] install: sudo apt-get install -y ydotoold" >&2
    return 0
  fi

  if ! pgrep -x ydotoold >/dev/null 2>&1; then
    if [ "${VOICE_AUTO_START_YDOTOOLD:-0}" = "1" ]; then
      echo "[run] starting ydotoold (requires sudo)..." >&2
      sudo ydotoold --socket-path=/tmp/.ydotool_socket >/dev/null 2>&1 &
      sleep 0.2
    else
      echo "[run] ydotoold not running; auto-typing may fail." >&2
      echo "[run] start: sudo ydotoold --socket-path=/tmp/.ydotool_socket" >&2
      echo "[run] or set VOICE_PASTE_MODE=clipboard" >&2
    fi
  fi

  if [ ! -w /dev/uinput ]; then
    echo "[run] no write access to /dev/uinput; ydotool may fail." >&2
    if [ ! -f /etc/udev/rules.d/99-uinput.rules ]; then
      echo "[run] create uinput rule:" >&2
      echo "[run] sudo sh -c 'printf \"KERNEL==\\\"uinput\\\", GROUP=\\\"input\\\", MODE=\\\"0660\\\"\\n\" > /etc/udev/rules.d/99-uinput.rules'" >&2
      echo "[run] then: sudo udevadm control --reload-rules && sudo udevadm trigger" >&2
    fi
  fi
}

ensure_python_env() {
  local install_requirements=0

  if [[ -x "$PYTHON_BIN" ]] && ! "$PYTHON_BIN" -c 'import sys' >/dev/null 2>&1; then
    echo "[run] Python venv is incompatible with this platform: $VENV_ROOT" >&2
    echo "[run] repairing it with the selected python3 interpreter..." >&2
    python3 -m venv --clear "$VENV_ROOT"
    install_requirements=1
  fi

  if [[ ! -x "$PYTHON_BIN" ]]; then
    case "${VOICE_CREATE_VENV:-1}" in
      0|false|no|none|null|off)
        echo "[run] Python venv missing: $PYTHON_BIN" >&2
        echo "[run] run ./install.sh or unset VOICE_CREATE_VENV=off." >&2
        exit 1
        ;;
    esac

    if ! command -v python3 >/dev/null 2>&1; then
      echo "[run] python3 not found. Install Python 3.10+ and retry." >&2
      exit 1
    fi

    echo "[run] creating Python venv: $VENV_ROOT" >&2
    if ! python3 -m venv "$VENV_ROOT"; then
      if [[ "${VOICE_PLATFORM:-linux}" == "macos" ]]; then
        echo "[run] unable to create venv. Install Python with: brew install python" >&2
      else
        echo "[run] unable to create venv. On Ubuntu, install python3-venv:" >&2
        echo "[run]   sudo apt-get install -y python3-venv" >&2
      fi
      exit 1
    fi
    install_requirements=1
  fi

  if [[ "$install_requirements" != "1" ]]; then
    if ! "$PYTHON_BIN" - <<'PY'
import importlib.util
import sys

required = ("numpy", "sounddevice", "soundfile", "pynput", "onnx_asr", "sherpa_onnx")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    print("[run] missing Python dependencies: " + ", ".join(missing), file=sys.stderr)
    raise SystemExit(1)
PY
    then
      install_requirements=1
    fi
  fi

  if [[ "$install_requirements" == "1" ]]; then
    echo "[run] installing default Python runtime dependencies..." >&2
    "$PYTHON_BIN" -m pip install --upgrade pip
    case "${VOICE_INSTALL_FULL_REQUIREMENTS:-0}" in
      1|true|yes|on)
        "$PYTHON_BIN" -m pip install -r "$ROOT/requirements.txt"
        ;;
      *)
        "$PYTHON_BIN" -m pip install \
          numpy \
          sounddevice \
          soundfile \
          pynput \
          'onnx-asr[cpu,hub]' \
          sherpa-onnx
        ;;
    esac
  fi
}

ensure_faster_whisper() {
  local py="$PYTHON_BIN"
  if "$py" - <<'PY'
import json
import os
import sys

path = os.environ.get("VOICE_HOTKEY_CONFIG")
try:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
except Exception:
    sys.exit(0)

backend = (os.environ.get("VOICE_TRANSCRIBE_BACKEND") or cfg.get("transcribe_backend") or "").strip().lower()
if backend != "faster-whisper":
    sys.exit(0)

try:
    import faster_whisper  # noqa: F401
except Exception:
    sys.exit(1)
sys.exit(0)
PY
  then
    return 0
  fi

  echo "[run] installing faster-whisper..." >&2
  "$PYTHON_BIN" -m pip install faster-whisper
}

ensure_nemo_canary() {
  local py="$PYTHON_BIN"
  if "$py" - <<'PY'
import json
import os
import sys

path = os.environ.get("VOICE_HOTKEY_CONFIG")
try:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
except Exception:
    sys.exit(0)

backend = (os.environ.get("VOICE_TRANSCRIBE_BACKEND") or cfg.get("transcribe_backend") or "").strip().lower()
if backend not in ("nemo", "canary", "nemo-canary"):
    sys.exit(0)

try:
    import nemo  # noqa: F401
except Exception:
    sys.exit(1)
sys.exit(0)
PY
  then
    return 0
  fi

  echo "[run] installing nemo (canary)..." >&2
  "$PYTHON_BIN" -m pip install "nemo_toolkit[asr,tts] @ git+https://github.com/NVIDIA/NeMo.git"
}

ensure_parakeet_onnx() {
  local py="$PYTHON_BIN"
  if "$py" - <<'PY'
import json
import os
import sys

path = os.environ.get("VOICE_HOTKEY_CONFIG")
try:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
except Exception:
    cfg = {}

backend = (os.environ.get("VOICE_TRANSCRIBE_BACKEND") or cfg.get("transcribe_backend") or "").strip().lower()
fallback = (os.environ.get("VOICE_FALLBACK_BACKEND") or cfg.get("fallback_backend") or "").strip().lower()
if backend not in ("parakeet", "parakeet-onnx") and fallback not in ("parakey", "parakeet", "parakeet-onnx"):
    sys.exit(0)

try:
    import onnx_asr  # noqa: F401
except Exception:
    sys.exit(1)
sys.exit(0)
PY
  then
    return 0
  fi

  echo "[run] installing onnx-asr (parakeet-onnx)..." >&2
  "$PYTHON_BIN" -m pip install 'onnx-asr[cpu,hub]'
}

ensure_parakeet_onnx_model() {
  case "${VOICE_PARAKEET_ONNX_DOWNLOAD:-1}" in
    0|false|no|none|null|off)
      return 0
      ;;
  esac

  "$PYTHON_BIN" - "$ROOT" <<'PY'
import json
import os
import sys

root = sys.argv[1]
sys.path.insert(0, root)


def normalize_backend(value):
    value = str(value or "").strip().lower()
    if value in ("parakey", "parakeet"):
        return "parakeet-onnx"
    return value


path = os.environ.get("VOICE_HOTKEY_CONFIG")
try:
    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
except Exception:
    config = {}

backend = normalize_backend(
    os.environ.get("VOICE_TRANSCRIBE_BACKEND") or config.get("transcribe_backend")
)
if backend != "parakeet-onnx":
    raise SystemExit(0)

import app

model_name = app.get_parakeet_onnx_model(config)
quantization = app.get_parakeet_onnx_quantization(config)
print("[run] ensuring parakeet-onnx model is downloaded...", flush=True)
app.load_parakeet_onnx_model(model_name, quantization)
print("[run] parakeet-onnx model ready.", flush=True)
PY
}

maybe_enable_remote_backend() {
  if [ -n "${VOICE_TRANSCRIBE_BACKEND:-}" ]; then
    return 0
  fi

  if [ "${VOICE_REMOTE_AUTO:-1}" != "1" ]; then
    return 0
  fi

  local health_url="${VOICE_REMOTE_URL%/*}/health"
  if ! "$PYTHON_BIN" - <<'PY'
import os
import sys
import urllib.error
import urllib.request

url = os.environ.get("VOICE_REMOTE_URL", "http://127.0.0.1:8765/transcribe")
health_url = url.rsplit("/", 1)[0] + "/health"
timeout = float(os.environ.get("VOICE_REMOTE_PROBE_TIMEOUT", "2"))

try:
    with urllib.request.urlopen(health_url, timeout=timeout) as resp:
        status = getattr(resp, "status", 200)
    raise SystemExit(0 if status == 200 else 1)
except (urllib.error.URLError, TimeoutError, ValueError):
    raise SystemExit(1)
except Exception:
    raise SystemExit(1)
PY
  then
    echo "[run] remote STT not ready at $health_url; using local backend." >&2
    return 0
  fi

  export VOICE_TRANSCRIBE_BACKEND=remote
  echo "[run] preferring remote STT at $VOICE_REMOTE_URL" >&2
}

select_default_transcribe_backend() {
  if [ -n "${VOICE_TRANSCRIBE_BACKEND:-}" ]; then
    return 0
  fi

  case "$(to_lower "$VOICE_DEFAULT_TRANSCRIBE_BACKEND")" in
    ""|0|false|no|none|null|off)
      return 0
      ;;
  esac

  local configured_backend
  configured_backend="$("$PYTHON_BIN" - <<'PY'
import json
import os

path = os.environ.get("VOICE_HOTKEY_CONFIG")
try:
    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
except Exception:
    config = {}

value = config.get("transcribe_backend")
print(str(value).strip().lower() if value is not None else "")
PY
)"
  case "$configured_backend" in
    ""|whisper|sherpa|vosk|parakey|parakeet|parakeet-onnx)
      ;;
    *)
      echo "[run] config backend -> ${configured_backend}" >&2
      return 0
      ;;
  esac

  case "$(to_lower "$VOICE_DEFAULT_TRANSCRIBE_BACKEND")" in
    parakey|parakeet)
      export VOICE_TRANSCRIBE_BACKEND="parakeet-onnx"
      ;;
    *)
      export VOICE_TRANSCRIBE_BACKEND="$VOICE_DEFAULT_TRANSCRIBE_BACKEND"
      ;;
  esac
  echo "[run] default local backend -> ${VOICE_TRANSCRIBE_BACKEND}" >&2
}

log_backend_fallback() {
  if [ -z "${VOICE_FALLBACK_BACKEND:-}" ]; then
    return 0
  fi
  case "$(to_lower "$VOICE_FALLBACK_BACKEND")" in
    0|false|no|none|null|off)
      echo "[run] backend fallback disabled." >&2
      ;;
    *)
      echo "[run] backend fallback -> ${VOICE_FALLBACK_BACKEND}" >&2
      ;;
  esac
}

ensure_auto_vad_model() {
  case "${VOICE_AUTO_VAD_BACKEND}" in
    sherpa|silero|silero-vad|sherpa-vad)
      if [[ ! -f "$VAD_MODEL" && "${VOICE_AUTO_VAD_DOWNLOAD:-1}" != "0" && "${VOICE_AUTO_VAD_DOWNLOAD:-1}" != "off" ]]; then
        mkdir -p "$(dirname "$VAD_MODEL")"
        tmp_model="${VAD_MODEL}.tmp.$$"
        if command -v curl >/dev/null 2>&1; then
          if ! curl -L --fail -o "$tmp_model" "$VAD_URL"; then
            rm -f "$tmp_model"
            echo "[run] unable to download Sherpa VAD; falling back in app." >&2
          fi
        elif command -v wget >/dev/null 2>&1; then
          if ! wget -O "$tmp_model" "$VAD_URL"; then
            rm -f "$tmp_model"
            echo "[run] unable to download Sherpa VAD; falling back in app." >&2
          fi
        else
          echo "[run] curl or wget is required to download Sherpa VAD; falling back in app." >&2
        fi
        if [[ -f "$tmp_model" ]]; then
          mv "$tmp_model" "$VAD_MODEL"
        fi
      fi
      ;;
  esac
}

ensure_transcript_correction_assets() {
  local exports
  exports="$("$PYTHON_BIN" - "$ROOT" <<'PY'
import json
import os
import shlex
import shutil
import sys

root = sys.argv[1]
path = os.environ.get("VOICE_HOTKEY_CONFIG")
try:
    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
except Exception:
    config = {}

backend = os.environ.get("VOICE_TRANSCRIPT_CORRECTION_BACKEND")
if backend is None:
    backend = config.get("transcript_correction_backend", "off")
backend = str(backend or "").strip().lower()
if backend in ("gemma", "gemma4", "gemma-4", "llama", "llamacpp", "llama.cpp"):
    backend = "llama-cpp"
correction_uses_llama = backend == "llama-cpp"
correction_is_off = backend in ("", "0", "false", "no", "none", "null", "off")
if not correction_is_off and not correction_uses_llama:
    print(
        f"[run] transcript correction backend '{backend}' is not managed by run.sh.",
        file=sys.stderr,
    )


def config_bool(env_name, config_name, default):
    value = os.environ.get(env_name)
    if value is None:
        value = config.get(config_name, default)
    return str(value or "").strip().lower() in ("1", "true", "yes", "on")


summary_enabled = config_bool(
    "VOICE_AUTO_TMUX_SUMMARY_ENABLED",
    "auto_tmux_summary_enabled",
    True,
)
if not correction_uses_llama and not summary_enabled:
    raise SystemExit(0)


def configured_path(env_name, config_name, default):
    value = os.environ.get(env_name)
    if value in (None, ""):
        value = config.get(config_name, default)
    return os.path.expanduser(str(value or "").strip())


def resolve_path(value, *, executable=False):
    if not value:
        return ""
    if os.path.isabs(value):
        return value
    if os.sep not in value:
        found = shutil.which(value)
        if found:
            return found
    return os.path.join(root, value)


binary = resolve_path(
    configured_path(
        "VOICE_TRANSCRIPT_CORRECTION_LLAMA_CPP_PATH",
        "transcript_correction_llama_cpp_path",
        "llama-cli",
    ),
    executable=True,
)
model = resolve_path(
    configured_path(
        "VOICE_TRANSCRIPT_CORRECTION_LLAMA_CPP_MODEL",
        "transcript_correction_llama_cpp_model",
        "models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q4_0.gguf",
    )
)
model_repo = configured_path(
    "VOICE_TRANSCRIPT_CORRECTION_LLAMA_CPP_MODEL_REPO",
    "transcript_correction_llama_cpp_model_repo",
    "ggml-org/gemma-4-E2B-it-GGUF",
)
model_download = config_bool(
    "VOICE_TRANSCRIPT_CORRECTION_LLAMA_CPP_MODEL_DOWNLOAD",
    "transcript_correction_llama_cpp_model_download",
    True,
)

print("export VOICE_LLAMA_ASSETS_REQUIRED=1")
print(
    "export VOICE_LLAMA_ASSET_BINARY="
    + shlex.quote(binary)
)
print(
    "export VOICE_LLAMA_ASSET_MODEL="
    + shlex.quote(model)
)
print(
    "export VOICE_LLAMA_ASSET_MODEL_REPO="
    + shlex.quote(model_repo)
)
print(
    "export VOICE_LLAMA_ASSET_MODEL_DOWNLOAD="
    + ("1" if model_download else "0")
)
PY
)"
  if [[ -n "$exports" ]]; then
    eval "$exports"
  fi
  if [[ "${VOICE_LLAMA_ASSETS_REQUIRED:-0}" != "1" ]]; then
    return 0
  fi
  if [[ ! -x "${VOICE_LLAMA_ASSET_BINARY:-}" ]]; then
    echo "[run] llama.cpp binary is missing or not executable: ${VOICE_LLAMA_ASSET_BINARY:-<unset>}" >&2
    return 1
  fi
  if [[ ! -f "${VOICE_LLAMA_ASSET_MODEL:-}" ]]; then
    if is_truthy "${VOICE_LLAMA_ASSET_MODEL_DOWNLOAD:-0}"; then
      "$PYTHON_BIN" "$ROOT/scripts/prefetch_llama_cpp_model.py" \
        --model "$VOICE_LLAMA_ASSET_MODEL" \
        --repo "$VOICE_LLAMA_ASSET_MODEL_REPO"
    else
      echo "[run] llama.cpp model is missing and automatic download is disabled: ${VOICE_LLAMA_ASSET_MODEL:-<unset>}" >&2
      return 1
    fi
  fi
  if [[ ! -f "${VOICE_LLAMA_ASSET_MODEL:-}" ]]; then
    echo "[run] llama.cpp model download did not create: ${VOICE_LLAMA_ASSET_MODEL:-<unset>}" >&2
    return 1
  fi
  export VOICE_TRANSCRIPT_CORRECTION_LLAMA_CPP_PATH="$VOICE_LLAMA_ASSET_BINARY"
  export VOICE_TRANSCRIPT_CORRECTION_LLAMA_CPP_MODEL="$VOICE_LLAMA_ASSET_MODEL"
  echo "[run] llama.cpp binary: $VOICE_TRANSCRIPT_CORRECTION_LLAMA_CPP_PATH" >&2
  echo "[run] llama.cpp model: $VOICE_TRANSCRIPT_CORRECTION_LLAMA_CPP_MODEL" >&2
}

is_truthy() {
  case "${1:-}" in
    1|true|yes|on)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

if is_wayland; then
  check_ydotoold
fi

ensure_python_env
if is_truthy "${VOICE_DISABLE_STT:-0}"; then
  echo "[run] STT disabled; skipping STT backend and VAD setup." >&2
else
  maybe_enable_remote_backend
  select_default_transcribe_backend
  ensure_faster_whisper
  ensure_nemo_canary
  ensure_parakeet_onnx
  ensure_parakeet_onnx_model
  log_backend_fallback
  ensure_auto_vad_model
fi
ensure_transcript_correction_assets
if is_truthy "${VOICE_PREFETCH_ONLY:-0}"; then
  echo "[run] model prefetch complete." >&2
  exit 0
fi

exec "$PYTHON_BIN" "$ROOT/app.py" "$@"
