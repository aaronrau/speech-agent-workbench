#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${VOICE_INSTALL_PLATFORM_DISPATCHED:-0}" != "1" ]]; then
  case "${VOICE_PLATFORM_OVERRIDE:-$(uname -s)}" in
    Linux)
      exec "$ROOT/scripts/linux/install.sh" "$@"
      ;;
    Darwin)
      exec "$ROOT/scripts/macos/install.sh" "$@"
      ;;
    *)
      echo "Unsupported operating system: ${VOICE_PLATFORM_OVERRIDE:-$(uname -s)}" >&2
      exit 1
      ;;
  esac
fi

VENV_DIR="${VOICE_VENV:-$ROOT/.venv}"

install_system_deps() {
  if [[ "${VOICE_PLATFORM:-linux}" == "macos" ]]; then
    install_macos_system_deps
    return
  fi

  if ! command -v apt-get >/dev/null 2>&1; then
    cat >&2 <<'EOF'
apt-get not found. Install these packages manually:
  python3 python3-venv portaudio19-dev ffmpeg tmux build-essential cmake git
  xdotool xclip wtype wl-clipboard ydotool
EOF
    return 0
  fi

  sudo apt-get update
  sudo apt-get install -y \
    python3 \
    python3-venv \
    portaudio19-dev \
    ffmpeg \
    tmux \
    build-essential \
    cmake \
    git

  for pkg in xdotool xclip wtype wl-clipboard ydotool; do
    if ! sudo apt-get install -y "$pkg"; then
      echo "Optional package not installed: $pkg" >&2
    fi
  done
}

install_macos_system_deps() {
  if ! command -v brew >/dev/null 2>&1; then
    cat >&2 <<'EOF'
Homebrew is required to install macOS dependencies.
Install it from https://brew.sh and retry, or run with INSTALL_SYSTEM_DEPS=0
after manually installing Python 3.10+, tmux, ffmpeg, and PortAudio.
EOF
    return 1
  fi

  local packages=""
  command -v python3 >/dev/null 2>&1 || packages="$packages python"
  command -v tmux >/dev/null 2>&1 || packages="$packages tmux"
  command -v ffmpeg >/dev/null 2>&1 || packages="$packages ffmpeg"
  brew list --versions portaudio >/dev/null 2>&1 || packages="$packages portaudio"
  if ! command -v llama-cli >/dev/null 2>&1 || ! command -v llama-server >/dev/null 2>&1; then
    packages="$packages llama.cpp"
  fi

  if [[ -n "$packages" ]]; then
    echo "[install] installing macOS packages:$packages"
    # Package names above are fixed, not user input; intentional word splitting.
    # shellcheck disable=SC2086
    brew install $packages
  fi
}

ensure_tmux() {
  if command -v tmux >/dev/null 2>&1; then
    echo "[install] tmux ready: $(command -v tmux) ($(tmux -V))"
    return 0
  fi

  if [[ "${INSTALL_SYSTEM_DEPS:-1}" == "0" ]]; then
    echo "tmux is missing and INSTALL_SYSTEM_DEPS=0 skipped its installation." >&2
  else
    echo "tmux installation completed without providing an executable on PATH." >&2
  fi

  if [[ "${VOICE_PLATFORM:-linux}" == "macos" ]]; then
    echo "Install it with 'brew install tmux' and retry." >&2
  else
    echo "Install it with 'sudo apt-get install -y tmux' and retry." >&2
  fi
  return 1
}

llama_cpp_ready() {
  command -v llama-cli >/dev/null 2>&1 && command -v llama-server >/dev/null 2>&1
}

install_llama_cpp() {
  case "${VOICE_INSTALL_LLAMA_CPP:-1}" in
    0|false|no|off)
      echo "[install] skipping llama.cpp (VOICE_INSTALL_LLAMA_CPP=${VOICE_INSTALL_LLAMA_CPP})."
      return 0
      ;;
  esac

  if llama_cpp_ready; then
    echo "[install] llama.cpp ready: $(command -v llama-cli)"
    return 0
  fi

  if [[ "${VOICE_PLATFORM:-linux}" == "macos" ]]; then
    if [[ "${INSTALL_SYSTEM_DEPS:-1}" == "0" ]]; then
      echo "llama.cpp is missing and INSTALL_SYSTEM_DEPS=0 skipped its Homebrew install." >&2
      echo "Install it with 'brew install llama.cpp' or set VOICE_INSTALL_LLAMA_CPP=0." >&2
      return 1
    fi
    echo "llama.cpp installation completed without providing llama-cli and llama-server." >&2
    return 1
  fi

  local source_dir="${VOICE_LLAMA_CPP_SOURCE_DIR:-$ROOT/models/llama.cpp}"
  local build_dir="${VOICE_LLAMA_CPP_BUILD_DIR:-$source_dir/build}"
  local build_jobs="${VOICE_LLAMA_CPP_BUILD_JOBS:-2}"

  if ! command -v git >/dev/null 2>&1 || ! command -v cmake >/dev/null 2>&1; then
    echo "git and cmake are required to build llama.cpp on Linux." >&2
    echo "Install system dependencies or set VOICE_INSTALL_LLAMA_CPP=0." >&2
    return 1
  fi

  if [[ ! -d "$source_dir/.git" ]]; then
    if [[ -e "$source_dir" ]]; then
      echo "Cannot install llama.cpp: $source_dir exists but is not a Git checkout." >&2
      return 1
    fi
    mkdir -p "$(dirname "$source_dir")"
    echo "[install] downloading llama.cpp source: $source_dir"
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$source_dir"
  fi

  echo "[install] building llama-cli and llama-server..."
  cmake -S "$source_dir" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_CURL=OFF
  cmake --build "$build_dir" --config Release --parallel "$build_jobs" \
    --target llama-cli llama-server
  ln -sf "$build_dir/bin/llama-cli" "$VENV_DIR/bin/llama-cli"
  ln -sf "$build_dir/bin/llama-server" "$VENV_DIR/bin/llama-server"
  export PATH="$VENV_DIR/bin:$PATH"

  if ! llama_cpp_ready; then
    echo "llama.cpp build finished, but llama-cli or llama-server is unavailable." >&2
    return 1
  fi
  echo "[install] llama.cpp ready: $(command -v llama-cli)"
}

install_stt_model() {
  case "${VOICE_INSTALL_STT_MODEL:-1}" in
    0|false|no|off)
      echo "[install] skipping STT model prefetch (VOICE_INSTALL_STT_MODEL=${VOICE_INSTALL_STT_MODEL})."
      return 0
      ;;
  esac

  "$VENV_DIR/bin/python" "$ROOT/scripts/prefetch_stt_model.py" \
    --repo-root "$ROOT" \
    --config "$ROOT/config.json"
}

if [[ "${INSTALL_SYSTEM_DEPS:-1}" != "0" ]]; then
  install_system_deps
fi
ensure_tmux

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found. Install Python 3.10+ and retry." >&2
  exit 1
fi

if ! python3 - <<'PY'
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
then
  echo "Python 3.10+ is required; found $(python3 -V 2>&1)." >&2
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
elif [[ ! -x "$VENV_DIR/bin/python" ]] || ! "$VENV_DIR/bin/python" -c 'import sys' >/dev/null 2>&1; then
  echo "[install] repairing incompatible Python environment: $VENV_DIR"
  python3 -m venv --clear "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --upgrade pip
case "${VOICE_INSTALL_FULL_REQUIREMENTS:-1}" in
  0|false|no|off)
    "$VENV_DIR/bin/python" -m pip install \
      numpy \
      sounddevice \
      soundfile \
      pynput \
      'onnx-asr[cpu,hub]' \
      sherpa-onnx
    ;;
  *)
    "$VENV_DIR/bin/python" -m pip install -r "$ROOT/requirements.txt"
    ;;
esac

export PATH="$VENV_DIR/bin:$PATH"
install_llama_cpp

if [[ ! -f "$ROOT/config.json" ]]; then
  cp "$ROOT/config.example.json" "$ROOT/config.json"
  echo "Created config.json"
fi

if [[ ! -f "$ROOT/models/silero_vad.onnx" ]]; then
  echo "Missing VAD model file: $ROOT/models/silero_vad.onnx" >&2
  exit 1
fi

install_stt_model

echo "Install complete."
echo "Run: ./run.sh"
echo "Workbench: ./run-auto.sh"
