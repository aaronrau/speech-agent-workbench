#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VOICE_VENV:-$ROOT/.venv}"

install_system_deps() {
  if ! command -v apt-get >/dev/null 2>&1; then
    cat >&2 <<'EOF'
apt-get not found. Install these packages manually:
  python3 python3-venv portaudio19-dev ffmpeg tmux
  xdotool xclip wtype wl-clipboard ydotool
EOF
    return 0
  fi

  sudo apt-get update
  sudo apt-get install -y python3 python3-venv portaudio19-dev ffmpeg tmux

  for pkg in xdotool xclip wtype wl-clipboard ydotool; do
    if ! sudo apt-get install -y "$pkg"; then
      echo "Optional package not installed: $pkg" >&2
    fi
  done
}

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found. Install Python 3.9+ and retry." >&2
  exit 1
fi

if [[ "${INSTALL_SYSTEM_DEPS:-1}" != "0" ]]; then
  install_system_deps
fi

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -r "$ROOT/requirements.txt"

if [[ ! -f "$ROOT/config.json" ]]; then
  cp "$ROOT/config.example.json" "$ROOT/config.json"
  echo "Created config.json"
fi

missing=0
for path in \
  "$ROOT/models/silero_vad.onnx" \
  "$ROOT/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/encoder.int8.onnx" \
  "$ROOT/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/decoder.int8.onnx" \
  "$ROOT/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/joiner.int8.onnx" \
  "$ROOT/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/tokens.txt"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing model file: $path" >&2
    missing=1
  fi
done

if [[ "$missing" == "1" ]]; then
  echo "Install finished, but model files are incomplete." >&2
  exit 1
fi

echo "Install complete."
echo "Run: ./run.sh"
echo "Workbench: ./run-auto.sh"
