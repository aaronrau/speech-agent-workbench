#!/bin/bash
set -euo pipefail

tmux_ready() {
  command -v tmux >/dev/null 2>&1 && tmux -V >/dev/null 2>&1
}

if tmux_ready; then
  echo "[auto] tmux ready: $(command -v tmux) ($(tmux -V))"
  exit 0
fi

case "${VOICE_AUTO_INSTALL_TMUX:-1}" in
  0|false|no|none|null|off)
    echo "[auto] tmux is missing and automatic installation is disabled." >&2
    exit 1
    ;;
esac

case "${VOICE_PLATFORM:-$(uname -s)}" in
  macos|Darwin)
    if ! command -v brew >/dev/null 2>&1; then
      echo "[auto] tmux is missing and Homebrew is unavailable." >&2
      echo "[auto] install Homebrew, then run: brew install tmux" >&2
      exit 1
    fi
    echo "[auto] tmux is missing; installing it with Homebrew..."
    brew install tmux
    ;;
  linux|Linux)
    if ! command -v apt-get >/dev/null 2>&1; then
      echo "[auto] tmux is missing and apt-get is unavailable." >&2
      echo "[auto] install tmux with your system package manager and retry." >&2
      exit 1
    fi
    echo "[auto] tmux is missing; installing it with apt-get..."
    if [[ "$(id -u)" == "0" ]]; then
      apt-get update
      apt-get install -y tmux
    elif command -v sudo >/dev/null 2>&1; then
      sudo apt-get update
      sudo apt-get install -y tmux
    else
      echo "[auto] sudo is required to install tmux with apt-get." >&2
      exit 1
    fi
    ;;
  *)
    echo "[auto] tmux is missing on unsupported platform: ${VOICE_PLATFORM:-$(uname -s)}" >&2
    exit 1
    ;;
esac

hash -r
if ! tmux_ready; then
  echo "[auto] tmux installation finished, but tmux is still unavailable on PATH." >&2
  exit 1
fi

echo "[auto] tmux ready: $(command -v tmux) ($(tmux -V))"
