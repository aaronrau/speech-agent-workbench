#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/local/sbin${PATH:+:$PATH}"
export VOICE_PLATFORM=macos
export VOICE_INSTALL_PLATFORM_DISPATCHED=1
export VOICE_INSTALL_FULL_REQUIREMENTS="${VOICE_INSTALL_FULL_REQUIREMENTS:-0}"
exec /bin/bash "$ROOT/install.sh" "$@"
