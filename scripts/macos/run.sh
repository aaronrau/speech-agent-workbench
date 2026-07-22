#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/local/sbin${PATH:+:$PATH}"
export VOICE_PLATFORM=macos
export VOICE_RUN_PLATFORM_DISPATCHED=1
export VOICE_PASTE_MODE="${VOICE_PASTE_MODE:-auto}"
exec /bin/bash "$ROOT/run.sh" "$@"
