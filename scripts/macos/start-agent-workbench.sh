#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/local/sbin${PATH:+:$PATH}"
export VOICE_PLATFORM=macos
export VOICE_WORKBENCH_PLATFORM_DISPATCHED=1
exec /bin/bash "$ROOT/start-agent-workbench.sh" "$@"
