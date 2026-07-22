#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export VOICE_PLATFORM=linux
export VOICE_RUN_AUTO_PLATFORM_DISPATCHED=1
exec /bin/bash "$ROOT/run-auto.sh" "$@"
