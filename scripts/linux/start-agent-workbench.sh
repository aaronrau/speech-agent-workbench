#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export VOICE_PLATFORM=linux
export VOICE_WORKBENCH_PLATFORM_DISPATCHED=1
exec /bin/bash "$ROOT/start-agent-workbench.sh" "$@"
