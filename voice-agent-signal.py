#!/usr/bin/env python3
import json
import os
import sys
import time


def main():
    path = os.environ.get("VOICE_AGENT_COMPLETION_LOG", "").strip()
    if not path:
        print("VOICE_AGENT_COMPLETION_LOG is not configured.", file=sys.stderr)
        return 1

    agent = os.environ.get("VOICE_AGENT_NAME", "").strip() or "agent"
    status = sys.argv[1].strip() if len(sys.argv) > 1 and sys.argv[1].strip() else "done"
    message = " ".join(arg.strip() for arg in sys.argv[2:] if arg.strip())
    record = {
        "ts": time.time(),
        "agent": agent,
        "status": status,
        "message": message,
    }

    directory = os.path.dirname(os.path.expanduser(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(os.path.expanduser(path), "a", encoding="utf-8") as handle:
        json.dump(record, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
