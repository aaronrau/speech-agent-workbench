#!/usr/bin/env python3
import json
import os
import sys
import time


def prefixed_chunk(text, prefix, at_line_start):
    output = []
    for char in text:
        if at_line_start:
            output.append(prefix)
            at_line_start = False
        output.append(char)
        if char in ("\n", "\r"):
            at_line_start = True
    return "".join(output), at_line_start


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "tmux"
    prefix = f"[tmux][{label}] "
    at_line_start = True
    stdin_fd = sys.stdin.buffer.fileno()

    while True:
        chunk = os.read(stdin_fd, 4096)
        if not chunk:
            break
        text = chunk.decode("utf-8", errors="replace").replace("\x00", "")
        output, at_line_start = prefixed_chunk(text, prefix, at_line_start)
        if not output:
            continue
        record = {
            "ts": time.time(),
            "data": output,
        }
        sys.stdout.write(json.dumps(record, separators=(",", ":")) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
