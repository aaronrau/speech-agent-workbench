import json
import os
import shutil
import subprocess
import tempfile
import time
import unittest
import uuid
from unittest import mock

import app


@unittest.skipUnless(shutil.which("tmux"), "real tmux is not installed")
class RealTmuxWorkbenchTests(unittest.TestCase):
    def run_tmux(self, *args, check=True):
        return subprocess.run(
            ["tmux", *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=check,
        )

    def test_real_workbench_routes_command_to_real_pane(self):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        session = "voice-integration-" + uuid.uuid4().hex[:12]
        window = "Integration"
        recent_commands = dict(app.TMUX_RECENT_COMMANDS)

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = os.path.join(tmp_dir, "config.json")
            marker_path = os.path.join(tmp_dir, "routed.txt")
            config = {
                "agent_workbench": {
                    "session_name": session,
                    "layout": "panes",
                    "panes_window": window,
                    "agent_command": "sh",
                    "agents": [
                        {"name": "Flux", "path": tmp_dir},
                        {"name": "Brock", "path": tmp_dir},
                        {"name": "Pike", "path": tmp_dir},
                    ],
                    "voice": {"name": "Wolf", "path": repo_root},
                }
            }
            with open(config_path, "w", encoding="utf-8") as handle:
                json.dump(config, handle)

            env = os.environ.copy()
            env.update(
                {
                    "AGENTS_CONFIG_PROMPT": "0",
                    "ATTACH": "0",
                    "AUTO_STT": "0",
                    "VOICE_HOTKEY_CONFIG": config_path,
                    "VOICE_PLATFORM_OVERRIDE": "Darwin" if os.uname().sysname == "Darwin" else "Linux",
                }
            )

            try:
                result = subprocess.run(
                    [os.path.join(repo_root, "start-agent-workbench.sh")],
                    cwd=repo_root,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                    timeout=20,
                )
                self.assertEqual(
                    result.returncode,
                    0,
                    msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
                )

                panes = self.run_tmux(
                    "list-panes",
                    "-t",
                    f"{session}:{window}",
                    "-F",
                    "#{@agent_name}|#{pane_id}|#{pane_current_command}",
                ).stdout.splitlines()
                self.assertEqual(len(panes), 4, msg="\n".join(panes))
                pane_index = {
                    parts[0]: parts[1]
                    for line in panes
                    if len(parts := line.split("|", 2)) == 3
                }
                self.assertEqual(set(pane_index), {"Flux", "Brock", "Pike", "Wolf"})

                command = {
                    "label": "Flux",
                    "argv": ["tmux", "select-pane", "-t", pane_index["Flux"]],
                    "tmux_send_target": pane_index["Flux"],
                }
                commands = {"flux": command}
                routed_text = f"printf routed-through-real-tmux > {marker_path}"
                with mock.patch.dict(
                    os.environ,
                    {
                        "VOICE_AUTO_REFOCUS_TERMINAL": "0",
                        "VOICE_SUBMIT_ENTER_DELAY": "0",
                    },
                ):
                    response = app.route_api_message_to_tmux(
                        "Flux", routed_text, commands
                    )
                self.assertTrue(response["ok"], msg=response)
                self.assertTrue(response["sent"], msg=response)

                deadline = time.monotonic() + 5
                while time.monotonic() < deadline and not os.path.exists(marker_path):
                    time.sleep(0.05)
                self.assertTrue(os.path.exists(marker_path), "routed command did not run")
                with open(marker_path, "r", encoding="utf-8") as handle:
                    self.assertEqual(handle.read(), "routed-through-real-tmux")
            finally:
                self.run_tmux("kill-session", "-t", session, check=False)
                app.TMUX_RECENT_COMMANDS.clear()
                app.TMUX_RECENT_COMMANDS.update(recent_commands)

    def test_run_auto_disable_stt_starts_real_voice_pane(self):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        session = "voice-auto-" + uuid.uuid4().hex[:12]
        window = "AutoIntegration"

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = os.path.join(tmp_dir, "config.json")
            config = {
                "agent_workbench": {
                    "session_name": session,
                    "layout": "panes",
                    "panes_window": window,
                    "agent_command": "sh",
                    "agents": [
                        {"name": "Flux", "path": tmp_dir},
                        {"name": "Brock", "path": tmp_dir},
                        {"name": "Pike", "path": tmp_dir},
                    ],
                    "voice": {"name": "Wolf", "path": repo_root},
                },
                "api_enabled": False,
                "auto_tmux_summary_enabled": False,
            }
            with open(config_path, "w", encoding="utf-8") as handle:
                json.dump(config, handle)

            env = os.environ.copy()
            env.update(
                {
                    "AGENTS_CONFIG_PROMPT": "0",
                    "ATTACH": "0",
                    "AUTO_LOG": os.path.join(tmp_dir, "auto.log"),
                    "AUTO_CONSOLE_LOG": os.path.join(tmp_dir, "console.log"),
                    "AUTO_COMPLETION_LOG": os.path.join(tmp_dir, "completion.log"),
                    "AUTO_READY_TIMEOUT": "5",
                    "VOICE_AUTO_PREFETCH_MODELS": "0",
                    "VOICE_HOTKEY_CONFIG": config_path,
                    "VOICE_PLATFORM_OVERRIDE": "Darwin" if os.uname().sysname == "Darwin" else "Linux",
                }
            )

            try:
                result = subprocess.run(
                    [os.path.join(repo_root, "run-auto.sh"), "--disable-stt"],
                    cwd=repo_root,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                    timeout=30,
                )
                self.assertEqual(
                    result.returncode,
                    0,
                    msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
                )

                pane_lines = self.run_tmux(
                    "list-panes",
                    "-t",
                    f"{session}:{window}",
                    "-F",
                    "#{@agent_name}|#{pane_id}",
                ).stdout.splitlines()
                voice_pane = next(
                    line.split("|", 1)[1]
                    for line in pane_lines
                    if line.startswith("Wolf|")
                )
                deadline = time.monotonic() + 10
                captured = ""
                while time.monotonic() < deadline:
                    captured = self.run_tmux(
                        "capture-pane",
                        "-p",
                        "-t",
                        voice_pane,
                        "-S",
                        "-100",
                    ).stdout
                    if "STT disabled by --disable-stt" in captured:
                        break
                    time.sleep(0.1)
                self.assertIn("STT disabled by --disable-stt", captured)
                self.assertTrue(
                    "auto STT running in pane" in result.stdout
                    or "auto STT scheduled in current voice pane" in result.stdout,
                    msg=result.stdout,
                )
            finally:
                self.run_tmux("kill-session", "-t", session, check=False)


if __name__ == "__main__":
    unittest.main()
