import json
import os
import stat
import subprocess
import tempfile
import unittest


class WorkbenchConfigTests(unittest.TestCase):
    def write_executable(self, directory, name, content):
        path = os.path.join(directory, name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.chmod(
            path,
            stat.S_IRUSR
            | stat.S_IWUSR
            | stat.S_IXUSR
            | stat.S_IRGRP
            | stat.S_IXGRP
            | stat.S_IROTH
            | stat.S_IXOTH,
        )
        return path

    def write_fake_tmux(self, directory):
        return self.write_executable(
            directory,
            "tmux",
            """#!/usr/bin/env bash
case "$1" in
  has-session)
    exit 1
    ;;
  list-panes|list-windows)
    exit 0
    ;;
  display-message)
    case "$*" in
      *pane_current_command*) echo bash ;;
      *) echo %1 ;;
    esac
    ;;
  split-window)
    echo %2
    ;;
esac
exit 0
""",
        )

    def test_start_workbench_migrates_legacy_codex_agents_config(self):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with tempfile.TemporaryDirectory() as tmp_dir:
            fake_bin = os.path.join(tmp_dir, "bin")
            os.mkdir(fake_bin)
            agent1_dir = os.path.join(tmp_dir, "all")
            agent2_dir = os.path.join(agent1_dir, "backend")
            agent3_dir = os.path.join(agent1_dir, "flutter")
            voice_dir = os.path.join(tmp_dir, "voice")
            for path in (agent1_dir, agent2_dir, agent3_dir, voice_dir):
                os.makedirs(path)

            self.write_fake_tmux(fake_bin)
            self.write_executable(
                fake_bin,
                "agent-cli",
                "#!/usr/bin/env bash\nexit 0\n",
            )

            config_path = os.path.join(tmp_dir, "config.json")
            with open(config_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "codex_agents": {
                            "session_name": "legacy-session",
                            "layout": "panes",
                            "panes_window": "LegacyPanes",
                            "codex_command": "agent-cli",
                            "agents": [
                                {"name": "North", "path": agent1_dir},
                                {"name": "South", "path": agent2_dir},
                                {"name": "West", "path": agent3_dir},
                            ],
                            "voice": {"name": "Listener", "path": voice_dir},
                        },
                        "device": 4,
                        "device_name": "USB Microphone",
                        "device_hostapi": "ALSA",
                        "evdev_device_name": "USB Keyboard",
                        "hotkey": "right_ctrl",
                    },
                    handle,
                )

            env = os.environ.copy()
            env.update(
                {
                    "AGENTS_CONFIG_PROMPT": "0",
                    "ATTACH": "0",
                    "AUTO_STT": "0",
                    "PATH": f"{fake_bin}{os.pathsep}{env.get('PATH', '')}",
                    "VOICE_HOTKEY_CONFIG": config_path,
                }
            )
            result = subprocess.run(
                [os.path.join(repo_root, "start-agent-workbench.sh")],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

            self.assertEqual(
                result.returncode,
                0,
                msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
            )
            with open(config_path, "r", encoding="utf-8") as handle:
                config = json.load(handle)

            self.assertEqual(
                config["agent_workbench"],
                {
                    "session_name": "legacy-session",
                    "layout": "panes",
                    "panes_window": "LegacyPanes",
                    "agent_command": "agent-cli",
                    "agents": [
                        {"name": "North", "path": agent1_dir},
                        {"name": "South", "path": agent2_dir},
                        {"name": "West", "path": agent3_dir},
                    ],
                    "voice": {"name": "Listener", "path": voice_dir},
                },
            )
            self.assertEqual(config["device_name"], "USB Microphone")
            self.assertEqual(config["device_hostapi"], "ALSA")
            self.assertEqual(config["evdev_device_name"], "USB Keyboard")
            self.assertEqual(config["hotkey"], "right_ctrl")

    def test_start_workbench_ignores_stt_disable_alias_as_agent_path(self):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with tempfile.TemporaryDirectory() as tmp_dir:
            fake_bin = os.path.join(tmp_dir, "bin")
            os.mkdir(fake_bin)
            agent1_dir = os.path.join(tmp_dir, "all")
            agent2_dir = os.path.join(agent1_dir, "backend")
            agent3_dir = os.path.join(agent1_dir, "flutter")
            voice_dir = os.path.join(tmp_dir, "voice")
            for path in (agent1_dir, agent2_dir, agent3_dir, voice_dir):
                os.makedirs(path)

            self.write_fake_tmux(fake_bin)
            self.write_executable(fake_bin, "agent-cli", "#!/usr/bin/env bash\nexit 0\n")

            config_path = os.path.join(tmp_dir, "config.json")
            workbench_config = {
                "session_name": "alias-session",
                "layout": "panes",
                "panes_window": "AliasPanes",
                "agent_command": "agent-cli",
                "agents": [
                    {"name": "Flux", "path": agent1_dir},
                    {"name": "Brock", "path": agent2_dir},
                    {"name": "Pike", "path": agent3_dir},
                ],
                "voice": {"name": "Wolf", "path": voice_dir},
            }
            with open(config_path, "w", encoding="utf-8") as handle:
                json.dump({"agent_workbench": workbench_config}, handle)

            env = os.environ.copy()
            env.update(
                {
                    "AGENTS_CONFIG_PROMPT": "0",
                    "ATTACH": "0",
                    "AUTO_STT": "0",
                    "PATH": f"{fake_bin}{os.pathsep}{env.get('PATH', '')}",
                    "VOICE_HOTKEY_CONFIG": config_path,
                }
            )
            result = subprocess.run(
                [os.path.join(repo_root, "start-agent-workbench.sh"), "--", "stt-disable"],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

            self.assertEqual(
                result.returncode,
                0,
                msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
            )
            with open(config_path, "r", encoding="utf-8") as handle:
                config = json.load(handle)

            self.assertEqual(config["agent_workbench"], workbench_config)


if __name__ == "__main__":
    unittest.main()
