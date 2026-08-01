import json
import os
import subprocess
import tempfile
import unittest
from unittest import mock

import app


class ConfigPathTests(unittest.TestCase):
    def setUp(self):
        self.repo_root = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )

    def test_python_default_uses_xdg_config_home(self):
        with mock.patch.dict(
            os.environ,
            {"XDG_CONFIG_HOME": "/tmp/workbench-user-config"},
        ):
            path = app.get_default_config_path()

        self.assertEqual(
            path,
            "/tmp/workbench-user-config/speech-agent-workbench/config.json",
        )

    def test_python_default_falls_back_to_home_dot_config(self):
        with mock.patch.dict(
            os.environ,
            {"HOME": "/tmp/workbench-home", "XDG_CONFIG_HOME": ""},
        ):
            path = app.get_default_config_path()

        self.assertEqual(
            path,
            "/tmp/workbench-home/.config/speech-agent-workbench/config.json",
        )

    def test_shell_launchers_share_the_user_config_default(self):
        env = os.environ.copy()
        env.update(
            {
                "HOME": "/tmp/workbench-home",
                "XDG_CONFIG_HOME": "/tmp/workbench-user-config",
            }
        )
        result = subprocess.run(
            [
                "/bin/bash",
                "-c",
                'source scripts/config-path.sh; default_voice_config_path',
            ],
            cwd=self.repo_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertEqual(
            result.stdout.strip(),
            "/tmp/workbench-user-config/speech-agent-workbench/config.json",
        )

    def test_runtime_entrypoints_use_the_shared_config_path(self):
        for filename in (
            "install.sh",
            "run.sh",
            "run-auto.sh",
            "start-agent-workbench.sh",
        ):
            with self.subTest(filename=filename):
                path = os.path.join(self.repo_root, filename)
                with open(path, "r", encoding="utf-8") as handle:
                    launcher = handle.read()

                self.assertIn('source "$ROOT/scripts/config-path.sh"', launcher)
                self.assertIn("default_voice_config_path", launcher)
                self.assertNotIn("$ROOT/config.json", launcher)

    def test_save_config_creates_the_user_config_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(
                temp_dir,
                "speech-agent-workbench",
                "config.json",
            )
            app.save_config(path, {"api_enabled": True})

            with open(path, "r", encoding="utf-8") as handle:
                saved = json.load(handle)

        self.assertEqual(saved, {"api_enabled": True})

    def test_runtime_paths_expand_home_relative_values(self):
        with mock.patch.dict(os.environ, {"HOME": "/tmp/workbench-home"}):
            path = app.resolve_local_runtime_path("~/bin/llama-cli")

        self.assertEqual(path, "/tmp/workbench-home/bin/llama-cli")


if __name__ == "__main__":
    unittest.main()
