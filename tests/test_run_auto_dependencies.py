import os
import stat
import subprocess
import tempfile
import unittest


class RunAutoDependencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.ensure_tmux = os.path.join(cls.repo_root, "scripts", "ensure-tmux.sh")

    def write_executable(self, directory, name, content):
        path = os.path.join(directory, name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)
        return path

    def test_macos_preflight_installs_and_verifies_missing_tmux(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bin_dir = os.path.join(temp_dir, "bin")
            os.makedirs(bin_dir)
            brew_log = os.path.join(temp_dir, "brew.log")
            self.write_executable(
                bin_dir,
                "brew",
                """#!/bin/bash
printf '%s\n' "$*" >> "$TMUX_TEST_BREW_LOG"
printf '#!/bin/bash\necho tmux test-version\n' > "$TMUX_TEST_BIN/tmux"
chmod +x "$TMUX_TEST_BIN/tmux"
""",
            )

            env = os.environ.copy()
            env.update(
                {
                    "PATH": bin_dir + ":/usr/bin:/bin",
                    "TMUX_TEST_BIN": bin_dir,
                    "TMUX_TEST_BREW_LOG": brew_log,
                    "VOICE_PLATFORM": "macos",
                }
            )
            result = subprocess.run(
                ["/bin/bash", self.ensure_tmux],
                cwd=self.repo_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn("installing it with Homebrew", result.stdout)
            self.assertIn("tmux ready", result.stdout)
            with open(brew_log, "r", encoding="utf-8") as handle:
                self.assertEqual(handle.read().strip(), "install tmux")

    def test_preflight_honors_disabled_automatic_install(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env = os.environ.copy()
            env.update(
                {
                    "PATH": temp_dir + ":/usr/bin:/bin",
                    "VOICE_AUTO_INSTALL_TMUX": "0",
                    "VOICE_PLATFORM": "macos",
                }
            )
            result = subprocess.run(
                ["/bin/bash", self.ensure_tmux],
                cwd=self.repo_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 1)
            self.assertIn("automatic installation is disabled", result.stderr)


if __name__ == "__main__":
    unittest.main()
