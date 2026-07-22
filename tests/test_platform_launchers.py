import os
import subprocess
import unittest


class PlatformLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def run_workbench_help(self, platform):
        env = os.environ.copy()
        env["VOICE_PLATFORM_OVERRIDE"] = platform
        return subprocess.run(
            [os.path.join(self.repo_root, "start-agent-workbench.sh"), "--help"],
            cwd=self.repo_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

    def test_linux_dispatcher_runs_linux_launcher(self):
        result = self.run_workbench_help("Linux")
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("Starts a tmux workbench", result.stdout)

    def test_macos_dispatcher_runs_macos_launcher_with_apple_bash(self):
        result = self.run_workbench_help("Darwin")
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("Starts a tmux workbench", result.stdout)

    def test_dispatcher_rejects_unknown_platform(self):
        result = self.run_workbench_help("Plan9")
        self.assertEqual(result.returncode, 1)
        self.assertIn("Unsupported operating system: Plan9", result.stderr)

    def test_all_launchers_parse_with_macos_bash(self):
        scripts = [
            "install.sh",
            "setup.sh",
            "run.sh",
            "run-auto.sh",
            "start-agent-workbench.sh",
            "scripts/linux/install.sh",
            "scripts/linux/run.sh",
            "scripts/linux/run-auto.sh",
            "scripts/linux/start-agent-workbench.sh",
            "scripts/macos/install.sh",
            "scripts/macos/run.sh",
            "scripts/macos/run-auto.sh",
            "scripts/macos/start-agent-workbench.sh",
        ]
        result = subprocess.run(
            ["/bin/bash", "-n", *scripts],
            cwd=self.repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)


if __name__ == "__main__":
    unittest.main()
