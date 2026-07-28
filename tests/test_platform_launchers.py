import os
import subprocess
import tempfile
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

    def test_run_auto_local_env_preserves_inherited_disable_stt(self):
        path = os.path.join(self.repo_root, "run-auto.sh")
        with open(path, encoding="utf-8") as handle:
            launcher = handle.read()
        function_start = launcher.index("load_local_env() {")
        function_end = launcher.index("\n}\n\nload_local_env", function_start) + 2
        load_local_env_function = launcher[function_start:function_end]

        with tempfile.TemporaryDirectory() as temp_dir:
            with open(
                os.path.join(temp_dir, ".env"),
                "w",
                encoding="utf-8",
            ) as handle:
                handle.write(
                    "VOICE_DISABLE_STT=0\n"
                    "VOICE_TEST_LOCAL_VALUE=loaded\n"
                )
            env = os.environ.copy()
            env.update(
                {
                    "TEST_ROOT": temp_dir,
                    "VOICE_DISABLE_STT": "1",
                }
            )
            result = subprocess.run(
                [
                    "/bin/bash",
                    "-c",
                    load_local_env_function
                    + '\nROOT="$TEST_ROOT"\n'
                    + "load_local_env\n"
                    + 'printf "%s|%s\\n" "$VOICE_DISABLE_STT" '
                    + '"$VOICE_TEST_LOCAL_VALUE"\n',
                ],
                cwd=self.repo_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertEqual(result.stdout.strip(), "1|loaded")

    def test_installer_fails_when_tmux_is_missing_after_package_stage(self):
        env = os.environ.copy()
        env.update(
            {
                "INSTALL_SYSTEM_DEPS": "0",
                "PATH": "/usr/bin:/bin",
                "VOICE_INSTALL_PLATFORM_DISPATCHED": "1",
                "VOICE_PLATFORM": "macos",
            }
        )
        result = subprocess.run(
            ["/bin/bash", os.path.join(self.repo_root, "install.sh")],
            cwd=self.repo_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("tmux is missing", result.stderr)
        self.assertIn("brew install tmux", result.stderr)


if __name__ == "__main__":
    unittest.main()
