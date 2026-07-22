import subprocess
import unittest
from unittest import mock

import app


class MacOSDesktopTests(unittest.TestCase):
    def completed(self, argv, returncode=0):
        return subprocess.CompletedProcess(argv, returncode, stdout="", stderr="")

    def test_set_clipboard_uses_pbcopy(self):
        with mock.patch.object(app, "is_macos", return_value=True):
            with mock.patch.object(app.shutil, "which", return_value="/usr/bin/pbcopy"):
                with mock.patch.object(
                    app, "run_command", side_effect=lambda argv, **kwargs: self.completed(argv)
                ) as run:
                    self.assertTrue(app.set_clipboard("hello from macOS"))

        self.assertEqual(run.call_args.args[0], ["pbcopy"])
        self.assertEqual(run.call_args.kwargs["input_text"], "hello from macOS")

    def test_paste_hotkey_uses_command_v(self):
        with mock.patch.object(app, "is_macos", return_value=True):
            with mock.patch.object(app.shutil, "which", return_value="/usr/bin/osascript"):
                with mock.patch.object(
                    app, "run_command", side_effect=lambda argv, **kwargs: self.completed(argv)
                ) as run:
                    self.assertTrue(app.paste_via_hotkey())

        argv = run.call_args.args[0]
        self.assertEqual(argv[0], "osascript")
        self.assertTrue(
            any('keystroke "v" using command down' in item for item in argv)
        )

    def test_press_enter_uses_macos_return_key(self):
        with mock.patch.object(app, "is_macos", return_value=True):
            with mock.patch.object(app.shutil, "which", return_value="/usr/bin/osascript"):
                with mock.patch.object(
                    app, "run_command", side_effect=lambda argv, **kwargs: self.completed(argv)
                ) as run:
                    self.assertTrue(app.press_enter())

        self.assertIn('tell application "System Events" to key code 36', run.call_args.args[0])

    def test_focus_activates_detected_terminal_application(self):
        env = {"TERM_PROGRAM": "iTerm.app"}
        with mock.patch.dict("os.environ", env, clear=True):
            with mock.patch.object(app.shutil, "which", return_value="/usr/bin/open"):
                with mock.patch.object(
                    app, "run_command", side_effect=lambda argv, **kwargs: self.completed(argv)
                ) as run:
                    self.assertTrue(app.focus_macos_terminal_application())

        run.assert_called_once_with(["open", "-a", "iTerm"], timeout=2.0)


if __name__ == "__main__":
    unittest.main()
