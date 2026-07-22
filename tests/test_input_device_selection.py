import json
import os
import tempfile
import unittest
from unittest.mock import patch

import app
from app import prompt_for_auto_pause_hotkey, prompt_for_device


class _FakeDefault:
    device = [2, None]


class _FakeSoundDevice:
    default = _FakeDefault()

    @staticmethod
    def query_devices():
        return [
            {
                "name": "Built-in Monitor",
                "hostapi": 0,
                "max_input_channels": 0,
                "default_samplerate": 48000,
            },
            {
                "name": "Webcam Microphone",
                "hostapi": 0,
                "max_input_channels": 1,
                "default_samplerate": 48000,
            },
            {
                "name": "Laptop Microphone",
                "hostapi": 0,
                "max_input_channels": 2,
                "default_samplerate": 48000,
            },
        ]

    @staticmethod
    def query_hostapis():
        return [{"name": "PipeWire"}]


class InputDeviceSelectionTests(unittest.TestCase):
    def write_config(self, config):
        handle = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        )
        self.addCleanup(
            lambda: os.path.exists(handle.name) and os.unlink(handle.name)
        )
        json.dump(config, handle)
        handle.close()
        return handle.name

    def read_config(self, path):
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    @patch("app.prompt_for_device")
    @patch("app.load_config", return_value={"device": None})
    def test_configure_audio_input_cli_only_prompts_for_device(
        self, load_config, prompt
    ):
        with patch.dict(
            os.environ,
            {"VOICE_HOTKEY_CONFIG": "/tmp/workbench-config.json"},
        ):
            with patch(
                "sys.argv", ["app.py", "--configure-audio-input"]
            ):
                app.main()

        load_config.assert_called_once_with("/tmp/workbench-config.json")
        prompt.assert_called_once_with(
            "/tmp/workbench-config.json", {"device": None}
        )

    @patch("app.prompt_for_auto_pause_hotkey")
    @patch("app.prompt_for_device")
    @patch(
        "app.load_config",
        return_value={"device": None, "auto_pause_hotkey": "ctrl"},
    )
    def test_configure_macos_inputs_prompts_for_audio_and_keyboard(
        self, load_config, prompt_device, prompt_hotkey
    ):
        config = {"device": None, "auto_pause_hotkey": "ctrl"}
        with patch.dict(
            os.environ,
            {"VOICE_HOTKEY_CONFIG": "/tmp/workbench-config.json"},
        ):
            with patch("sys.argv", ["app.py", "--configure-macos-inputs"]):
                app.main()

        load_config.assert_called_once_with("/tmp/workbench-config.json")
        prompt_device.assert_called_once_with(
            "/tmp/workbench-config.json", config, step_label="Step 1/2:"
        )
        prompt_hotkey.assert_called_once_with(
            "/tmp/workbench-config.json", config, step_label="Step 2/2:"
        )

    @patch("app.parse_auto_pause_hotkey_pynput")
    @patch("app.capture_hotkey_pynput", return_value="right_cmd")
    @patch("app.prompt_change_saved_value", return_value=True)
    @patch("app.get_auto_pause_hotkey", return_value="ctrl")
    @patch("sys.stdin.isatty", return_value=True)
    def test_pause_hotkey_selection_is_saved(
        self, _isatty, _get_hotkey, _change, _capture, parse_hotkey
    ):
        config = {"auto_pause_hotkey": "ctrl"}
        path = self.write_config(config)

        selected = prompt_for_auto_pause_hotkey(path, config)

        self.assertEqual(selected, "right_cmd")
        parse_hotkey.assert_called_once_with("right_cmd")
        self.assertEqual(
            self.read_config(path)["auto_pause_hotkey"], "right_cmd"
        )

    @patch("app.sd", _FakeSoundDevice)
    @patch("sys.stdin.isatty", return_value=True)
    @patch("builtins.input", return_value="")
    def test_keeps_missing_saved_named_input_when_default_is_used(
        self, _input, _isatty
    ):
        config = {
            "device": 7,
            "device_name": "Razer Seiren Mini",
            "device_hostapi": "PipeWire",
        }
        path = self.write_config(config)

        selected = prompt_for_device(path, config)

        self.assertEqual(selected, 2)
        self.assertEqual(
            self.read_config(path),
            {
                "device": 7,
                "device_name": "Razer Seiren Mini",
                "device_hostapi": "PipeWire",
            },
        )

    @patch("app.sd", _FakeSoundDevice)
    @patch("sys.stdin.isatty", return_value=True)
    @patch("builtins.input", return_value="1")
    def test_explicit_selection_replaces_missing_saved_named_input(
        self, _input, _isatty
    ):
        config = {
            "device": 7,
            "device_name": "Razer Seiren Mini",
            "device_hostapi": "PipeWire",
        }
        path = self.write_config(config)

        selected = prompt_for_device(path, config)

        self.assertEqual(selected, 1)
        self.assertEqual(
            self.read_config(path),
            {
                "device": 1,
                "device_name": "Webcam Microphone",
                "device_hostapi": "PipeWire",
            },
        )

    @patch("app.sd", _FakeSoundDevice)
    @patch("sys.stdin.isatty", return_value=True)
    @patch("builtins.input", return_value="")
    def test_keeps_missing_legacy_index_when_default_is_used(
        self, _input, _isatty
    ):
        config = {"device": 7}
        path = self.write_config(config)

        selected = prompt_for_device(path, config)

        self.assertEqual(selected, 2)
        self.assertEqual(self.read_config(path), {"device": 7})


if __name__ == "__main__":
    unittest.main()
