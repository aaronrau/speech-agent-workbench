import json
import os
import tempfile
import unittest
from unittest.mock import patch

from app import prompt_for_device


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
