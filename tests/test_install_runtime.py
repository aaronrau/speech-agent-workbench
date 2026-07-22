import contextlib
import io
import unittest

from scripts.prefetch_stt_model import prefetch_stt_model


class FakeWorkbenchApp:
    def __init__(self):
        self.loaded_config_path = None
        self.loaded_model = None

    def load_config(self, path):
        self.loaded_config_path = path
        return {
            "parakeet_onnx_model": "example/parakeet-onnx",
            "parakeet_onnx_quantization": "int8",
            "parakeet_onnx_provider": "cpu",
        }

    def get_parakeet_onnx_model(self, config):
        return config["parakeet_onnx_model"]

    def get_parakeet_onnx_quantization(self, config):
        return config["parakeet_onnx_quantization"]

    def get_parakeet_onnx_provider(self, config):
        return config["parakeet_onnx_provider"]

    def load_parakeet_onnx_model(self, model_name, quantization, provider):
        self.loaded_model = (model_name, quantization, provider)


class InstallRuntimeTests(unittest.TestCase):
    def test_prefetch_loads_configured_stt_model(self):
        workbench_app = FakeWorkbenchApp()
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            prefetch_stt_model("/tmp/config.json", workbench_app)

        self.assertEqual(workbench_app.loaded_config_path, "/tmp/config.json")
        self.assertEqual(
            workbench_app.loaded_model,
            ("example/parakeet-onnx", "int8", "cpu"),
        )
        self.assertIn("Parakeet ONNX STT model ready", output.getvalue())


if __name__ == "__main__":
    unittest.main()
