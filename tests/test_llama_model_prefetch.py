import os
import tempfile
import unittest

from scripts.prefetch_llama_cpp_model import prefetch_llama_cpp_model


class LlamaCppModelPrefetchTests(unittest.TestCase):
    def test_rejects_empty_model_path(self):
        with self.assertRaisesRegex(ValueError, "model path is empty"):
            prefetch_llama_cpp_model("", "example/test-GGUF")

    def test_downloads_missing_model_to_configured_directory(self):
        calls = []

        def download_fn(**kwargs):
            calls.append(kwargs)
            path = os.path.join(kwargs["local_dir"], kwargs["filename"])
            with open(path, "wb") as handle:
                handle.write(b"GGUF")
            return path

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "models", "test-Q4_0.gguf")
            result = prefetch_llama_cpp_model(
                model_path,
                "example/test-GGUF",
                download_fn=download_fn,
            )

            self.assertEqual(result, model_path)
            self.assertTrue(os.path.isfile(model_path))

        self.assertEqual(
            calls,
            [
                {
                    "repo_id": "example/test-GGUF",
                    "filename": "test-Q4_0.gguf",
                    "local_dir": os.path.dirname(model_path),
                }
            ],
        )

    def test_reuses_existing_model_without_downloading(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "model.gguf")
            with open(model_path, "wb") as handle:
                handle.write(b"GGUF")

            result = prefetch_llama_cpp_model(
                model_path,
                "example/test-GGUF",
                download_fn=lambda **_kwargs: self.fail("unexpected download"),
            )

        self.assertEqual(result, model_path)
