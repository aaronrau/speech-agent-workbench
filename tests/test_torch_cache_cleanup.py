import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

from app import (
    empty_torch_cache_after_transcribe,
    parse_bool,
    should_empty_torch_cache_after_transcribe,
)


class TorchCacheCleanupTests(unittest.TestCase):
    def setUp(self):
        self._torch_module = sys.modules.get("torch")
        os.environ.pop("VOICE_TORCH_EMPTY_CACHE_AFTER_TRANSCRIBE", None)

    def tearDown(self):
        if self._torch_module is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = self._torch_module
        os.environ.pop("VOICE_TORCH_EMPTY_CACHE_AFTER_TRANSCRIBE", None)

    def install_fake_torch(self, hip=None):
        fake_cuda = SimpleNamespace(
            is_available=mock.Mock(return_value=True),
            empty_cache=mock.Mock(),
            ipc_collect=mock.Mock(),
        )
        fake_torch = SimpleNamespace(
            version=SimpleNamespace(hip=hip),
            cuda=fake_cuda,
        )
        sys.modules["torch"] = fake_torch
        return fake_torch

    def cuda_model(self):
        return SimpleNamespace(device=SimpleNamespace(type="cuda"))

    def test_parse_bool_accepts_common_values(self):
        self.assertTrue(parse_bool("yes"))
        self.assertTrue(parse_bool("1"))
        self.assertFalse(parse_bool("off"))
        self.assertFalse(parse_bool("false"))
        self.assertIsNone(parse_bool("auto"))

    def test_defaults_to_cleanup_for_rocm_cuda_model(self):
        self.install_fake_torch(hip="7.1.1")

        self.assertTrue(
            should_empty_torch_cache_after_transcribe({}, self.cuda_model())
        )

    def test_does_not_default_to_cleanup_for_cuda_runtime(self):
        self.install_fake_torch(hip=None)

        self.assertFalse(
            should_empty_torch_cache_after_transcribe({}, self.cuda_model())
        )

    def test_env_override_disables_rocm_cleanup(self):
        self.install_fake_torch(hip="7.1.1")
        os.environ["VOICE_TORCH_EMPTY_CACHE_AFTER_TRANSCRIBE"] = "0"

        self.assertFalse(
            should_empty_torch_cache_after_transcribe({}, self.cuda_model())
        )

    def test_cleanup_releases_cuda_cache_when_enabled(self):
        fake_torch = self.install_fake_torch(hip=None)

        empty_torch_cache_after_transcribe(
            {"torch_empty_cache_after_transcribe": True},
            self.cuda_model(),
        )

        fake_torch.cuda.empty_cache.assert_called_once_with()
        fake_torch.cuda.ipc_collect.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
