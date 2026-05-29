import os
import unittest
from tempfile import NamedTemporaryFile
from unittest.mock import patch
import wave

from app import RemoteTranscribeError, resolve_initial_transcriber, transcribe_remote


class ResolveInitialTranscriberTests(unittest.TestCase):
    @patch("app.log_backend_fallback")
    @patch("app.get_fallback_backend", return_value="parakeet-onnx")
    @patch("app.wait_for_remote_ready", return_value=False)
    @patch("app.build_transcriber")
    def test_switches_to_fallback_when_remote_is_not_ready(
        self,
        build_transcriber,
        wait_for_remote_ready,
        get_fallback_backend,
        log_backend_fallback,
    ):
        config = {"remote_url": "http://127.0.0.1:8765/transcribe"}
        build_transcriber.side_effect = [
            ("remote_fn", "remote", "http://127.0.0.1:8765/transcribe"),
            ("fallback_fn", "parakeet-onnx", "parakeet model"),
        ]

        result = resolve_initial_transcriber(config, 16000, 1)

        self.assertEqual(
            result,
            ("fallback_fn", "parakeet-onnx", "parakeet model"),
        )
        self.assertEqual(build_transcriber.call_count, 2)
        self.assertEqual(
            build_transcriber.call_args_list[1].kwargs,
            {"backend_override": "parakeet-onnx", "allow_fallback": False},
        )
        wait_for_remote_ready.assert_called_once_with(config)
        get_fallback_backend.assert_called_once_with(config)
        log_backend_fallback.assert_called_once()

    @patch("app.log_backend_fallback")
    @patch("app.get_fallback_backend", return_value="parakeet-onnx")
    @patch("app.wait_for_remote_ready", return_value=True)
    @patch("app.build_transcriber")
    def test_keeps_remote_when_it_is_ready(
        self,
        build_transcriber,
        wait_for_remote_ready,
        get_fallback_backend,
        log_backend_fallback,
    ):
        config = {"remote_url": "http://127.0.0.1:8765/transcribe"}
        build_transcriber.return_value = (
            "remote_fn",
            "remote",
            "http://127.0.0.1:8765/transcribe",
        )

        result = resolve_initial_transcriber(config, 16000, 1)

        self.assertEqual(
            result,
            ("remote_fn", "remote", "http://127.0.0.1:8765/transcribe"),
        )
        build_transcriber.assert_called_once_with(config, 16000, 1)
        wait_for_remote_ready.assert_called_once_with(config)
        get_fallback_backend.assert_called_once_with(config)
        log_backend_fallback.assert_not_called()


class _FakeResponse:
    def __init__(self, status=200, body=b""):
        self.status = status
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class TranscribeRemoteTests(unittest.TestCase):
    def create_test_wav(self):
        handle = NamedTemporaryFile(suffix=".wav", delete=False)
        handle.close()
        with wave.open(handle.name, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(b"\x00\x00" * 160)
        self.addCleanup(lambda: os.unlink(handle.name))
        return handle.name

    @patch("urllib.request.urlopen")
    @patch("app.get_fallback_backend", return_value="parakeet-onnx")
    def test_transcribe_remote_checks_health_before_upload(
        self, get_fallback_backend, urlopen
    ):
        wav_path = self.create_test_wav()
        urlopen.return_value = _FakeResponse(status=503, body=b"loading")

        with self.assertRaises(RemoteTranscribeError):
            transcribe_remote(wav_path, {"remote_url": "http://127.0.0.1:8765/transcribe"})

        self.assertEqual(urlopen.call_count, 1)
        request = urlopen.call_args.args[0]
        self.assertEqual(request, "http://127.0.0.1:8765/health")
        get_fallback_backend.assert_called()

    @patch("urllib.request.urlopen")
    @patch("app.get_fallback_backend", return_value="parakeet-onnx")
    def test_transcribe_remote_posts_audio_after_health_check(
        self, get_fallback_backend, urlopen
    ):
        wav_path = self.create_test_wav()
        urlopen.side_effect = [
            _FakeResponse(status=200, body=b"ok"),
            _FakeResponse(status=200, body=b'{"text": "hello world"}'),
        ]

        text = transcribe_remote(
            wav_path, {"remote_url": "http://127.0.0.1:8765/transcribe"}
        )

        self.assertEqual(text, "hello world")
        self.assertEqual(urlopen.call_count, 2)
        health_request = urlopen.call_args_list[0].args[0]
        post_request = urlopen.call_args_list[1].args[0]
        self.assertEqual(health_request, "http://127.0.0.1:8765/health")
        self.assertEqual(post_request.full_url, "http://127.0.0.1:8765/transcribe")
        self.assertIsNotNone(post_request.data)
        get_fallback_backend.assert_called()

    @patch("urllib.request.urlopen")
    @patch("app.get_fallback_backend", return_value="parakeet-onnx")
    def test_transcribe_remote_caps_post_timeout_to_request_timeout(
        self, get_fallback_backend, urlopen
    ):
        wav_path = self.create_test_wav()
        urlopen.side_effect = [
            _FakeResponse(status=200, body=b"ok"),
            _FakeResponse(status=200, body=b'{"text": "hello world"}'),
        ]

        text = transcribe_remote(
            wav_path,
            {
                "remote_url": "http://127.0.0.1:8765/transcribe",
                "remote_timeout": 600,
                "transcribe_request_timeout": 2,
            },
        )

        self.assertEqual(text, "hello world")
        self.assertEqual(urlopen.call_args_list[1].kwargs["timeout"], 2)
        get_fallback_backend.assert_called()


if __name__ == "__main__":
    unittest.main()
