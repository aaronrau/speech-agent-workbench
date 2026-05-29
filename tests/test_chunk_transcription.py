import unittest

import numpy as np

from app import choose_chunk_end_sample


class ChooseChunkEndSampleTests(unittest.TestCase):
    def test_waits_until_audio_reaches_twenty_seconds(self):
        sample_rate = 16000
        speech_a = np.full((int(4.6 * sample_rate), 1), 4000, dtype=np.int16)
        silence = np.zeros((int(0.5 * sample_rate), 1), dtype=np.int16)
        speech_b = np.full((int(1.5 * sample_rate), 1), 5000, dtype=np.int16)
        samples = np.concatenate([speech_a, silence, speech_b], axis=0)

        boundary = choose_chunk_end_sample(
            samples,
            sample_rate,
            target_seconds=20.0,
            pause_window_seconds=1.0,
            frame_ms=30,
            silence_ms=240,
            silence_threshold=0.012,
        )

        self.assertIsNone(boundary)

    def test_falls_back_to_target_when_no_pause_exists(self):
        sample_rate = 16000
        samples = np.full((int(21.5 * sample_rate), 1), 6000, dtype=np.int16)

        boundary = choose_chunk_end_sample(
            samples,
            sample_rate,
            target_seconds=20.0,
            pause_window_seconds=1.0,
            frame_ms=30,
            silence_ms=240,
            silence_threshold=0.012,
        )

        self.assertEqual(boundary, int(20.0 * sample_rate))


if __name__ == "__main__":
    unittest.main()
