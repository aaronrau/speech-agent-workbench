import unittest

import numpy as np

from app import Recorder, get_auto_initial_scan_sample


class RecorderMemoryTests(unittest.TestCase):
    def test_auto_initial_scan_includes_pre_roll_buffer(self):
        self.assertEqual(get_auto_initial_scan_sample(1200, 500), 700)

    def test_auto_initial_scan_clamps_to_start_when_pre_roll_exceeds_buffer(self):
        self.assertEqual(get_auto_initial_scan_sample(300, 500), 0)

    def test_clear_if_idle_releases_buffers_for_completed_recording(self):
        recorder = Recorder(sample_rate=16000, channels=1)
        recorder._recording_id = 3
        recorder._frames = [np.ones((16000, 1), dtype=np.int16)]
        recorder._frame_offsets = [0]
        recorder._sample_count = 16000
        recorder._level = 0.5

        cleared = recorder.clear_if_idle(3)

        self.assertTrue(cleared)
        self.assertEqual(recorder._frames, [])
        self.assertEqual(recorder._frame_offsets, [])
        self.assertEqual(recorder._sample_count, 0)
        self.assertEqual(recorder._level, 0.0)

    def test_clear_if_idle_does_not_clear_newer_recording(self):
        recorder = Recorder(sample_rate=16000, channels=1)
        recorder._recording_id = 4
        frame = np.ones((16000, 1), dtype=np.int16)
        recorder._frames = [frame]
        recorder._frame_offsets = [0]
        recorder._sample_count = 16000

        cleared = recorder.clear_if_idle(3)

        self.assertFalse(cleared)
        self.assertEqual(recorder._frames, [frame])
        self.assertEqual(recorder._sample_count, 16000)

    def test_discard_before_releases_old_complete_frames(self):
        recorder = Recorder(sample_rate=16000, channels=1)
        old_frame = np.ones((100, 1), dtype=np.int16)
        kept_frame = np.ones((100, 1), dtype=np.int16) * 2
        recorder._frames = [old_frame, kept_frame]
        recorder._frame_offsets = [0, 100]
        recorder._sample_count = 200

        recorder.discard_before(100)
        samples, total = recorder.get_samples_since(0)

        self.assertEqual(total, 200)
        self.assertEqual(len(recorder._frames), 1)
        self.assertIs(recorder._frames[0], kept_frame)
        self.assertTrue(np.array_equal(samples, kept_frame))


if __name__ == "__main__":
    unittest.main()
