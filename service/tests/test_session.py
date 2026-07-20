import unittest

from samm_server.session import SessionTracker


class Clock:
    def __init__(self, value=0):
        self.value = value

    def __call__(self):
        return self.value

    def tick(self, seconds):
        self.value += seconds


class SessionTrackerTest(unittest.TestCase):
    def test_touch_requires_session_id(self):
        tracker = SessionTracker(now=Clock())

        result = tracker.touch({"autosave": {}})

        self.assertEqual(result.status_code, 400)
        self.assertEqual(result.payload, {"error": "session_id must be a non-empty string"})

    def test_touch_tracks_active_session(self):
        tracker = SessionTracker(timeout_seconds=600, now=Clock())

        result = tracker.touch({"session_id": "slicer-1"})

        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.payload["active_sessions"], 1)
        self.assertFalse(tracker.expired())

    def test_expires_when_session_is_idle(self):
        clock = Clock()
        tracker = SessionTracker(timeout_seconds=600, now=clock)
        tracker.touch({"session_id": "slicer-1"})

        clock.tick(601)

        self.assertTrue(tracker.expired())

    def test_expires_without_any_session(self):
        clock = Clock()
        tracker = SessionTracker(timeout_seconds=600, now=clock)

        clock.tick(600)

        self.assertTrue(tracker.expired())

    def test_keeps_latest_autosave_after_expiry(self):
        clock = Clock()
        tracker = SessionTracker(timeout_seconds=600, now=clock)
        autosave = {"segmentation": "autosave/segmented_volume.seg.nrrd", "embeddings": "autosave/segmented_volume_embeddings"}
        tracker.touch({"session_id": "slicer-1", "autosave": autosave})

        clock.tick(601)
        tracker.expired()

        self.assertEqual(tracker.autosaves(), {"slicer-1": autosave})


if __name__ == "__main__":
    unittest.main()
