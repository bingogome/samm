from contextlib import nullcontext
import types
import unittest

from samm_server.backends.base import BackendError
from samm_server.backends.sam3 import Sam3Backend


class TrackingLock:
    def __init__(self):
        self.held = False
        self.entries = 0

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.release()
        return False

    def acquire(self):
        if self.held:
            raise AssertionError("lock entered twice")
        self.held = True
        self.entries += 1

    def release(self):
        if not self.held:
            raise AssertionError("lock released while not held")
        self.held = False


class CancelEvent:
    def __init__(self, value=False):
        self.value = value

    def is_set(self):
        return self.value

    def set(self):
        self.value = True


class FakeProbabilities:
    def __init__(self, best=0):
        self.best = best

    def __len__(self):
        return 1

    def argmax(self):
        return self.best


class FakeVideoModel:
    def __init__(self, lock):
        self.lock = lock
        self.device = "cuda:0"
        self.init_calls = []
        self.prompt_calls = []
        self.propagate_calls = []
        self.reset_calls = 0

    def init_state(self, **kwargs):
        self.assert_locked()
        self.init_calls.append(kwargs)
        return {"num_frames": len(kwargs["resource_path"])}

    def add_prompt(self, **kwargs):
        self.assert_locked()
        self.prompt_calls.append(kwargs)
        position = kwargs["frame_idx"]
        return position, {
            "out_obj_ids": [7],
            "out_probs": FakeProbabilities(),
            "value": position + 1,
        }

    def propagate_in_video(self, state, start_frame_idx, reverse):
        self.assert_locked()
        self.propagate_calls.append((start_frame_idx, reverse))
        positions = (
            range(start_frame_idx - 1, -1, -1)
            if reverse
            else range(start_frame_idx, state["num_frames"])
        )
        for position in positions:
            self.assert_locked()
            yield position, {"value": position + 1}

    def reset_state(self, state):
        self.assert_locked()
        self.reset_calls += 1

    def assert_locked(self):
        if not self.lock.held:
            raise AssertionError("SAM3 video model used without its weight lock")


class HarnessSam3Backend(Sam3Backend):
    def __init__(self):
        super().__init__(enable_video_propagation=True)

    def image(self, image_bytes, shape):
        return (image_bytes, tuple(shape))

    def inference_context(self, weight):
        return nullcontext()

    def video_mask_payload(self, frame_index, shape, outputs):
        return {
            "frame_index": frame_index,
            "shape": list(shape),
            "data": bytes([outputs["value"]]),
        }


def video_frames():
    return [
        {"frame_index": 30, "image_bytes": b"\x03\x03\x03", "shape": [1, 1, 3]},
        {"frame_index": 10, "image_bytes": b"\x01\x01\x01", "shape": [1, 1, 3]},
        {"frame_index": 20, "image_bytes": b"\x02\x02\x02", "shape": [1, 1, 3]},
    ]


class Sam3VideoBackendTest(unittest.TestCase):
    def backend(self):
        backend = HarnessSam3Backend()
        weight = types.SimpleNamespace(id="sam3", label="SAM 3")
        lock = TrackingLock()
        model = FakeVideoModel(lock)
        backend.video_models[weight.id] = model
        backend.locks[weight.id] = lock
        return backend, weight, model, lock

    def test_both_directions_stream_seed_forward_then_backward(self):
        backend, weight, model, lock = self.backend()

        results = list(backend.propagate_video(
            weight,
            video_frames(),
            {"frame_index": 20, "box": [0, 0, 1, 1]},
            direction="both",
            offload_video_to_cpu=False,
            offload_state_to_cpu=True,
        ))

        self.assertEqual([result["frame_index"] for result in results], [20, 30, 10])
        self.assertEqual([result["data"] for result in results], [b"\x02", b"\x03", b"\x01"])
        self.assertEqual(model.propagate_calls, [(1, False), (1, True)])
        self.assertEqual(model.reset_calls, 1)
        self.assertEqual(model.init_calls[0]["offload_video_to_cpu"], False)
        self.assertEqual(model.init_calls[0]["offload_state_to_cpu"], True)
        self.assertEqual(lock.entries, 1)
        self.assertFalse(lock.held)

    def test_box_and_points_seed_then_refine_the_box_object(self):
        backend, weight, model, _lock = self.backend()

        results = list(backend.propagate_video(
            weight,
            video_frames(),
            {
                "frame_index": 20,
                "points": [[0.25, 0.75]],
                "labels": [1],
                "box": [0, 0, 1, 1],
            },
            direction="forward",
        ))

        self.assertEqual([result["frame_index"] for result in results], [20, 30])
        self.assertEqual(len(model.prompt_calls), 2)
        self.assertEqual(model.prompt_calls[0]["boxes_xywh"], [[0.0, 0.0, 1.0, 1.0]])
        self.assertEqual(model.prompt_calls[0]["box_labels"], [1])
        self.assertEqual(model.prompt_calls[1]["points"], [[0.25, 0.75]])
        self.assertEqual(model.prompt_calls[1]["point_labels"], [1])
        self.assertEqual(model.prompt_calls[1]["obj_id"], 7)
        self.assertFalse(model.prompt_calls[1]["rel_coordinates"])

    def test_text_and_box_are_sent_as_one_sam3_semantic_prompt(self):
        backend, weight, model, _lock = self.backend()

        list(backend.propagate_video(
            weight,
            video_frames(),
            {"frame_index": 10, "text": "brain tumor", "box": [0, 0, 1, 1]},
            direction="backward",
        ))

        self.assertEqual(len(model.prompt_calls), 1)
        self.assertEqual(model.prompt_calls[0]["text_str"], "brain tumor")
        self.assertEqual(model.prompt_calls[0]["boxes_xywh"], [[0.0, 0.0, 1.0, 1.0]])

    def test_cancellation_releases_the_model_lock(self):
        backend, weight, model, lock = self.backend()
        cancel = CancelEvent()
        iterator = backend.propagate_video(
            weight,
            video_frames(),
            {"frame_index": 10, "text": "tumor"},
            cancel_event=cancel,
        )

        self.assertEqual(next(iterator)["frame_index"], 10)
        self.assertTrue(lock.held)
        cancel.set()
        self.assertEqual(list(iterator), [])
        self.assertEqual(model.reset_calls, 1)
        self.assertFalse(lock.held)

    def test_pre_cancelled_run_does_not_touch_the_model(self):
        backend, weight, model, lock = self.backend()

        results = list(backend.propagate_video(
            weight,
            video_frames(),
            {"frame_index": 10, "text": "tumor"},
            cancel_event=CancelEvent(True),
        ))

        self.assertEqual(results, [])
        self.assertEqual(model.init_calls, [])
        self.assertEqual(lock.entries, 0)

    def test_validation_rejects_mask_and_text_with_points(self):
        backend = Sam3Backend(enable_video_propagation=True)

        with self.assertRaisesRegex(BackendError, "does not support mask"):
            backend.propagate_video(
                None,
                video_frames(),
                {"frame_index": 10, "mask": {"shape": [1, 1], "data": b"\x01"}},
            )
        with self.assertRaisesRegex(BackendError, "cannot be combined"):
            backend.propagate_video(
                None,
                video_frames(),
                {"frame_index": 10, "text": "tumor", "points": [[0, 0]], "labels": [1]},
            )

    def test_video_mask_payload_unions_all_sam3_objects(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is installed in the service environments")
        backend = Sam3Backend(enable_video_propagation=True)
        outputs = {
            "out_binary_masks": np.array(
                [
                    [[True, False], [False, False]],
                    [[False, True], [False, False]],
                ]
            )
        }

        payload = backend.video_mask_payload(4, [2, 2], outputs)

        self.assertEqual(payload["frame_index"], 4)
        self.assertEqual(payload["shape"], [2, 2])
        self.assertEqual(payload["data"], bytes([1, 1, 0, 0]))


if __name__ == "__main__":
    unittest.main()
