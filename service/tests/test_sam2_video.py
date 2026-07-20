import types
import unittest

from samm_server.backends.base import BackendError
from samm_server.backends.sam2 import Sam2Backend


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


class FakeVideoPredictor:
    def __init__(self, lock):
        self.lock = lock
        self.seed_calls = []
        self.propagate_calls = []
        self.reset_calls = 0

    def add_new_points_or_box(self, state, frame_idx, obj_id, points, labels, box):
        self.assert_locked()
        self.seed_calls.append({
            "state": state,
            "frame_idx": frame_idx,
            "obj_id": obj_id,
            "points": points,
            "labels": labels,
            "box": box,
        })
        return frame_idx, [obj_id], frame_idx

    def propagate_in_video(self, state, start_frame_idx, reverse):
        self.assert_locked()
        self.propagate_calls.append((start_frame_idx, reverse))
        if reverse:
            positions = range(start_frame_idx, -1, -1)
        else:
            positions = range(start_frame_idx, state["num_frames"])
        for position in positions:
            self.assert_locked()
            yield position, [1], position

    def reset_state(self, state):
        self.assert_locked()
        self.reset_calls += 1

    def assert_locked(self):
        if not self.lock.held:
            raise AssertionError("video model used without its weight lock")


class HarnessSam2Backend(Sam2Backend):
    def __init__(self):
        super().__init__()
        self.preprocess_calls = []
        self.initialize_calls = []

    def required_module(self, name, package_name=None):
        return types.SimpleNamespace(uint8="uint8")

    def preprocess_video_frames(self, frames, predictor, offload_video_to_cpu, torch, np):
        predictor.assert_locked()
        self.preprocess_calls.append((frames, offload_video_to_cpu))
        return "prepared-images"

    def initialize_video_state(
        self,
        predictor,
        images,
        video_height,
        video_width,
        offload_video_to_cpu,
        offload_state_to_cpu,
        torch,
    ):
        predictor.assert_locked()
        self.initialize_calls.append((
            images,
            video_height,
            video_width,
            offload_video_to_cpu,
            offload_state_to_cpu,
        ))
        return {"num_frames": len(self.preprocess_calls[-1][0])}

    def video_mask_payload(self, frame_index, shape, mask_logits, torch):
        return {
            "frame_index": frame_index,
            "shape": list(shape),
            "data": bytes([mask_logits]),
        }


class FakeStatePredictor:
    def __init__(self):
        self.device = "cuda:0"
        self.feature_calls = []
        self.init_calls = []

    def _get_image_feature(self, state, frame_idx, batch_size):
        self.feature_calls.append((state, frame_idx, batch_size))

    def init_state(self, images, height, width, **kwargs):
        self.init_calls.append((images, height, width, kwargs))
        return {"native": True}


class FakeTorch:
    @staticmethod
    def device(name):
        return f"device:{name}"


class FakeMaskArray:
    def __init__(self, data):
        self.data = data
        self.shape = None

    def reshape(self, shape):
        self.shape = list(shape)
        return self

    def copy(self):
        return self


class FakeNumpy:
    uint8 = "uint8"

    @staticmethod
    def frombuffer(data, dtype):
        return FakeMaskArray(data)


class FakeMaskPromptPredictor:
    def __init__(self):
        self.calls = []

    def add_new_mask(self, state, frame_idx, obj_id, mask):
        self.calls.append((state, frame_idx, obj_id, mask))


class FakePreparedVideoModel:
    def __init__(self, config, checkpoint, device):
        self.config = config
        self.checkpoint = checkpoint
        self.device = device


class FakePreparedImagePredictor:
    def __init__(self, model):
        self.model = model


def fake_prepare_modules(name, package_name=None):
    if name == "sam2.build_sam":
        return types.SimpleNamespace(
            build_sam2_video_predictor=lambda config, checkpoint, device: FakePreparedVideoModel(
                config,
                checkpoint,
                device,
            ),
        )
    if name == "sam2.sam2_image_predictor":
        return types.SimpleNamespace(SAM2ImagePredictor=FakePreparedImagePredictor)
    raise AssertionError(name)


def video_frames():
    return [
        {"frame_index": 30, "image_bytes": b"\x03\x03\x03", "shape": [1, 1, 3]},
        {"frame_index": 10, "image_bytes": b"\x01\x01\x01", "shape": [1, 1, 3]},
        {"frame_index": 20, "image_bytes": b"\x02\x02\x02", "shape": [1, 1, 3]},
    ]


class Sam2VideoBackendTest(unittest.TestCase):
    def backend(self):
        backend = HarnessSam2Backend()
        weight = types.SimpleNamespace(id="weight", label="Weight")
        lock = TrackingLock()
        predictor = FakeVideoPredictor(lock)
        backend.video_predictors[weight.id] = predictor
        backend.locks[weight.id] = lock
        return backend, weight, predictor, lock

    def test_both_directions_reset_reseed_and_dedupe_seed(self):
        backend, weight, predictor, lock = self.backend()
        prompt = {
            "frame_index": 20,
            "points": [[0, 0]],
            "labels": [1],
            "box": [0, 0, 1, 1],
        }

        results = list(backend.propagate_video(
            weight,
            video_frames(),
            prompt,
            direction="both",
            offload_video_to_cpu=False,
            offload_state_to_cpu=True,
        ))

        self.assertEqual([result["frame_index"] for result in results], [20, 30, 10])
        self.assertEqual([result["data"] for result in results], [b"\x01", b"\x02", b"\x00"])
        self.assertEqual(predictor.propagate_calls, [(1, False), (1, True)])
        self.assertEqual([call["frame_idx"] for call in predictor.seed_calls], [1, 1])
        self.assertEqual(predictor.seed_calls[0]["points"], [[0, 0]])
        self.assertEqual(predictor.seed_calls[0]["labels"], [1])
        self.assertEqual(predictor.seed_calls[0]["box"], [0, 0, 1, 1])
        self.assertEqual(predictor.reset_calls, 2)
        self.assertEqual(backend.initialize_calls, [("prepared-images", 1, 1, False, True)])
        self.assertEqual(lock.entries, 1)
        self.assertFalse(lock.held)

    def test_prepare_retires_prior_weight_and_rejects_queued_stale_run(self):
        backend = Sam2Backend()
        backend.required_module = fake_prepare_modules
        weight_a = types.SimpleNamespace(id="a", label="A", backend="sam2", model_type="config-a")
        weight_b = types.SimpleNamespace(id="b", label="B", backend="sam2", model_type="config-b")
        backend.prepare(weight_a, "a.pt", "cpu")
        backend.embeddings["old-embedding"] = {"weight_id": "a"}
        backend.embedding_ids[("a", "image")] = "old-embedding"
        queued = backend.propagate_video(
            weight_a,
            video_frames(),
            {"frame_index": 10, "box": [0, 0, 1, 1]},
        )

        backend.prepare(weight_b, "b.pt", "cpu")

        self.assertEqual(set(backend.models), {"b"})
        self.assertEqual(set(backend.video_predictors), {"b"})
        self.assertEqual(set(backend.predictors), {"b"})
        self.assertEqual(set(backend.predictor_types), {"b"})
        self.assertEqual(set(backend.locks), {"b"})
        self.assertEqual(backend.image_keys, {})
        self.assertEqual(backend.embeddings, {})
        self.assertEqual(backend.embedding_ids, {})
        self.assertIs(backend.predictors["b"].model, backend.video_predictors["b"])
        with self.assertRaisesRegex(BackendError, "model not prepared"):
            list(queued)

    def test_forward_and_backward_propagate_only_requested_side(self):
        for direction, expected_indices, expected_reverse in (
            ("forward", [20, 30], False),
            ("backward", [20, 10], True),
        ):
            with self.subTest(direction=direction):
                backend, weight, predictor, _lock = self.backend()
                prompt = {"frame_index": 20, "box": [0, 0, 1, 1]}

                results = list(backend.propagate_video(weight, video_frames(), prompt, direction))

                self.assertEqual([result["frame_index"] for result in results], expected_indices)
                self.assertEqual(predictor.propagate_calls, [(1, expected_reverse)])
                self.assertEqual(len(predictor.seed_calls), 1)
                self.assertEqual(predictor.reset_calls, 1)

    def test_backward_from_first_frame_still_yields_seed(self):
        backend, weight, predictor, _lock = self.backend()

        results = list(backend.propagate_video(
            weight,
            video_frames(),
            {"frame_index": 10, "box": [0, 0, 1, 1]},
            direction="backward",
        ))

        self.assertEqual([result["frame_index"] for result in results], [10])
        self.assertEqual(predictor.propagate_calls, [(0, True)])

    def test_cancellation_stops_between_frames_and_releases_lock(self):
        backend, weight, predictor, lock = self.backend()
        cancel_event = CancelEvent()
        iterator = backend.propagate_video(
            weight,
            video_frames(),
            {"frame_index": 10, "box": [0, 0, 1, 1]},
            cancel_event=cancel_event,
        )

        first = next(iterator)
        self.assertEqual(first["frame_index"], 10)
        self.assertTrue(lock.held)
        cancel_event.set()
        self.assertEqual(list(iterator), [])

        self.assertFalse(lock.held)
        self.assertEqual(predictor.reset_calls, 1)

    def test_pre_cancelled_run_does_not_touch_model(self):
        backend, weight, predictor, lock = self.backend()

        results = list(backend.propagate_video(
            weight,
            video_frames(),
            {"frame_index": 10, "box": [0, 0, 1, 1]},
            cancel_event=CancelEvent(True),
        ))

        self.assertEqual(results, [])
        self.assertEqual(lock.entries, 0)
        self.assertEqual(predictor.seed_calls, [])
        self.assertEqual(backend.preprocess_calls, [])

    def test_video_input_validation(self):
        backend = Sam2Backend()
        valid_prompt = {"frame_index": 10, "box": [0, 0, 1, 1]}
        cases = [
            ([], valid_prompt, "both", "at least one frame"),
            (video_frames(), valid_prompt, "sideways", "direction"),
            (
                [
                    {"frame_index": 10, "image_bytes": b"rgb", "shape": [1, 1, 3]},
                    {"frame_index": 10, "image_bytes": b"rgb", "shape": [1, 1, 3]},
                ],
                valid_prompt,
                "both",
                "unique",
            ),
            (
                [{"frame_index": True, "image_bytes": b"rgb", "shape": [1, 1, 3]}],
                {"frame_index": True, "box": [0, 0, 1, 1]},
                "both",
                "integers",
            ),
            (video_frames(), {"frame_index": 20, "points": [[0, 0]], "labels": []}, "both", "same length"),
            (
                video_frames(),
                {
                    "frame_index": 20,
                    "mask": {"shape": [2, 2], "data": b"\x01\x00\x00\x01"},
                },
                "both",
                "match the seed frame",
            ),
        ]

        for frames, prompt, direction, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(BackendError, message):
                    backend.propagate_video(frames=frames, weight=None, prompt=prompt, direction=direction)

    def test_preprocess_video_frames_handles_multiple_batches(self):
        try:
            import numpy as np
            import torch
        except ImportError:
            self.skipTest("torch is available in the SAM2 environments")
        backend = Sam2Backend()
        predictor = types.SimpleNamespace(image_size=4, device=torch.device("cpu"))
        frames = [
            {
                "frame_index": index,
                "shape": [2, 3, 3],
                "image_bytes": bytes([index] * 18),
            }
            for index in range(10)
        ]

        images = backend.preprocess_video_frames(
            frames,
            predictor,
            offload_video_to_cpu=True,
            torch=torch,
            np=np,
        )

        self.assertEqual(tuple(images.shape), (10, 3, 4, 4))
        self.assertEqual(images.device.type, "cpu")
        self.assertTrue(torch.isfinite(images).all())
        self.assertFalse(torch.equal(images[0], images[8]))

    def test_base_sam2_state_is_initialized_from_in_memory_images(self):
        backend = Sam2Backend()
        predictor = FakeStatePredictor()
        images = ["frame-0", "frame-1"]

        state = backend.initialize_video_state(
            predictor,
            images,
            12,
            16,
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
            torch=FakeTorch,
        )

        self.assertIs(state["images"], images)
        self.assertEqual(state["num_frames"], 2)
        self.assertEqual(state["video_height"], 12)
        self.assertEqual(state["video_width"], 16)
        self.assertEqual(state["storage_device"], "device:cpu")
        self.assertEqual(predictor.feature_calls, [(state, 0, 1)])

    def test_medsam2_uses_native_npz_state_initializer(self):
        backend = Sam2Backend("MedSAM2")
        predictor = FakeStatePredictor()
        images = object()

        state = backend.initialize_video_state(
            predictor,
            images,
            12,
            16,
            offload_video_to_cpu=False,
            offload_state_to_cpu=True,
            torch=FakeTorch,
        )

        self.assertEqual(state, {"native": True})
        self.assertEqual(predictor.init_calls, [(
            images,
            12,
            16,
            {"offload_video_to_cpu": False, "offload_state_to_cpu": True},
        )])
        self.assertEqual(predictor.feature_calls, [])

    def test_mask_seed_uses_raw_lossless_mask_bytes(self):
        backend = Sam2Backend()
        predictor = FakeMaskPromptPredictor()
        state = object()
        prompt = {
            "mask": {
                "shape": [2, 2],
                "data": b"\x01\x00\x00\x01",
            },
        }

        backend.seed_video_prompt(predictor, state, 3, prompt, FakeNumpy)

        called_state, frame_idx, obj_id, mask = predictor.calls[0]
        self.assertIs(called_state, state)
        self.assertEqual(frame_idx, 3)
        self.assertEqual(obj_id, 1)
        self.assertEqual(mask.data, prompt["mask"]["data"])
        self.assertEqual(mask.shape, [2, 2])


if __name__ == "__main__":
    unittest.main()
