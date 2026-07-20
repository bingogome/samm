from base64 import b64decode, b64encode
from threading import Event
import time
import unittest

from samm_server.video_prediction_jobs import VideoPredictionJobService


class FakePreparer:
    def __init__(self, prepared=None, results=None, error=None, block=False):
        self.prepared = prepared
        self.results = results
        self.error = error
        self.block = block
        self.calls = []
        self.started = Event()

    def prepared_payload(self):
        return self.prepared

    def propagate_video(self, frames, prompt, **kwargs):
        self.calls.append((frames, prompt, kwargs))
        self.started.set()
        if self.block:
            kwargs["cancel_event"].wait(1)
        if self.error:
            raise self.error
        results = self.results
        if results is None:
            results = [
                {
                    "frame_index": frame["frame_index"],
                    "shape": frame["shape"][:2],
                    "data": bytes([frame["frame_index"] + 1] * (frame["shape"][0] * frame["shape"][1])),
                }
                for frame in frames
            ]
        yield from results


class VideoPredictionJobServiceTest(unittest.TestCase):
    def test_start_requires_video_capable_prepared_model(self):
        unprepared = VideoPredictionJobService(FakePreparer()).start({"total": 2})
        unsupported = VideoPredictionJobService(
            FakePreparer({"weight_id": "sam_vit_b", "capabilities": {"video_propagation": False}})
        ).start({"total": 2})

        self.assertEqual(unprepared.status_code, 409)
        self.assertEqual(unprepared.payload, {"error": "model not prepared"})
        self.assertEqual(unsupported.status_code, 409)
        self.assertEqual(unsupported.payload["error"], "prepared model does not support video propagation")

    def test_start_validates_total_and_enters_uploading_state(self):
        service = VideoPredictionJobService(video_preparer())

        invalid = service.start({"total": True})
        started = service.start({"total": 2})

        self.assertEqual(invalid.status_code, 400)
        self.assertEqual(started.status_code, 202)
        self.assertEqual(started.payload["status"], "uploading")
        self.assertEqual(started.payload["weight_id"], "sam2_hiera_tiny")
        self.assertEqual(started.payload["submitted"], 0)
        self.assertEqual(started.payload["next_cursor"], 0)

    def test_bounds_frame_count_input_bytes_and_concurrent_jobs(self):
        frame_limited = VideoPredictionJobService(video_preparer(), max_frames=1)
        too_many_frames = frame_limited.start({"total": 2})
        self.assertEqual(too_many_frames.status_code, 400)
        self.assertEqual(too_many_frames.payload["max_frames"], 1)

        byte_limited = VideoPredictionJobService(video_preparer(), max_input_bytes=11)
        started = byte_limited.start({"total": 1})
        too_many_bytes = byte_limited.add_frames(started.payload["job_id"], {"frames": [frame(0)]})
        self.assertEqual(too_many_bytes.status_code, 413)
        self.assertEqual(too_many_bytes.payload["max_input_bytes"], 11)
        self.assertEqual(byte_limited.state(started.payload["job_id"]).payload["submitted"], 0)

        job_limited = VideoPredictionJobService(video_preparer(), max_jobs=1)
        first = job_limited.start({"total": 1})
        rejected = job_limited.start({"total": 1})
        self.assertEqual(rejected.status_code, 429)
        job_limited.cancel(first.payload["job_id"])
        replacement = job_limited.start({"total": 1})
        self.assertEqual(replacement.status_code, 202)
        self.assertEqual(job_limited.state(first.payload["job_id"]).status_code, 404)

    def test_prunes_stale_terminal_and_uploading_jobs(self):
        now = [100.0]
        service = VideoPredictionJobService(
            video_preparer(),
            terminal_retention_seconds=5,
            clock=lambda: now[0],
        )
        started = service.start({"total": 1})
        job_id = started.payload["job_id"]
        service.cancel(job_id)

        now[0] += 4.9
        self.assertEqual(service.state(job_id).status_code, 200)
        now[0] += 0.2
        self.assertEqual(service.state(job_id).status_code, 404)

        uploading = VideoPredictionJobService(
            video_preparer(),
            upload_retention_seconds=5,
            clock=lambda: now[0],
        )
        upload_job_id = uploading.start({"total": 1}).payload["job_id"]
        now[0] += 5
        self.assertEqual(uploading.state(upload_job_id).status_code, 404)

    def test_uploads_frames_atomically_and_in_any_order(self):
        service, job_id = started_service(2)

        result = service.add_frames(job_id, {"frames": [frame(1, bytes(range(12))), frame(0, bytes(12))]})

        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.payload["submitted"], 2)
        self.assertEqual(result.payload["status"], "uploading")
        self.assertEqual(sorted(service.jobs[job_id]["frames"]), [0, 1])

    def test_rejects_duplicate_out_of_range_and_mismatched_frames(self):
        service, job_id = started_service(2)

        duplicate_batch = service.add_frames(job_id, {"frames": [frame(0), frame(0)]})
        self.assertEqual(duplicate_batch.status_code, 400)
        self.assertEqual(service.state(job_id).payload["submitted"], 0)

        out_of_range = service.add_frames(job_id, {"frames": [frame(2)]})
        self.assertEqual(out_of_range.status_code, 400)
        self.assertEqual(service.state(job_id).payload["submitted"], 0)

        mismatch = service.add_frames(job_id, {"frames": [frame(0), frame(1, shape=[3, 2, 3])]})
        self.assertEqual(mismatch.status_code, 400)
        self.assertEqual(service.state(job_id).payload["submitted"], 0)

        service.add_frames(job_id, {"frames": [frame(0)]})
        repeated = service.add_frames(job_id, {"frames": [frame(0)]})
        self.assertEqual(repeated.status_code, 409)
        self.assertEqual(service.state(job_id).payload["submitted"], 1)

    def test_run_requires_all_frames_and_valid_prompt(self):
        service, job_id = started_service(2)
        service.add_frames(job_id, {"frames": [frame(0)]})

        incomplete = service.run(job_id, run_payload())
        invalid_prompt = service.run(job_id, {"prompt": {"frame_index": 0, "points": [], "labels": []}})

        self.assertEqual(incomplete.status_code, 409)
        self.assertEqual(incomplete.payload["submitted"], 1)
        self.assertEqual(invalid_prompt.status_code, 400)
        self.assertEqual(invalid_prompt.payload["error"], "prompt points/labels, box, mask, or text is required")

    def test_run_validates_options_and_mask_shape(self):
        service, job_id = ready_service(1)

        bad_direction = service.run(job_id, {**run_payload(), "direction": "sideways"})
        bad_offload = service.run(job_id, {**run_payload(), "offload_video_to_cpu": 1})
        bad_mask = run_payload()
        bad_mask["prompt"] = {
            "frame_index": 0,
            "points": [],
            "labels": [],
            "mask": {"shape": [1, 1], "data": b64encode(bytes(1)).decode("ascii")},
        }
        mask_result = service.run(job_id, bad_mask)

        self.assertEqual(bad_direction.status_code, 400)
        self.assertEqual(bad_offload.status_code, 400)
        self.assertEqual(mask_result.status_code, 400)
        self.assertEqual(mask_result.payload["error"], "prompt.mask.shape must match video frame shape")

    def test_run_rejects_mask_combined_with_points_or_box(self):
        service, job_id = ready_service(1)
        payload = run_payload()
        payload["prompt"]["mask"] = {
            "shape": [2, 2],
            "data": b64encode(bytes(4)).decode("ascii"),
        }

        result = service.run(job_id, payload)

        self.assertEqual(result.status_code, 400)
        self.assertEqual(result.payload["error"], "prompt.mask cannot be combined with points, box, or text")

    def test_text_prompt_is_normalized_and_forwarded(self):
        preparer = video_preparer(
            prepared=video_prepared(
                "sam3",
                video_text=True,
                video_mask=False,
            )
        )
        service, job_id = ready_service(1, preparer)

        accepted = service.run(
            job_id,
            {"prompt": {"frame_index": 0, "text": "  brain tumor  "}},
        )
        state = wait_for_terminal_state(service, job_id)

        self.assertEqual(accepted.status_code, 202)
        self.assertEqual(state.payload["status"], "complete")
        self.assertEqual(
            preparer.calls[0][1],
            {
                "frame_index": 0,
                "points": [],
                "labels": [],
                "box": None,
                "mask": None,
                "text": "brain tumor",
            },
        )

    def test_model_specific_video_prompt_capabilities_are_enforced(self):
        preparer = video_preparer(
            prepared=video_prepared(
                "sam3",
                video_text=True,
                video_mask=False,
            )
        )
        service, job_id = ready_service(1, preparer)
        payload = {
            "prompt": {
                "frame_index": 0,
                "mask": {
                    "shape": [2, 2],
                    "data": b64encode(bytes(4)).decode("ascii"),
                },
            }
        }

        result = service.run(job_id, payload)

        self.assertEqual(result.status_code, 400)
        self.assertEqual(
            result.payload["error"],
            "prepared model does not support mask prompts for video propagation",
        )
        self.assertEqual(preparer.calls, [])

        point_service, point_job_id = ready_service(1, preparer)
        point_result = point_service.run(
            point_job_id,
            {
                "prompt": {
                    "frame_index": 0,
                    "points": [[1, 1]],
                    "labels": [1],
                    "text": "tumor",
                }
            },
        )
        self.assertEqual(point_result.status_code, 400)
        self.assertEqual(
            point_result.payload["error"],
            "video text prompts cannot be combined with points",
        )

    def test_runs_video_backend_and_polls_results_by_cursor(self):
        preparer = video_preparer()
        service, job_id = ready_service(2, preparer)
        payload = run_payload()
        payload.update(
            {
                "direction": "forward",
                "offload_video_to_cpu": False,
                "offload_state_to_cpu": True,
            }
        )

        accepted = service.run(job_id, payload)
        state = wait_for_terminal_state(service, job_id)
        second_page = service.state(job_id, 1)

        self.assertEqual(accepted.status_code, 202)
        self.assertEqual(accepted.payload["status"], "queued")
        self.assertEqual(state.payload["status"], "complete")
        self.assertEqual(state.payload["completed"], 2)
        self.assertEqual([item["sequence"] for item in state.payload["results"]], [0, 1])
        self.assertEqual([item["frame_index"] for item in state.payload["results"]], [0, 1])
        self.assertEqual(b64decode(state.payload["results"][1]["data"]), bytes([2] * 4))
        self.assertEqual(second_page.payload["next_cursor"], 2)
        self.assertEqual([item["sequence"] for item in second_page.payload["results"]], [1])

        frames, prompt, options = preparer.calls[0]
        self.assertEqual([item["frame_index"] for item in frames], [0, 1])
        self.assertEqual(frames[0]["image_bytes"], bytes(12))
        self.assertEqual(prompt, {"frame_index": 0, "points": [[1, 1]], "labels": [1], "box": None, "mask": None})
        self.assertEqual(options["direction"], "forward")
        self.assertFalse(options["offload_video_to_cpu"])
        self.assertTrue(options["offload_state_to_cpu"])
        self.assertEqual(options["expected_weight_id"], "sam2_hiera_tiny")
        self.assertNotIn("image_bytes", service.jobs[job_id]["frames"][0])
        self.assertIsNone(service.jobs[job_id]["run_payload"])

    def test_run_fails_when_prepared_weight_changed(self):
        preparer = video_preparer()
        service, job_id = ready_service(1, preparer)
        preparer.prepared = video_prepared("medsam2_2411")

        result = service.run(job_id, run_payload())

        self.assertEqual(result.status_code, 409)
        self.assertEqual(result.payload["status"], "failed")
        self.assertEqual(result.payload["expected_weight_id"], "sam2_hiera_tiny")
        self.assertEqual(result.payload["actual_weight_id"], "medsam2_2411")
        self.assertEqual(preparer.calls, [])
        self.assertNotIn("image_bytes", service.jobs[job_id]["frames"][0])

    def test_backend_error_marks_job_failed(self):
        preparer = video_preparer(error=RuntimeError("video backend failed"))
        service, job_id = ready_service(1, preparer)

        service.run(job_id, run_payload())
        state = wait_for_terminal_state(service, job_id)

        self.assertEqual(state.payload["status"], "failed")
        self.assertEqual(state.payload["error"], "video backend failed")
        self.assertNotIn("image_bytes", service.jobs[job_id]["frames"][0])

    def test_cancel_is_idempotent_while_uploading(self):
        service, job_id = started_service(2)
        service.add_frames(job_id, {"frames": [frame(0)]})

        first = service.cancel(job_id)
        second = service.cancel(job_id)
        upload = service.add_frames(job_id, {"frames": [frame(1)]})

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(first.payload["status"], "cancelled")
        self.assertEqual(upload.status_code, 409)
        self.assertEqual(first.payload["submitted"], 1)
        self.assertNotIn("image_bytes", service.jobs[job_id]["frames"][0])

    def test_cancel_running_signals_backend_and_keeps_cancelled_state(self):
        preparer = video_preparer(block=True)
        service, job_id = ready_service(1, preparer)
        service.run(job_id, run_payload())
        self.assertTrue(preparer.started.wait(1))

        cancelled = service.cancel(job_id)
        state = wait_for_terminal_state(service, job_id)

        self.assertEqual(cancelled.payload["status"], "cancelled")
        self.assertEqual(state.payload["status"], "cancelled")
        self.assertTrue(preparer.calls[0][2]["cancel_event"].is_set())
        self.assertEqual(state.payload["results"], [])
        self.assertNotIn("image_bytes", service.jobs[job_id]["frames"][0])

    def test_cancel_terminal_returns_clear_conflict_error(self):
        service, job_id = ready_service(1)
        service.run(job_id, run_payload())
        wait_for_terminal_state(service, job_id)

        result = service.cancel(job_id)

        self.assertEqual(result.status_code, 409)
        self.assertEqual(result.payload["error"], "video prediction job is already terminal")

    def test_state_validates_cursor_and_unknown_job(self):
        service, job_id = started_service(1)

        invalid = service.state(job_id, -1)
        future = service.state(job_id, 1)
        unknown = service.state("missing")

        self.assertEqual(invalid.status_code, 400)
        self.assertEqual(invalid.payload, {"error": "cursor must be a non-negative integer"})
        self.assertEqual(future.status_code, 400)
        self.assertEqual(future.payload["next_cursor"], 0)
        self.assertEqual(unknown.status_code, 404)


def video_prepared(weight_id="sam2_hiera_tiny", **capabilities):
    return {
        "weight_id": weight_id,
        "capabilities": {"video_propagation": True, **capabilities},
    }


def video_preparer(**kwargs):
    prepared = kwargs.pop("prepared", video_prepared())
    return FakePreparer(prepared, **kwargs)


def started_service(total, preparer=None):
    service = VideoPredictionJobService(preparer or video_preparer())
    result = service.start({"total": total})
    return service, result.payload["job_id"]


def ready_service(total, preparer=None):
    service, job_id = started_service(total, preparer)
    service.add_frames(job_id, {"frames": [frame(index) for index in range(total)]})
    return service, job_id


def frame(frame_index, image_bytes=None, shape=None):
    shape = shape or [2, 2, 3]
    image_bytes = image_bytes if image_bytes is not None else bytes(shape[0] * shape[1] * shape[2])
    return {
        "frame_index": frame_index,
        "key": f"slice-{frame_index}",
        "slice_spec": {"view": "Red", "axis": 0, "index": frame_index},
        "image": {"shape": shape, "data": b64encode(image_bytes).decode("ascii")},
    }


def run_payload():
    return {"prompt": {"frame_index": 0, "points": [[1, 1]], "labels": [1]}}


def wait_for_terminal_state(service, job_id):
    state = None
    for _ in range(100):
        state = service.state(job_id)
        if state.payload["status"] in ("complete", "failed", "cancelled"):
            return state
        time.sleep(0.01)
    return state


if __name__ == "__main__":
    unittest.main()
