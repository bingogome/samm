from base64 import b64decode, b64encode
import time
import unittest

from samm_server.backends.base import BackendError
from samm_server.prediction_jobs import PredictionJobService


class FakePreparer:
    def __init__(self, prepared=None, error=None):
        self.prepared = prepared
        self.error = error
        self.calls = []

    def prepared_payload(self):
        return self.prepared

    def predict(self, image_bytes, shape, points, labels, box=None, mask=None, text=None):
        if self.error:
            raise self.error
        self.calls.append((image_bytes, shape, points, labels, box, mask, text))
        return bytes([len(self.calls)] * (shape[0] * shape[1]))


class PredictionJobServiceTest(unittest.TestCase):
    def test_job_requires_prepared_model(self):
        result = PredictionJobService(FakePreparer()).start({"total": 1})

        self.assertEqual(result.status_code, 409)
        self.assertEqual(result.payload, {"error": "model not prepared"})

    def test_job_rejects_invalid_total(self):
        result = PredictionJobService(FakePreparer({"weight_id": "sam_vit_b"})).start({"total": 0})

        self.assertEqual(result.status_code, 400)
        self.assertEqual(result.payload, {"error": "total must be a positive integer"})

    def test_job_accepts_streamed_box_items(self):
        preparer = FakePreparer({"weight_id": "sam_vit_b"})
        service = PredictionJobService(preparer)
        result = service.start({"total": 2})

        self.assertEqual(result.status_code, 202)
        job_id = result.payload["job_id"]
        service.add_items(job_id, {"items": [job_item("slice-1", bytes(12))]})
        service.add_items(job_id, {"items": [job_item("slice-2", bytes(range(12)))]})
        state = self.wait_for_terminal_state(service, job_id)

        self.assertEqual(state.payload["status"], "complete")
        self.assertEqual(state.payload["completed"], 2)
        self.assertEqual([item["key"] for item in state.payload["results"]], ["slice-1", "slice-2"])
        self.assertEqual(b64decode(state.payload["results"][0]["data"]), bytes([1, 1, 1, 1]))
        self.assertEqual(preparer.calls[0], (bytes(12), [2, 2, 3], [], [], [0, 0, 1, 1], None, None))

    def test_job_accepts_streamed_mask_items(self):
        preparer = FakePreparer({"weight_id": "sam_vit_b"})
        service = PredictionJobService(preparer)
        result = service.start({"total": 1})
        item = job_item("slice-1")
        item["box"] = None
        item["mask"] = {"shape": [2, 2], "data": b64encode(bytes([0, 1, 1, 0])).decode("ascii")}
        service.add_items(result.payload["job_id"], {"items": [item]})

        state = self.wait_for_terminal_state(service, result.payload["job_id"])

        self.assertEqual(state.payload["status"], "complete")
        self.assertEqual(preparer.calls[0], (bytes(12), [2, 2, 3], [], [], None, {"shape": [2, 2], "data": bytes([0, 1, 1, 0])}, None))

    def test_job_accepts_streamed_text_items(self):
        preparer = FakePreparer({"weight_id": "sam3"})
        service = PredictionJobService(preparer)
        result = service.start({"total": 1})
        item = job_item("slice-1")
        item["box"] = None
        item["text"] = "kidney"
        service.add_items(result.payload["job_id"], {"items": [item]})

        state = self.wait_for_terminal_state(service, result.payload["job_id"])

        self.assertEqual(state.payload["status"], "complete")
        self.assertEqual(preparer.calls[0], (bytes(12), [2, 2, 3], [], [], None, None, "kidney"))

    def test_job_rejects_too_many_streamed_items(self):
        service = PredictionJobService(FakePreparer({"weight_id": "sam_vit_b"}))
        result = service.start({"total": 1})

        too_many = service.add_items(result.payload["job_id"], {"items": [job_item("slice-1"), job_item("slice-2")]})

        self.assertEqual(too_many.status_code, 400)
        self.assertEqual(too_many.payload["error"], "too many prediction items")

    def test_job_reports_backend_error(self):
        service = PredictionJobService(FakePreparer({"weight_id": "sam_vit_b"}, BackendError("backend failed")))
        result = service.start({"total": 1})
        service.add_items(result.payload["job_id"], {"items": [job_item("slice-1")]})

        state = self.wait_for_terminal_state(service, result.payload["job_id"])
        self.assertEqual(state.payload["status"], "failed")
        self.assertEqual(state.payload["error"], "backend failed")

    def test_unknown_job_returns_404(self):
        result = PredictionJobService(FakePreparer({"weight_id": "sam_vit_b"})).state("missing")

        self.assertEqual(result.status_code, 404)
        self.assertEqual(result.payload, {"error": "prediction job not found", "job_id": "missing"})

    def wait_for_terminal_state(self, service, job_id):
        state = None
        for _ in range(50):
            state = service.state(job_id)
            if state.payload["status"] in ("complete", "failed"):
                return state
            time.sleep(0.01)
        return state


def job_item(key, image_bytes=None):
    image_bytes = image_bytes or bytes(12)
    return {
        "key": key,
        "slice_spec": {"view": "Red", "axis": 0, "index": 1},
        "image": {"shape": [2, 2, 3], "data": b64encode(image_bytes).decode("ascii")},
        "points": [],
        "labels": [],
        "box": [0, 0, 1, 1],
    }


if __name__ == "__main__":
    unittest.main()
