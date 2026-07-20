from base64 import b64encode
import time
import unittest

from samm_server.backends.base import BackendError
from samm_server.embedding_jobs import EmbeddingJobService


class FakePreparer:
    def __init__(self, prepared=None, error=None):
        self.prepared = prepared
        self.error = error
        self.calls = []

    def prepared_payload(self):
        return self.prepared

    def embed(self, image_bytes, shape):
        if self.error:
            raise self.error
        self.calls.append((image_bytes, shape))
        return {"embedding_id": f"emb-{len(self.calls)}", "shape": [shape[0], shape[1]]}


class EmbeddingJobServiceTest(unittest.TestCase):
    def test_job_requires_prepared_model(self):
        result = EmbeddingJobService(FakePreparer()).start({"total": 1})

        self.assertEqual(result.status_code, 409)
        self.assertEqual(result.payload, {"error": "model not prepared"})

    def test_job_rejects_invalid_total(self):
        result = EmbeddingJobService(FakePreparer({"weight_id": "sam_vit_b"})).start({"total": 0})

        self.assertEqual(result.status_code, 400)
        self.assertEqual(result.payload, {"error": "total must be a positive integer"})

    def test_job_rejects_empty_items(self):
        service = EmbeddingJobService(FakePreparer({"weight_id": "sam_vit_b"}))
        result = service.start({"total": 1})
        empty = service.add_items(result.payload["job_id"], {"items": []})

        self.assertEqual(empty.status_code, 400)
        self.assertEqual(empty.payload, {"error": "items must be a non-empty list"})

    def test_job_accepts_streamed_items(self):
        preparer = FakePreparer({"weight_id": "sam_vit_b"})
        service = EmbeddingJobService(preparer)
        result = service.start({"total": 2})

        self.assertEqual(result.status_code, 202)
        self.assertEqual(result.payload["submitted"], 0)
        job_id = result.payload["job_id"]
        first = service.add_items(job_id, {"items": [job_item("slice-1", bytes(12))]})
        second = service.add_items(job_id, {"items": [job_item("slice-2", bytes(range(12)))]})

        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.payload["submitted"], 1)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(second.payload["submitted"], 2)
        state = self.wait_for_terminal_state(service, job_id)
        self.assertEqual(state.payload["status"], "complete")
        self.assertEqual(state.payload["completed"], 2)
        self.assertEqual(state.payload["submitted"], 2)
        self.assertEqual(state.payload["total"], 2)
        self.assertEqual([item["key"] for item in state.payload["results"]], ["slice-1", "slice-2"])
        self.assertEqual([item["embedding_id"] for item in state.payload["results"]], ["emb-1", "emb-2"])

    def test_job_rejects_too_many_streamed_items(self):
        service = EmbeddingJobService(FakePreparer({"weight_id": "sam_vit_b"}))
        result = service.start({"total": 1})

        too_many = service.add_items(result.payload["job_id"], {"items": job_payload()["items"]})

        self.assertEqual(too_many.status_code, 400)
        self.assertEqual(too_many.payload["error"], "too many embedding items")

    def test_job_reports_backend_error(self):
        service = EmbeddingJobService(FakePreparer({"weight_id": "sam_vit_b"}, BackendError("backend failed")))
        result = service.start({"total": 2})
        service.add_items(result.payload["job_id"], job_payload())

        state = self.wait_for_terminal_state(service, result.payload["job_id"])
        self.assertEqual(state.payload["status"], "failed")
        self.assertEqual(state.payload["error"], "backend failed")

    def test_late_items_return_failed_job_state(self):
        service = EmbeddingJobService(FakePreparer({"weight_id": "sam_vit_b"}, BackendError("backend failed")))
        result = service.start({"total": 2})
        job_id = result.payload["job_id"]
        service.add_items(job_id, {"items": [job_item("slice-1", bytes(12))]})
        self.wait_for_terminal_state(service, job_id)

        late = service.add_items(job_id, {"items": [job_item("slice-2", bytes(range(12)))]})

        self.assertEqual(late.status_code, 200)
        self.assertEqual(late.payload["status"], "failed")
        self.assertEqual(late.payload["error"], "backend failed")

    def test_unknown_job_returns_404(self):
        result = EmbeddingJobService(FakePreparer({"weight_id": "sam_vit_b"})).state("missing")

        self.assertEqual(result.status_code, 404)
        self.assertEqual(result.payload, {"error": "embedding job not found", "job_id": "missing"})

    def wait_for_terminal_state(self, service, job_id):
        state = None
        for _ in range(50):
            state = service.state(job_id)
            if state.payload["status"] in ("complete", "failed"):
                return state
            time.sleep(0.01)
        return state


def job_payload():
    return {"items": [job_item("slice-1", bytes(12)), job_item("slice-2", bytes(range(12)))]}


def job_item(key, image_bytes):
    return {
        "key": key,
        "slice_spec": {"view": "Red", "axis": 0, "index": 1},
        "image_shape": [2, 2, 3],
        "image_digest": "00",
        "image": {"shape": [2, 2, 3], "data": b64encode(image_bytes).decode("ascii")},
    }


if __name__ == "__main__":
    unittest.main()
