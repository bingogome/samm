from base64 import b64decode, b64encode
import unittest

from samm_server.backends.base import BackendError
from samm_server.prediction import PredictionService


class FakePreparer:
    def __init__(self, prepared=None, mask=b"", error=None):
        self.prepared = prepared
        self.mask = mask
        self.error = error
        self.calls = []
        self.embedding_calls = []

    def prepared_payload(self):
        return self.prepared

    def predict(self, image_bytes, shape, points, labels, box=None, mask=None, text=None):
        if self.error:
            raise self.error
        self.calls.append((image_bytes, shape, points, labels, box, mask, text))
        return self.mask

    def predict_embedding(self, embedding_id, points, labels, box=None, mask=None, text=None):
        if self.error:
            raise self.error
        self.embedding_calls.append((embedding_id, points, labels, box, mask, text))
        return self.mask, [2, 3]


class PredictionServiceTest(unittest.TestCase):
    def test_predict_requires_prepared_model(self):
        result = PredictionService(FakePreparer()).predict(image_payload())

        self.assertEqual(result.status_code, 409)
        self.assertEqual(result.payload, {"error": "model not prepared"})

    def test_predict_calls_prepared_backend(self):
        preparer = FakePreparer({"weight_id": "sam_vit_b"}, bytes([0, 1, 0, 1, 0, 1]))
        payload = image_payload(3, 2, bytes(range(18)))
        result = PredictionService(preparer).predict(payload)

        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.payload["shape"], [2, 3])
        self.assertEqual(b64decode(result.payload["data"]), preparer.mask)
        self.assertEqual(preparer.calls, [(bytes(range(18)), [2, 3, 3], [[1, 1]], [1], None, None, None)])

    def test_predict_calls_embedding_backend(self):
        preparer = FakePreparer({"weight_id": "sam_vit_b"}, bytes([0, 1, 0, 1, 0, 1]))
        payload = {"embedding_id": "abc123", "points": [[1, 1]], "labels": [1]}
        result = PredictionService(preparer).predict(payload)

        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.payload["shape"], [2, 3])
        self.assertEqual(b64decode(result.payload["data"]), preparer.mask)
        self.assertEqual(preparer.calls, [])
        self.assertEqual(preparer.embedding_calls, [("abc123", [[1, 1]], [1], None, None, None)])

    def test_predict_calls_backend_with_box(self):
        preparer = FakePreparer({"weight_id": "sam_vit_b"}, bytes([0, 1, 0, 1]))
        payload = image_payload()
        payload["points"] = []
        payload["labels"] = []
        payload["box"] = [0, 0, 1, 1]
        result = PredictionService(preparer).predict(payload)

        self.assertEqual(result.status_code, 200)
        self.assertEqual(preparer.calls, [(bytes(12), [2, 2, 3], [], [], [0, 0, 1, 1], None, None)])

    def test_predict_calls_embedding_backend_with_box(self):
        preparer = FakePreparer({"weight_id": "sam_vit_b"}, bytes([0, 1, 0, 1]))
        payload = {"embedding_id": "abc123", "points": [], "labels": [], "box": [0, 0, 1, 1]}
        result = PredictionService(preparer).predict(payload)

        self.assertEqual(result.status_code, 200)
        self.assertEqual(preparer.embedding_calls, [("abc123", [], [], [0, 0, 1, 1], None, None)])

    def test_predict_calls_backend_with_mask(self):
        preparer = FakePreparer({"weight_id": "sam_vit_b"}, bytes([0, 1, 0, 1]))
        payload = image_payload()
        payload["points"] = []
        payload["labels"] = []
        payload["mask"] = mask_payload(2, 2, bytes([0, 1, 1, 0]))
        result = PredictionService(preparer).predict(payload)

        self.assertEqual(result.status_code, 200)
        self.assertEqual(preparer.calls, [(bytes(12), [2, 2, 3], [], [], None, {"shape": [2, 2], "data": bytes([0, 1, 1, 0])}, None)])

    def test_predict_calls_embedding_backend_with_mask(self):
        preparer = FakePreparer({"weight_id": "sam_vit_b"}, bytes([0, 1, 0, 1]))
        payload = {"embedding_id": "abc123", "points": [], "labels": [], "mask": mask_payload(2, 3, bytes(6))}
        result = PredictionService(preparer).predict(payload)

        self.assertEqual(result.status_code, 200)
        self.assertEqual(preparer.embedding_calls, [("abc123", [], [], None, {"shape": [3, 2], "data": bytes(6)}, None)])

    def test_predict_calls_backend_with_text(self):
        preparer = FakePreparer({"weight_id": "sam3"}, bytes([0, 1, 0, 1]))
        payload = image_payload()
        payload["points"] = []
        payload["labels"] = []
        payload["text"] = "kidney"
        result = PredictionService(preparer).predict(payload)

        self.assertEqual(result.status_code, 200)
        self.assertEqual(preparer.calls, [(bytes(12), [2, 2, 3], [], [], None, None, "kidney")])

    def test_predict_calls_embedding_backend_with_text(self):
        preparer = FakePreparer({"weight_id": "sam3"}, bytes([0, 1, 0, 1]))
        payload = {"embedding_id": "abc123", "points": [], "labels": [], "text": "kidney"}
        result = PredictionService(preparer).predict(payload)

        self.assertEqual(result.status_code, 200)
        self.assertEqual(preparer.embedding_calls, [("abc123", [], [], None, None, "kidney")])

    def test_predict_rejects_empty_embedding_id(self):
        payload = {"embedding_id": "", "points": [[1, 1]], "labels": [1]}
        result = PredictionService(FakePreparer({"weight_id": "sam_vit_b"})).predict(payload)

        self.assertEqual(result.status_code, 400)
        self.assertEqual(result.payload["error"], "embedding_id must be a non-empty string")

    def test_predict_rejects_image_byte_count_mismatch(self):
        payload = image_payload(2, 2)
        payload["image"]["data"] = b64encode(bytes(3)).decode("ascii")
        result = PredictionService(FakePreparer({"weight_id": "sam_vit_b"})).predict(payload)

        self.assertEqual(result.status_code, 400)
        self.assertEqual(result.payload["error"], "image.data length does not match image.shape")
        self.assertEqual(result.payload["expected"], 12)
        self.assertEqual(result.payload["actual"], 3)

    def test_predict_rejects_prompt_count_mismatch(self):
        payload = image_payload()
        payload["labels"] = []
        result = PredictionService(FakePreparer({"weight_id": "sam_vit_b"})).predict(payload)

        self.assertEqual(result.status_code, 400)
        self.assertEqual(result.payload["error"], "points and labels must be equal length lists of [x, y] and 0/1 labels")

    def test_predict_rejects_empty_prompts(self):
        payload = image_payload()
        payload["points"] = []
        payload["labels"] = []
        result = PredictionService(FakePreparer({"weight_id": "sam_vit_b"})).predict(payload)

        self.assertEqual(result.status_code, 400)
        self.assertEqual(result.payload["error"], "points/labels, box, mask, or text prompt is required")

    def test_predict_rejects_empty_text(self):
        payload = image_payload()
        payload["text"] = ""
        result = PredictionService(FakePreparer({"weight_id": "sam_vit_b"})).predict(payload)

        self.assertEqual(result.status_code, 400)
        self.assertEqual(result.payload["error"], "text must be a non-empty string")

    def test_predict_rejects_invalid_box(self):
        payload = image_payload()
        payload["box"] = [2, 0, 1, 1]
        result = PredictionService(FakePreparer({"weight_id": "sam_vit_b"})).predict(payload)

        self.assertEqual(result.status_code, 400)
        self.assertEqual(result.payload["error"], "box must be [x0, y0, x1, y1]")

    def test_predict_rejects_mask_shape_mismatch(self):
        payload = image_payload()
        payload["mask"] = mask_payload(1, 1, bytes(1))
        result = PredictionService(FakePreparer({"weight_id": "sam_vit_b"})).predict(payload)

        self.assertEqual(result.status_code, 400)
        self.assertEqual(result.payload["error"], "mask.shape must match image shape")

    def test_predict_reports_backend_error(self):
        result = PredictionService(FakePreparer({"weight_id": "sam_vit_b"}, error=BackendError("backend failed"))).predict(
            image_payload()
        )

        self.assertEqual(result.status_code, 500)
        self.assertEqual(result.payload, {"error": "backend failed"})


def image_payload(width=2, height=2, image_bytes=None):
    image_bytes = image_bytes or bytes(width * height * 3)
    image = b64encode(image_bytes).decode("ascii")
    return {
        "image": {"shape": [height, width, 3], "data": image},
        "points": [[1, 1]],
        "labels": [1],
    }


def mask_payload(width, height, mask_bytes):
    return {"shape": [height, width], "data": b64encode(mask_bytes).decode("ascii")}


if __name__ == "__main__":
    unittest.main()
