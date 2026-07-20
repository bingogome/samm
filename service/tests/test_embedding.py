from base64 import b64encode
import unittest

from samm_server.backends.base import BackendError
from samm_server.embedding import EmbeddingService


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
        return {"embedding_id": "emb-1", "shape": [shape[0], shape[1]]}


class EmbeddingServiceTest(unittest.TestCase):
    def test_embed_requires_prepared_model(self):
        result = EmbeddingService(FakePreparer()).embed(image_payload())

        self.assertEqual(result.status_code, 409)
        self.assertEqual(result.payload, {"error": "model not prepared"})

    def test_embed_calls_prepared_backend(self):
        preparer = FakePreparer({"weight_id": "sam_vit_b"})
        result = EmbeddingService(preparer).embed(image_payload(3, 2, bytes(range(18))))

        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.payload, {"embedding_id": "emb-1", "shape": [2, 3]})
        self.assertEqual(preparer.calls, [(bytes(range(18)), [2, 3, 3])])

    def test_embed_rejects_image_byte_count_mismatch(self):
        payload = image_payload(2, 2)
        payload["image"]["data"] = b64encode(bytes(3)).decode("ascii")
        result = EmbeddingService(FakePreparer({"weight_id": "sam_vit_b"})).embed(payload)

        self.assertEqual(result.status_code, 400)
        self.assertEqual(result.payload["error"], "image.data length does not match image.shape")

    def test_embed_reports_backend_error(self):
        result = EmbeddingService(FakePreparer({"weight_id": "sam_vit_b"}, BackendError("backend failed"))).embed(
            image_payload()
        )

        self.assertEqual(result.status_code, 500)
        self.assertEqual(result.payload, {"error": "backend failed"})


def image_payload(width=2, height=2, image_bytes=None):
    image_bytes = image_bytes or bytes(width * height * 3)
    image = b64encode(image_bytes).decode("ascii")
    return {"image": {"shape": [height, width, 3], "data": image}}


if __name__ == "__main__":
    unittest.main()
