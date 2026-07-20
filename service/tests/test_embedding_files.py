import unittest

from samm_server.backends.base import BackendError
from samm_server.embedding_files import EmbeddingFileService


class FakePreparer:
    def __init__(self, prepared=None, error=None):
        self.prepared = prepared
        self.error = error
        self.saved = []
        self.loaded = []

    def prepared_payload(self):
        return self.prepared

    def save_embeddings(self, items, path):
        if self.error:
            raise self.error
        self.saved.append((items, path))
        return {"path": str(path), "count": len(items)}

    def load_embeddings(self, path):
        if self.error:
            raise self.error
        self.loaded.append(path)
        return {"path": str(path), "count": 1, "items": [file_item("emb-2")]}


class EmbeddingFileServiceTest(unittest.TestCase):
    def test_save_requires_prepared_model(self):
        result = EmbeddingFileService(FakePreparer()).save(save_payload())

        self.assertEqual(result.status_code, 409)
        self.assertEqual(result.payload, {"error": "model not prepared"})

    def test_save_calls_preparer(self):
        preparer = FakePreparer({"weight_id": "sam_vit_b"})
        result = EmbeddingFileService(preparer).save(save_payload())

        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.payload, {"path": "/tmp/embeddings", "count": 1})
        self.assertEqual(str(preparer.saved[0][1]), "/tmp/embeddings")
        self.assertEqual(preparer.saved[0][0], [file_item()])

    def test_save_rejects_missing_items(self):
        result = EmbeddingFileService(FakePreparer({"weight_id": "sam_vit_b"})).save({"path": "/tmp/embeddings"})

        self.assertEqual(result.status_code, 400)
        self.assertEqual(result.payload, {"error": "items must be a non-empty list"})

    def test_load_requires_prepared_model(self):
        result = EmbeddingFileService(FakePreparer()).load({"path": "/tmp/embeddings"})

        self.assertEqual(result.status_code, 409)
        self.assertEqual(result.payload, {"error": "model not prepared"})

    def test_load_calls_preparer(self):
        preparer = FakePreparer({"weight_id": "sam_vit_b"})
        result = EmbeddingFileService(preparer).load({"path": "/tmp/embeddings"})

        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.payload, {"path": "/tmp/embeddings", "count": 1, "items": [file_item("emb-2")]})
        self.assertEqual(str(preparer.loaded[0]), "/tmp/embeddings")

    def test_load_reports_backend_error(self):
        service = EmbeddingFileService(FakePreparer({"weight_id": "sam_vit_b"}, BackendError("bad file")))
        result = service.load({"path": "/tmp/embeddings"})

        self.assertEqual(result.status_code, 500)
        self.assertEqual(result.payload, {"error": "bad file"})


def save_payload():
    return {"path": "/tmp/embeddings", "items": [file_item()]}


def file_item(embedding_id="emb-1"):
    return {
        "embedding_id": embedding_id,
        "slice_spec": {"view": "Red", "axis": 0, "index": 1},
        "image_shape": [2, 2, 3],
        "image_digest": "00",
    }


if __name__ == "__main__":
    unittest.main()
