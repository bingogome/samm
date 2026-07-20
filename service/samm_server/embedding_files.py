from dataclasses import dataclass
from pathlib import Path

from .backends.base import BackendError


@dataclass(frozen=True)
class EmbeddingFileResult:
    status_code: int
    payload: dict


class EmbeddingFileService:
    def __init__(self, model_preparer):
        self.model_preparer = model_preparer

    def save(self, payload):
        if not self.model_preparer.prepared_payload():
            return EmbeddingFileResult(409, {"error": "model not prepared"})
        if not isinstance(payload, dict):
            return EmbeddingFileResult(400, {"error": "payload must be an object"})
        path = payload.get("path")
        items = payload.get("items")
        if not isinstance(path, str) or not path:
            return EmbeddingFileResult(400, {"error": "path must be a non-empty string"})
        if not valid_items(items):
            return EmbeddingFileResult(400, {"error": "items must be a non-empty list"})
        try:
            result = self.model_preparer.save_embeddings(items, Path(path))
        except BackendError as exc:
            return EmbeddingFileResult(500, {"error": str(exc)})
        return EmbeddingFileResult(200, result)

    def load(self, payload):
        if not self.model_preparer.prepared_payload():
            return EmbeddingFileResult(409, {"error": "model not prepared"})
        if not isinstance(payload, dict):
            return EmbeddingFileResult(400, {"error": "payload must be an object"})
        path = payload.get("path")
        if not isinstance(path, str) or not path:
            return EmbeddingFileResult(400, {"error": "path must be a non-empty string"})
        try:
            result = self.model_preparer.load_embeddings(Path(path))
        except BackendError as exc:
            return EmbeddingFileResult(500, {"error": str(exc)})
        return EmbeddingFileResult(200, result)


def valid_items(items):
    return isinstance(items, list) and bool(items) and all(valid_item(item) for item in items)


def valid_item(item):
    return (
        isinstance(item, dict)
        and isinstance(item.get("embedding_id"), str)
        and bool(item["embedding_id"])
        and isinstance(item.get("slice_spec"), dict)
        and isinstance(item.get("image_shape"), list)
        and isinstance(item.get("image_digest"), str)
        and bool(item["image_digest"])
    )
