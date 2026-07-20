from dataclasses import dataclass

from .backends.base import BackendError
from .request_payload import image_payload


@dataclass(frozen=True)
class EmbeddingResult:
    status_code: int
    payload: dict


class EmbeddingService:
    def __init__(self, model_preparer):
        self.model_preparer = model_preparer

    def embed(self, payload):
        if not self.model_preparer.prepared_payload():
            return EmbeddingResult(409, {"error": "model not prepared"})
        if not isinstance(payload, dict):
            return EmbeddingResult(400, {"error": "payload must be an object"})

        error, image_bytes, shape = image_payload(payload)
        if error:
            return EmbeddingResult(400, error)

        try:
            embedding = self.model_preparer.embed(image_bytes, shape)
        except BackendError as exc:
            return EmbeddingResult(500, {"error": str(exc)})
        return EmbeddingResult(200, embedding)
