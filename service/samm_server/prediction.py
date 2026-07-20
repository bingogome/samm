from base64 import b64encode
from dataclasses import dataclass

from .backends.base import BackendError
from .request_payload import image_payload, prompt_payload


@dataclass(frozen=True)
class PredictionResult:
    status_code: int
    payload: dict


class PredictionService:
    def __init__(self, model_preparer):
        self.model_preparer = model_preparer

    def predict(self, payload):
        if not self.model_preparer.prepared_payload():
            return PredictionResult(409, {"error": "model not prepared"})
        if not isinstance(payload, dict):
            return bad("payload must be an object")

        error, points, labels, box, mask_prompt, text = prompt_payload(payload)
        if error:
            return PredictionResult(400, error)

        try:
            if "embedding_id" in payload:
                embedding_id = payload.get("embedding_id")
                if not isinstance(embedding_id, str) or not embedding_id:
                    return bad("embedding_id must be a non-empty string")
                mask, shape = self.model_preparer.predict_embedding(embedding_id, points, labels, box, mask_prompt, text)
                height, width = shape
            else:
                error, image_bytes, shape = image_payload(payload)
                if error:
                    return PredictionResult(400, error)
                height, width, _ = shape
                if mask_prompt and mask_prompt["shape"] != [height, width]:
                    return bad("mask.shape must match image shape")
                mask = self.model_preparer.predict(image_bytes, shape, points, labels, box, mask_prompt, text)
            mask = b64encode(mask).decode("ascii")
        except BackendError as exc:
            return PredictionResult(500, {"error": str(exc)})
        return PredictionResult(200, {"shape": [height, width], "data": mask})


def bad(message, **details):
    payload = {"error": message}
    payload.update(details)
    return PredictionResult(400, payload)
