from dataclasses import dataclass
from threading import Lock

from .backends import default_backends
from .backends.base import BackendError
from .model_registry import checkpoint_path, resolve_model_dir, weight_definition, weight_payload
from .protocol import offload_payload


@dataclass(frozen=True)
class PreparationResult:
    status_code: int
    payload: dict


class ModelPreparer:
    def __init__(self, model_dir, device="cpu", backends=None):
        self.model_dir = resolve_model_dir(model_dir)
        self.device = device
        self.backends = backends or default_backends()
        self.prepared_weight_id = None
        self.prepared_weight = None
        self.prepared = None
        self.lifecycle_lock = Lock()

    def prepare(self, weight_id):
        with self.lifecycle_lock:
            return self._prepare(weight_id)

    def _prepare(self, weight_id):
        model, weight = weight_definition(weight_id, self.model_dir)
        if not weight:
            return PreparationResult(404, {"error": "weight not found", "weight_id": weight_id})

        payload = weight_payload(model, weight, self.model_dir)
        if not payload["available"]:
            return PreparationResult(
                409,
                {
                    "error": "checkpoint missing",
                    "model_id": model.id,
                    "weight_id": weight.id,
                    "checkpoint": weight.checkpoint,
                },
            )

        checkpoint_file = checkpoint_path(weight, self.model_dir)
        backend = self.backends.get(weight.backend)
        if not backend:
            return PreparationResult(
                501,
                {
                    "error": "backend unsupported",
                    "model_id": model.id,
                    "weight_id": weight.id,
                    "backend": weight.backend,
                },
            )

        try:
            prepared = backend.prepare(weight, checkpoint_file, self.device)
        except BackendError as exc:
            return PreparationResult(
                500,
                {
                    "error": str(exc),
                    "model_id": model.id,
                    "weight_id": weight.id,
                    "backend": weight.backend,
                },
            )

        self.prepared_weight_id = weight.id
        self.prepared_weight = weight
        self.prepared = {
            "model_id": model.id,
            "weight_id": weight.id,
            "status": "prepared",
            "checkpoint": weight.checkpoint,
            "backend": prepared["backend"],
            "model_type": prepared["model_type"],
            "device": prepared["device"],
            "capabilities": dict(weight.capabilities),
        }
        return PreparationResult(200, self.prepared)

    def prepared_payload(self):
        return self.prepared

    def offload(self):
        with self.lifecycle_lock:
            return self._offload()

    def _offload(self):
        for backend in self.backends.values():
            backend.offload()
        self.prepared_weight_id = None
        self.prepared_weight = None
        self.prepared = None
        return PreparationResult(200, offload_payload())

    def predict(self, image_bytes, shape, points, labels, box=None, mask=None, text=None):
        backend = self.backends[self.prepared_weight.backend]
        return backend.predict(self.prepared_weight, image_bytes, shape, points, labels, box, mask, text)

    def embed(self, image_bytes, shape):
        backend = self.backends[self.prepared_weight.backend]
        return backend.embed(self.prepared_weight, image_bytes, shape)

    def predict_embedding(self, embedding_id, points, labels, box=None, mask=None, text=None):
        backend = self.backends[self.prepared_weight.backend]
        return backend.predict_embedding(self.prepared_weight, embedding_id, points, labels, box, mask, text)

    def propagate_video(
        self,
        frames,
        prompt,
        direction="both",
        cancel_event=None,
        offload_video_to_cpu=True,
        offload_state_to_cpu=False,
        expected_weight_id=None,
    ):
        with self.lifecycle_lock:
            pinned_weight_id = self._validate_video_weight(expected_weight_id)

        def generate():
            with self.lifecycle_lock:
                self._validate_video_weight(pinned_weight_id)
                backend = self.backends[self.prepared_weight.backend]
                results = backend.propagate_video(
                    self.prepared_weight,
                    frames,
                    prompt,
                    direction,
                    cancel_event,
                    offload_video_to_cpu,
                    offload_state_to_cpu,
                )
                try:
                    yield from results
                finally:
                    close = getattr(results, "close", None)
                    if close is not None:
                        close()

        return generate()

    def _validate_video_weight(self, expected_weight_id):
        if self.prepared_weight is None:
            raise BackendError("model not prepared")
        if expected_weight_id is not None and expected_weight_id != self.prepared_weight_id:
            raise BackendError("prepared model changed")
        return self.prepared_weight_id

    def save_embeddings(self, items, path):
        backend = self.backends[self.prepared_weight.backend]
        return backend.save_embeddings(self.prepared_weight, items, path)

    def load_embeddings(self, path):
        backend = self.backends[self.prepared_weight.backend]
        return backend.load_embeddings(self.prepared_weight, path)
