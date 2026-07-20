from hashlib import sha256
from importlib import import_module
import json
from pathlib import Path
import sys
from threading import Lock
from uuid import uuid4

from .base import Backend, BackendError


EMBEDDINGS_FILE = "embeddings.pt"
METADATA_FILE = "metadata.json"
FOLDER_FORMAT = "samm-embeddings-folder"
DATA_FORMAT = "samm-medsam-embedding-data"
MEDSAM_SOURCE = Path(__file__).resolve().parents[3] / "sam_variants" / "MedSAM"


class MedSamBackend(Backend):
    def __init__(self):
        self.models = {}
        self.locks = {}
        self.embeddings = {}
        self.embedding_ids = {}
        self.embedding_lock = Lock()

    def prepare(self, weight, checkpoint_path, device):
        self.use_source()
        try:
            module = self.required_module("segment_anything")
            model = module.sam_model_registry[weight.model_type](checkpoint=str(checkpoint_path))
            model.to(device=device)
            model.eval()
        finally:
            self.clear_source_modules()
        self.clear_weight_embeddings(weight.id)
        self.models[weight.id] = model
        self.locks[weight.id] = Lock()
        return {
            "backend": weight.backend,
            "model_type": weight.model_type,
            "device": device,
            "checkpoint_path": str(checkpoint_path),
        }

    def offload(self):
        self.models.clear()
        self.locks.clear()
        self.embeddings.clear()
        self.embedding_ids.clear()

    def predict(self, weight, image_bytes, shape, points, labels, box=None, mask=None, text=None):
        self.validate_prompt(points, box, mask, text)
        torch = self.required_module("torch")
        features = self.image_embedding(weight, image_bytes, shape, torch)
        with self.locks[weight.id]:
            return self.predict_mask(weight, features, shape[:2], box, torch)

    def embed(self, weight, image_bytes, shape):
        torch = self.required_module("torch")
        image_key = self.image_key(image_bytes, shape)
        cache_key = (weight.id, image_key)
        with self.embedding_lock:
            embedding_id = self.embedding_ids.get(cache_key)
            if embedding_id in self.embeddings:
                return self.embedding_payload(embedding_id)
            embedding_id = uuid4().hex
            self.embeddings[embedding_id] = {
                "weight_id": weight.id,
                "features": self.image_embedding(weight, image_bytes, shape, torch),
                "shape": [shape[0], shape[1]],
                "lock": Lock(),
            }
            self.embedding_ids[cache_key] = embedding_id
            return self.embedding_payload(embedding_id)

    def predict_embedding(self, weight, embedding_id, points, labels, box=None, mask=None, text=None):
        self.validate_prompt(points, box, mask, text)
        torch = self.required_module("torch")
        embedding = self.embeddings.get(embedding_id)
        if not embedding or embedding["weight_id"] != weight.id:
            raise BackendError("embedding not found")
        with embedding["lock"]:
            output = self.predict_mask(weight, embedding["features"], embedding["shape"], box, torch)
        return output, embedding["shape"]

    def save_embeddings(self, weight, items, path):
        torch = self.required_module("torch")
        path.mkdir(parents=True, exist_ok=True)
        metadata_items = []
        data_items = []
        with self.embedding_lock:
            for index, item in enumerate(items):
                embedding = self.embeddings.get(item["embedding_id"])
                if not embedding or embedding["weight_id"] != weight.id:
                    raise BackendError("embedding not found")
                item_id = f"embedding_{index:06d}"
                metadata_items.append({
                    "id": item_id,
                    "slice_spec": item["slice_spec"],
                    "image_shape": item["image_shape"],
                    "image_digest": item["image_digest"],
                    "shape": embedding["shape"],
                })
                data_items.append({"id": item_id, "features": embedding["features"].detach().cpu()})
        metadata = {
            "format": FOLDER_FORMAT,
            "backend": weight.backend,
            "backend_format": DATA_FORMAT,
            "weight_id": weight.id,
            "data_file": EMBEDDINGS_FILE,
            "count": len(metadata_items),
            "items": metadata_items,
        }
        torch.save({"format": DATA_FORMAT, "weight_id": weight.id, "items": data_items}, path / EMBEDDINGS_FILE)
        (path / METADATA_FILE).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        return {
            "path": str(path),
            "metadata_path": str(path / METADATA_FILE),
            "data_path": str(path / EMBEDDINGS_FILE),
            "count": len(metadata_items),
        }

    def load_embeddings(self, weight, path):
        torch = self.required_module("torch")
        metadata = json.loads((path / METADATA_FILE).read_text(encoding="utf-8"))
        if metadata["format"] != FOLDER_FORMAT or metadata["backend"] != weight.backend or metadata["weight_id"] != weight.id:
            raise BackendError("embedding folder does not match prepared model")
        bundle = torch.load(path / metadata["data_file"], map_location=self.device_for(weight))
        if bundle["format"] != metadata["backend_format"] or bundle["weight_id"] != weight.id:
            raise BackendError("embedding folder does not match prepared model")
        data_by_id = {item["id"]: item for item in bundle["items"]}
        loaded = []
        with self.embedding_lock:
            for item in metadata["items"]:
                embedding_id = uuid4().hex
                self.embeddings[embedding_id] = {
                    "weight_id": weight.id,
                    "features": data_by_id[item["id"]]["features"],
                    "shape": item["shape"],
                    "lock": Lock(),
                }
                self.embedding_ids[(weight.id, (tuple(item["image_shape"]), bytes.fromhex(item["image_digest"])))] = embedding_id
                loaded.append({
                    "embedding_id": embedding_id,
                    "slice_spec": item["slice_spec"],
                    "image_shape": item["image_shape"],
                    "image_digest": item["image_digest"],
                    "shape": item["shape"],
                })
        return {"path": str(path), "count": len(loaded), "items": loaded}

    def image_embedding(self, weight, image_bytes, shape, torch):
        image = torch.as_tensor(self.image_array(image_bytes, shape), dtype=torch.float32, device=self.device_for(weight))
        image = image.permute(2, 0, 1).unsqueeze(0)
        image = torch.nn.functional.interpolate(image, size=(1024, 1024), mode="bilinear", align_corners=False)
        image = (image - image.amin()) / torch.clamp(image.amax() - image.amin(), min=1e-8)
        with torch.no_grad():
            return self.models[weight.id].image_encoder(image)

    def predict_mask(self, weight, features, shape, box, torch):
        model = self.models[weight.id]
        height, width = shape
        box_torch = torch.as_tensor([box], dtype=torch.float32, device=features.device)
        box_torch = box_torch / torch.tensor([width, height, width, height], dtype=torch.float32, device=features.device) * 1024
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = model.prompt_encoder(points=None, boxes=box_torch[:, None, :], masks=None)
            low_res_logits, _ = model.mask_decoder(
                image_embeddings=features,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            logits = torch.sigmoid(low_res_logits)
            logits = torch.nn.functional.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)
        return (logits.squeeze() > 0.5).to(torch.uint8).cpu().numpy().tobytes()

    def validate_prompt(self, points, box, mask, text=None):
        if points:
            raise BackendError("MedSAM does not support point prompts")
        if mask:
            raise BackendError("MedSAM does not support mask input")
        if text:
            raise BackendError("MedSAM does not support text prompts")
        if not box:
            raise BackendError("MedSAM requires a box prompt")

    def image_array(self, image_bytes, shape):
        np = self.required_module("numpy")
        return np.frombuffer(image_bytes, dtype=np.uint8).reshape(shape)

    def image_key(self, image_bytes, shape):
        return tuple(shape), sha256(image_bytes).digest()

    def embedding_payload(self, embedding_id):
        return {"embedding_id": embedding_id, "shape": self.embeddings[embedding_id]["shape"]}

    def device_for(self, weight):
        return next(self.models[weight.id].parameters()).device

    def required_module(self, name):
        try:
            return import_module(name)
        except ModuleNotFoundError as exc:
            if exc.name == name:
                raise BackendError(f"{name} is not installed") from exc
            raise

    def use_source(self):
        source = str(MEDSAM_SOURCE)
        if source not in sys.path:
            sys.path.insert(0, source)

    def clear_source_modules(self):
        source = str(MEDSAM_SOURCE)
        sys.path = [item for item in sys.path if item != source]
        for name in [name for name in sys.modules if name == "segment_anything" or name.startswith("segment_anything.")]:
            sys.modules.pop(name)

    def clear_weight_embeddings(self, weight_id):
        stale = [embedding_id for embedding_id, item in self.embeddings.items() if item["weight_id"] == weight_id]
        for embedding_id in stale:
            self.embeddings.pop(embedding_id)
        self.embedding_ids = {key: value for key, value in self.embedding_ids.items() if key[0] != weight_id}
