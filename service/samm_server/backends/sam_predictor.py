from hashlib import sha256
from importlib import import_module
import json
from threading import Lock
from uuid import uuid4

from .base import Backend, BackendError


EMBEDDINGS_FILE = "embeddings.pt"
METADATA_FILE = "metadata.json"
FOLDER_FORMAT = "samm-embeddings-folder"


class SamPredictorBackend(Backend):
    def __init__(self, module_name, package_name, data_format):
        self.module_name = module_name
        self.package_name = package_name
        self.data_format = data_format
        self.models = {}
        self.predictors = {}
        self.predictor_types = {}
        self.image_keys = {}
        self.locks = {}
        self.embeddings = {}
        self.embedding_ids = {}
        self.embedding_lock = Lock()

    def prepare(self, weight, checkpoint_path, device):
        module = self.module()
        model = module.sam_model_registry[weight.model_type](checkpoint=str(checkpoint_path))
        model.to(device=device)
        self.clear_weight_embeddings(weight.id)
        self.models[weight.id] = model
        self.predictor_types[weight.id] = module.SamPredictor
        self.predictors[weight.id] = module.SamPredictor(model)
        self.image_keys.pop(weight.id, None)
        self.locks[weight.id] = Lock()
        return {
            "backend": weight.backend,
            "model_type": weight.model_type,
            "device": device,
            "checkpoint_path": str(checkpoint_path),
        }

    def offload(self):
        self.models.clear()
        self.predictors.clear()
        self.predictor_types.clear()
        self.image_keys.clear()
        self.locks.clear()
        self.embeddings.clear()
        self.embedding_ids.clear()

    def predict(self, weight, image_bytes, shape, points, labels, box=None, mask=None, text=None):
        if text:
            raise BackendError(f"{weight.label} does not support text prompts")
        np = self.required_module("numpy")
        predictor = self.predictors[weight.id]
        image_key = self.image_key(image_bytes, shape)
        with self.locks[weight.id]:
            if self.image_keys.get(weight.id) != image_key:
                image = np.frombuffer(image_bytes, dtype=np.uint8).reshape(shape)
                predictor.set_image(image)
                self.image_keys[weight.id] = image_key
            return self.predict_with_predictor(predictor, points, labels, box, mask, np)

    def embed(self, weight, image_bytes, shape):
        np = self.required_module("numpy")
        image_key = self.image_key(image_bytes, shape)
        cache_key = (weight.id, image_key)
        with self.embedding_lock:
            embedding_id = self.embedding_ids.get(cache_key)
            if embedding_id in self.embeddings:
                return self.embedding_payload(embedding_id)
            predictor = self.predictor_types[weight.id](self.models[weight.id])
            predictor.set_image(np.frombuffer(image_bytes, dtype=np.uint8).reshape(shape))
            embedding_id = uuid4().hex
            self.embeddings[embedding_id] = {
                "weight_id": weight.id,
                "predictor": predictor,
                "shape": [shape[0], shape[1]],
                "lock": Lock(),
            }
            self.embedding_ids[cache_key] = embedding_id
            return self.embedding_payload(embedding_id)

    def predict_embedding(self, weight, embedding_id, points, labels, box=None, mask=None, text=None):
        if text:
            raise BackendError(f"{weight.label} does not support text prompts")
        np = self.required_module("numpy")
        embedding = self.embeddings.get(embedding_id)
        if not embedding or embedding["weight_id"] != weight.id:
            raise BackendError("embedding not found")
        if mask and mask["shape"] != embedding["shape"]:
            raise BackendError("mask shape does not match embedding")
        with embedding["lock"]:
            output = self.predict_with_predictor(embedding["predictor"], points, labels, box, mask, np)
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
                predictor = embedding["predictor"]
                item_id = f"embedding_{index:06d}"
                metadata_items.append({
                    "id": item_id,
                    "slice_spec": item["slice_spec"],
                    "image_shape": item["image_shape"],
                    "image_digest": item["image_digest"],
                    "shape": embedding["shape"],
                })
                data_items.append({
                    "id": item_id,
                    "features": predictor.features.detach().cpu(),
                    "original_size": list(predictor.original_size),
                    "input_size": list(predictor.input_size),
                })
        metadata = {
            "format": FOLDER_FORMAT,
            "backend": weight.backend,
            "backend_format": self.data_format,
            "weight_id": weight.id,
            "data_file": EMBEDDINGS_FILE,
            "count": len(metadata_items),
            "items": metadata_items,
        }
        torch.save({"format": self.data_format, "weight_id": weight.id, "items": data_items}, path / EMBEDDINGS_FILE)
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
                data = data_by_id[item["id"]]
                predictor = self.predictor_types[weight.id](self.models[weight.id])
                predictor.features = data["features"]
                predictor.original_size = tuple(data["original_size"])
                predictor.input_size = tuple(data["input_size"])
                predictor.is_image_set = True
                embedding_id = uuid4().hex
                self.embeddings[embedding_id] = {
                    "weight_id": weight.id,
                    "predictor": predictor,
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

    def module(self):
        return self.required_module(self.module_name, self.package_name)

    def required_module(self, name, package_name=None):
        try:
            return import_module(name)
        except ModuleNotFoundError as exc:
            if exc.name == name:
                raise BackendError(f"{package_name or name} is not installed") from exc
            raise

    def image_key(self, image_bytes, shape):
        return tuple(shape), sha256(image_bytes).digest()

    def device_for(self, weight):
        return next(self.models[weight.id].parameters()).device

    def embedding_payload(self, embedding_id):
        return {"embedding_id": embedding_id, "shape": self.embeddings[embedding_id]["shape"]}

    def predict_with_predictor(self, predictor, points, labels, box, mask, np):
        masks, scores, _ = predictor.predict(
            point_coords=np.array(points, dtype=np.float32) if points else None,
            point_labels=np.array(labels, dtype=np.int32) if labels else None,
            box=np.array(box, dtype=np.float32) if box else None,
            mask_input=self.mask_input(mask, np),
            multimask_output=True,
        )
        return masks[int(scores.argmax())].astype(np.uint8).tobytes()

    def mask_input(self, mask, np):
        if not mask:
            return None
        source = np.frombuffer(mask["data"], dtype=np.uint8).reshape(mask["shape"])
        if mask["shape"] != [256, 256]:
            source = self.resize_mask(source, mask["shape"], np)
        return np.array([source], dtype=np.float32)

    def resize_mask(self, source, shape, np):
        rows = (np.arange(256) * shape[0] // 256).astype(np.int64)
        columns = (np.arange(256) * shape[1] // 256).astype(np.int64)
        return source[rows][:, columns]

    def clear_weight_embeddings(self, weight_id):
        stale = [embedding_id for embedding_id, item in self.embeddings.items() if item["weight_id"] == weight_id]
        for embedding_id in stale:
            self.embeddings.pop(embedding_id)
        self.embedding_ids = {key: value for key, value in self.embedding_ids.items() if key[0] != weight_id}
