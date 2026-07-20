from collections import OrderedDict
from contextlib import contextmanager
from hashlib import sha256
from importlib import import_module
import json
from threading import Lock
from uuid import uuid4

from .base import Backend, BackendError


EMBEDDINGS_FILE = "embeddings.pt"
METADATA_FILE = "metadata.json"
FOLDER_FORMAT = "samm-embeddings-folder"
DATA_FORMAT = "samm-sam2-embedding-data"


class Sam2Backend(Backend):
    def __init__(self, package_name="sam2", data_format=DATA_FORMAT):
        self.package_name = package_name
        self.data_format = data_format
        self.models = {}
        self.video_predictors = {}
        self.predictors = {}
        self.predictor_types = {}
        self.image_keys = {}
        self.locks = {}
        self.embeddings = {}
        self.embedding_ids = {}
        self.embedding_lock = Lock()
        self.lifecycle_lock = Lock()

    def prepare(self, weight, checkpoint_path, device):
        with self.embedding_lock:
            with self.lifecycle_lock:
                active_locks = list(self.locks.values())
                for lock in active_locks:
                    lock.acquire()
                try:
                    build_module = self.required_module("sam2.build_sam", self.package_name)
                    predictor_module = self.required_module("sam2.sam2_image_predictor", self.package_name)
                    if self.package_name.casefold() == "medsam2":
                        model = build_module.build_sam2_video_predictor_npz(
                            weight.model_type,
                            str(checkpoint_path),
                            device=device,
                        )
                    else:
                        model = build_module.build_sam2_video_predictor(
                            weight.model_type,
                            str(checkpoint_path),
                            device=device,
                        )
                    self.models.clear()
                    self.video_predictors.clear()
                    self.predictors.clear()
                    self.predictor_types.clear()
                    self.image_keys.clear()
                    self.locks.clear()
                    self.embeddings.clear()
                    self.embedding_ids.clear()
                    self.models[weight.id] = model
                    self.video_predictors[weight.id] = model
                    self.predictor_types[weight.id] = predictor_module.SAM2ImagePredictor
                    self.predictors[weight.id] = predictor_module.SAM2ImagePredictor(model)
                    self.image_keys.pop(weight.id, None)
                    self.locks[weight.id] = Lock()
                finally:
                    for lock in reversed(active_locks):
                        lock.release()
        return {
            "backend": weight.backend,
            "model_type": weight.model_type,
            "device": device,
            "checkpoint_path": str(checkpoint_path),
        }

    def offload(self):
        with self.embedding_lock:
            with self.lifecycle_lock:
                active_locks = list(self.locks.values())
                for lock in active_locks:
                    lock.acquire()
                try:
                    self.models.clear()
                    self.video_predictors.clear()
                    self.predictors.clear()
                    self.predictor_types.clear()
                    self.image_keys.clear()
                    self.locks.clear()
                    self.embeddings.clear()
                    self.embedding_ids.clear()
                finally:
                    for lock in reversed(active_locks):
                        lock.release()

    def predict(self, weight, image_bytes, shape, points, labels, box=None, mask=None, text=None):
        if text:
            raise BackendError(f"{weight.label} does not support text prompts")
        np = self.required_module("numpy")
        image_key = self.image_key(image_bytes, shape)
        with self.model_access(weight.id):
            predictor = self.predictors[weight.id]
            if self.image_keys.get(weight.id) != image_key:
                predictor.set_image(np.frombuffer(image_bytes, dtype=np.uint8).reshape(shape))
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
            with self.model_access(weight.id):
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
            with self.model_access(weight.id):
                if self.embeddings.get(embedding_id) is not embedding:
                    raise BackendError("embedding not found")
                output = self.predict_with_predictor(embedding["predictor"], points, labels, box, mask, np)
        return output, embedding["shape"]

    def propagate_video(
        self,
        weight,
        frames,
        prompt,
        direction="both",
        cancel_event=None,
        offload_video_to_cpu=True,
        offload_state_to_cpu=False,
    ):
        ordered_frames, seed_position = self.validate_video_input(frames, prompt, direction)

        def generate():
            if self.cancelled(cancel_event):
                return
            with self.model_access(weight.id):
                predictor = self.video_predictors[weight.id]
                if self.cancelled(cancel_event):
                    return
                torch = self.required_module("torch")
                np = self.required_module("numpy")
                images = self.preprocess_video_frames(
                    ordered_frames,
                    predictor,
                    offload_video_to_cpu,
                    torch,
                    np,
                )
                if self.cancelled(cancel_event):
                    return
                state = self.initialize_video_state(
                    predictor,
                    images,
                    ordered_frames[0]["shape"][0],
                    ordered_frames[0]["shape"][1],
                    offload_video_to_cpu,
                    offload_state_to_cpu,
                    torch,
                )
                seen = set()
                try:
                    for direction_index, reverse in enumerate(self.propagation_directions(direction)):
                        if direction_index:
                            predictor.reset_state(state)
                        seed_output = self.seed_video_prompt(predictor, state, seed_position, prompt, np)
                        if seed_position not in seen:
                            if self.cancelled(cancel_event):
                                return
                            position, _object_ids, mask_logits = seed_output
                            seen.add(position)
                            frame = ordered_frames[position]
                            yield self.video_mask_payload(
                                frame["frame_index"],
                                frame["shape"][:2],
                                mask_logits,
                                torch,
                            )
                        outputs = predictor.propagate_in_video(
                            state,
                            start_frame_idx=seed_position,
                            reverse=reverse,
                        )
                        try:
                            for position, _object_ids, mask_logits in outputs:
                                if self.cancelled(cancel_event):
                                    return
                                if position in seen:
                                    continue
                                seen.add(position)
                                frame = ordered_frames[position]
                                yield self.video_mask_payload(
                                    frame["frame_index"],
                                    frame["shape"][:2],
                                    mask_logits,
                                    torch,
                                )
                        finally:
                            close = getattr(outputs, "close", None)
                            if close is not None:
                                close()
                finally:
                    predictor.reset_state(state)

        return generate()

    def validate_video_input(self, frames, prompt, direction):
        if direction not in ("both", "forward", "backward"):
            raise BackendError("video direction must be both, forward, or backward")
        if not frames:
            raise BackendError("video requires at least one frame")
        try:
            ordered_frames = sorted(frames, key=lambda frame: frame["frame_index"])
        except (KeyError, TypeError) as exc:
            raise BackendError("each video frame requires a frame_index") from exc
        frame_indices = [frame["frame_index"] for frame in ordered_frames]
        if any(not self.is_integer(frame_index) for frame_index in frame_indices):
            raise BackendError("video frame_index values must be integers")
        if len(set(frame_indices)) != len(frame_indices):
            raise BackendError("video frame_index values must be unique")

        expected_shape = None
        for frame in ordered_frames:
            shape = frame.get("shape")
            image_bytes = frame.get("image_bytes")
            if not isinstance(shape, (list, tuple)) or len(shape) != 3 or shape[2] != 3:
                raise BackendError("video frames must have RGB shape [height, width, 3]")
            if any(not self.is_integer(value) or value <= 0 for value in shape):
                raise BackendError("video frame shape values must be positive integers")
            if expected_shape is None:
                expected_shape = list(shape)
            elif list(shape) != expected_shape:
                raise BackendError("video frames must all have the same shape")
            if not isinstance(image_bytes, bytes) or len(image_bytes) != shape[0] * shape[1] * shape[2]:
                raise BackendError("video frame data does not match its shape")

        if not isinstance(prompt, dict):
            raise BackendError("video prompt is required")
        seed_frame_index = prompt.get("frame_index")
        if not self.is_integer(seed_frame_index):
            raise BackendError("video prompt frame_index must be an integer")
        try:
            seed_position = frame_indices.index(seed_frame_index)
        except ValueError as exc:
            raise BackendError("video prompt frame_index is not present in frames") from exc
        points = prompt.get("points") or []
        labels = prompt.get("labels") or []
        box = prompt.get("box")
        mask = prompt.get("mask")
        if not isinstance(points, (list, tuple)) or not isinstance(labels, (list, tuple)):
            raise BackendError("video prompt points and labels must be lists")
        if len(points) != len(labels):
            raise BackendError("video prompt points and labels must have the same length")
        if any(
            not isinstance(point, (list, tuple))
            or len(point) != 2
            or any(not self.is_number(value) for value in point)
            for point in points
        ):
            raise BackendError("video prompt points must contain x, y coordinate pairs")
        if any(not self.is_integer(label) or label not in (0, 1) for label in labels):
            raise BackendError("video prompt labels must contain only 0 or 1")
        if box is not None and (
            not isinstance(box, (list, tuple))
            or len(box) != 4
            or any(not self.is_number(value) for value in box)
        ):
            raise BackendError("video prompt box must contain four coordinates")
        if mask and (points or box):
            raise BackendError("video mask prompts cannot be combined with points or a box")
        if not mask and not points and not box:
            raise BackendError("video propagation requires points, a box, or a mask prompt")
        if mask:
            mask_shape = mask.get("shape") if isinstance(mask, dict) else None
            mask_data = mask.get("data") if isinstance(mask, dict) else None
            if (
                not isinstance(mask_shape, (list, tuple))
                or len(mask_shape) != 2
                or any(not self.is_integer(value) or value <= 0 for value in mask_shape)
                or not isinstance(mask_data, bytes)
                or len(mask_data) != mask_shape[0] * mask_shape[1]
            ):
                raise BackendError("video prompt mask data does not match its shape")
            if list(mask_shape) != expected_shape[:2]:
                raise BackendError("video prompt mask shape must match the seed frame")
        return ordered_frames, seed_position

    def preprocess_video_frames(self, frames, predictor, offload_video_to_cpu, torch, np):
        storage_device = torch.device("cpu") if offload_video_to_cpu else predictor.device
        image_size = predictor.image_size
        images = torch.empty(
            (len(frames), 3, image_size, image_size),
            dtype=torch.float32,
            device=storage_device,
        )
        mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32, device=storage_device)[:, None, None]
        std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32, device=storage_device)[:, None, None]
        batch_size = 8
        for start in range(0, len(frames), batch_size):
            batch_frames = frames[start:start + batch_size]
            arrays = [
                np.frombuffer(frame["image_bytes"], dtype=np.uint8).reshape(frame["shape"])
                for frame in batch_frames
            ]
            batch = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).float() / 255.0
            batch = torch.nn.functional.interpolate(
                batch,
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            ).to(storage_device)
            batch = (batch - mean) / std
            images[start:start + len(batch_frames)].copy_(batch)
        return images

    def initialize_video_state(
        self,
        predictor,
        images,
        video_height,
        video_width,
        offload_video_to_cpu,
        offload_state_to_cpu,
        torch,
    ):
        if self.package_name.casefold() == "medsam2":
            return predictor.init_state(
                images,
                video_height,
                video_width,
                offload_video_to_cpu=offload_video_to_cpu,
                offload_state_to_cpu=offload_state_to_cpu,
            )
        compute_device = predictor.device
        state = {
            "images": images,
            "num_frames": len(images),
            "offload_video_to_cpu": offload_video_to_cpu,
            "offload_state_to_cpu": offload_state_to_cpu,
            "video_height": video_height,
            "video_width": video_width,
            "device": compute_device,
            "storage_device": torch.device("cpu") if offload_state_to_cpu else compute_device,
            "point_inputs_per_obj": {},
            "mask_inputs_per_obj": {},
            "cached_features": {},
            "constants": {},
            "obj_id_to_idx": OrderedDict(),
            "obj_idx_to_id": OrderedDict(),
            "obj_ids": [],
            "output_dict_per_obj": {},
            "temp_output_dict_per_obj": {},
            "frames_tracked_per_obj": {},
        }
        predictor._get_image_feature(state, frame_idx=0, batch_size=1)
        return state

    def seed_video_prompt(self, predictor, state, frame_position, prompt, np):
        mask = prompt.get("mask")
        if mask:
            source = np.frombuffer(mask["data"], dtype=np.uint8).reshape(mask["shape"]).copy()
            return predictor.add_new_mask(state, frame_position, obj_id=1, mask=source)
        points = prompt.get("points") or None
        return predictor.add_new_points_or_box(
            state,
            frame_position,
            obj_id=1,
            points=points,
            labels=(prompt.get("labels") or None) if points else None,
            box=prompt.get("box"),
        )

    def video_mask_payload(self, frame_index, shape, mask_logits, torch):
        mask = (mask_logits[0] > 0.0).to(torch.uint8).cpu().numpy().reshape(shape)
        return {"frame_index": frame_index, "shape": list(shape), "data": mask.tobytes()}

    def propagation_directions(self, direction):
        if direction == "both":
            return (False, True)
        return (direction == "backward",)

    def cancelled(self, cancel_event):
        return cancel_event is not None and cancel_event.is_set()

    @contextmanager
    def model_access(self, weight_id):
        with self.lifecycle_lock:
            lock = self.locks.get(weight_id)
            if lock is None:
                raise BackendError("model not prepared")
            lock.acquire()
        try:
            yield
        finally:
            lock.release()

    def is_integer(self, value):
        return isinstance(value, int) and not isinstance(value, bool)

    def is_number(self, value):
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    def save_embeddings(self, weight, items, path):
        torch = self.required_module("torch")
        path.mkdir(parents=True, exist_ok=True)
        metadata_items, data_items = [], []
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
                    "features": predictor._features,
                    "orig_hw": predictor._orig_hw,
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
        loaded = []
        with self.embedding_lock:
            with self.model_access(weight.id):
                bundle = torch.load(path / metadata["data_file"], map_location=self.device_for(weight))
                if bundle["format"] != self.data_format or bundle["weight_id"] != weight.id:
                    raise BackendError("embedding folder does not match prepared model")
                data_by_id = {item["id"]: item for item in bundle["items"]}
                for item in metadata["items"]:
                    data = data_by_id[item["id"]]
                    predictor = self.predictor_types[weight.id](self.models[weight.id])
                    predictor._features = data["features"]
                    predictor._orig_hw = data["orig_hw"]
                    predictor._is_image_set = True
                    predictor._is_batch = False
                    embedding_id = uuid4().hex
                    self.embeddings[embedding_id] = {
                        "weight_id": weight.id,
                        "predictor": predictor,
                        "shape": item["shape"],
                        "lock": Lock(),
                    }
                    self.embedding_ids[
                        (weight.id, (tuple(item["image_shape"]), bytes.fromhex(item["image_digest"])))
                    ] = embedding_id
                    loaded.append({
                        "embedding_id": embedding_id,
                        "slice_spec": item["slice_spec"],
                        "image_shape": item["image_shape"],
                        "image_digest": item["image_digest"],
                        "shape": item["shape"],
                    })
        return {"path": str(path), "count": len(loaded), "items": loaded}

    def predict_with_predictor(self, predictor, points, labels, box, mask, np):
        masks, scores, _ = predictor.predict(
            point_coords=np.array(points, dtype=np.float32) if points else None,
            point_labels=np.array(labels, dtype=np.int32) if points else None,
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

    def required_module(self, name, package_name=None):
        try:
            return import_module(name)
        except ModuleNotFoundError as exc:
            if exc.name in (name, name.split(".", 1)[0]):
                raise BackendError(f"{package_name or name} is not installed") from exc
            raise

    def image_key(self, image_bytes, shape):
        return tuple(shape), sha256(image_bytes).digest()

    def embedding_payload(self, embedding_id):
        return {"embedding_id": embedding_id, "shape": self.embeddings[embedding_id]["shape"]}

    def device_for(self, weight):
        return next(self.models[weight.id].parameters()).device

    def clear_weight_embeddings(self, weight_id):
        stale = [embedding_id for embedding_id, item in self.embeddings.items() if item["weight_id"] == weight_id]
        for embedding_id in stale:
            self.embeddings.pop(embedding_id)
        self.embedding_ids = {key: value for key, value in self.embedding_ids.items() if key[0] != weight_id}
