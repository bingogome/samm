from contextlib import nullcontext
from hashlib import sha256
from importlib import import_module
import json
from threading import Lock
from uuid import uuid4

from .base import Backend, BackendError


EMBEDDINGS_FILE = "embeddings.pt"
METADATA_FILE = "metadata.json"
FOLDER_FORMAT = "samm-embeddings-folder"
DATA_FORMAT = "samm-sam3-embedding-data"


class Sam3Backend(Backend):
    def __init__(
        self,
        enable_inst_interactivity=True,
        enable_video_propagation=False,
        confidence_threshold=0.5,
        grounding_mask_mode="union",
    ):
        if grounding_mask_mode not in ("union", "best"):
            raise ValueError(f"unsupported SAM3 grounding mask mode: {grounding_mask_mode}")
        self.enable_inst_interactivity = enable_inst_interactivity
        self.enable_video_propagation = enable_video_propagation
        self.confidence_threshold = confidence_threshold
        self.grounding_mask_mode = grounding_mask_mode
        self.models = {}
        self.video_models = {}
        self.processors = {}
        self.locks = {}
        self.embeddings = {}
        self.embedding_ids = {}
        self.embedding_lock = Lock()

    def prepare(self, weight, checkpoint_path, device):
        builder = self.required_module("sam3.model_builder", "sam3")
        processor_module = self.required_module("sam3.model.sam3_image_processor", "sam3")
        if self.enable_video_propagation:
            interactive_module = self.required_module(
                "sam3.model.sam1_task_predictor",
                "sam3",
            )
            video_model = builder.build_sam3_video_model(
                device=device,
                checkpoint_path=str(checkpoint_path),
                load_from_HF=False,
            )
            model = video_model.detector
            if self.enable_inst_interactivity:
                model.inst_interactive_predictor = (
                    interactive_module.SAM3InteractiveImagePredictor(video_model.tracker)
                )
            self.video_models[weight.id] = video_model
        else:
            model = builder.build_sam3_image_model(
                device=device,
                checkpoint_path=str(checkpoint_path),
                load_from_HF=False,
                enable_inst_interactivity=self.enable_inst_interactivity,
            )
        self.clear_weight_embeddings(weight.id)
        self.models[weight.id] = model
        self.processors[weight.id] = processor_module.Sam3Processor(
            model,
            device=device,
            confidence_threshold=self.confidence_threshold,
        )
        self.locks[weight.id] = Lock()
        return {
            "backend": weight.backend,
            "model_type": weight.model_type,
            "device": device,
            "checkpoint_path": str(checkpoint_path),
        }

    def offload(self):
        self.models.clear()
        self.video_models.clear()
        self.processors.clear()
        self.locks.clear()
        self.embeddings.clear()
        self.embedding_ids.clear()

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
        if not self.enable_video_propagation:
            raise BackendError("this SAM3 checkpoint does not support video propagation")
        ordered_frames, seed_position = self.validate_video_input(frames, prompt, direction)

        def generate():
            if self.cancelled(cancel_event):
                return
            lock = self.locks.get(weight.id)
            video_model = self.video_models.get(weight.id)
            if lock is None or video_model is None:
                raise BackendError("model not prepared")
            with lock, self.inference_context(weight):
                if self.cancelled(cancel_event):
                    return
                images = [
                    self.image(frame["image_bytes"], frame["shape"])
                    for frame in ordered_frames
                ]
                effective_video_offload = offload_video_to_cpu or not str(video_model.device).startswith("cuda")
                state = video_model.init_state(
                    resource_path=images,
                    offload_video_to_cpu=effective_video_offload,
                    offload_state_to_cpu=offload_state_to_cpu,
                )
                del images
                seen = set()
                try:
                    seed_frame, seed_outputs = self.seed_video_prompt(
                        video_model,
                        state,
                        seed_position,
                        prompt,
                        ordered_frames[seed_position]["shape"][:2],
                    )
                    if not self.cancelled(cancel_event):
                        seen.add(seed_frame)
                        frame = ordered_frames[seed_frame]
                        yield self.video_mask_payload(
                            frame["frame_index"],
                            frame["shape"][:2],
                            seed_outputs,
                        )
                    for reverse in self.propagation_directions(direction):
                        outputs = video_model.propagate_in_video(
                            state,
                            start_frame_idx=seed_position,
                            reverse=reverse,
                        )
                        try:
                            for position, frame_outputs in outputs:
                                if self.cancelled(cancel_event):
                                    return
                                if position in seen:
                                    continue
                                seen.add(position)
                                frame = ordered_frames[position]
                                yield self.video_mask_payload(
                                    frame["frame_index"],
                                    frame["shape"][:2],
                                    frame_outputs,
                                )
                        finally:
                            close = getattr(outputs, "close", None)
                            if close is not None:
                                close()
                finally:
                    video_model.reset_state(state)

        return generate()

    def seed_video_prompt(self, video_model, state, frame_position, prompt, shape):
        points = prompt.get("points") or []
        box = prompt.get("box")
        text = prompt.get("text")
        outputs = None
        if text or box:
            boxes = [self.normalized_video_box(box, shape)] if box else None
            _frame_position, outputs = video_model.add_prompt(
                inference_state=state,
                frame_idx=frame_position,
                text_str=text,
                boxes_xywh=boxes,
                box_labels=[1] if boxes else None,
            )
        if points:
            object_id = self.video_object_id(outputs)
            _frame_position, outputs = video_model.add_prompt(
                inference_state=state,
                frame_idx=frame_position,
                points=points,
                point_labels=prompt.get("labels"),
                obj_id=object_id,
                rel_coordinates=False,
            )
        return frame_position, outputs

    def video_object_id(self, outputs):
        if not outputs:
            return 1
        object_ids = outputs.get("out_obj_ids")
        if object_ids is None or len(object_ids) == 0:
            return 1
        probabilities = outputs.get("out_probs")
        if probabilities is not None and len(probabilities) == len(object_ids):
            return int(object_ids[int(probabilities.argmax())])
        return int(object_ids[0])

    def video_mask_payload(self, frame_index, shape, outputs):
        np = self.required_module("numpy")
        masks = None if outputs is None else outputs.get("out_binary_masks")
        if masks is None or len(masks) == 0:
            mask = np.zeros(shape, dtype=np.uint8)
        else:
            mask = np.asarray(masks).any(axis=0).astype(np.uint8)
        return {
            "frame_index": frame_index,
            "shape": list(shape),
            "data": mask.reshape(shape).tobytes(),
        }

    def normalized_video_box(self, box, shape):
        height, width = shape
        x0, y0, x1, y1 = box
        return [x0 / width, y0 / height, (x1 - x0) / width, (y1 - y0) / height]

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
        text = prompt.get("text")
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
            or box[0] >= box[2]
            or box[1] >= box[3]
        ):
            raise BackendError("video prompt box must contain x0, y0, x1, y1 coordinates")
        if text is not None and (not isinstance(text, str) or not text.strip()):
            raise BackendError("video prompt text must be a non-empty string")
        if mask is not None:
            raise BackendError("SAM3 video propagation does not support mask prompts")
        if text and points:
            raise BackendError("SAM3 video text prompts cannot be combined with points")
        if not points and box is None and text is None:
            raise BackendError("SAM3 video propagation requires points, a box, or text")
        return ordered_frames, seed_position

    def propagation_directions(self, direction):
        if direction == "both":
            return (False, True)
        return (direction == "backward",)

    def cancelled(self, cancel_event):
        return cancel_event is not None and cancel_event.is_set()

    def is_integer(self, value):
        return isinstance(value, int) and not isinstance(value, bool)

    def is_number(self, value):
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    def predict(self, weight, image_bytes, shape, points, labels, box=None, mask=None, text=None):
        self.validate_prompt(points, box, mask, text)
        with self.locks[weight.id], self.inference_context(weight):
            state = self.processors[weight.id].set_image(self.image(image_bytes, shape))
            return self.predict_state(weight, state, shape[:2], points, labels, box, mask, text)

    def embed(self, weight, image_bytes, shape):
        image_key = self.image_key(image_bytes, shape)
        cache_key = (weight.id, image_key)
        with self.embedding_lock:
            embedding_id = self.embedding_ids.get(cache_key)
            if embedding_id in self.embeddings:
                return self.embedding_payload(embedding_id)
            with self.locks[weight.id], self.inference_context(weight):
                state = self.processors[weight.id].set_image(self.image(image_bytes, shape))
            embedding_id = uuid4().hex
            self.embeddings[embedding_id] = {
                "weight_id": weight.id,
                "state": state,
                "shape": [shape[0], shape[1]],
                "lock": Lock(),
            }
            self.embedding_ids[cache_key] = embedding_id
            return self.embedding_payload(embedding_id)

    def predict_embedding(self, weight, embedding_id, points, labels, box=None, mask=None, text=None):
        self.validate_prompt(points, box, mask, text)
        embedding = self.embeddings.get(embedding_id)
        if not embedding or embedding["weight_id"] != weight.id:
            raise BackendError("embedding not found")
        with embedding["lock"]:
            with self.locks[weight.id], self.inference_context(weight):
                self.processors[weight.id].reset_all_prompts(embedding["state"])
                output = self.predict_state(weight, embedding["state"], embedding["shape"], points, labels, box, mask, text)
        return output, embedding["shape"]

    def save_embeddings(self, weight, items, path):
        torch = self.required_module("torch")
        path.mkdir(parents=True, exist_ok=True)
        metadata_items, data_items = [], []
        with self.embedding_lock:
            for index, item in enumerate(items):
                embedding = self.embeddings.get(item["embedding_id"])
                if not embedding or embedding["weight_id"] != weight.id:
                    raise BackendError("embedding not found")
                with embedding["lock"]:
                    self.processors[weight.id].reset_all_prompts(embedding["state"])
                    item_id = f"embedding_{index:06d}"
                    metadata_items.append({
                        "id": item_id,
                        "slice_spec": item["slice_spec"],
                        "image_shape": item["image_shape"],
                        "image_digest": item["image_digest"],
                        "shape": embedding["shape"],
                    })
                    data_items.append({"id": item_id, "state": embedding["state"]})
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
        if bundle["format"] != DATA_FORMAT or bundle["weight_id"] != weight.id:
            raise BackendError("embedding folder does not match prepared model")
        data_by_id = {item["id"]: item for item in bundle["items"]}
        loaded = []
        with self.embedding_lock:
            for item in metadata["items"]:
                embedding_id = uuid4().hex
                self.embeddings[embedding_id] = {
                    "weight_id": weight.id,
                    "state": data_by_id[item["id"]]["state"],
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

    def predict_state(self, weight, state, shape, points, labels, box, mask, text):
        if points or mask:
            return self.predict_interactive(weight, state, points, labels, box, mask)
        return self.predict_grounding(weight, state, shape, box, text)

    def predict_grounding(self, weight, state, shape, box, text):
        processor = self.processors[weight.id]
        if text:
            state = processor.set_text_prompt(text, state)
        if box:
            state = processor.add_geometric_prompt(self.normalized_box(box, shape), True, state)
        return self.grounding_mask(state, shape)

    def predict_interactive(self, weight, state, points, labels, box, mask):
        np = self.required_module("numpy")
        masks, scores, _ = self.models[weight.id].predict_inst(
            state,
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

    def grounding_mask(self, state, shape):
        np = self.required_module("numpy")
        torch = self.required_module("torch")
        masks = state["masks"]
        if masks.shape[0] == 0:
            return np.zeros(shape, dtype=np.uint8).tobytes()
        if self.grounding_mask_mode == "best":
            mask = masks[int(state["scores"].argmax())]
        else:
            mask = masks.any(dim=0)
        mask = mask.squeeze().to(dtype=torch.uint8).cpu().numpy()
        return mask.reshape(shape).tobytes()

    def normalized_box(self, box, shape):
        height, width = shape
        x0, y0, x1, y1 = box
        return [
            ((x0 + x1) / 2) / width,
            ((y0 + y1) / 2) / height,
            (x1 - x0) / width,
            (y1 - y0) / height,
        ]

    def validate_prompt(self, points, box, mask, text):
        if not self.enable_inst_interactivity and (points or mask):
            raise BackendError("this SAM3 checkpoint does not support point or mask prompts")
        if text and (points or mask):
            raise BackendError("SAM3 text prompts cannot be combined with points or mask input")
        if not points and not box and not mask and not text:
            raise BackendError("SAM3 requires a point, box, mask, or text prompt")

    def image(self, image_bytes, shape):
        np = self.required_module("numpy")
        image_module = self.required_module("PIL.Image", "pillow")
        return image_module.fromarray(np.frombuffer(image_bytes, dtype=np.uint8).reshape(shape))

    def image_key(self, image_bytes, shape):
        return tuple(shape), sha256(image_bytes).digest()

    def embedding_payload(self, embedding_id):
        return {"embedding_id": embedding_id, "shape": self.embeddings[embedding_id]["shape"]}

    def device_for(self, weight):
        return next(self.models[weight.id].parameters()).device

    def inference_context(self, weight):
        if not str(self.device_for(weight)).startswith("cuda"):
            return nullcontext()
        torch = self.required_module("torch")
        return torch.autocast("cuda", dtype=torch.bfloat16)

    def required_module(self, name, package_name=None):
        try:
            return import_module(name)
        except ModuleNotFoundError as exc:
            if exc.name in (name, name.split(".", 1)[0]):
                raise BackendError(f"{package_name or name} is not installed") from exc
            raise

    def clear_weight_embeddings(self, weight_id):
        stale = [embedding_id for embedding_id, item in self.embeddings.items() if item["weight_id"] == weight_id]
        for embedding_id in stale:
            self.embeddings.pop(embedding_id)
        self.embedding_ids = {key: value for key, value in self.embedding_ids.items() if key[0] != weight_id}
