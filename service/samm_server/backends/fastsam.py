from importlib import import_module
from os import environ
from pathlib import Path
from sys import path as import_paths
from threading import Lock
from warnings import filterwarnings

from .base import Backend, BackendError


class FastSamBackend(Backend):
    def __init__(self, confidence=0.4, iou=0.9, image_size=1024):
        self.confidence = confidence
        self.iou = iou
        self.image_size = image_size
        self.models = {}
        self.prompts = {}
        self.devices = {}
        self.locks = {}

    def prepare(self, weight, checkpoint_path, device):
        self.prepare_environment()
        self.add_source_path(checkpoint_path)
        module = self.fastsam_module()
        self.models[weight.id] = self.load_model(module, checkpoint_path)
        self.prompts[weight.id] = module.FastSAMPrompt
        self.devices[weight.id] = device
        self.locks[weight.id] = Lock()
        return {
            "backend": weight.backend,
            "model_type": weight.model_type,
            "device": device,
            "checkpoint_path": str(checkpoint_path),
        }

    def offload(self):
        self.models.clear()
        self.prompts.clear()
        self.devices.clear()
        self.locks.clear()

    def load_model(self, module, checkpoint_path):
        torch = self.required_module("torch")
        original_load = torch.load

        def trusted_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original_load(*args, **kwargs)

        torch.load = trusted_load
        try:
            return module.FastSAM(str(checkpoint_path))
        finally:
            torch.load = original_load

    def predict(self, weight, image_bytes, shape, points, labels, box=None, mask=None, text=None):
        self.validate_prompt(weight, points, box, mask, text)
        np = self.required_module("numpy")
        image = np.frombuffer(image_bytes, dtype=np.uint8).reshape(shape)
        with self.locks[weight.id]:
            results = self.models[weight.id](
                image,
                device=self.devices[weight.id],
                retina_masks=True,
                imgsz=self.image_size,
                conf=self.confidence,
                iou=self.iou,
            )
        if results is None:
            raise BackendError("FastSAM prediction failed")
        if not self.has_masks(results):
            return np.zeros(shape[:2], dtype=np.uint8).tobytes()
        prompt = self.prompts[weight.id](image, results, device=self.devices[weight.id])
        masks = self.prompt_masks(prompt, points, labels, box, text)
        return self.mask_bytes(masks, shape[:2], np)

    def embed(self, weight, image_bytes, shape):
        raise BackendError("FastSAM does not support cached embeddings")

    def predict_embedding(self, weight, embedding_id, points, labels, box=None, mask=None, text=None):
        raise BackendError("FastSAM does not support cached embeddings")

    def save_embeddings(self, weight, items, path):
        raise BackendError("FastSAM does not support cached embeddings")

    def load_embeddings(self, weight, path):
        raise BackendError("FastSAM does not support cached embeddings")

    def validate_prompt(self, weight, points, box, mask, text):
        if mask:
            raise BackendError(f"{weight.label} does not support mask prompts")
        count = int(bool(points)) + int(box is not None) + int(bool(text))
        if count != 1:
            raise BackendError("FastSAM requires exactly one point, box, or text prompt")

    def prompt_masks(self, prompt, points, labels, box, text):
        if text:
            self.required_module("fastsam.prompt").clip = self.required_module("clip", "CLIP")
            return prompt.text_prompt(text=text)
        if box is not None:
            return prompt.box_prompt(bboxes=[list(box)])
        return prompt.point_prompt(points=points, pointlabel=labels)

    def mask_bytes(self, masks, shape, np):
        array = masks.detach().cpu().numpy() if hasattr(masks, "detach") else np.asarray(masks)
        if array.size == 0:
            return np.zeros(shape, dtype=np.uint8).tobytes()
        if array.ndim == 2:
            mask = array > 0
        elif array.ndim == 3:
            mask = np.any(array > 0, axis=0)
        else:
            raise BackendError(f"FastSAM returned unsupported mask shape: {array.shape}")
        if list(mask.shape) != list(shape):
            mask = self.resize_mask(mask, shape, np)
        return mask.astype(np.uint8).tobytes()

    def resize_mask(self, mask, shape, np):
        rows = (np.arange(shape[0]) * mask.shape[0] // shape[0]).astype(np.int64)
        columns = (np.arange(shape[1]) * mask.shape[1] // shape[1]).astype(np.int64)
        return mask[rows][:, columns]

    def has_masks(self, results):
        return bool(results and getattr(results[0], "masks", None) and len(results[0].masks.data))

    def add_source_path(self, checkpoint_path):
        source = Path(__file__).resolve().parents[3] / "sam_variants" / "FastSAM"
        if str(source) not in import_paths:
            import_paths.insert(0, str(source))

    def prepare_environment(self):
        environ.setdefault("MPLCONFIGDIR", "/tmp/samm-matplotlib")
        environ.setdefault("YOLO_CONFIG_DIR", "/tmp/samm-ultralytics")
        filterwarnings("ignore", message="pkg_resources is deprecated as an API.*", category=UserWarning)

    def fastsam_module(self):
        temp_home = Path("/tmp/samm-fastsam-home")
        (temp_home / ".config").mkdir(parents=True, exist_ok=True)
        original_home = environ.get("HOME")
        environ["HOME"] = str(temp_home)
        try:
            return self.required_module("fastsam", "FastSAM")
        finally:
            if original_home is None:
                environ.pop("HOME", None)
            else:
                environ["HOME"] = original_home

    def required_module(self, name, package_name=None):
        try:
            return import_module(name)
        except ModuleNotFoundError as exc:
            if exc.name in (name, name.split(".", 1)[0]):
                raise BackendError(f"{package_name or name} is not installed") from exc
            raise
