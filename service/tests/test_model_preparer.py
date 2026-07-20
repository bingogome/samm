from pathlib import Path
import builtins
import json
import sys
import tempfile
from threading import Event, Thread
import types
import unittest
from unittest.mock import patch

from samm_server.backends.base import BackendError
from samm_server.backends.medsam_text import MedSamTextBackend
from samm_server.backends.sam3 import Sam3Backend
from samm_server.model_preparer import ModelPreparer
from samm_server.model_registry import MODEL_DEFINITIONS


class FakeSamModel:
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.device = None

    def to(self, device):
        self.device = device

    def eval(self):
        self.evaluated = True

    def parameters(self):
        return iter([types.SimpleNamespace(device=self.device or "cpu")])


class FakeSamPredictor:
    def __init__(self, model):
        self.model = model
        self.image = None
        self.images = []
        self.calls = []

    def set_image(self, image):
        self.image = image
        self.images.append(image)
        self.features = FakeFeatures(image.data)
        self.original_size = tuple(image.shape[:2])
        self.input_size = tuple(image.shape[:2])

    def predict(self, point_coords=None, point_labels=None, box=None, mask_input=None, multimask_output=True):
        self.calls.append((point_coords, point_labels, box, mask_input, multimask_output))
        return FakeMasks([FakeMask(b"low"), FakeMask(b"best")]), FakeScores(1), None


class FakeSam2Model:
    def __init__(self, config, checkpoint, device):
        self.config = config
        self.checkpoint = checkpoint
        self.device = device

    def parameters(self):
        return iter([types.SimpleNamespace(device=self.device)])


class FakeSam2Predictor(FakeSamPredictor):
    def set_image(self, image):
        self.image = image
        self.images.append(image)
        self._features = {"image": FakeFeatures(image.data)}
        self._orig_hw = [tuple(image.shape[:2])]
        self._is_image_set = True
        self._is_batch = False


class FakeSam3Model:
    def __init__(self, checkpoint_path, device, enable_inst_interactivity):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.enable_inst_interactivity = enable_inst_interactivity
        self.inst_calls = []

    def parameters(self):
        return iter([types.SimpleNamespace(device=self.device)])

    def predict_inst(self, state, **kwargs):
        self.inst_calls.append((state, kwargs))
        return FakeMasks([FakeMask(b"low"), FakeMask(b"best")]), FakeScores(1), None


class FakeSam3VideoModel:
    def __init__(self, checkpoint_path, device):
        self.detector = FakeSam3Model(checkpoint_path, device, True)
        self.tracker = types.SimpleNamespace()
        self.device = device


class FakeSam3InteractiveImagePredictor:
    def __init__(self, tracker):
        self.model = tracker


class FakeSam3Processor:
    def __init__(self, model, device, confidence_threshold=0.5):
        self.model = model
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.images = []
        self.reset_calls = []
        self.text_calls = []
        self.box_calls = []

    def set_image(self, image):
        self.images.append(image)
        return {"image": image, "original_height": 2, "original_width": 2, "backbone_out": {}}

    def reset_all_prompts(self, state):
        self.reset_calls.append(state)

    def set_text_prompt(self, prompt, state):
        self.text_calls.append((prompt, state))
        return self.grounding_state(state)

    def add_geometric_prompt(self, box, label, state):
        self.box_calls.append((box, label, state))
        return self.grounding_state(state)

    def grounding_state(self, state):
        state["masks"] = FakeMasks(
            [
                FakeMask(bytes([1, 0, 0, 0])),
                FakeMask(bytes([0, 1, 1, 0])),
            ],
            union_data=bytes([1, 1, 1, 0]),
        )
        state["scores"] = FakeScores(1)
        return state


class FakeFastSamModel:
    def __init__(self, checkpoint):
        import torch
        self.checkpoint = checkpoint
        self.load_kwargs = torch.load(checkpoint)
        self.calls = []

    def __call__(self, image, **kwargs):
        self.calls.append((image, kwargs))
        return [types.SimpleNamespace(masks=types.SimpleNamespace(data=[1]))]


class FakeFastSamPrompt:
    instances = []

    def __init__(self, image, results, device="cuda"):
        self.image = image
        self.results = results
        self.device = device
        self.calls = []
        self.instances.append(self)

    def text_prompt(self, text):
        self.calls.append(("text", text))
        return FakeFastMaskArray(bytes([0, 1, 1, 0]))

    def box_prompt(self, bboxes=None):
        self.calls.append(("box", bboxes))
        return FakeFastMaskArray(bytes([1, 0, 0, 1]))

    def point_prompt(self, points, pointlabel):
        self.calls.append(("points", points, pointlabel))
        return FakeFastMaskArray(bytes([1, 1, 0, 0]))


class FakeFastMaskArray:
    def __init__(self, data):
        self.data = data
        self.size = len(data)
        self.ndim = 2
        self.shape = [2, 2]
        self.dtype = None

    def __gt__(self, value):
        return self

    def astype(self, dtype):
        self.dtype = dtype
        return self

    def tobytes(self):
        return self.data


class FakePillowImageModule:
    @staticmethod
    def fromarray(array):
        return types.SimpleNamespace(array=array)


class FakeArray:
    def __init__(self, data, dtype=None):
        self.data = data
        self.dtype = dtype
        self.shape = None

    def reshape(self, shape):
        self.shape = shape
        return self


class FakeFeatures:
    def __init__(self, data):
        self.data = data

    def detach(self):
        return self

    def cpu(self):
        return self


class FakeMask:
    def __init__(self, data):
        self.data = data
        self.dtype = None

    def astype(self, dtype):
        self.dtype = dtype
        return self

    def squeeze(self):
        return self

    def to(self, dtype):
        self.dtype = dtype
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self

    def reshape(self, shape):
        self.shape = shape
        return self

    def tobytes(self):
        return self.data


class FakeMasks:
    def __init__(self, masks, union_data=b"union"):
        self.masks = masks
        self.union_data = union_data
        self.shape = [len(masks)]

    def __getitem__(self, index):
        return self.masks[index]

    def any(self, dim):
        self.any_dim = dim
        return FakeMask(self.union_data)


class FakeScores:
    def __init__(self, best_index):
        self.best_index = best_index

    def argmax(self):
        return self.best_index


class FakeBackend:
    def __init__(self):
        self.calls = []
        self.embed_calls = []
        self.embedding_calls = []
        self.video_calls = []
        self.save_calls = []
        self.load_calls = []
        self.offloaded = False

    def prepare(self, weight, checkpoint_path, device):
        return {"backend": weight.backend, "model_type": weight.model_type, "device": device}

    def offload(self):
        self.offloaded = True

    def predict(self, weight, image_bytes, shape, points, labels, box=None, mask=None, text=None):
        self.calls.append((weight, image_bytes, shape, points, labels, box, mask, text))
        return b"mask"

    def embed(self, weight, image_bytes, shape):
        self.embed_calls.append((weight, image_bytes, shape))
        return {"embedding_id": "emb-1", "shape": [shape[0], shape[1]]}

    def predict_embedding(self, weight, embedding_id, points, labels, box=None, mask=None, text=None):
        self.embedding_calls.append((weight, embedding_id, points, labels, box, mask, text))
        return b"mask", [2, 3]

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
        self.video_calls.append((
            weight,
            frames,
            prompt,
            direction,
            cancel_event,
            offload_video_to_cpu,
            offload_state_to_cpu,
        ))
        return iter([{"frame_index": 0, "shape": [1, 1], "data": b"\x01"}])

    def save_embeddings(self, weight, items, path):
        self.save_calls.append((weight, items, path))
        return {"path": str(path), "count": len(items)}

    def load_embeddings(self, weight, path):
        self.load_calls.append((weight, path))
        return {"path": str(path), "count": 1, "items": [{"embedding_id": "emb-1"}]}


def fake_segment_anything():
    module = types.ModuleType("segment_anything")
    module.SamPredictor = FakeSamPredictor
    module.sam_model_registry = {
        "vit_b": lambda checkpoint: FakeSamModel(checkpoint),
        "vit_l": lambda checkpoint: FakeSamModel(checkpoint),
        "vit_h": lambda checkpoint: FakeSamModel(checkpoint),
    }
    return module


def fake_mobile_sam():
    module = types.ModuleType("mobile_sam")
    module.SamPredictor = FakeSamPredictor
    module.sam_model_registry = {"vit_t": lambda checkpoint: FakeSamModel(checkpoint)}
    return module


def fake_sam2_modules():
    sam2 = types.ModuleType("sam2")
    build = types.ModuleType("sam2.build_sam")
    predictor = types.ModuleType("sam2.sam2_image_predictor")
    built = []

    def build_sam2_video_predictor(config_file, ckpt_path, device="cuda"):
        item = FakeSam2Model(config_file, ckpt_path, device)
        built.append((item, config_file, ckpt_path, device, "video"))
        return item

    def build_sam2_video_predictor_npz(config_file, ckpt_path, device="cuda"):
        item = FakeSam2Model(config_file, ckpt_path, device)
        built.append((item, config_file, ckpt_path, device, "video_npz"))
        return item

    build.build_sam2_video_predictor = build_sam2_video_predictor
    build.build_sam2_video_predictor_npz = build_sam2_video_predictor_npz
    predictor.SAM2ImagePredictor = FakeSam2Predictor
    return {
        "sam2": sam2,
        "sam2.build_sam": build,
        "sam2.sam2_image_predictor": predictor,
    }, built


def fake_sam3_modules():
    sam3 = types.ModuleType("sam3")
    model = types.ModuleType("sam3.model")
    builder = types.ModuleType("sam3.model_builder")
    processor = types.ModuleType("sam3.model.sam3_image_processor")
    interactive = types.ModuleType("sam3.model.sam1_task_predictor")
    built = []

    def build_sam3_image_model(**kwargs):
        item = FakeSam3Model(
            kwargs["checkpoint_path"],
            kwargs["device"],
            kwargs["enable_inst_interactivity"],
        )
        built.append((item, kwargs))
        return item

    def build_sam3_video_model(**kwargs):
        item = FakeSam3VideoModel(kwargs["checkpoint_path"], kwargs["device"])
        built.append((item, kwargs))
        return item

    builder.build_sam3_image_model = build_sam3_image_model
    builder.build_sam3_video_model = build_sam3_video_model
    processor.Sam3Processor = FakeSam3Processor
    interactive.SAM3InteractiveImagePredictor = FakeSam3InteractiveImagePredictor
    return {
        "sam3": sam3,
        "sam3.model": model,
        "sam3.model_builder": builder,
        "sam3.model.sam3_image_processor": processor,
        "sam3.model.sam1_task_predictor": interactive,
    }, built


def fake_pillow():
    pil = types.ModuleType("PIL")
    image = types.ModuleType("PIL.Image")
    image.fromarray = FakePillowImageModule.fromarray
    return {"PIL": pil, "PIL.Image": image}


def fake_sam3_runtime():
    modules, built = fake_sam3_modules()
    modules.update(fake_pillow())
    modules["numpy"] = fake_numpy()
    modules["torch"] = fake_sam3_torch()
    return modules, built


def fake_sam3_torch():
    module = types.ModuleType("torch")
    module.bfloat16 = "bfloat16"
    module.uint8 = "uint8"
    module.autocast_calls = []

    class Autocast:
        def __init__(self, device_type, dtype):
            self.device_type = device_type
            self.dtype = dtype

        def __enter__(self):
            module.autocast_calls.append((self.device_type, self.dtype))

        def __exit__(self, exc_type, exc, traceback):
            return False

    module.autocast = Autocast
    return module


def fake_fastsam_runtime():
    module = types.ModuleType("fastsam")
    prompt = types.ModuleType("fastsam.prompt")
    clip = types.ModuleType("clip")
    FakeFastSamPrompt.instances = []
    module.FastSAM = FakeFastSamModel
    module.FastSAMPrompt = FakeFastSamPrompt
    return {
        "fastsam": module,
        "fastsam.prompt": prompt,
        "clip": clip,
        "numpy": fake_fastsam_numpy(),
        "torch": fake_fastsam_torch(),
    }


def fake_sam2_runtime():
    modules, built = fake_sam2_modules()
    modules["numpy"] = fake_numpy()
    return modules, built


def fake_fastsam_numpy():
    module = types.ModuleType("numpy")
    module.uint8 = "uint8"
    module.frombuffer = lambda data, dtype: FakeArray(data, dtype)
    module.asarray = lambda data: data
    module.zeros = lambda shape, dtype: FakeFastMaskArray(bytes(shape[0] * shape[1]))
    return module


def fake_fastsam_torch():
    module = types.ModuleType("torch")
    module.load = lambda path, **kwargs: dict(kwargs)
    return module


def fake_numpy():
    module = types.ModuleType("numpy")
    module.uint8 = "uint8"
    module.float32 = "float32"
    module.int32 = "int32"
    module.frombuffer = lambda data, dtype: FakeArray(data, dtype)
    module.array = lambda data, dtype: FakeArray(data, dtype)
    return module


def fake_torch():
    module = types.ModuleType("torch")
    saved = {}

    def save(payload, path):
        saved[str(path)] = payload
        Path(path).write_text("torch\n", encoding="utf-8")

    def load(path, map_location=None):
        return saved[str(path)]

    module.save = save
    module.load = load
    return module


class ModelPreparerTest(unittest.TestCase):
    def test_unknown_model_returns_404(self):
        with tempfile.TemporaryDirectory() as model_dir:
            result = ModelPreparer(model_dir).prepare("unknown")

        self.assertEqual(result.status_code, 404)
        self.assertEqual(result.payload, {"error": "weight not found", "weight_id": "unknown"})

    def test_missing_checkpoint_returns_409(self):
        model = MODEL_DEFINITIONS[0]
        weight = model.weights[0]
        with tempfile.TemporaryDirectory() as model_dir:
            result = ModelPreparer(model_dir).prepare(weight.id)

        self.assertEqual(result.status_code, 409)
        self.assertEqual(result.payload["model_id"], model.id)
        self.assertEqual(result.payload["weight_id"], weight.id)
        self.assertEqual(result.payload["checkpoint"], weight.checkpoint)

    def test_available_model_prepares(self):
        model = MODEL_DEFINITIONS[0]
        weight = model.weights[0]
        with tempfile.TemporaryDirectory() as model_dir:
            checkpoint = Path(model_dir, weight.checkpoint)
            checkpoint.touch()
            with patch.dict(sys.modules, {"segment_anything": fake_segment_anything()}):
                preparer = ModelPreparer(model_dir, "cuda")
                result = preparer.prepare(weight.id)
                prepared_model = preparer.backends["sam1"].models[weight.id]
                predictor = preparer.backends["sam1"].predictors[weight.id]

        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.payload["model_id"], model.id)
        self.assertEqual(result.payload["weight_id"], weight.id)
        self.assertEqual(result.payload["status"], "prepared")
        self.assertEqual(result.payload["backend"], weight.backend)
        self.assertEqual(result.payload["model_type"], weight.model_type)
        self.assertEqual(result.payload["device"], "cuda")
        self.assertEqual(result.payload["capabilities"], weight.capabilities)
        self.assertEqual(preparer.prepared_weight_id, weight.id)
        self.assertEqual(preparer.prepared_weight, weight)
        self.assertEqual(preparer.prepared_payload(), result.payload)
        self.assertEqual(prepared_model.checkpoint, str(checkpoint))
        self.assertEqual(prepared_model.device, "cuda")
        self.assertEqual(predictor.model, prepared_model)

    def test_sam2_model_prepares(self):
        model = MODEL_DEFINITIONS[1]
        weight = model.weights[0]
        with tempfile.TemporaryDirectory() as model_dir:
            checkpoint = Path(model_dir, weight.checkpoint)
            checkpoint.touch()
            modules, built = fake_sam2_runtime()
            with patch.dict(sys.modules, modules):
                preparer = ModelPreparer(model_dir, "cuda")
                result = preparer.prepare(weight.id)
                prepared_model = preparer.backends["sam2"].models[weight.id]
                predictor = preparer.backends["sam2"].predictors[weight.id]

        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.payload["model_id"], model.id)
        self.assertEqual(result.payload["backend"], "sam2")
        self.assertEqual(result.payload["model_type"], "configs/sam2.1/sam2.1_hiera_t.yaml")
        self.assertEqual(prepared_model.config, weight.model_type)
        self.assertEqual(prepared_model.checkpoint, str(checkpoint))
        self.assertEqual(prepared_model.device, "cuda")
        self.assertEqual(predictor.model, prepared_model)
        self.assertEqual(built[0][1], weight.model_type)
        self.assertEqual(built[0][4], "video")
        self.assertIs(preparer.backends["sam2"].video_predictors[weight.id], prepared_model)

    def test_sam2_predict_uses_predictor(self):
        weight = MODEL_DEFINITIONS[1].weights[0]
        with tempfile.TemporaryDirectory() as model_dir:
            Path(model_dir, weight.checkpoint).touch()
            modules, _ = fake_sam2_runtime()
            with patch.dict(sys.modules, modules):
                preparer = ModelPreparer(model_dir, backends=None)
                preparer.prepare(weight.id)
                mask = preparer.predict(bytes(range(12)), [2, 2, 3], [[1, 1], [0, 0]], [1, 0], [0, 0, 1, 1])
                predictor = preparer.backends["sam2"].predictors[weight.id]

        point_coords, point_labels, box, mask_input, multimask_output = predictor.calls[0]
        self.assertEqual(mask, b"best")
        self.assertEqual(predictor.image.data, bytes(range(12)))
        self.assertEqual(predictor.image.dtype, "uint8")
        self.assertEqual(predictor.image.shape, [2, 2, 3])
        self.assertEqual(point_coords.data, [[1, 1], [0, 0]])
        self.assertEqual(point_labels.data, [1, 0])
        self.assertEqual(box.data, [0, 0, 1, 1])
        self.assertIsNone(mask_input)
        self.assertTrue(multimask_output)

    def test_sam2_predict_embedding_uses_stored_predictor(self):
        weight = MODEL_DEFINITIONS[1].weights[0]
        with tempfile.TemporaryDirectory() as model_dir:
            Path(model_dir, weight.checkpoint).touch()
            modules, _ = fake_sam2_runtime()
            with patch.dict(sys.modules, modules):
                preparer = ModelPreparer(model_dir, backends=None)
                preparer.prepare(weight.id)
                embedding = preparer.embed(b"same-image-12", [2, 2, 3])
                mask, shape = preparer.predict_embedding(embedding["embedding_id"], [[0, 0]], [0])
                predictor = preparer.backends["sam2"].embeddings[embedding["embedding_id"]]["predictor"]

        self.assertEqual(mask, b"best")
        self.assertEqual(shape, [2, 2])
        self.assertEqual(len(predictor.images), 1)
        self.assertEqual(len(predictor.calls), 1)
        self.assertEqual(predictor.calls[0][0].data, [[0, 0]])
        self.assertEqual(predictor.calls[0][1].data, [0])

    def test_sam2_save_load_embeddings_folder(self):
        weight = MODEL_DEFINITIONS[1].weights[0]
        with tempfile.TemporaryDirectory() as model_dir:
            Path(model_dir, weight.checkpoint).touch()
            modules, _ = fake_sam2_runtime()
            modules["torch"] = fake_torch()
            with patch.dict(sys.modules, modules):
                preparer = ModelPreparer(model_dir, backends=None)
                preparer.prepare(weight.id)
                embedding = preparer.embed(b"same-image-12", [2, 2, 3])
                path = Path(model_dir, "embeddings")
                result = preparer.save_embeddings([embedding_item(embedding["embedding_id"])], path)
                metadata = json.loads(Path(path, "metadata.json").read_text(encoding="utf-8"))
                loaded = preparer.load_embeddings(path)

        self.assertEqual(result["path"], str(path))
        self.assertEqual(result["count"], 1)
        self.assertEqual(metadata["backend"], "sam2")
        self.assertEqual(metadata["backend_format"], "samm-sam2-embedding-data")
        self.assertEqual(loaded["count"], 1)
        self.assertEqual(loaded["items"][0]["image_digest"], "00")

    def test_mobile_sam_model_prepares(self):
        model = MODEL_DEFINITIONS[2]
        weight = model.weights[0]
        with tempfile.TemporaryDirectory() as model_dir:
            checkpoint = Path(model_dir, weight.checkpoint)
            checkpoint.touch()
            with patch.dict(sys.modules, {"mobile_sam": fake_mobile_sam()}):
                preparer = ModelPreparer(model_dir, "cuda")
                result = preparer.prepare(weight.id)
                prepared_model = preparer.backends["mobile_sam"].models[weight.id]
                predictor = preparer.backends["mobile_sam"].predictors[weight.id]

        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.payload["model_id"], model.id)
        self.assertEqual(result.payload["backend"], "mobile_sam")
        self.assertEqual(result.payload["model_type"], "vit_t")
        self.assertEqual(prepared_model.checkpoint, str(checkpoint))
        self.assertEqual(prepared_model.device, "cuda")
        self.assertEqual(predictor.model, prepared_model)

    def test_medsam_model_prepares(self):
        model = MODEL_DEFINITIONS[3]
        weight = model.weights[0]
        with tempfile.TemporaryDirectory() as model_dir:
            checkpoint = Path(model_dir, weight.checkpoint)
            checkpoint.touch()
            with patch.dict(sys.modules, {"segment_anything": fake_segment_anything()}):
                preparer = ModelPreparer(model_dir, "cuda")
                result = preparer.prepare(weight.id)
                prepared_model = preparer.backends["medsam"].models[weight.id]

        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.payload["model_id"], model.id)
        self.assertEqual(result.payload["backend"], "medsam")
        self.assertEqual(result.payload["model_type"], "vit_b")
        self.assertEqual(prepared_model.checkpoint, str(checkpoint))
        self.assertEqual(prepared_model.device, "cuda")

    def test_medsam_text_prompt_validation(self):
        backend = MedSamTextBackend()

        backend.validate_prompt([], None, None, "liver")
        with self.assertRaisesRegex(BackendError, "text prompts only"):
            backend.validate_prompt([[1, 1]], None, None, "liver")
        with self.assertRaisesRegex(BackendError, "text prompts only"):
            backend.validate_prompt([], [0, 0, 1, 1], None, "liver")
        with self.assertRaisesRegex(BackendError, "requires a text prompt"):
            backend.validate_prompt([], None, None, None)

    def test_medsam2_model_prepares(self):
        model = MODEL_DEFINITIONS[4]
        weight = model.weights[0]
        with tempfile.TemporaryDirectory() as model_dir:
            checkpoint = Path(model_dir, weight.checkpoint)
            checkpoint.touch()
            modules, built = fake_sam2_runtime()
            with patch.dict(sys.modules, modules):
                preparer = ModelPreparer(model_dir, "cuda")
                result = preparer.prepare(weight.id)
                prepared_model = preparer.backends["medsam2"].models[weight.id]
                predictor = preparer.backends["medsam2"].predictors[weight.id]

        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.payload["model_id"], model.id)
        self.assertEqual(result.payload["backend"], "medsam2")
        self.assertEqual(result.payload["model_type"], "configs/sam2.1_hiera_t512.yaml")
        self.assertEqual(prepared_model.config, weight.model_type)
        self.assertEqual(prepared_model.checkpoint, str(checkpoint))
        self.assertEqual(prepared_model.device, "cuda")
        self.assertEqual(predictor.model, prepared_model)
        self.assertEqual(built[0][1], weight.model_type)
        self.assertEqual(built[0][4], "video_npz")
        self.assertIs(preparer.backends["medsam2"].video_predictors[weight.id], prepared_model)

    def test_finetuned_medsam2_model_prepares(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "checkpoints"
            run_dir = root / "finetuning_runs" / "liver"
            checkpoint = run_dir / "checkpoints" / "checkpoint.pt"
            model_dir.mkdir()
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()
            (run_dir / "samm_model.json").write_text(json.dumps({
                "id": "medsam2_liver",
                "label": "Liver",
                "model_id": "medsam2",
                "backend": "medsam2",
                "checkpoint": "finetuning_runs/liver/checkpoints/checkpoint.pt",
                "model_type": "configs/sam2.1_hiera_t512.yaml",
            }), encoding="utf-8")
            modules, _ = fake_sam2_runtime()
            with patch.dict(sys.modules, modules):
                preparer = ModelPreparer(model_dir, "cuda")
                result = preparer.prepare("medsam2_liver")
                prepared_model = preparer.backends["medsam2"].models["medsam2_liver"]

        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.payload["model_id"], "medsam2")
        self.assertEqual(result.payload["weight_id"], "medsam2_liver")
        self.assertEqual(result.payload["backend"], "medsam2")
        self.assertEqual(prepared_model.checkpoint, str(checkpoint.resolve()))

    def test_sam3_model_prepares_with_instance_interactivity(self):
        model = MODEL_DEFINITIONS[5]
        weight = model.weights[0]
        with tempfile.TemporaryDirectory() as model_dir:
            checkpoint = Path(model_dir, weight.checkpoint)
            checkpoint.touch()
            modules, built = fake_sam3_runtime()
            with patch.dict(sys.modules, modules):
                preparer = ModelPreparer(model_dir, "cuda")
                result = preparer.prepare(weight.id)
                backend = preparer.backends["sam3"]
                prepared_model = backend.models[weight.id]
                processor = backend.processors[weight.id]

        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.payload["model_id"], model.id)
        self.assertEqual(result.payload["backend"], "sam3")
        self.assertEqual(result.payload["model_type"], "sam3")
        self.assertEqual(prepared_model.checkpoint_path, str(checkpoint))
        self.assertEqual(prepared_model.device, "cuda")
        self.assertTrue(prepared_model.enable_inst_interactivity)
        self.assertEqual(processor.confidence_threshold, 0.5)
        self.assertEqual(backend.grounding_mask_mode, "union")
        self.assertIs(backend.video_models[weight.id], built[0][0])
        self.assertIs(prepared_model, backend.video_models[weight.id].detector)
        self.assertIs(
            prepared_model.inst_interactive_predictor.model,
            backend.video_models[weight.id].tracker,
        )
        self.assertEqual(built[0][1]["load_from_HF"], False)

    def test_sam3_predict_uses_interactive_points_and_box(self):
        weight = MODEL_DEFINITIONS[5].weights[0]
        with tempfile.TemporaryDirectory() as model_dir:
            Path(model_dir, weight.checkpoint).touch()
            modules, _ = fake_sam3_runtime()
            with patch.dict(sys.modules, modules):
                preparer = ModelPreparer(model_dir, backends=None)
                preparer.prepare(weight.id)
                mask = preparer.predict(bytes(range(12)), [2, 2, 3], [[1, 1], [0, 0]], [1, 0], [0, 0, 1, 1])
                backend = preparer.backends["sam3"]
                model = backend.models[weight.id]
                image = backend.processors[weight.id].images[0]

        state, call = model.inst_calls[0]
        self.assertEqual(mask, b"best")
        self.assertIs(state["image"], image)
        self.assertEqual(image.array.data, bytes(range(12)))
        self.assertEqual(image.array.dtype, "uint8")
        self.assertEqual(image.array.shape, [2, 2, 3])
        self.assertEqual(call["point_coords"].data, [[1, 1], [0, 0]])
        self.assertEqual(call["point_coords"].dtype, "float32")
        self.assertEqual(call["point_labels"].data, [1, 0])
        self.assertEqual(call["point_labels"].dtype, "int32")
        self.assertEqual(call["box"].data, [0, 0, 1, 1])
        self.assertEqual(call["box"].dtype, "float32")
        self.assertIsNone(call["mask_input"])
        self.assertTrue(call["multimask_output"])

    def test_sam3_cuda_predict_uses_bfloat16_autocast(self):
        weight = MODEL_DEFINITIONS[5].weights[0]
        with tempfile.TemporaryDirectory() as model_dir:
            Path(model_dir, weight.checkpoint).touch()
            modules, _ = fake_sam3_runtime()
            with patch.dict(sys.modules, modules):
                preparer = ModelPreparer(model_dir, "cuda")
                preparer.prepare(weight.id)
                preparer.predict(bytes(range(12)), [2, 2, 3], [[1, 1]], [1])

        self.assertEqual(modules["torch"].autocast_calls, [("cuda", "bfloat16")])

    def test_sam3_predict_uses_interactive_mask_input(self):
        weight = MODEL_DEFINITIONS[5].weights[0]
        prompt_mask = {"shape": [256, 256], "data": bytes([1]) * (256 * 256)}
        with tempfile.TemporaryDirectory() as model_dir:
            Path(model_dir, weight.checkpoint).touch()
            modules, _ = fake_sam3_runtime()
            with patch.dict(sys.modules, modules):
                preparer = ModelPreparer(model_dir, backends=None)
                preparer.prepare(weight.id)
                mask = preparer.predict(bytes(range(12)), [2, 2, 3], [], [], None, prompt_mask)
                model = preparer.backends["sam3"].models[weight.id]

        call = model.inst_calls[0][1]
        self.assertEqual(mask, b"best")
        self.assertIsNone(call["point_coords"])
        self.assertIsNone(call["point_labels"])
        self.assertIsNone(call["box"])
        self.assertEqual(call["mask_input"].data[0].data, prompt_mask["data"])
        self.assertEqual(call["mask_input"].dtype, "float32")

    def test_sam3_predict_embedding_uses_interactive_points(self):
        weight = MODEL_DEFINITIONS[5].weights[0]
        with tempfile.TemporaryDirectory() as model_dir:
            Path(model_dir, weight.checkpoint).touch()
            modules, _ = fake_sam3_runtime()
            with patch.dict(sys.modules, modules):
                preparer = ModelPreparer(model_dir, backends=None)
                preparer.prepare(weight.id)
                embedding = preparer.embed(bytes(range(12)), [2, 2, 3])
                mask, shape = preparer.predict_embedding(embedding["embedding_id"], [[1, 1]], [1])
                backend = preparer.backends["sam3"]
                model = backend.models[weight.id]
                processor = backend.processors[weight.id]

        state, call = model.inst_calls[0]
        self.assertEqual(mask, b"best")
        self.assertEqual(shape, [2, 2])
        self.assertIs(processor.reset_calls[0], state)
        self.assertEqual(call["point_coords"].data, [[1, 1]])
        self.assertEqual(call["point_labels"].data, [1])

    def test_sam3_rejects_text_with_points(self):
        weight = MODEL_DEFINITIONS[5].weights[0]
        with tempfile.TemporaryDirectory() as model_dir:
            Path(model_dir, weight.checkpoint).touch()
            modules, _ = fake_sam3_runtime()
            with patch.dict(sys.modules, modules):
                preparer = ModelPreparer(model_dir, backends=None)
                preparer.prepare(weight.id)
                with self.assertRaisesRegex(BackendError, "cannot be combined"):
                    preparer.predict(bytes(range(12)), [2, 2, 3], [[1, 1]], [1], None, None, "kidney")

    def test_medical_sam3_model_prepares_with_sam3_backend(self):
        model = MODEL_DEFINITIONS[6]
        weight = model.weights[0]
        with tempfile.TemporaryDirectory() as model_dir:
            checkpoint = Path(model_dir, weight.checkpoint)
            checkpoint.touch()
            modules, built = fake_sam3_runtime()
            with patch.dict(sys.modules, modules):
                preparer = ModelPreparer(model_dir, "cuda")
                result = preparer.prepare(weight.id)
                backend = preparer.backends["medical_sam3"]
                prepared_model = backend.models[weight.id]
                processor = backend.processors[weight.id]

        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.payload["model_id"], "medical_sam3")
        self.assertEqual(result.payload["backend"], "medical_sam3")
        self.assertEqual(result.payload["model_type"], "medical_sam3")
        self.assertEqual(prepared_model.checkpoint_path, str(checkpoint))
        self.assertEqual(prepared_model.device, "cuda")
        self.assertFalse(prepared_model.enable_inst_interactivity)
        self.assertEqual(processor.confidence_threshold, 0.1)
        self.assertEqual(backend.grounding_mask_mode, "best")
        self.assertEqual(backend.video_models, {})
        self.assertEqual(built[0][1]["load_from_HF"], False)

    def test_medical_sam3_rejects_unsupported_interactive_prompts(self):
        backend = Sam3Backend(enable_inst_interactivity=False)

        backend.validate_prompt([], [0, 0, 1, 1], None, None)
        backend.validate_prompt([], None, None, "kidney")
        with self.assertRaisesRegex(BackendError, "does not support point or mask"):
            backend.validate_prompt([[1, 1]], None, None, None)
        with self.assertRaisesRegex(BackendError, "does not support point or mask"):
            backend.validate_prompt([], None, {"shape": [2, 2], "data": b"mask"}, None)

    def test_medical_sam3_box_prediction_uses_grounding_and_best_mask(self):
        weight = MODEL_DEFINITIONS[6].weights[0]
        with tempfile.TemporaryDirectory() as model_dir:
            Path(model_dir, weight.checkpoint).touch()
            modules, _ = fake_sam3_runtime()
            with patch.dict(sys.modules, modules):
                preparer = ModelPreparer(model_dir)
                preparer.prepare(weight.id)
                mask = preparer.predict(bytes(range(12)), [2, 2, 3], [], [], [0, 0, 2, 2])
                backend = preparer.backends["medical_sam3"]
                processor = backend.processors[weight.id]
                model = backend.models[weight.id]

        self.assertEqual(mask, bytes([0, 1, 1, 0]))
        self.assertEqual(processor.box_calls[0][:2], ([0.5, 0.5, 1.0, 1.0], True))
        self.assertEqual(model.inst_calls, [])

    def test_medical_sam3_embedding_text_prediction_uses_best_mask(self):
        weight = MODEL_DEFINITIONS[6].weights[0]
        with tempfile.TemporaryDirectory() as model_dir:
            Path(model_dir, weight.checkpoint).touch()
            modules, _ = fake_sam3_runtime()
            with patch.dict(sys.modules, modules):
                preparer = ModelPreparer(model_dir)
                preparer.prepare(weight.id)
                embedding = preparer.embed(bytes(range(12)), [2, 2, 3])
                mask, shape = preparer.predict_embedding(embedding["embedding_id"], [], [], text="tumor")
                processor = preparer.backends["medical_sam3"].processors[weight.id]

        self.assertEqual(mask, bytes([0, 1, 1, 0]))
        self.assertEqual(shape, [2, 2])
        self.assertEqual(processor.text_calls[0][0], "tumor")
        self.assertIs(processor.reset_calls[0], processor.text_calls[0][1])

    def test_sam3_grounding_mask_mode_can_select_best_or_union(self):
        masks = FakeMasks([FakeMask(b"first"), FakeMask(b"best")])
        state = {"masks": masks, "scores": FakeScores(1)}
        modules = types.SimpleNamespace(uint8="uint8")

        best_backend = Sam3Backend(grounding_mask_mode="best")
        with patch.object(best_backend, "required_module", return_value=modules):
            best = best_backend.grounding_mask(state, [2, 2])

        union_backend = Sam3Backend(grounding_mask_mode="union")
        with patch.object(union_backend, "required_module", return_value=modules):
            union = union_backend.grounding_mask(state, [2, 2])

        self.assertEqual(best, b"best")
        self.assertEqual(union, b"union")

    def test_fastsam_model_prepares(self):
        model = MODEL_DEFINITIONS[7]
        weight = model.weights[0]
        with tempfile.TemporaryDirectory() as model_dir:
            checkpoint = Path(model_dir, weight.checkpoint)
            checkpoint.touch()
            with patch.dict(sys.modules, fake_fastsam_runtime()):
                preparer = ModelPreparer(model_dir, "cuda")
                result = preparer.prepare(weight.id)
                prepared_model = preparer.backends["fastsam"].models[weight.id]

        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.payload["model_id"], model.id)
        self.assertEqual(result.payload["backend"], "fastsam")
        self.assertEqual(result.payload["model_type"], "fastsam_x")
        self.assertEqual(prepared_model.checkpoint, str(checkpoint))
        self.assertEqual(prepared_model.load_kwargs["weights_only"], False)

    def test_fastsam_predict_uses_text_prompt(self):
        weight = MODEL_DEFINITIONS[7].weights[0]
        with tempfile.TemporaryDirectory() as model_dir:
            Path(model_dir, weight.checkpoint).touch()
            with patch.dict(sys.modules, fake_fastsam_runtime()):
                preparer = ModelPreparer(model_dir, "cuda")
                preparer.prepare(weight.id)
                mask = preparer.predict(bytes(range(12)), [2, 2, 3], [], [], None, None, "kidney")
                backend = preparer.backends["fastsam"]
                model = backend.models[weight.id]
                prompt = backend.prompts[weight.id].instances[0]

        self.assertEqual(mask, bytes([0, 1, 1, 0]))
        self.assertEqual(model.calls[0][0].data, bytes(range(12)))
        self.assertEqual(model.calls[0][0].shape, [2, 2, 3])
        self.assertEqual(model.calls[0][1]["device"], "cuda")
        self.assertEqual(model.calls[0][1]["retina_masks"], True)
        self.assertEqual(prompt.device, "cuda")
        self.assertEqual(prompt.calls, [("text", "kidney")])

    def test_fastsam_predict_uses_box_prompt(self):
        weight = MODEL_DEFINITIONS[7].weights[0]
        with tempfile.TemporaryDirectory() as model_dir:
            Path(model_dir, weight.checkpoint).touch()
            with patch.dict(sys.modules, fake_fastsam_runtime()):
                preparer = ModelPreparer(model_dir, "cuda")
                preparer.prepare(weight.id)
                mask = preparer.predict(bytes(range(12)), [2, 2, 3], [], [], [0, 0, 1, 1])
                prompt = preparer.backends["fastsam"].prompts[weight.id].instances[0]

        self.assertEqual(mask, bytes([1, 0, 0, 1]))
        self.assertEqual(prompt.calls, [("box", [[0, 0, 1, 1]])])

    def test_fastsam_rejects_mask_and_embeddings(self):
        weight = MODEL_DEFINITIONS[7].weights[0]
        prompt_mask = {"shape": [2, 2], "data": bytes([1, 0, 0, 1])}
        with tempfile.TemporaryDirectory() as model_dir:
            Path(model_dir, weight.checkpoint).touch()
            with patch.dict(sys.modules, fake_fastsam_runtime()):
                preparer = ModelPreparer(model_dir, "cuda")
                preparer.prepare(weight.id)
                with self.assertRaisesRegex(BackendError, "does not support mask"):
                    preparer.predict(bytes(range(12)), [2, 2, 3], [], [], None, prompt_mask)
                with self.assertRaisesRegex(BackendError, "cached embeddings"):
                    preparer.embed(bytes(range(12)), [2, 2, 3])

    def test_sam1_predict_uses_predictor(self):
        weight = MODEL_DEFINITIONS[0].weights[0]
        with tempfile.TemporaryDirectory() as model_dir:
            Path(model_dir, weight.checkpoint).touch()
            modules = {"segment_anything": fake_segment_anything(), "numpy": fake_numpy()}
            with patch.dict(sys.modules, modules):
                preparer = ModelPreparer(model_dir, backends=None)
                preparer.prepare(weight.id)
                mask = preparer.predict(bytes(range(12)), [2, 2, 3], [[1, 1], [0, 0]], [1, 0])
                predictor = preparer.backends["sam1"].predictors[weight.id]

        point_coords, point_labels, box, mask_input, multimask_output = predictor.calls[0]
        self.assertEqual(mask, b"best")
        self.assertEqual(predictor.image.data, bytes(range(12)))
        self.assertEqual(predictor.image.dtype, "uint8")
        self.assertEqual(predictor.image.shape, [2, 2, 3])
        self.assertEqual(point_coords.data, [[1, 1], [0, 0]])
        self.assertEqual(point_coords.dtype, "float32")
        self.assertEqual(point_labels.data, [1, 0])
        self.assertEqual(point_labels.dtype, "int32")
        self.assertIsNone(box)
        self.assertIsNone(mask_input)
        self.assertTrue(multimask_output)
        self.assertEqual(len(predictor.images), 1)

    def test_sam1_predict_uses_box(self):
        weight = MODEL_DEFINITIONS[0].weights[0]
        with tempfile.TemporaryDirectory() as model_dir:
            Path(model_dir, weight.checkpoint).touch()
            modules = {"segment_anything": fake_segment_anything(), "numpy": fake_numpy()}
            with patch.dict(sys.modules, modules):
                preparer = ModelPreparer(model_dir, backends=None)
                preparer.prepare(weight.id)
                mask = preparer.predict(bytes(range(12)), [2, 2, 3], [], [], [0, 0, 1, 1])
                predictor = preparer.backends["sam1"].predictors[weight.id]

        point_coords, point_labels, box, mask_input, multimask_output = predictor.calls[0]
        self.assertEqual(mask, b"best")
        self.assertIsNone(point_coords)
        self.assertIsNone(point_labels)
        self.assertEqual(box.data, [0, 0, 1, 1])
        self.assertEqual(box.dtype, "float32")
        self.assertIsNone(mask_input)
        self.assertTrue(multimask_output)

    def test_sam1_predict_uses_mask_input(self):
        weight = MODEL_DEFINITIONS[0].weights[0]
        prompt_mask = {"shape": [256, 256], "data": bytes([1]) * (256 * 256)}
        with tempfile.TemporaryDirectory() as model_dir:
            Path(model_dir, weight.checkpoint).touch()
            modules = {"segment_anything": fake_segment_anything(), "numpy": fake_numpy()}
            with patch.dict(sys.modules, modules):
                preparer = ModelPreparer(model_dir, backends=None)
                preparer.prepare(weight.id)
                mask = preparer.predict(bytes(range(12)), [2, 2, 3], [], [], None, prompt_mask)
                predictor = preparer.backends["sam1"].predictors[weight.id]

        point_coords, point_labels, box, mask_input, multimask_output = predictor.calls[0]
        self.assertEqual(mask, b"best")
        self.assertIsNone(point_coords)
        self.assertIsNone(point_labels)
        self.assertIsNone(box)
        self.assertEqual(mask_input.data[0].data, prompt_mask["data"])
        self.assertEqual(mask_input.dtype, "float32")
        self.assertTrue(multimask_output)

    def test_sam1_predict_reuses_embedding_for_same_image(self):
        weight = MODEL_DEFINITIONS[0].weights[0]
        with tempfile.TemporaryDirectory() as model_dir:
            Path(model_dir, weight.checkpoint).touch()
            modules = {"segment_anything": fake_segment_anything(), "numpy": fake_numpy()}
            with patch.dict(sys.modules, modules):
                preparer = ModelPreparer(model_dir, backends=None)
                preparer.prepare(weight.id)
                preparer.predict(b"same-image-12", [2, 2, 3], [[1, 1]], [1])
                preparer.predict(b"same-image-12", [2, 2, 3], [[0, 0]], [0])
                predictor = preparer.backends["sam1"].predictors[weight.id]

        self.assertEqual(len(predictor.images), 1)
        self.assertEqual(len(predictor.calls), 2)
        self.assertEqual(predictor.calls[1][0].data, [[0, 0]])
        self.assertEqual(predictor.calls[1][1].data, [0])

    def test_sam1_predict_refreshes_embedding_for_new_image(self):
        weight = MODEL_DEFINITIONS[0].weights[0]
        with tempfile.TemporaryDirectory() as model_dir:
            Path(model_dir, weight.checkpoint).touch()
            modules = {"segment_anything": fake_segment_anything(), "numpy": fake_numpy()}
            with patch.dict(sys.modules, modules):
                preparer = ModelPreparer(model_dir, backends=None)
                preparer.prepare(weight.id)
                preparer.predict(b"first-image!", [2, 2, 3], [[1, 1]], [1])
                preparer.predict(b"second-image", [2, 2, 3], [[1, 1]], [1])
                predictor = preparer.backends["sam1"].predictors[weight.id]

        self.assertEqual(len(predictor.images), 2)
        self.assertEqual(predictor.images[0].data, b"first-image!")
        self.assertEqual(predictor.images[1].data, b"second-image")

    def test_sam1_embed_reuses_embedding_for_same_image(self):
        weight = MODEL_DEFINITIONS[0].weights[0]
        with tempfile.TemporaryDirectory() as model_dir:
            Path(model_dir, weight.checkpoint).touch()
            modules = {"segment_anything": fake_segment_anything(), "numpy": fake_numpy()}
            with patch.dict(sys.modules, modules):
                preparer = ModelPreparer(model_dir, backends=None)
                preparer.prepare(weight.id)
                first = preparer.embed(b"same-image-12", [2, 2, 3])
                second = preparer.embed(b"same-image-12", [2, 2, 3])
                backend = preparer.backends["sam1"]

        self.assertEqual(first, second)
        self.assertEqual(first["shape"], [2, 2])
        self.assertEqual(len(backend.embeddings), 1)

    def test_sam1_predict_embedding_uses_stored_predictor(self):
        weight = MODEL_DEFINITIONS[0].weights[0]
        with tempfile.TemporaryDirectory() as model_dir:
            Path(model_dir, weight.checkpoint).touch()
            modules = {"segment_anything": fake_segment_anything(), "numpy": fake_numpy()}
            with patch.dict(sys.modules, modules):
                preparer = ModelPreparer(model_dir, backends=None)
                preparer.prepare(weight.id)
                embedding = preparer.embed(b"same-image-12", [2, 2, 3])
                mask, shape = preparer.predict_embedding(embedding["embedding_id"], [[0, 0]], [0])
                predictor = preparer.backends["sam1"].embeddings[embedding["embedding_id"]]["predictor"]

        self.assertEqual(mask, b"best")
        self.assertEqual(shape, [2, 2])
        self.assertEqual(len(predictor.images), 1)
        self.assertEqual(len(predictor.calls), 1)
        self.assertEqual(predictor.calls[0][0].data, [[0, 0]])
        self.assertEqual(predictor.calls[0][1].data, [0])

    def test_sam1_save_load_embeddings_folder(self):
        weight = MODEL_DEFINITIONS[0].weights[0]
        with tempfile.TemporaryDirectory() as model_dir:
            Path(model_dir, weight.checkpoint).touch()
            modules = {"segment_anything": fake_segment_anything(), "numpy": fake_numpy(), "torch": fake_torch()}
            with patch.dict(sys.modules, modules):
                preparer = ModelPreparer(model_dir, backends=None)
                preparer.prepare(weight.id)
                embedding = preparer.embed(b"same-image-12", [2, 2, 3])
                path = Path(model_dir, "embeddings")
                result = preparer.save_embeddings([embedding_item(embedding["embedding_id"])], path)
                data_file_exists = Path(path, "embeddings.pt").exists()
                metadata = json.loads(Path(path, "metadata.json").read_text(encoding="utf-8"))
                loaded = preparer.load_embeddings(path)

        self.assertEqual(result["path"], str(path))
        self.assertEqual(result["count"], 1)
        self.assertTrue(data_file_exists)
        self.assertEqual(metadata["format"], "samm-embeddings-folder")
        self.assertEqual(metadata["weight_id"], weight.id)
        self.assertEqual(metadata["data_file"], "embeddings.pt")
        self.assertEqual(metadata["items"][0]["slice_spec"], embedding_item("unused")["slice_spec"])
        self.assertEqual(loaded["count"], 1)
        self.assertEqual(loaded["items"][0]["image_digest"], "00")

    def test_offload_clears_prepared_model(self):
        weight = MODEL_DEFINITIONS[0].weights[0]
        backend = FakeBackend()
        with tempfile.TemporaryDirectory() as model_dir:
            Path(model_dir, weight.checkpoint).touch()
            preparer = ModelPreparer(model_dir, backends={"sam1": backend})
            preparer.prepare(weight.id)
            result = preparer.offload()

        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.payload, {"status": "offloaded"})
        self.assertTrue(backend.offloaded)
        self.assertIsNone(preparer.prepared_weight_id)
        self.assertIsNone(preparer.prepared_weight)
        self.assertIsNone(preparer.prepared_payload())

    def test_predict_uses_prepared_backend(self):
        weight = MODEL_DEFINITIONS[0].weights[0]
        backend = FakeBackend()
        with tempfile.TemporaryDirectory() as model_dir:
            Path(model_dir, weight.checkpoint).touch()
            preparer = ModelPreparer(model_dir, backends={"sam1": backend})
            preparer.prepare(weight.id)
            mask = preparer.predict(b"image", [1, 2, 3], [[4, 5]], [1])

        self.assertEqual(mask, b"mask")
        self.assertEqual(backend.calls, [(weight, b"image", [1, 2, 3], [[4, 5]], [1], None, None, None)])

    def test_propagate_video_uses_prepared_backend(self):
        weight = MODEL_DEFINITIONS[0].weights[0]
        backend = FakeBackend()
        frames = [{"frame_index": 0, "image_bytes": b"rgb", "shape": [1, 1, 3]}]
        prompt = {"frame_index": 0, "points": [[0, 0]], "labels": [1]}
        cancel_event = object()
        with tempfile.TemporaryDirectory() as model_dir:
            Path(model_dir, weight.checkpoint).touch()
            preparer = ModelPreparer(model_dir, backends={"sam1": backend})
            preparer.prepare(weight.id)
            results = list(preparer.propagate_video(
                frames,
                prompt,
                direction="forward",
                cancel_event=cancel_event,
                offload_video_to_cpu=False,
                offload_state_to_cpu=True,
                expected_weight_id=weight.id,
            ))

        self.assertEqual(results, [{"frame_index": 0, "shape": [1, 1], "data": b"\x01"}])
        self.assertEqual(backend.video_calls, [(
            weight,
            frames,
            prompt,
            "forward",
            cancel_event,
            False,
            True,
        )])

    def test_propagate_video_rejects_stale_or_missing_prepared_model(self):
        preparer = ModelPreparer(".", backends={})
        with self.assertRaisesRegex(BackendError, "model not prepared"):
            preparer.propagate_video([], {})

        weight = MODEL_DEFINITIONS[0].weights[0]
        backend = FakeBackend()
        with tempfile.TemporaryDirectory() as model_dir:
            Path(model_dir, weight.checkpoint).touch()
            preparer = ModelPreparer(model_dir, backends={"sam1": backend})
            preparer.prepare(weight.id)
            with self.assertRaisesRegex(BackendError, "prepared model changed"):
                preparer.propagate_video([], {}, expected_weight_id="another-weight")

        self.assertEqual(backend.video_calls, [])

    def test_propagate_video_rechecks_pinned_weight_when_iteration_starts(self):
        weight = MODEL_DEFINITIONS[0].weights[0]
        backend = FakeBackend()
        with tempfile.TemporaryDirectory() as model_dir:
            Path(model_dir, weight.checkpoint).touch()
            preparer = ModelPreparer(model_dir, backends={"sam1": backend})
            preparer.prepare(weight.id)
            results = preparer.propagate_video(
                [{"frame_index": 0, "image_bytes": b"rgb", "shape": [1, 1, 3]}],
                {"frame_index": 0, "points": [[0, 0]], "labels": [1]},
                expected_weight_id=weight.id,
            )
            preparer.prepared_weight_id = "another-weight"

            with self.assertRaisesRegex(BackendError, "prepared model changed"):
                list(results)

        self.assertEqual(backend.video_calls, [])

    def test_prepare_waits_for_running_video_generator(self):
        class BlockingVideoBackend(FakeBackend):
            def __init__(self):
                super().__init__()
                self.video_started = Event()
                self.release_video = Event()

            def propagate_video(self, weight, frames, prompt, *args, **kwargs):
                self.video_calls.append((weight, frames, prompt, args, kwargs))

                def generate():
                    self.video_started.set()
                    if not self.release_video.wait(1):
                        raise RuntimeError("timed out waiting to release video test")
                    yield {"frame_index": 0, "shape": [1, 1], "data": b"\x01"}

                return generate()

        first, second = MODEL_DEFINITIONS[0].weights[:2]
        backend = BlockingVideoBackend()
        prepare_done = Event()
        prepare_results = []
        video_errors = []

        with tempfile.TemporaryDirectory() as model_dir:
            Path(model_dir, first.checkpoint).touch()
            Path(model_dir, second.checkpoint).touch()
            preparer = ModelPreparer(model_dir, backends={"sam1": backend})
            preparer.prepare(first.id)

            def run_video():
                try:
                    list(preparer.propagate_video(
                        [{"frame_index": 0, "image_bytes": b"rgb", "shape": [1, 1, 3]}],
                        {"frame_index": 0, "points": [[0, 0]], "labels": [1]},
                        expected_weight_id=first.id,
                    ))
                except Exception as exc:
                    video_errors.append(exc)

            video_thread = Thread(target=run_video)
            video_thread.start()
            self.assertTrue(backend.video_started.wait(1))

            def prepare_second():
                prepare_results.append(preparer.prepare(second.id))
                prepare_done.set()

            prepare_thread = Thread(target=prepare_second)
            prepare_thread.start()
            self.assertFalse(prepare_done.wait(0.05))
            backend.release_video.set()
            video_thread.join(1)
            prepare_thread.join(1)

            self.assertFalse(video_thread.is_alive())
            self.assertFalse(prepare_thread.is_alive())
            self.assertEqual(video_errors, [])
            self.assertTrue(prepare_done.is_set())
            self.assertEqual(prepare_results[0].status_code, 200)
            self.assertEqual(preparer.prepared_weight_id, second.id)

    def test_embed_uses_prepared_backend(self):
        weight = MODEL_DEFINITIONS[0].weights[0]
        backend = FakeBackend()
        with tempfile.TemporaryDirectory() as model_dir:
            Path(model_dir, weight.checkpoint).touch()
            preparer = ModelPreparer(model_dir, backends={"sam1": backend})
            preparer.prepare(weight.id)
            embedding = preparer.embed(b"image", [1, 2, 3])

        self.assertEqual(embedding, {"embedding_id": "emb-1", "shape": [1, 2]})
        self.assertEqual(backend.embed_calls, [(weight, b"image", [1, 2, 3])])

    def test_save_embeddings_uses_prepared_backend(self):
        weight = MODEL_DEFINITIONS[0].weights[0]
        backend = FakeBackend()
        items = [{"embedding_id": "emb-1"}]
        with tempfile.TemporaryDirectory() as model_dir:
            Path(model_dir, weight.checkpoint).touch()
            preparer = ModelPreparer(model_dir, backends={"sam1": backend})
            preparer.prepare(weight.id)
            path = Path(model_dir, "embeddings")
            result = preparer.save_embeddings(items, path)

        self.assertEqual(result, {"path": str(path), "count": 1})
        self.assertEqual(backend.save_calls, [(weight, items, path)])

    def test_load_embeddings_uses_prepared_backend(self):
        weight = MODEL_DEFINITIONS[0].weights[0]
        backend = FakeBackend()
        with tempfile.TemporaryDirectory() as model_dir:
            Path(model_dir, weight.checkpoint).touch()
            preparer = ModelPreparer(model_dir, backends={"sam1": backend})
            preparer.prepare(weight.id)
            path = Path(model_dir, "embeddings")
            result = preparer.load_embeddings(path)

        self.assertEqual(result, {"path": str(path), "count": 1, "items": [{"embedding_id": "emb-1"}]})
        self.assertEqual(backend.load_calls, [(weight, path)])

    def test_predict_embedding_uses_prepared_backend(self):
        weight = MODEL_DEFINITIONS[0].weights[0]
        backend = FakeBackend()
        with tempfile.TemporaryDirectory() as model_dir:
            Path(model_dir, weight.checkpoint).touch()
            preparer = ModelPreparer(model_dir, backends={"sam1": backend})
            preparer.prepare(weight.id)
            mask, shape = preparer.predict_embedding("emb-1", [[4, 5]], [1])

        self.assertEqual(mask, b"mask")
        self.assertEqual(shape, [2, 3])
        self.assertEqual(backend.embedding_calls, [(weight, "emb-1", [[4, 5]], [1], None, None, None)])

    def test_missing_sam_dependency_returns_500(self):
        weight = MODEL_DEFINITIONS[0].weights[0]
        real_import = builtins.__import__

        def import_without_sam(name, *args, **kwargs):
            if name == "segment_anything":
                raise ModuleNotFoundError(name)
            return real_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as model_dir:
            Path(model_dir, weight.checkpoint).touch()
            with patch("builtins.__import__", import_without_sam):
                result = ModelPreparer(model_dir).prepare(weight.id)

        self.assertEqual(result.status_code, 500)
        self.assertEqual(result.payload["error"], "segment-anything is not installed")

    def test_unsupported_backend_returns_501(self):
        model = MODEL_DEFINITIONS[3]
        weight = model.weights[0]
        with tempfile.TemporaryDirectory() as model_dir:
            Path(model_dir, weight.checkpoint).touch()
            result = ModelPreparer(model_dir, backends={"sam1": FakeBackend()}).prepare(weight.id)

        self.assertEqual(result.status_code, 501)
        self.assertEqual(result.payload["error"], "backend unsupported")
        self.assertEqual(result.payload["backend"], weight.backend)


def embedding_item(embedding_id):
    return {
        "embedding_id": embedding_id,
        "slice_spec": {"view": "Red", "axis": 0, "index": 1},
        "image_shape": [2, 2, 3],
        "image_digest": "00",
    }


if __name__ == "__main__":
    unittest.main()
