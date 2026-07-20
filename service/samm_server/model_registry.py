import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WeightDefinition:
    id: str
    label: str
    checkpoint: str
    backend: str
    model_type: str
    capabilities: dict


@dataclass(frozen=True)
class ModelDefinition:
    id: str
    label: str
    weights: tuple[WeightDefinition, ...]


SAM_PROMPTS = {
    "points": True,
    "box": True,
    "box_3d": True,
    "mask": True,
    "text": False,
    "auto_predict_2d": True,
    "embeddings": True,
    "video_propagation": False,
}

SAM2_PROMPTS = {
    "points": True,
    "box": True,
    "box_3d": True,
    "mask": True,
    "text": False,
    "auto_predict_2d": True,
    "embeddings": True,
    "video_propagation": True,
}

MEDSAM_PROMPTS = {
    "points": False,
    "box": True,
    "box_3d": True,
    "mask": False,
    "text": False,
    "auto_predict_2d": True,
    "embeddings": True,
    "video_propagation": False,
}

MEDSAM_TEXT_PROMPTS = {
    "points": False,
    "box": False,
    "box_3d": False,
    "mask": False,
    "text": True,
    "auto_predict_2d": True,
    "embeddings": True,
    "video_propagation": False,
}

MEDSAM2_PROMPTS = SAM2_PROMPTS
MEDSAM2_CONFIG = "configs/sam2.1_hiera_t512.yaml"
FINETUNED_RUNS_DIR = "finetuning_runs"
FINETUNED_MODEL_FILE = "samm_model.json"

SAM3_PROMPTS = {
    "points": True,
    "box": True,
    "box_3d": True,
    "mask": True,
    "text": True,
    "auto_predict_2d": True,
    "embeddings": True,
    "video_propagation": True,
    "video_mask": False,
    "video_text": True,
}

MEDICAL_SAM3_PROMPTS = {
    "points": False,
    "box": True,
    "box_3d": True,
    "mask": False,
    "text": True,
    "auto_predict_2d": True,
    "embeddings": True,
    "video_propagation": False,
}

FASTSAM_PROMPTS = {
    "points": True,
    "box": True,
    "box_3d": True,
    "mask": False,
    "text": True,
    "auto_predict_2d": True,
    "embeddings": False,
    "video_propagation": False,
}


MODEL_DEFINITIONS = (
    ModelDefinition(
        "sam1",
        "SAM 1",
        (
            WeightDefinition("sam_vit_b", "ViT-B", "sam_vit_b_01ec64.pth", "sam1", "vit_b", SAM_PROMPTS),
            WeightDefinition("sam_vit_l", "ViT-L", "sam_vit_l_0b3195.pth", "sam1", "vit_l", SAM_PROMPTS),
            WeightDefinition("sam_vit_h", "ViT-H", "sam_vit_h_4b8939.pth", "sam1", "vit_h", SAM_PROMPTS),
        ),
    ),
    ModelDefinition(
        "sam2",
        "SAM 2",
        (
            WeightDefinition(
                "sam2_1_hiera_tiny",
                "Hiera Tiny",
                "sam2.1_hiera_tiny.pt",
                "sam2",
                "configs/sam2.1/sam2.1_hiera_t.yaml",
                SAM2_PROMPTS,
            ),
            WeightDefinition(
                "sam2_1_hiera_small",
                "Hiera Small",
                "sam2.1_hiera_small.pt",
                "sam2",
                "configs/sam2.1/sam2.1_hiera_s.yaml",
                SAM2_PROMPTS,
            ),
            WeightDefinition(
                "sam2_1_hiera_base_plus",
                "Hiera Base+",
                "sam2.1_hiera_base_plus.pt",
                "sam2",
                "configs/sam2.1/sam2.1_hiera_b+.yaml",
                SAM2_PROMPTS,
            ),
            WeightDefinition(
                "sam2_1_hiera_large",
                "Hiera Large",
                "sam2.1_hiera_large.pt",
                "sam2",
                "configs/sam2.1/sam2.1_hiera_l.yaml",
                SAM2_PROMPTS,
            ),
        ),
    ),
    ModelDefinition(
        "mobile_sam",
        "MobileSAM",
        (WeightDefinition("mobile_sam_vit_t", "ViT-T", "mobile_sam.pt", "mobile_sam", "vit_t", SAM_PROMPTS),),
    ),
    ModelDefinition(
        "medsam",
        "MedSAM",
        (
            WeightDefinition("medsam_vit_b", "ViT-B", "medsam_vit_b.pth", "medsam", "vit_b", MEDSAM_PROMPTS),
            WeightDefinition(
                "medsam_text_flare22",
                "Text FLARE22",
                "medsam_text_prompt_flare22.pth",
                "medsam_text",
                "vit_b",
                MEDSAM_TEXT_PROMPTS,
            ),
        ),
    ),
    ModelDefinition(
        "medsam2",
        "MedSAM2",
        (
            WeightDefinition("medsam2_latest", "Latest", "MedSAM2_latest.pt", "medsam2", MEDSAM2_CONFIG, MEDSAM2_PROMPTS),
            WeightDefinition("medsam2_2411", "Nov 2024", "MedSAM2_2411.pt", "medsam2", MEDSAM2_CONFIG, MEDSAM2_PROMPTS),
            WeightDefinition(
                "medsam2_ct_lesion",
                "CT Lesion",
                "MedSAM2_CTLesion.pt",
                "medsam2",
                MEDSAM2_CONFIG,
                MEDSAM2_PROMPTS,
            ),
            WeightDefinition(
                "medsam2_mri_liver_lesion",
                "MRI Liver Lesion",
                "MedSAM2_MRI_LiverLesion.pt",
                "medsam2",
                MEDSAM2_CONFIG,
                MEDSAM2_PROMPTS,
            ),
            WeightDefinition(
                "medsam2_us_heart",
                "US Heart",
                "MedSAM2_US_Heart.pt",
                "medsam2",
                MEDSAM2_CONFIG,
                MEDSAM2_PROMPTS,
            ),
        ),
    ),
    ModelDefinition(
        "sam3",
        "SAM 3",
        (WeightDefinition("sam3", "SAM 3", "sam3.pt", "sam3", "sam3", SAM3_PROMPTS),),
    ),
    ModelDefinition(
        "medical_sam3",
        "Medical-SAM3",
        (
            WeightDefinition(
                "medical_sam3",
                "Medical-SAM3",
                "medical_sam3.pt",
                "medical_sam3",
                "medical_sam3",
                MEDICAL_SAM3_PROMPTS,
            ),
        ),
    ),
    ModelDefinition(
        "fastsam",
        "FastSAM",
        (
            WeightDefinition("fastsam_x", "FastSAM-x", "FastSAM-x.pt", "fastsam", "fastsam_x", FASTSAM_PROMPTS),
            WeightDefinition("fastsam_s", "FastSAM-s", "FastSAM-s.pt", "fastsam", "fastsam_s", FASTSAM_PROMPTS),
        ),
    ),
)


def resolve_model_dir(model_dir):
    path = Path(model_dir).expanduser().resolve()
    if not path.is_dir():
        raise NotADirectoryError(f"Model directory not found: {path}")
    return path


def model_payloads(model_dir):
    root = resolve_model_dir(model_dir)
    return [model_payload(model, root) for model in model_definitions(root)]


def weight_payload_for_id(weight_id, model_dir):
    root = resolve_model_dir(model_dir)
    model, weight = weight_definition(weight_id, root)
    return weight_payload(model, weight, root) if weight else None


def weight_definition(weight_id, model_dir=None):
    models = model_definitions(model_dir) if model_dir else MODEL_DEFINITIONS
    for model in models:
        for weight in model.weights:
            if weight.id == weight_id:
                return model, weight
    return None, None


def model_definitions(model_dir):
    root = resolve_model_dir(model_dir)
    weights = finetuned_weights(root)
    if not weights:
        return MODEL_DEFINITIONS
    return tuple(model_with_finetuned_weights(model, weights.get(model.id, ())) for model in MODEL_DEFINITIONS)


def model_with_finetuned_weights(model, weights):
    return ModelDefinition(model.id, model.label, model.weights + tuple(weights))


def finetuned_weights(root):
    project_root = project_root_for_model_dir(root)
    runs_root = project_root / FINETUNED_RUNS_DIR
    if not runs_root.is_dir():
        return {}

    seen = {weight.id for model in MODEL_DEFINITIONS for weight in model.weights}
    weights = {}
    for path in sorted(runs_root.glob(f"*/{FINETUNED_MODEL_FILE}")):
        model_id, weight = finetuned_weight(project_root, path)
        if weight.id in seen:
            raise ValueError(f"duplicate weight id: {weight.id}")
        seen.add(weight.id)
        weights.setdefault(model_id, []).append(weight)
    return {key: tuple(value) for key, value in weights.items()}


def finetuned_weight(project_root, path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    missing = [key for key in ("id", "label", "model_id", "backend", "checkpoint", "model_type") if key not in payload]
    if missing:
        raise ValueError(f"{path} missing keys: {', '.join(missing)}")
    if payload["model_id"] != "medsam2" or payload["backend"] != "medsam2":
        raise ValueError(f"{path} must register a MedSAM2 weight")
    checkpoint = metadata_path(project_root, payload["checkpoint"])
    return payload["model_id"], WeightDefinition(
        payload["id"],
        payload["label"],
        str(checkpoint),
        payload["backend"],
        payload["model_type"],
        MEDSAM2_PROMPTS,
    )


def project_root_for_model_dir(root):
    return root.parent if root.name == "checkpoints" else root


def metadata_path(project_root, value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def model_payload(model, root):
    return {
        "id": model.id,
        "label": model.label,
        "capabilities": model_capabilities(model),
        "weights": [weight_payload(model, weight, root) for weight in model.weights],
    }


def weight_payload(model, weight, root):
    return {
        "id": weight.id,
        "model_id": model.id,
        "label": weight.label,
        "checkpoint": weight.checkpoint,
        "backend": weight.backend,
        "model_type": weight.model_type,
        "available": checkpoint_path(weight, root).is_file(),
        "capabilities": dict(weight.capabilities),
    }


def checkpoint_path(weight, model_dir):
    path = Path(weight.checkpoint).expanduser()
    return path.resolve() if path.is_absolute() else resolve_model_dir(model_dir) / path


def model_capabilities(model):
    keys = {key for weight in model.weights for key in weight.capabilities}
    return {key: any(weight.capabilities.get(key, False) for weight in model.weights) for key in sorted(keys)}
