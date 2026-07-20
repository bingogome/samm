FINETUNE_BASE_MODELS = {
    "SAM2.1 Hiera Tiny": "checkpoints/sam2.1_hiera_tiny.pt",
}

FINETUNE_BASE_MODEL_LABELS = tuple(FINETUNE_BASE_MODELS)


def finetune_base_checkpoint(label):
    return FINETUNE_BASE_MODELS[label]
