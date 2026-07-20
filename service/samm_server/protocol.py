from .model_registry import model_payloads, weight_payload_for_id


SERVICE_NAME = "SAMM service"
SERVICE_VERSION = "1.2.0"
PROTOCOL_VERSION = "1.2.0"
HEALTH_PATH = "/health"
MODELS_PATH = "/models"
SESSION_PATH = "/session"
EMBEDDINGS_PATH = "/embeddings"
EMBEDDING_JOBS_PATH = "/embedding-jobs"
EMBEDDING_FILES_LOAD_PATH = "/embedding-files/load"
EMBEDDING_FILES_SAVE_PATH = "/embedding-files/save"
OFFLOAD_PATH = "/offload"
PREPARED_PATH = "/prepared"
PREDICT_PATH = "/predict"
PREDICTION_JOBS_PATH = "/prediction-jobs"
VIDEO_PREDICTION_JOBS_PATH = "/video-prediction-jobs"
FINETUNING_DATASETS_PATH = "/finetuning/datasets"
FINETUNING_REPORTS_PATH = "/finetuning/reports"
FINETUNING_TRAIN_PATH = "/finetuning/train"
FINETUNING_EVAL_PATH = "/finetuning/eval"
FINETUNING_JOBS_PATH = "/finetuning/jobs"


def health_payload():
    return {
        "name": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "status": "ready",
        "protocol_version": PROTOCOL_VERSION,
    }


def models_payload(model_dir):
    return {"models": model_payloads(model_dir)}


def weight_payload(model_dir, weight_id):
    return weight_payload_for_id(weight_id, model_dir)


def prepared_payload(prepared):
    return {"prepared": prepared}


def offload_payload():
    return {"status": "offloaded"}
