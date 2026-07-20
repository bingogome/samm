from urllib.parse import unquote

from .protocol import (
    EMBEDDINGS_PATH,
    EMBEDDING_FILES_LOAD_PATH,
    EMBEDDING_FILES_SAVE_PATH,
    EMBEDDING_JOBS_PATH,
    FINETUNING_DATASETS_PATH,
    FINETUNING_JOBS_PATH,
    FINETUNING_REPORTS_PATH,
    MODELS_PATH,
    OFFLOAD_PATH,
    PREDICT_PATH,
    PREDICTION_JOBS_PATH,
    PREPARED_PATH,
    VIDEO_PREDICTION_JOBS_PATH,
)


def worker_get_path(path):
    return (
        path == PREPARED_PATH
        or bool(embedding_job_id_from_path(path))
        or bool(prediction_job_id_from_path(path))
        or bool(video_prediction_job_id_from_path(path))
    )


def worker_post_path(path):
    paths = (
        EMBEDDINGS_PATH,
        EMBEDDING_FILES_LOAD_PATH,
        EMBEDDING_FILES_SAVE_PATH,
        EMBEDDING_JOBS_PATH,
        OFFLOAD_PATH,
        PREDICT_PATH,
        PREDICTION_JOBS_PATH,
        VIDEO_PREDICTION_JOBS_PATH,
    )
    return (
        path in paths
        or bool(embedding_job_items_id_from_path(path))
        or bool(prediction_job_items_id_from_path(path))
        or bool(video_prediction_job_frames_id_from_path(path))
        or bool(video_prediction_job_run_id_from_path(path))
        or bool(video_prediction_job_cancel_id_from_path(path))
    )


def weight_id_from_path(path):
    prefix = f"{MODELS_PATH}/"
    if not path.startswith(prefix):
        return None
    weight_id = unquote(path[len(prefix):])
    return weight_id if weight_id and "/" not in weight_id else None


def prepare_weight_id_from_path(path):
    prefix = f"{MODELS_PATH}/"
    suffix = "/prepare"
    if not path.startswith(prefix) or not path.endswith(suffix):
        return None
    weight_id = unquote(path[len(prefix):-len(suffix)])
    return weight_id if weight_id and "/" not in weight_id else None


def embedding_job_id_from_path(path):
    prefix = f"{EMBEDDING_JOBS_PATH}/"
    if not path.startswith(prefix):
        return None
    job_id = unquote(path[len(prefix):])
    return job_id if job_id and "/" not in job_id else None


def embedding_job_items_id_from_path(path):
    prefix = f"{EMBEDDING_JOBS_PATH}/"
    suffix = "/items"
    if not path.startswith(prefix) or not path.endswith(suffix):
        return None
    job_id = unquote(path[len(prefix):-len(suffix)])
    return job_id if job_id and "/" not in job_id else None


def prediction_job_id_from_path(path):
    prefix = f"{PREDICTION_JOBS_PATH}/"
    if not path.startswith(prefix):
        return None
    job_id = unquote(path[len(prefix):])
    return job_id if job_id and "/" not in job_id else None


def prediction_job_items_id_from_path(path):
    prefix = f"{PREDICTION_JOBS_PATH}/"
    suffix = "/items"
    if not path.startswith(prefix) or not path.endswith(suffix):
        return None
    job_id = unquote(path[len(prefix):-len(suffix)])
    return job_id if job_id and "/" not in job_id else None


def video_prediction_job_id_from_path(path):
    prefix = f"{VIDEO_PREDICTION_JOBS_PATH}/"
    if not path.startswith(prefix):
        return None
    job_id = unquote(path[len(prefix):])
    return job_id if job_id and "/" not in job_id else None


def video_prediction_job_frames_id_from_path(path):
    return video_prediction_job_action_id_from_path(path, "frames")


def video_prediction_job_run_id_from_path(path):
    return video_prediction_job_action_id_from_path(path, "run")


def video_prediction_job_cancel_id_from_path(path):
    return video_prediction_job_action_id_from_path(path, "cancel")


def video_prediction_job_action_id_from_path(path, action):
    prefix = f"{VIDEO_PREDICTION_JOBS_PATH}/"
    suffix = f"/{action}"
    if not path.startswith(prefix) or not path.endswith(suffix):
        return None
    job_id = unquote(path[len(prefix):-len(suffix)])
    return job_id if job_id and "/" not in job_id else None


def finetuning_dataset_from_path(path):
    prefix = f"{FINETUNING_DATASETS_PATH}/"
    if not path.startswith(prefix):
        return None
    name = unquote(path[len(prefix):])
    return name if name and "/" not in name else None


def finetuning_dataset_build_from_path(path):
    prefix = f"{FINETUNING_DATASETS_PATH}/"
    suffix = "/build"
    if not path.startswith(prefix) or not path.endswith(suffix):
        return None
    name = unquote(path[len(prefix):-len(suffix)])
    return name if name and "/" not in name else None


def finetuning_report_from_path(path):
    prefix = f"{FINETUNING_REPORTS_PATH}/"
    if not path.startswith(prefix):
        return None
    run = unquote(path[len(prefix):])
    return run if run and "/" not in run else None


def finetuning_job_from_path(path):
    prefix = f"{FINETUNING_JOBS_PATH}/"
    if not path.startswith(prefix):
        return None
    job_id = unquote(path[len(prefix):])
    return job_id if job_id and "/" not in job_id else None


def finetuning_job_cancel_from_path(path):
    prefix = f"{FINETUNING_JOBS_PATH}/"
    suffix = "/cancel"
    if not path.startswith(prefix) or not path.endswith(suffix):
        return None
    job_id = unquote(path[len(prefix):-len(suffix)])
    return job_id if job_id and "/" not in job_id else None
