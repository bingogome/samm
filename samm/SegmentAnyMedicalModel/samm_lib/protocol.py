SERVICE_NAME = "SAMM service"
PROTOCOL_VERSION = "1.2.0"
GATEWAY_HOST = "127.0.0.1"
GATEWAY_PORT = 8799
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
HEALTH_TIMEOUT_SECONDS = 2
REQUEST_TIMEOUT_SECONDS = 600
MODEL_IDS = ("sam1", "sam2", "mobile_sam", "medsam", "medsam2", "sam3", "medical_sam3", "fastsam")
WEIGHT_IDS = (
    "sam_vit_b",
    "sam_vit_l",
    "sam_vit_h",
    "sam2_1_hiera_tiny",
    "sam2_1_hiera_small",
    "sam2_1_hiera_base_plus",
    "sam2_1_hiera_large",
    "mobile_sam_vit_t",
    "medsam_vit_b",
    "medsam_text_flare22",
    "medsam2_latest",
    "medsam2_2411",
    "medsam2_ct_lesion",
    "medsam2_mri_liver_lesion",
    "medsam2_us_heart",
    "sam3",
    "medical_sam3",
    "fastsam_x",
    "fastsam_s",
)
SLICE_VIEWS = ("Red", "Green", "Yellow")
