from dataclasses import dataclass
import json
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from .protocol import (
    GATEWAY_HOST,
    GATEWAY_PORT,
    HEALTH_PATH,
    EMBEDDINGS_PATH,
    EMBEDDING_FILES_LOAD_PATH,
    EMBEDDING_FILES_SAVE_PATH,
    EMBEDDING_JOBS_PATH,
    FINETUNING_DATASETS_PATH,
    FINETUNING_EVAL_PATH,
    FINETUNING_JOBS_PATH,
    FINETUNING_REPORTS_PATH,
    FINETUNING_TRAIN_PATH,
    HEALTH_TIMEOUT_SECONDS,
    MODELS_PATH,
    OFFLOAD_PATH,
    PREDICT_PATH,
    PREPARED_PATH,
    PROTOCOL_VERSION,
    PREDICTION_JOBS_PATH,
    REQUEST_TIMEOUT_SECONDS,
    SESSION_PATH,
    SERVICE_NAME,
    VIDEO_PREDICTION_JOBS_PATH,
)


@dataclass(frozen=True)
class ServiceStatus:
    connected: bool
    message: str


class ServiceClient:
    def __init__(self, host: str = GATEWAY_HOST, port: int = GATEWAY_PORT):
        self.host = host
        self.port = port

    @property
    def health_url(self) -> str:
        return f"http://{self.host}:{self.port}{HEALTH_PATH}"

    def url(self, path: str) -> str:
        return f"http://{self.host}:{self.port}{path}"

    def get_json(self, path: str, timeout=REQUEST_TIMEOUT_SECONDS):
        try:
            with urlopen(self.url(path), timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            payload = json.loads(exc.read().decode("utf-8"))
            raise RuntimeError(self.error_message(exc.code, payload)) from None
        except URLError as exc:
            raise ConnectionError(f"Could not connect to {self.url(path)}: {exc.reason}") from None

    def post_json(self, path: str, payload=None, timeout=REQUEST_TIMEOUT_SECONDS):
        data = b"" if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(self.url(path), data=data, method="POST")
        if payload is not None:
            request.add_header("Content-Type", "application/json")
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            payload = json.loads(exc.read().decode("utf-8"))
            raise RuntimeError(self.error_message(exc.code, payload)) from None
        except URLError as exc:
            raise ConnectionError(f"Could not connect to {self.url(path)}: {exc.reason}") from None

    def error_message(self, code, payload):
        message = f"{code}: {payload['error']}"
        if "checkpoint" in payload:
            message += f" ({payload['checkpoint']})"
        return message

    def connect(self) -> ServiceStatus:
        payload = self.get_json(HEALTH_PATH, HEALTH_TIMEOUT_SECONDS)

        if payload["name"] != SERVICE_NAME:
            raise RuntimeError(f"Unexpected service: {payload['name']}")
        if payload["protocol_version"] != PROTOCOL_VERSION:
            raise RuntimeError(f"Unsupported protocol: {payload['protocol_version']}")

        return ServiceStatus(True, f"Connected to {payload['name']} {payload['version']}")

    def list_models(self):
        return self.get_json(MODELS_PATH)["models"]

    def get_weight(self, weight_id: str):
        return self.get_json(f"{MODELS_PATH}/{quote(weight_id, safe='')}")

    def prepare_weight(self, weight_id: str):
        return self.post_json(f"{MODELS_PATH}/{quote(weight_id, safe='')}/prepare")

    def offload_model(self):
        return self.post_json(OFFLOAD_PATH)

    def touch_session(self, session_id, autosave=None):
        return self.post_json(SESSION_PATH, {"session_id": session_id, "autosave": autosave or {}})

    def embed(self, payload):
        return self.post_json(EMBEDDINGS_PATH, payload)

    def start_embedding_job(self, payload):
        return self.post_json(EMBEDDING_JOBS_PATH, payload)

    def add_embedding_job_items(self, job_id, payload):
        return self.post_json(f"{EMBEDDING_JOBS_PATH}/{quote(job_id, safe='')}/items", payload)

    def embedding_job(self, job_id):
        return self.get_json(f"{EMBEDDING_JOBS_PATH}/{quote(job_id, safe='')}")

    def start_prediction_job(self, payload):
        return self.post_json(PREDICTION_JOBS_PATH, payload)

    def add_prediction_job_items(self, job_id, payload):
        return self.post_json(f"{PREDICTION_JOBS_PATH}/{quote(job_id, safe='')}/items", payload)

    def prediction_job(self, job_id):
        return self.get_json(f"{PREDICTION_JOBS_PATH}/{quote(job_id, safe='')}")

    def start_video_prediction_job(self, payload):
        return self.post_json(VIDEO_PREDICTION_JOBS_PATH, payload)

    def add_video_prediction_job_frames(self, job_id, payload):
        job_path = f"{VIDEO_PREDICTION_JOBS_PATH}/{quote(job_id, safe='')}"
        return self.post_json(f"{job_path}/frames", payload)

    def run_video_prediction_job(self, job_id, payload):
        job_path = f"{VIDEO_PREDICTION_JOBS_PATH}/{quote(job_id, safe='')}"
        return self.post_json(f"{job_path}/run", payload)

    def video_prediction_job(self, job_id, cursor=0):
        job_path = f"{VIDEO_PREDICTION_JOBS_PATH}/{quote(job_id, safe='')}"
        return self.get_json(f"{job_path}?{urlencode({'cursor': cursor})}")

    def cancel_video_prediction_job(self, job_id):
        job_path = f"{VIDEO_PREDICTION_JOBS_PATH}/{quote(job_id, safe='')}"
        return self.post_json(f"{job_path}/cancel")

    def save_embeddings(self, payload):
        return self.post_json(EMBEDDING_FILES_SAVE_PATH, payload)

    def load_embeddings(self, payload):
        return self.post_json(EMBEDDING_FILES_LOAD_PATH, payload)

    def prepared(self):
        return self.get_json(PREPARED_PATH)

    def predict(self, payload):
        return self.post_json(PREDICT_PATH, payload)

    def list_finetune_datasets(self):
        return self.get_json(FINETUNING_DATASETS_PATH)["datasets"]

    def finetune_dataset_status(self, name):
        return self.get_json(f"{FINETUNING_DATASETS_PATH}/{quote(name, safe='')}")

    def build_finetune_dataset(self, name, payload):
        return self.post_json(f"{FINETUNING_DATASETS_PATH}/{quote(name, safe='')}/build", payload)

    def finetune_report(self, run):
        return self.get_json(f"{FINETUNING_REPORTS_PATH}/{quote(run, safe='')}")

    def start_finetune_training(self, payload):
        return self.post_json(FINETUNING_TRAIN_PATH, payload)

    def start_finetune_eval(self, payload):
        return self.post_json(FINETUNING_EVAL_PATH, payload)

    def finetune_job(self, job_id):
        return self.get_json(f"{FINETUNING_JOBS_PATH}/{quote(job_id, safe='')}")

    def cancel_finetune_job(self, job_id):
        return self.post_json(f"{FINETUNING_JOBS_PATH}/{quote(job_id, safe='')}/cancel")
