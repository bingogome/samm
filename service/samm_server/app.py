import argparse
import os
from threading import Thread
import time
from http.server import ThreadingHTTPServer
from urllib.parse import urlparse

from .http_handler import JsonHandler
from .model_registry import resolve_model_dir, weight_definition, weight_payload as registry_weight_payload
from .finetuning import FinetuningService
from .protocol import (
    HEALTH_PATH,
    EMBEDDINGS_PATH,
    EMBEDDING_FILES_LOAD_PATH,
    EMBEDDING_FILES_SAVE_PATH,
    EMBEDDING_JOBS_PATH,
    FINETUNING_EVAL_PATH,
    FINETUNING_DATASETS_PATH,
    FINETUNING_TRAIN_PATH,
    MODELS_PATH,
    OFFLOAD_PATH,
    PREDICT_PATH,
    PREDICTION_JOBS_PATH,
    PREPARED_PATH,
    SESSION_PATH,
    VIDEO_PREDICTION_JOBS_PATH,
    health_payload,
    models_payload,
    offload_payload,
    prepared_payload,
    weight_payload,
)
from .session import SessionTracker
from .routes import (
    embedding_job_id_from_path,
    embedding_job_items_id_from_path,
    finetuning_dataset_build_from_path,
    finetuning_dataset_from_path,
    finetuning_job_cancel_from_path,
    finetuning_job_from_path,
    finetuning_report_from_path,
    prediction_job_id_from_path,
    prediction_job_items_id_from_path,
    prepare_weight_id_from_path,
    video_prediction_job_cancel_id_from_path,
    video_prediction_job_frames_id_from_path,
    video_prediction_job_id_from_path,
    video_prediction_job_run_id_from_path,
    weight_id_from_path,
)
from .worker_runtime import WorkerProxy, WorkerRuntime


class GatewayHandler(JsonHandler):
    def do_GET(self):
        path = urlparse(self.path).path
        if path == HEALTH_PATH:
            self.send_json(200, health_payload())
            return

        if path == MODELS_PATH:
            self.send_json(200, models_payload(self.server.model_dir))
            return

        weight_id = weight_id_from_path(path)
        if weight_id:
            payload = weight_payload(self.server.model_dir, weight_id)
            if payload:
                self.send_json(200, payload)
            else:
                self.send_json(404, {"error": "weight not found", "weight_id": weight_id})
            return

        if path == PREPARED_PATH:
            if self.server.worker_runtime.backend:
                self.proxy("GET")
            else:
                self.send_json(200, prepared_payload(None))
            return

        if path == FINETUNING_DATASETS_PATH:
            self.send_result(self.server.finetuning.datasets())
            return

        dataset_name = finetuning_dataset_from_path(path)
        if dataset_name:
            self.send_result(self.server.finetuning.dataset_status(dataset_name))
            return

        report_name = finetuning_report_from_path(path)
        if report_name:
            self.send_result(self.server.finetuning.report(report_name))
            return

        job_id = finetuning_job_from_path(path)
        if job_id:
            self.send_result(self.server.finetuning.job(job_id))
            return

        if embedding_job_id_from_path(path):
            if self.server.worker_runtime.backend:
                self.proxy("GET")
            else:
                self.send_json(409, {"error": "model not prepared"})
            return
        if prediction_job_id_from_path(path):
            if self.server.worker_runtime.backend:
                self.proxy("GET")
            else:
                self.send_json(409, {"error": "model not prepared"})
            return
        if video_prediction_job_id_from_path(path):
            if self.server.worker_runtime.backend:
                self.proxy("GET")
            else:
                self.send_json(409, {"error": "model not prepared"})
            return

        self.send_json(404, {"error": "not found"})

    def do_POST(self):
        path = urlparse(self.path).path
        weight_id = prepare_weight_id_from_path(path)
        if weight_id:
            self.prepare_weight(weight_id)
            return
        if path == OFFLOAD_PATH:
            self.offload_model()
            return
        if path == SESSION_PATH:
            result = self.server.session_tracker.touch(self.read_json())
            self.send_json(result.status_code, result.payload)
            return
        dataset_name = finetuning_dataset_build_from_path(path)
        if dataset_name:
            self.run_finetuning(lambda: self.server.finetuning.build_dataset(dataset_name, self.read_json()))
            return
        if path == FINETUNING_TRAIN_PATH:
            self.send_result(self.server.finetuning.start_train(self.read_json()))
            return
        if path == FINETUNING_EVAL_PATH:
            self.send_result(self.server.finetuning.start_eval(self.read_json()))
            return
        job_id = finetuning_job_cancel_from_path(path)
        if job_id:
            self.send_result(self.server.finetuning.cancel(job_id))
            return
        if path == EMBEDDINGS_PATH:
            if self.server.worker_runtime.backend:
                self.proxy("POST", self.read_json())
            else:
                self.send_json(409, {"error": "model not prepared"})
            return
        if path in (EMBEDDING_FILES_LOAD_PATH, EMBEDDING_FILES_SAVE_PATH):
            if self.server.worker_runtime.backend:
                self.proxy("POST", self.read_json())
            else:
                self.send_json(409, {"error": "model not prepared"})
            return
        if path == EMBEDDING_JOBS_PATH:
            if self.server.worker_runtime.backend:
                self.proxy("POST", self.read_json())
            else:
                self.send_json(409, {"error": "model not prepared"})
            return
        if embedding_job_items_id_from_path(path):
            if self.server.worker_runtime.backend:
                self.proxy("POST", self.read_json())
            else:
                self.send_json(409, {"error": "model not prepared"})
            return
        if path == PREDICTION_JOBS_PATH:
            if self.server.worker_runtime.backend:
                self.proxy("POST", self.read_json())
            else:
                self.send_json(409, {"error": "model not prepared"})
            return
        if prediction_job_items_id_from_path(path):
            if self.server.worker_runtime.backend:
                self.proxy("POST", self.read_json())
            else:
                self.send_json(409, {"error": "model not prepared"})
            return
        if path == VIDEO_PREDICTION_JOBS_PATH:
            if self.server.worker_runtime.backend:
                self.proxy("POST", self.read_json())
            else:
                self.send_json(409, {"error": "model not prepared"})
            return
        if (
            video_prediction_job_frames_id_from_path(path)
            or video_prediction_job_run_id_from_path(path)
            or video_prediction_job_cancel_id_from_path(path)
        ):
            if self.server.worker_runtime.backend:
                self.proxy("POST", self.read_json())
            else:
                self.send_json(409, {"error": "model not prepared"})
            return
        if path == PREDICT_PATH:
            if self.server.worker_runtime.backend:
                self.proxy("POST", self.read_json())
            else:
                self.send_json(409, {"error": "model not prepared"})
            return
        self.send_json(404, {"error": "not found"})

    def send_result(self, result):
        self.send_json(result.status_code, result.payload)

    def run_finetuning(self, callback):
        try:
            self.send_result(callback())
        except ValueError as exc:
            self.send_json(400, {"error": str(exc)})
        except FileNotFoundError as exc:
            self.send_json(404, {"error": str(exc)})

    def prepare_weight(self, weight_id):
        model, weight = weight_definition(weight_id, self.server.model_dir)
        if not weight:
            self.send_json(404, {"error": "weight not found", "weight_id": weight_id})
            return

        payload = registry_weight_payload(model, weight, self.server.model_dir)
        if not payload["available"]:
            self.send_json(
                409,
                {
                    "error": "checkpoint missing",
                    "model_id": model.id,
                    "weight_id": weight.id,
                    "checkpoint": weight.checkpoint,
                },
            )
            return

        try:
            self.server.worker_runtime.use_backend(weight.backend)
        except KeyError:
            self.send_json(
                501,
                {
                    "error": "backend unsupported",
                    "model_id": model.id,
                    "weight_id": weight.id,
                    "backend": weight.backend,
                },
            )
            return
        except (ConnectionError, TimeoutError, RuntimeError) as exc:
            self.send_json(502, {"error": str(exc)})
            return

        self.proxy("POST")

    def offload_model(self):
        self.server.worker_runtime.offload()
        self.send_json(200, offload_payload())

    def proxy(self, method, payload=None):
        try:
            result = self.server.worker_proxy.request(method, self.path, payload)
        except (ConnectionError, TimeoutError, RuntimeError) as exc:
            self.send_json(502, {"error": str(exc)})
            return
        self.send_json(result.status_code, result.payload)


def serve(
    host="127.0.0.1",
    port=8799,
    worker_host="127.0.0.1",
    worker_port=8801,
    model_dir=".",
    device="cpu",
    idle_timeout_seconds=600,
    idle_check_seconds=10,
):
    model_dir = resolve_model_dir(model_dir)
    runtime = WorkerRuntime(worker_host, worker_port, model_dir, device)
    server = ThreadingHTTPServer((host, port), GatewayHandler)
    server.model_dir = model_dir
    server.finetuning = FinetuningService(model_dir)
    server.worker_runtime = runtime
    server.worker_proxy = WorkerProxy(runtime)
    server.session_tracker = SessionTracker(idle_timeout_seconds)
    start_idle_monitor(server, idle_check_seconds)
    print(f"SAMM gateway listening on http://{host}:{port}", flush=True)
    print(f"SAMM worker host: http://{worker_host}:{worker_port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.finetuning.stop_all()
        runtime.stop()


def start_idle_monitor(server, check_seconds):
    thread = Thread(target=idle_monitor, args=(server, check_seconds), daemon=True)
    thread.start()


def idle_monitor(server, check_seconds):
    while True:
        time.sleep(check_seconds)
        if server.session_tracker.expired():
            if server.finetuning.has_running_jobs():
                continue
            print(
                f"SAMM gateway idle timeout after {server.session_tracker.timeout_seconds} seconds",
                flush=True,
            )
            autosaves = server.session_tracker.autosaves()
            if autosaves:
                print(f"SAMM latest autosaves: {autosaves}", flush=True)
            server.shutdown()
            return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8799)
    parser.add_argument("--worker-host", default="127.0.0.1")
    parser.add_argument("--worker-port", type=int, default=8801)
    parser.add_argument("--model-dir", default=os.environ.get("SAMM_MODEL_DIR", "."))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--idle-timeout-seconds", type=float, default=600)
    parser.add_argument("--idle-check-seconds", type=float, default=10)
    args = parser.parse_args()

    try:
        serve(
            args.host,
            args.port,
            args.worker_host,
            args.worker_port,
            args.model_dir,
            args.device,
            args.idle_timeout_seconds,
            args.idle_check_seconds,
        )
    except KeyboardInterrupt:
        print("SAMM gateway stopped", flush=True)


if __name__ == "__main__":
    main()
