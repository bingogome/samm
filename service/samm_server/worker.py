import argparse
import os
from http.server import ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .http_handler import JsonHandler
from .embedding import EmbeddingService
from .embedding_files import EmbeddingFileService
from .embedding_jobs import EmbeddingJobService
from .model_preparer import ModelPreparer
from .model_registry import resolve_model_dir
from .prediction import PredictionService
from .prediction_jobs import PredictionJobService
from .video_prediction_jobs import VideoPredictionJobService
from .protocol import (
    HEALTH_PATH,
    EMBEDDINGS_PATH,
    EMBEDDING_FILES_LOAD_PATH,
    EMBEDDING_FILES_SAVE_PATH,
    EMBEDDING_JOBS_PATH,
    MODELS_PATH,
    OFFLOAD_PATH,
    PREDICT_PATH,
    PREDICTION_JOBS_PATH,
    PREPARED_PATH,
    VIDEO_PREDICTION_JOBS_PATH,
    health_payload,
    models_payload,
    prepared_payload,
    weight_payload,
)
from .routes import (
    embedding_job_id_from_path,
    embedding_job_items_id_from_path,
    prediction_job_id_from_path,
    prediction_job_items_id_from_path,
    prepare_weight_id_from_path,
    video_prediction_job_cancel_id_from_path,
    video_prediction_job_frames_id_from_path,
    video_prediction_job_id_from_path,
    video_prediction_job_run_id_from_path,
    weight_id_from_path,
)


class WorkerHandler(JsonHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path == HEALTH_PATH:
            self.send_json(200, health_payload())
            return
        if path == MODELS_PATH:
            self.send_json(200, models_payload(self.server.model_dir))
            return
        if path == PREPARED_PATH:
            self.send_json(200, prepared_payload(self.server.model_preparer.prepared_payload()))
            return
        job_id = embedding_job_id_from_path(path)
        if job_id:
            result = self.server.embedding_job_service.state(job_id)
            self.send_json(result.status_code, result.payload)
            return
        job_id = prediction_job_id_from_path(path)
        if job_id:
            result = self.server.prediction_job_service.state(job_id)
            self.send_json(result.status_code, result.payload)
            return
        job_id = video_prediction_job_id_from_path(path)
        if job_id:
            result = self.server.video_prediction_job_service.state(job_id, video_prediction_cursor(parsed.query))
            self.send_json(result.status_code, result.payload)
            return

        weight_id = weight_id_from_path(path)
        if weight_id:
            payload = weight_payload(self.server.model_dir, weight_id)
            if payload:
                self.send_json(200, payload)
            else:
                self.send_json(404, {"error": "weight not found", "weight_id": weight_id})
            return

        self.send_json(404, {"error": "not found"})

    def do_POST(self):
        path = urlparse(self.path).path
        if path == OFFLOAD_PATH:
            result = self.server.model_preparer.offload()
            self.send_json(result.status_code, result.payload)
            return
        if path == EMBEDDINGS_PATH:
            result = self.server.embedding_service.embed(self.read_json())
            self.send_json(result.status_code, result.payload)
            return
        if path == EMBEDDING_FILES_SAVE_PATH:
            result = self.server.embedding_file_service.save(self.read_json())
            self.send_json(result.status_code, result.payload)
            return
        if path == EMBEDDING_FILES_LOAD_PATH:
            result = self.server.embedding_file_service.load(self.read_json())
            self.send_json(result.status_code, result.payload)
            return
        if path == EMBEDDING_JOBS_PATH:
            result = self.server.embedding_job_service.start(self.read_json())
            self.send_json(result.status_code, result.payload)
            return
        job_id = embedding_job_items_id_from_path(path)
        if job_id:
            result = self.server.embedding_job_service.add_items(job_id, self.read_json())
            self.send_json(result.status_code, result.payload)
            return
        if path == PREDICTION_JOBS_PATH:
            result = self.server.prediction_job_service.start(self.read_json())
            self.send_json(result.status_code, result.payload)
            return
        job_id = prediction_job_items_id_from_path(path)
        if job_id:
            result = self.server.prediction_job_service.add_items(job_id, self.read_json())
            self.send_json(result.status_code, result.payload)
            return
        if path == VIDEO_PREDICTION_JOBS_PATH:
            result = self.server.video_prediction_job_service.start(self.read_json())
            self.send_json(result.status_code, result.payload)
            return
        job_id = video_prediction_job_frames_id_from_path(path)
        if job_id:
            result = self.server.video_prediction_job_service.add_frames(job_id, self.read_json())
            self.send_json(result.status_code, result.payload)
            return
        job_id = video_prediction_job_run_id_from_path(path)
        if job_id:
            result = self.server.video_prediction_job_service.run(job_id, self.read_json())
            self.send_json(result.status_code, result.payload)
            return
        job_id = video_prediction_job_cancel_id_from_path(path)
        if job_id:
            result = self.server.video_prediction_job_service.cancel(job_id)
            self.send_json(result.status_code, result.payload)
            return
        if path == PREDICT_PATH:
            result = self.server.prediction_service.predict(self.read_json())
            self.send_json(result.status_code, result.payload)
            return

        weight_id = prepare_weight_id_from_path(path)
        if weight_id:
            result = self.server.model_preparer.prepare(weight_id)
            self.send_json(result.status_code, result.payload)
            return

        self.send_json(404, {"error": "not found"})


def serve(host="127.0.0.1", port=8801, model_dir=".", device="cpu"):
    model_dir = resolve_model_dir(model_dir)
    server = ThreadingHTTPServer((host, port), WorkerHandler)
    server.model_dir = model_dir
    server.model_preparer = ModelPreparer(model_dir, device)
    server.embedding_service = EmbeddingService(server.model_preparer)
    server.embedding_file_service = EmbeddingFileService(server.model_preparer)
    server.embedding_job_service = EmbeddingJobService(server.model_preparer)
    server.prediction_service = PredictionService(server.model_preparer)
    server.prediction_job_service = PredictionJobService(server.model_preparer)
    server.video_prediction_job_service = VideoPredictionJobService(server.model_preparer)
    print(f"SAMM worker listening on http://{host}:{port}", flush=True)
    print(f"SAMM model directory: {model_dir}", flush=True)
    print(f"SAMM device: {device}", flush=True)
    server.serve_forever()


def video_prediction_cursor(query):
    values = parse_qs(query, keep_blank_values=True).get("cursor")
    if values is None:
        return 0
    if len(values) != 1 or not values[0].isdigit():
        return None
    return int(values[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8801)
    parser.add_argument("--model-dir", default=os.environ.get("SAMM_MODEL_DIR", "."))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    try:
        serve(args.host, args.port, args.model_dir, args.device)
    except KeyboardInterrupt:
        print("SAMM worker stopped", flush=True)


if __name__ == "__main__":
    main()
