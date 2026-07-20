from base64 import b64encode
from dataclasses import dataclass
from queue import Queue
from threading import Lock, Thread
from uuid import uuid4

from .request_payload import image_payload, prompt_payload


@dataclass(frozen=True)
class PredictionJobResult:
    status_code: int
    payload: dict


class PredictionJobService:
    def __init__(self, model_preparer):
        self.model_preparer = model_preparer
        self.jobs = {}
        self.lock = Lock()

    def start(self, payload):
        if not self.model_preparer.prepared_payload():
            return PredictionJobResult(409, {"error": "model not prepared"})
        if not isinstance(payload, dict):
            return PredictionJobResult(400, {"error": "payload must be an object"})
        total = payload.get("total")
        if not positive_int(total):
            return PredictionJobResult(400, {"error": "total must be a positive integer"})
        job = self.create_job(total)
        return PredictionJobResult(202, self.public(job))

    def add_items(self, job_id, payload):
        if not isinstance(payload, dict):
            return PredictionJobResult(400, {"error": "payload must be an object"})
        items = payload.get("items")
        if not valid_items(items):
            return PredictionJobResult(400, {"error": "items must be a non-empty list"})

        result = self.add_job_items(job_id, items)
        if result:
            return result
        state = self.state(job_id)
        if state.payload["submitted"] == state.payload["total"]:
            self.close_job(job_id)
            state = self.state(job_id)
        return state

    def create_job(self, total):
        job_id = uuid4().hex
        job = {
            "job_id": job_id,
            "status": "queued",
            "total": total,
            "submitted": 0,
            "completed": 0,
            "results": [],
            "error": None,
            "queue": Queue(),
            "closed": False,
        }
        with self.lock:
            self.jobs[job_id] = job
        Thread(target=self.run, args=(job_id,), daemon=True).start()
        return job

    def state(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return PredictionJobResult(404, {"error": "prediction job not found", "job_id": job_id})
            return PredictionJobResult(200, self.public(job))

    def add_job_items(self, job_id, items):
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return PredictionJobResult(404, {"error": "prediction job not found", "job_id": job_id})
            if job["status"] in ("complete", "failed") or job["closed"]:
                return PredictionJobResult(409, {"error": "prediction job is closed", "job_id": job_id})
            if job["submitted"] + len(items) > job["total"]:
                return PredictionJobResult(400, {"error": "too many prediction items", "job_id": job_id})
            job["submitted"] += len(items)
            queue = job["queue"]
        for item in items:
            queue.put(item)
        return None

    def close_job(self, job_id):
        with self.lock:
            job = self.jobs[job_id]
            job["closed"] = True
            queue = job["queue"]
        queue.put(None)

    def run(self, job_id):
        try:
            while True:
                item = self.next_item(job_id)
                if item is None:
                    break
                self.update(job_id, status="running")
                error, image_bytes, shape = image_payload(item)
                if error:
                    raise ValueError(error["error"])
                error, points, labels, box, mask_prompt, text = prompt_payload(item)
                if error:
                    raise ValueError(error["error"])
                if mask_prompt and mask_prompt["shape"] != shape[:2]:
                    raise ValueError("mask.shape must match image shape")
                mask = self.model_preparer.predict(image_bytes, shape, points, labels, box, mask_prompt, text)
                self.append_result(job_id, {
                    "key": item["key"],
                    "slice_spec": item["slice_spec"],
                    "shape": [shape[0], shape[1]],
                    "data": b64encode(mask).decode("ascii"),
                })
            self.update(job_id, status="complete")
        except Exception as exc:
            self.update(job_id, status="failed", error=str(exc))

    def next_item(self, job_id):
        with self.lock:
            queue = self.jobs[job_id]["queue"]
        return queue.get()

    def append_result(self, job_id, result):
        with self.lock:
            job = self.jobs[job_id]
            job["results"].append(result)
            job["completed"] = len(job["results"])

    def update(self, job_id, **values):
        with self.lock:
            self.jobs[job_id].update(values)

    def public(self, job):
        return {
            "job_id": job["job_id"],
            "status": job["status"],
            "total": job["total"],
            "submitted": job["submitted"],
            "completed": job["completed"],
            "results": list(job["results"]),
            "error": job["error"],
        }


def valid_items(items):
    return isinstance(items, list) and bool(items) and all(valid_item(item) for item in items)


def valid_item(item):
    return (
        isinstance(item, dict)
        and isinstance(item.get("key"), str)
        and bool(item["key"])
        and isinstance(item.get("slice_spec"), dict)
        and isinstance(item.get("image"), dict)
    )


def positive_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0
