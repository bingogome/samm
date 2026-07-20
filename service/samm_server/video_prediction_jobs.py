from base64 import b64encode
from copy import deepcopy
from dataclasses import dataclass
from threading import Event, Lock, Thread
from time import monotonic
from uuid import uuid4

from .request_payload import image_payload, is_box, is_int, mask_prompt, valid_prompts


TERMINAL_STATUSES = frozenset(("complete", "failed", "cancelled"))
VALID_DIRECTIONS = frozenset(("both", "forward", "backward"))
DEFAULT_MAX_JOBS = 4
DEFAULT_MAX_FRAMES = 4096
DEFAULT_MAX_INPUT_BYTES = 1024 * 1024 * 1024
DEFAULT_TERMINAL_RETENTION_SECONDS = 600
DEFAULT_UPLOAD_RETENTION_SECONDS = 600


@dataclass(frozen=True)
class VideoPredictionJobResult:
    status_code: int
    payload: dict


class VideoPredictionJobService:
    """Owns uploaded frame sequences and their stateful video predictions."""

    def __init__(
        self,
        model_preparer,
        max_jobs=DEFAULT_MAX_JOBS,
        max_frames=DEFAULT_MAX_FRAMES,
        max_input_bytes=DEFAULT_MAX_INPUT_BYTES,
        terminal_retention_seconds=DEFAULT_TERMINAL_RETENTION_SECONDS,
        upload_retention_seconds=DEFAULT_UPLOAD_RETENTION_SECONDS,
        clock=monotonic,
    ):
        self.model_preparer = model_preparer
        self.max_jobs = max_jobs
        self.max_frames = max_frames
        self.max_input_bytes = max_input_bytes
        self.terminal_retention_seconds = terminal_retention_seconds
        self.upload_retention_seconds = upload_retention_seconds
        self.clock = clock
        self.jobs = {}
        self.lock = Lock()

    def start(self, payload):
        prepared = self.model_preparer.prepared_payload()
        if not prepared:
            return VideoPredictionJobResult(409, {"error": "model not prepared"})
        if not isinstance(payload, dict):
            return VideoPredictionJobResult(400, {"error": "payload must be an object"})

        total = payload.get("total")
        if not positive_int(total):
            return VideoPredictionJobResult(400, {"error": "total must be a positive integer"})
        if total > self.max_frames:
            return VideoPredictionJobResult(
                400,
                {"error": "total exceeds the video frame limit", "total": total, "max_frames": self.max_frames},
            )

        weight_id = prepared.get("weight_id")
        if not isinstance(weight_id, str) or not weight_id:
            return VideoPredictionJobResult(409, {"error": "prepared model has no weight_id"})
        capabilities = prepared.get("capabilities")
        if not isinstance(capabilities, dict) or capabilities.get("video_propagation") is not True:
            return VideoPredictionJobResult(
                409,
                {
                    "error": "prepared model does not support video propagation",
                    "weight_id": weight_id,
                },
            )

        now = self.clock()
        job = {
            "job_id": uuid4().hex,
            "status": "uploading",
            "weight_id": weight_id,
            "capabilities": deepcopy(capabilities),
            "total": total,
            "frames": {},
            "frame_shape": None,
            "completed": 0,
            "results": [],
            "result_frame_indices": set(),
            "error": None,
            "cancel_event": Event(),
            "run_payload": None,
            "input_bytes": 0,
            "created_at": now,
            "updated_at": now,
        }
        with self.lock:
            self.prune_locked(make_room=True)
            if len(self.jobs) >= self.max_jobs:
                return VideoPredictionJobResult(
                    429,
                    {"error": "too many video prediction jobs", "max_jobs": self.max_jobs},
                )
            self.jobs[job["job_id"]] = job
            public = self.public(job, 0)
        return VideoPredictionJobResult(202, public)

    def add_frames(self, job_id, payload):
        if not isinstance(payload, dict):
            return VideoPredictionJobResult(400, {"error": "payload must be an object"})
        frames = payload.get("frames")
        if not isinstance(frames, list) or not frames:
            return VideoPredictionJobResult(400, {"error": "frames must be a non-empty list"})

        parsed = []
        indices = set()
        batch_shape = None
        batch_bytes = 0
        for item in frames:
            error, frame = parse_frame(item)
            if error:
                return VideoPredictionJobResult(400, error)
            frame_index = frame["frame_index"]
            if frame_index in indices:
                return VideoPredictionJobResult(400, {"error": "frame_index must be unique", "frame_index": frame_index})
            indices.add(frame_index)
            if batch_shape is None:
                batch_shape = frame["shape"]
            elif frame["shape"] != batch_shape:
                return VideoPredictionJobResult(400, {"error": "all video frames must have equal image.shape"})
            parsed.append(frame)
            batch_bytes += len(frame["image_bytes"])

        with self.lock:
            self.prune_locked()
            job = self.jobs.get(job_id)
            if not job:
                return self.not_found(job_id)
            if job["status"] != "uploading":
                return VideoPredictionJobResult(409, self.closed_payload(job))
            if len(job["frames"]) + len(parsed) > job["total"]:
                return VideoPredictionJobResult(400, {"error": "too many video frames", "job_id": job_id})
            if job["input_bytes"] + batch_bytes > self.max_input_bytes:
                return VideoPredictionJobResult(
                    413,
                    {
                        "error": "video frame data exceeds the input byte limit",
                        "input_bytes": job["input_bytes"] + batch_bytes,
                        "max_input_bytes": self.max_input_bytes,
                    },
                )
            for frame in parsed:
                frame_index = frame["frame_index"]
                if frame_index < 0 or frame_index >= job["total"]:
                    return VideoPredictionJobResult(
                        400,
                        {
                            "error": "frame_index must be within the declared total",
                            "frame_index": frame_index,
                            "total": job["total"],
                        },
                    )
                if frame_index in job["frames"]:
                    return VideoPredictionJobResult(
                        409,
                        {"error": "frame_index already submitted", "frame_index": frame_index, "job_id": job_id},
                    )
            if job["frame_shape"] is not None and batch_shape != job["frame_shape"]:
                return VideoPredictionJobResult(400, {"error": "all video frames must have equal image.shape"})

            if job["frame_shape"] is None:
                job["frame_shape"] = list(batch_shape)
            for frame in parsed:
                job["frames"][frame["frame_index"]] = frame
            job["input_bytes"] += batch_bytes
            self.touch(job)
            return VideoPredictionJobResult(200, self.public(job, 0))

    def run(self, job_id, payload):
        if not isinstance(payload, dict):
            return VideoPredictionJobResult(400, {"error": "payload must be an object"})

        error, run_payload = parse_run_payload(payload)
        if error:
            return VideoPredictionJobResult(400, error)

        prepared = self.model_preparer.prepared_payload()
        current_weight_id = prepared.get("weight_id") if isinstance(prepared, dict) else None
        with self.lock:
            self.prune_locked()
            job = self.jobs.get(job_id)
            if not job:
                return self.not_found(job_id)
            if job["status"] != "uploading":
                return VideoPredictionJobResult(409, self.closed_payload(job))
            if len(job["frames"]) != job["total"]:
                return VideoPredictionJobResult(
                    409,
                    {
                        "error": "all video frames must be submitted before run",
                        "job_id": job_id,
                        "submitted": len(job["frames"]),
                        "total": job["total"],
                    },
                )
            if current_weight_id != job["weight_id"]:
                job["status"] = "failed"
                job["error"] = "prepared model changed before video prediction run"
                self.release_inputs(job)
                self.touch(job)
                payload = self.public(job, 0)
                payload["expected_weight_id"] = job["weight_id"]
                payload["actual_weight_id"] = current_weight_id
                return VideoPredictionJobResult(409, payload)

            prompt_frame_index = run_payload["prompt"]["frame_index"]
            if prompt_frame_index not in job["frames"]:
                return VideoPredictionJobResult(
                    400,
                    {"error": "prompt.frame_index was not submitted", "frame_index": prompt_frame_index},
                )
            mask = run_payload["prompt"]["mask"]
            if mask is not None and mask["shape"] != job["frame_shape"][:2]:
                return VideoPredictionJobResult(400, {"error": "prompt.mask.shape must match video frame shape"})
            capability_error = video_prompt_capability_error(
                run_payload["prompt"],
                job["capabilities"],
            )
            if capability_error:
                return VideoPredictionJobResult(400, {"error": capability_error})

            job["run_payload"] = run_payload
            job["status"] = "queued"
            self.touch(job)
            public = self.public(job, 0)

        Thread(target=self._execute, args=(job_id,), daemon=True).start()
        return VideoPredictionJobResult(202, public)

    def state(self, job_id, cursor=0):
        if not nonnegative_int(cursor):
            return VideoPredictionJobResult(400, {"error": "cursor must be a non-negative integer"})
        with self.lock:
            self.prune_locked()
            job = self.jobs.get(job_id)
            if not job:
                return self.not_found(job_id)
            if cursor > len(job["results"]):
                return VideoPredictionJobResult(
                    400,
                    {
                        "error": "cursor exceeds available video prediction results",
                        "cursor": cursor,
                        "next_cursor": len(job["results"]),
                    },
                )
            return VideoPredictionJobResult(200, self.public(job, cursor))

    def cancel(self, job_id):
        with self.lock:
            self.prune_locked()
            job = self.jobs.get(job_id)
            if not job:
                return self.not_found(job_id)
            if job["status"] in ("complete", "failed"):
                payload = self.public(job, 0)
                payload["error"] = "video prediction job is already terminal"
                return VideoPredictionJobResult(
                    409,
                    payload,
                )
            if job["status"] != "cancelled":
                job["cancel_event"].set()
                job["status"] = "cancelled"
                self.release_inputs(job)
                self.touch(job)
            return VideoPredictionJobResult(200, self.public(job, 0))

    def _execute(self, job_id):
        with self.lock:
            job = self.jobs[job_id]
            if job["status"] != "queued" or job["cancel_event"].is_set():
                return
            job["status"] = "running"
            self.touch(job)
            frames = [deepcopy(job["frames"][index]) for index in sorted(job["frames"])]
            run_payload = deepcopy(job["run_payload"])
            self.release_inputs(job)
            cancel_event = job["cancel_event"]
            expected_weight_id = job["weight_id"]

        try:
            results = self.model_preparer.propagate_video(
                frames,
                run_payload["prompt"],
                direction=run_payload["direction"],
                cancel_event=cancel_event,
                offload_video_to_cpu=run_payload["offload_video_to_cpu"],
                offload_state_to_cpu=run_payload["offload_state_to_cpu"],
                expected_weight_id=expected_weight_id,
            )
            for result in results:
                if cancel_event.is_set():
                    break
                self._append_result(job_id, result)
            with self.lock:
                job = self.jobs[job_id]
                if cancel_event.is_set():
                    job["status"] = "cancelled"
                elif job["status"] != "cancelled":
                    job["status"] = "complete"
                self.release_inputs(job)
                self.touch(job)
        except Exception as exc:
            with self.lock:
                job = self.jobs[job_id]
                if cancel_event.is_set():
                    job["status"] = "cancelled"
                elif job["status"] != "cancelled":
                    job["status"] = "failed"
                    job["error"] = str(exc)
                self.release_inputs(job)
                self.touch(job)

    def _append_result(self, job_id, result):
        if not isinstance(result, dict):
            raise ValueError("video prediction result must be an object")
        frame_index = result.get("frame_index")
        shape = result.get("shape")
        data = result.get("data")
        if not nonnegative_int(frame_index):
            raise ValueError("video prediction result frame_index must be a non-negative integer")
        if not (
            isinstance(shape, (list, tuple))
            and len(shape) == 2
            and all(positive_int(value) for value in shape)
        ):
            raise ValueError("video prediction result shape must be [height, width]")
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise ValueError("video prediction result data must be bytes")
        data = bytes(data)
        if len(data) != shape[0] * shape[1]:
            raise ValueError("video prediction result data length does not match shape")

        with self.lock:
            job = self.jobs[job_id]
            frame = job["frames"].get(frame_index)
            if frame is None:
                raise ValueError("video prediction result frame_index was not submitted")
            if list(shape) != frame["shape"][:2]:
                raise ValueError("video prediction result shape does not match video frame shape")
            if frame_index in job["result_frame_indices"]:
                raise ValueError("video prediction returned a duplicate frame_index")
            if job["cancel_event"].is_set() or job["status"] == "cancelled":
                return

            sequence = len(job["results"])
            job["results"].append(
                {
                    "sequence": sequence,
                    "frame_index": frame_index,
                    "key": frame["key"],
                    "slice_spec": deepcopy(frame["slice_spec"]),
                    "shape": list(shape),
                    "data": b64encode(data).decode("ascii"),
                }
            )
            job["result_frame_indices"].add(frame_index)
            job["completed"] = len(job["results"])
            self.touch(job)

    def public(self, job, cursor):
        return {
            "job_id": job["job_id"],
            "status": job["status"],
            "weight_id": job["weight_id"],
            "total": job["total"],
            "submitted": len(job["frames"]),
            "completed": job["completed"],
            "results": deepcopy(job["results"][cursor:]),
            "next_cursor": len(job["results"]),
            "error": job["error"],
        }

    @staticmethod
    def release_inputs(job):
        for frame in job["frames"].values():
            frame.pop("image_bytes", None)
        job["run_payload"] = None
        job["input_bytes"] = 0

    def touch(self, job):
        job["updated_at"] = self.clock()

    def prune_locked(self, make_room=False):
        now = self.clock()
        expired = [
            job_id
            for job_id, job in self.jobs.items()
            if (
                job["status"] in TERMINAL_STATUSES
                and now - job["updated_at"] >= self.terminal_retention_seconds
            )
            or (
                job["status"] == "uploading"
                and now - job["updated_at"] >= self.upload_retention_seconds
            )
        ]
        for job_id in expired:
            del self.jobs[job_id]
        if not make_room or len(self.jobs) < self.max_jobs:
            return
        terminal = sorted(
            (
                (job["updated_at"], job_id)
                for job_id, job in self.jobs.items()
                if job["status"] in TERMINAL_STATUSES
            )
        )
        for _updated_at, job_id in terminal:
            if len(self.jobs) < self.max_jobs:
                break
            del self.jobs[job_id]

    @staticmethod
    def not_found(job_id):
        return VideoPredictionJobResult(404, {"error": "video prediction job not found", "job_id": job_id})

    @staticmethod
    def closed_payload(job):
        return {
            "error": "video prediction job is not accepting this operation",
            "job_id": job["job_id"],
            "status": job["status"],
        }


def parse_frame(item):
    if not isinstance(item, dict):
        return {"error": "video frame must be an object"}, None
    frame_index = item.get("frame_index")
    if not nonnegative_int(frame_index):
        return {"error": "frame_index must be a non-negative integer"}, None
    key = item.get("key")
    if not isinstance(key, str) or not key:
        return {"error": "key must be a non-empty string"}, None
    slice_spec = item.get("slice_spec")
    if not isinstance(slice_spec, dict):
        return {"error": "slice_spec must be an object"}, None
    error, image_bytes, shape = image_payload(item)
    if error:
        return error, None
    return None, {
        "frame_index": frame_index,
        "key": key,
        "slice_spec": deepcopy(slice_spec),
        "image_bytes": image_bytes,
        "shape": list(shape),
    }


def parse_run_payload(payload):
    prompt = payload.get("prompt")
    if not isinstance(prompt, dict):
        return {"error": "prompt must be an object"}, None
    frame_index = prompt.get("frame_index")
    if not nonnegative_int(frame_index):
        return {"error": "prompt.frame_index must be a non-negative integer"}, None

    points = prompt.get("points", [])
    labels = prompt.get("labels", [])
    if not valid_prompts(points, labels):
        return {"error": "prompt points and labels must be equal length lists of [x, y] and 0/1 labels"}, None
    box = prompt.get("box")
    if box is not None and not is_box(box):
        return {"error": "prompt.box must be [x0, y0, x1, y1]"}, None
    error, mask = mask_prompt(prompt.get("mask"))
    if error:
        error["error"] = f"prompt.{error['error']}"
        return error, None
    text = prompt.get("text")
    if text is not None and (not isinstance(text, str) or not text.strip()):
        return {"error": "prompt.text must be a non-empty string"}, None
    if not points and box is None and mask is None and text is None:
        return {"error": "prompt points/labels, box, mask, or text is required"}, None
    if mask is not None and (points or box is not None or text is not None):
        return {"error": "prompt.mask cannot be combined with points, box, or text"}, None

    direction = payload.get("direction", "both")
    if not isinstance(direction, str) or direction not in VALID_DIRECTIONS:
        return {"error": "direction must be both, forward, or backward"}, None
    offload_video_to_cpu = payload.get("offload_video_to_cpu", True)
    if not isinstance(offload_video_to_cpu, bool):
        return {"error": "offload_video_to_cpu must be a boolean"}, None
    offload_state_to_cpu = payload.get("offload_state_to_cpu", False)
    if not isinstance(offload_state_to_cpu, bool):
        return {"error": "offload_state_to_cpu must be a boolean"}, None

    parsed_prompt = {
        "frame_index": frame_index,
        "points": deepcopy(points),
        "labels": list(labels),
        "box": deepcopy(box),
        "mask": deepcopy(mask),
    }
    if text is not None:
        parsed_prompt["text"] = text.strip()
    return None, {
        "prompt": parsed_prompt,
        "direction": direction,
        "offload_video_to_cpu": offload_video_to_cpu,
        "offload_state_to_cpu": offload_state_to_cpu,
    }


def video_prompt_capability_error(prompt, capabilities):
    if prompt.get("mask") is not None and not capabilities.get(
        "video_mask", capabilities.get("mask", True)
    ):
        return "prepared model does not support mask prompts for video propagation"
    if prompt.get("text") is not None and capabilities.get("video_text") is not True:
        return "prepared model does not support text prompts for video propagation"
    if prompt.get("text") is not None and prompt.get("points"):
        return "video text prompts cannot be combined with points"
    return None


def positive_int(value):
    return is_int(value) and value > 0


def nonnegative_int(value):
    return is_int(value) and value >= 0
