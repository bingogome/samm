from argparse import Namespace
from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
from threading import Lock, Thread
from uuid import uuid4

from .model_registry import project_root_for_model_dir, resolve_model_dir
from .runtime_registry import pixi_program


CODE_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PROGRESS_RE = re.compile(r"Train Epoch: \[(\d+)\]\[(\d+)/(\d+)\]")


@dataclass(frozen=True)
class FinetuningResult:
    status_code: int
    payload: dict


class FinetuningService:
    def __init__(self, model_dir, pixi=None):
        self.model_dir = resolve_model_dir(model_dir)
        self.root = project_root_for_model_dir(self.model_dir)
        self.pixi = pixi or pixi_program()
        self.jobs = {}
        self.lock = Lock()

    def dataset_status(self, name):
        error = name_error(name, "dataset")
        if error:
            return FinetuningResult(400, error)
        return FinetuningResult(200, dataset_status(self.root, name))

    def datasets(self):
        return FinetuningResult(200, {"datasets": [dataset_status(self.root, name) for name in dataset_names(self.root)]})

    def build_dataset(self, name, payload):
        error = name_error(name, "dataset")
        if error:
            return FinetuningResult(400, error)
        if not isinstance(payload, dict):
            return FinetuningResult(400, {"error": "payload must be an object"})
        status = dataset_status(self.root, name)
        if status["source_segmented_volumes"] < 1:
            return FinetuningResult(400, {"error": f"No source segmented volumes found in {status['source']}"})

        try:
            args = export_args(status, payload)
            output = capture_output(lambda: export_dataset(args))
        except ValueError as exc:
            return FinetuningResult(400, {"error": str(exc)})
        except FileNotFoundError as exc:
            return FinetuningResult(404, {"error": str(exc)})
        return FinetuningResult(200, {
            "status": dataset_status(self.root, name),
            "output": output.strip(),
            "command": export_command(args),
        })

    def report(self, run):
        error = name_error(run, "run")
        if error:
            return FinetuningResult(400, error)
        run_path = self.root / "finetuning_runs" / run
        if not run_path.is_dir():
            return FinetuningResult(404, {"error": f"finetuning run not found: {run_path}", "run": str(run_path)})
        output = capture_output(lambda: report_run(run_path))
        return FinetuningResult(200, {"run": str(run_path), "output": output.strip()})

    def start_train(self, payload):
        if not isinstance(payload, dict):
            return FinetuningResult(400, {"error": "payload must be an object"})
        try:
            command, run = train_command(self.root, self.pixi, payload)
        except ValueError as exc:
            return FinetuningResult(400, {"error": str(exc)})
        return self.start_job("train", command, self.log_path("train", run))

    def start_eval(self, payload):
        if not isinstance(payload, dict):
            return FinetuningResult(400, {"error": "payload must be an object"})
        try:
            command, run = eval_command(self.root, self.pixi, payload)
        except ValueError as exc:
            return FinetuningResult(400, {"error": str(exc)})
        return self.start_job("eval", command, self.log_path("eval", run))

    def job(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return FinetuningResult(404, {"error": "finetuning job not found", "job_id": job_id})
            return FinetuningResult(200, self.public(job))

    def cancel(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return FinetuningResult(404, {"error": "finetuning job not found", "job_id": job_id})
            if job["status"] not in ("queued", "running"):
                return FinetuningResult(200, self.public(job))
            job["cancel_requested"] = True
            process = job["process"]
        if process and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
        return self.job(job_id)

    def has_running_jobs(self):
        with self.lock:
            return any(job["status"] in ("queued", "running") for job in self.jobs.values())

    def stop_all(self):
        with self.lock:
            jobs = list(self.jobs.values())
        for job in jobs:
            if job["status"] in ("queued", "running"):
                self.cancel(job["job_id"])

    def start_job(self, kind, command, log_path):
        job_id = uuid4().hex
        log_path.parent.mkdir(parents=True, exist_ok=True)
        job = {
            "job_id": job_id,
            "kind": kind,
            "status": "queued",
            "command": command,
            "log": str(log_path),
            "returncode": None,
            "error": None,
            "process": None,
            "cancel_requested": False,
        }
        with self.lock:
            self.jobs[job_id] = job
        Thread(target=self.run_job, args=(job_id,), daemon=True).start()
        return FinetuningResult(202, self.public(job))

    def run_job(self, job_id):
        with self.lock:
            job = self.jobs[job_id]
            command = job["command"]
            log_path = Path(job["log"])
            job["status"] = "running"
        try:
            with log_path.open("wb") as output:
                process = subprocess.Popen(
                    command,
                    cwd=self.root,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    env=job_environment(),
                    start_new_session=True,
                )
                with self.lock:
                    self.jobs[job_id]["process"] = process
                returncode = process.wait()
            with self.lock:
                job = self.jobs[job_id]
                job["returncode"] = returncode
                job["status"] = "canceled" if job["cancel_requested"] else ("complete" if returncode == 0 else "failed")
                job["error"] = None if returncode == 0 or job["cancel_requested"] else f"command exited with {returncode}"
        except Exception as exc:
            with self.lock:
                job = self.jobs[job_id]
                job["status"] = "failed"
                job["error"] = str(exc)

    def log_path(self, kind, run):
        return self.root / "logs" / "finetuning_jobs" / f"{run}_{kind}.log"

    def public(self, job):
        payload = {key: job[key] for key in ("job_id", "kind", "status", "command", "log", "returncode", "error")}
        payload["log_tail"] = log_tail(job["log"])
        payload["progress"] = job_progress(job, payload["log_tail"])
        return payload


def dataset_status(root, name):
    source = root / "segmented_volumes" / name
    dataset = root / "datasets" / name
    return {
        "name": name,
        "source": str(source),
        "dataset": str(dataset),
        "source_segmented_volumes": npz_count(source),
        "train_segmented_volumes": npz_count(dataset / "train_npz"),
        "val_segmented_volumes": npz_count(dataset / "val_npz"),
    }


def dataset_names(root):
    names = set()
    for folder in (root / "segmented_volumes", root / "datasets"):
        if folder.is_dir():
            names.update(path.name for path in folder.iterdir() if path.is_dir())
    return sorted(name for name in names if name_error(name, "dataset") is None)


def export_args(status, payload):
    return Namespace(
        input=Path(status["source"]),
        image=None,
        mask=None,
        output=Path(status["dataset"]),
        name=None,
        split="train",
        val_count=int(payload.get("val_count") or 0),
        val_fraction=0.0,
        seed=int(payload.get("seed") or 0),
        axes=axes(payload),
        window=window(payload),
        percentile_window=(0.5, 99.5),
    )


def axes(payload):
    values = payload.get("axes") or [0]
    if not isinstance(values, list) or not values:
        raise ValueError("axes must be a non-empty list")
    return [int(value) for value in values]


def window(payload):
    values = payload.get("window")
    if values is None:
        return None
    if not isinstance(values, list) or len(values) != 2:
        raise ValueError("window must be [min, max]")
    low, high = float(values[0]), float(values[1])
    if high <= low:
        raise ValueError("window max must be greater than min")
    return [low, high]


def export_command(args):
    command = [
        "pixi",
        "run",
        "export-finetune-dataset",
        "--input",
        str(args.input),
        "--output",
        str(args.output),
        "--axes",
        *[str(axis) for axis in args.axes],
    ]
    if args.val_count:
        command.extend(["--val-count", str(args.val_count)])
    if args.window:
        command.extend(["--window", str(args.window[0]), str(args.window[1])])
    return command


def train_command(root, pixi, payload):
    dataset = required_name(payload.get("dataset"), "dataset")
    run = required_name(payload.get("run") or f"{dataset}_v1", "run")
    command = [
        pixi,
        "run",
        "-e",
        "medsam2",
        "finetune-medsam2",
        "--dataset",
        str(root / "datasets" / dataset),
        "--name",
        run,
        "--checkpoint",
        str(checkpoint_path(root, payload.get("checkpoint") or "checkpoints/sam2.1_hiera_tiny.pt")),
        "--epochs",
        str(positive_int(payload.get("epochs") or 25, "epochs")),
        "--batch-size",
        str(positive_int(payload.get("batch_size") or 1, "batch_size")),
        "--num-workers",
        str(non_negative_int(payload.get("num_workers") or 0, "num_workers")),
        "--num-frames",
        str(positive_int(payload.get("num_frames") or 4, "num_frames")),
    ]
    if payload.get("max_objects"):
        command.extend(["--max-objects", str(positive_int(payload["max_objects"], "max_objects"))])
    return command, run


def eval_command(root, pixi, payload):
    dataset = required_name(payload.get("dataset"), "dataset")
    run = required_name(payload.get("run") or f"{dataset}_v1", "run")
    prompts = payload.get("prompts") or ["box", "point", "box-point"]
    if not isinstance(prompts, list) or any(prompt not in ("box", "point", "box-point") for prompt in prompts):
        raise ValueError("prompts must contain box, point, or box-point")
    command = [
        pixi,
        "run",
        "-e",
        "medsam2",
        "eval-finetuned",
        "--dataset",
        str(root / "datasets" / dataset),
        "--checkpoint",
        str(checkpoint_path(root, payload.get("checkpoint") or f"finetuning_runs/{run}/checkpoints/checkpoint.pt")),
        "--output",
        str(root / "finetuning_runs" / run / "eval.json"),
        "--prompt",
        *prompts,
    ]
    if payload.get("max_segmented_volumes"):
        command.extend(["--max-segmented-volumes", str(positive_int(payload["max_segmented_volumes"], "max_segmented_volumes"))])
    return command, run


def checkpoint_path(root, value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def required_name(value, label):
    if value is None:
        raise ValueError(f"{label} is required")
    value = str(value)
    error = name_error(value, label)
    if error:
        raise ValueError(error["error"])
    return value


def positive_int(value, label):
    value = int(value)
    if value <= 0:
        raise ValueError(f"{label} must be positive")
    return value


def non_negative_int(value, label):
    value = int(value)
    if value < 0:
        raise ValueError(f"{label} must be non-negative")
    return value


def job_environment():
    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    return env


def log_tail(path, limit=6000):
    path = Path(path)
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-limit:]


def job_progress(job, text):
    if job["kind"] == "train":
        return train_progress(job, text)
    if job["status"] in ("queued", "running"):
        return {"mode": "busy", "label": f"{job['kind']} {job['status']}"}
    current = 1 if job["status"] == "complete" else 0
    return {"mode": "determinate", "current": current, "total": 1, "label": job["status"]}


def train_progress(job, text):
    epochs = command_int(job["command"], "--epochs", 1)
    total = epochs * 1000
    current, label = train_log_progress(text, epochs)
    if job["status"] == "complete":
        current, label = total, f"{epochs} / {epochs} epochs"
    return {"mode": "determinate", "current": current, "total": total, "label": label}


def train_log_progress(text, epochs):
    matches = TRAIN_PROGRESS_RE.findall(text)
    if not matches:
        return 0, f"0 / {epochs} epochs"
    epoch, batch, batches = (int(value) for value in matches[-1])
    batches = max(batches, 1)
    batch = min(batch + 1, batches)
    epoch_number = min(epoch + 1, epochs)
    current = min(int((epoch + batch / batches) * 1000), epochs * 1000)
    return current, f"{epoch_number} / {epochs} epochs, batch {batch} / {batches}"


def command_int(command, option, default):
    return int(command[command.index(option) + 1]) if option in command else default


def capture_output(callback):
    output = StringIO()
    with redirect_stdout(output):
        callback()
    return output.getvalue()


def export_dataset(args):
    add_code_root()
    from finetuning.dataset import export_dataset as run
    return run(args)


def report_run(path):
    add_code_root()
    from finetuning.report import report
    return report(Namespace(run=path))


def add_code_root():
    if str(CODE_ROOT) not in sys.path:
        sys.path.insert(0, str(CODE_ROOT))


def npz_count(path):
    return len(list(path.glob("*.npz"))) if path.is_dir() else 0


def name_error(value, label):
    if not isinstance(value, str) or not value:
        return {"error": f"{label} name must be a non-empty string"}
    if clean_part(value) != value:
        return {"error": f"{label} name must use letters, numbers, dots, dashes, or underscores"}
    return None


def clean_part(value):
    return "".join(char if char.isalnum() or char in ".-_" else "_" for char in value).strip("._-")
