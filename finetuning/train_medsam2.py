from pathlib import Path
import json
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
MEDSAM2_ROOT = ROOT / "sam_variants" / "MedSAM2"
BASE_CONFIG = MEDSAM2_ROOT / "sam2" / "configs" / "sam2.1_hiera_tiny512_FLARE_RECIST.yaml"
CONFIG_DIR = MEDSAM2_ROOT / "sam2" / "configs" / "samm"
INFERENCE_CONFIG = "configs/sam2.1_hiera_t512.yaml"


def add_train_args(parser):
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "sam2.1_hiera_tiny.pt")
    parser.add_argument("--output-root", type=Path, default=ROOT / "finetuning_runs")
    parser.add_argument("--weight-id")
    parser.add_argument("--label")
    parser.add_argument("--registered-checkpoint", type=Path)
    parser.add_argument("--inference-config", default=INFERENCE_CONFIG)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--max-objects", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")


def train_medsam2(args):
    dataset = dataset_folder(args.dataset)
    name = safe_name(args.name)
    output = (args.output_root / name).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_name = write_config(args, dataset, output)
    registration = model_registration(args, output)
    write_json(output / "samm_model.json", registration)
    command = [
        "python",
        "training/train.py",
        "-c",
        config_name,
        "--dataset-path",
        str(dataset),
        "--output-path",
        str(output),
        "--use-cluster",
        "0",
        "--num-gpus",
        str(args.num_gpus),
        "--num-nodes",
        str(args.num_nodes),
    ]
    write_json(output / "samm_run.json", {
        "variant": "medsam2",
        "dataset": str(dataset),
        "checkpoint": str(args.checkpoint.resolve()),
        "config": config_name,
        "registered_weight": registration,
        "command": command,
    })
    print(" ".join(command))
    if not args.dry_run:
        subprocess.run(command, cwd=MEDSAM2_ROOT, check=True)


def model_registration(args, output):
    checkpoint = (args.registered_checkpoint or output / "checkpoints" / "checkpoint.pt").resolve()
    return {
        "id": args.weight_id or f"medsam2_{safe_name(args.name).lower()}",
        "label": args.label or args.name,
        "model_id": "medsam2",
        "backend": "medsam2",
        "checkpoint": repo_path(checkpoint),
        "model_type": args.inference_config,
    }


def dataset_folder(path):
    path = path.resolve()
    train = path / "train_npz"
    if train.is_dir():
        return train
    if path.is_dir():
        return path
    raise ValueError(f"dataset folder not found: {path}")


def write_config(args, dataset, output):
    if not BASE_CONFIG.is_file():
        raise FileNotFoundError(BASE_CONFIG)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    name = safe_name(args.name)
    target = CONFIG_DIR / f"{name}.yaml"
    text = BASE_CONFIG.read_text(encoding="utf-8")
    replacements = {
        "  train_video_batch_size: 2 # increase batch size based on your computing": f"  train_video_batch_size: {args.batch_size}",
        "  num_train_workers: 15": f"  num_train_workers: {args.num_workers}",
        "  num_frames: 8": f"  num_frames: {args.num_frames}",
        "  max_num_objects: 3": f"  max_num_objects: {args.max_objects}",
        "  num_epochs: 75": f"  num_epochs: {args.epochs}",
        "    prob_to_use_pt_input_for_train: 0.5": "    prob_to_use_pt_input_for_train: 1.0",
        "  folder:  # PATH to Med NPZ folder": f"  folder: {dataset}",
        "                folder: /home/jma/Documents/MedSAM2/data/RECIST_train_npz # must be absolute path": f"                folder: {dataset}",
        "        checkpoint_path: checkpoints/sam2.1_hiera_tiny.pt # PATH to SAM 2.1 checkpoint": f"        checkpoint_path: {args.checkpoint.resolve()}",
        "  experiment_log_dir: exp_log # Path to log directory, defaults to ./sam2_logs/${config_name}": f"  experiment_log_dir: {output}",
        "  gpus_per_node: 4": f"  gpus_per_node: {args.num_gpus}",
        "  num_nodes: 1": f"  num_nodes: {args.num_nodes}",
    }
    for old, new in replacements.items():
        if old not in text:
            raise ValueError(f"base MedSAM2 config changed; missing line: {old}")
        text = text.replace(old, new)
    target.write_text(text, encoding="utf-8")
    shutil.copyfile(target, output / "config.yaml")
    return f"configs/samm/{name}.yaml"


def safe_name(name):
    value = "".join(char if char.isalnum() or char in "-_" else "_" for char in name)
    return value.strip("_") or "run"


def repo_path(path):
    path = path.resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
