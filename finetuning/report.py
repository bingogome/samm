from pathlib import Path
import json
import re


ROOT = Path(__file__).resolve().parents[1]


def add_report_args(parser):
    parser.add_argument("--run", type=Path, required=True)


def report(args):
    print(format_report(collect_report(args.run)))


def collect_report(run, root=ROOT):
    run = run.resolve()
    if not run.is_dir():
        raise FileNotFoundError(run)
    model = read_json(run / "samm_model.json")
    run_info = read_json(run / "samm_run.json")
    checkpoint = checkpoint_path(model["checkpoint"], root)
    train = train_summary(read_json_lines(train_stats_path(run)))
    evals = [eval_summary(path) for path in eval_paths(run)]
    config = config_summary(run / "config.yaml")
    dataset = dataset_summary(run_info["dataset"])
    return {
        "name": run.name,
        "path": run,
        "model": model,
        "run": run_info,
        "checkpoint": checkpoint_summary(checkpoint),
        "dataset": dataset,
        "config": config,
        "train": train,
        "evals": evals,
    }


def format_report(data):
    lines = [
        f"Run: {data['name']}",
        f"Path: {display_path(data['path'])}",
        f"Weight: {data['model']['id']} ({data['model']['label']})",
        f"Checkpoint: {checkpoint_line(data['checkpoint'])}",
        f"Dataset: {data['dataset']['path']} ({data['dataset']['count']} segmented volume(s), {data['dataset']['status']})",
        f"Configured epochs: {value_or_unknown(data['config'].get('epochs'))}",
        "Training:",
        f"  epochs: {value_or_unknown(data['train'].get('epochs'))}",
        f"  steps: {value_or_unknown(data['train'].get('steps'))}",
        f"  final loss: {format_number(data['train'].get('final_loss'))}",
        f"  best loss: {format_number(data['train'].get('best_loss'))}{best_epoch_suffix(data['train'])}",
        "Evaluation:",
    ]
    if data["evals"]:
        for item in data["evals"]:
            lines.extend(format_eval(item))
    else:
        lines.append("  none")
    return "\n".join(lines)


def read_json(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_json_lines(path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def train_stats_path(run):
    for path in (run / "logs" / "train_stats.json", run / "train_stats.json"):
        if path.is_file():
            return path
    return run / "logs" / "train_stats.json"


def train_summary(rows):
    if not rows:
        return {}
    losses = [(row.get("Losses/train_all_loss"), row) for row in rows if row.get("Losses/train_all_loss") is not None]
    best_loss, best_row = min(losses, key=lambda item: item[0]) if losses else (None, {})
    final = rows[-1]
    return {
        "epochs": int(final["Trainer/epoch"]) + 1 if final.get("Trainer/epoch") is not None else len(rows),
        "steps": final.get("Trainer/steps_train"),
        "final_loss": final.get("Losses/train_all_loss"),
        "best_loss": best_loss,
        "best_epoch": best_row.get("Trainer/epoch"),
    }


def eval_paths(run):
    return sorted(path for path in run.glob("eval*.json") if path.is_file())


def eval_summary(path):
    data = read_json(path)
    return {"name": path.name, "split": data.get("split", "unknown"), "summary": data["summary"]}


def config_summary(path):
    if not path.is_file():
        return {}
    match = re.search(r"^\s*num_epochs:\s*(\d+)\s*$", path.read_text(encoding="utf-8"), re.MULTILINE)
    return {"epochs": int(match.group(1))} if match else {}


def dataset_summary(path):
    folder = Path(path)
    return {
        "path": str(folder),
        "status": "present" if folder.is_dir() else "missing",
        "count": len(list(folder.glob("*.npz"))) if folder.is_dir() else 0,
    }


def checkpoint_path(path, root):
    path = Path(path)
    return path if path.is_absolute() else root / path


def checkpoint_summary(path):
    return {
        "path": path,
        "status": "present" if path.is_file() else "missing",
        "size_mb": path.stat().st_size / 1024 / 1024 if path.is_file() else None,
    }


def checkpoint_line(data):
    if data["status"] == "present":
        return f"present, {data['size_mb']:.1f} MB ({display_path(data['path'])})"
    return f"missing ({display_path(data['path'])})"


def format_eval(item):
    summary = item["summary"]
    lines = [
        f"  {item['name']} [{item['split']}]",
        f"    all: n={summary['count']} mean_dice={format_number(summary.get('mean_dice'))}",
    ]
    for prompt, values in sorted(summary.get("prompts", {}).items()):
        lines.append(f"    {prompt}: n={values['count']} mean_dice={format_number(values.get('mean_dice'))}")
    return lines


def format_number(value):
    return "unknown" if value is None else f"{float(value):.4f}"


def value_or_unknown(value):
    return "unknown" if value is None else str(value)


def best_epoch_suffix(train):
    epoch = train.get("best_epoch")
    return "" if epoch is None else f" at epoch {epoch}"


def display_path(path):
    path = Path(path)
    return str(path.relative_to(ROOT)) if path.is_absolute() and path.is_relative_to(ROOT) else str(path)
