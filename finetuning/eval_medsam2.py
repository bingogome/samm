from pathlib import Path
import json

import numpy as np


def add_eval_args(parser):
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", default="configs/sam2.1_hiera_t512.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--split", choices=("val", "train"), default="val")
    parser.add_argument("--prompt", nargs="+", choices=("box", "point", "box-point"), default=["box"])
    parser.add_argument("--max-segmented-volumes", type=int, default=0)


def eval_finetuned(args):
    build_sam2, predictor_type = medsam2_modules()
    model = build_sam2(args.config, str(args.checkpoint.resolve()), device=args.device)
    predictor = predictor_type(model)
    results = []
    for path in segmented_volume_paths(args.dataset, args.max_segmented_volumes, args.split):
        results.extend(eval_segmented_volume(path, predictor, np, args.prompt))
    summary = summarize(results)
    payload = {"split": args.split, "prompts": args.prompt, "summary": summary, "items": results}
    print(json.dumps(summary, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def medsam2_modules():
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    return build_sam2, SAM2ImagePredictor


def segmented_volume_paths(dataset, max_segmented_volumes, split="val"):
    root = dataset.resolve()
    folder = root if root.name == f"{split}_npz" else root / f"{split}_npz"
    if not folder.is_dir():
        raise FileNotFoundError(f"{split} split not found: {folder}")
    paths = sorted(folder.rglob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no {split} segmented volumes found in {folder}")
    return paths[:max_segmented_volumes] if max_segmented_volumes else paths


def eval_segmented_volume(path, predictor, np_module, prompts=("box",)):
    data = np_module.load(path)
    images = data["imgs"]
    labels = data["gts"]
    items = []
    for z in range(images.shape[0]):
        for label in [int(value) for value in np_module.unique(labels[z]) if value != 0]:
            gt = labels[z] == label
            box = mask_box(gt, np_module)
            point = mask_point(gt, np_module)
            image = np_module.repeat(images[z][:, :, None], 3, axis=2)
            predictor.set_image(image)
            for prompt in prompts:
                masks, scores, _ = predictor.predict(**prompt_kwargs(prompt, box, point, np_module), multimask_output=True)
                pred = masks[int(scores.argmax())].astype(bool)
                items.append({
                    "segmented_volume": path.stem,
                    "slice": int(z),
                    "label": label,
                    "prompt": prompt,
                    "dice": dice(pred, gt),
                    "box": [int(value) for value in box],
                    "point": [int(value) for value in point],
                })
    return items


def prompt_kwargs(prompt, box, point, np_module):
    kwargs = {}
    if prompt in ("box", "box-point"):
        kwargs["box"] = np_module.array(box, dtype=np_module.float32)
    if prompt in ("point", "box-point"):
        kwargs["point_coords"] = np_module.array([point], dtype=np_module.float32)
        kwargs["point_labels"] = np_module.array([1], dtype=np_module.int32)
    return kwargs


def mask_box(mask, np_module):
    ys, xs = np_module.where(mask)
    if len(xs) == 0:
        raise ValueError("cannot build a box from an empty mask")
    return [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]


def mask_point(mask, np_module):
    ys, xs = np_module.where(mask)
    if len(xs) == 0:
        raise ValueError("cannot build a point from an empty mask")
    center_x, center_y = xs.mean(), ys.mean()
    index = int(np_module.argmin((xs - center_x) ** 2 + (ys - center_y) ** 2))
    return [xs[index], ys[index]]


def dice(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    size = pred.sum() + gt.sum()
    return float(2 * intersection / size) if size else 1.0


def summarize(items):
    if not items:
        return {"count": 0, "mean_dice": 0.0, "prompts": {}}
    values = [item["dice"] for item in items]
    prompts = {}
    for prompt in sorted({item["prompt"] for item in items}):
        prompt_values = [item["dice"] for item in items if item["prompt"] == prompt]
        prompts[prompt] = {"count": len(prompt_values), "mean_dice": float(sum(prompt_values) / len(prompt_values))}
    return {"count": len(items), "mean_dice": float(sum(values) / len(values)), "prompts": prompts}
