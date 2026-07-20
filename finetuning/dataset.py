from pathlib import Path
import json
import math

import numpy as np


IMAGE_KEYS = ("imgs", "image", "images", "volume")
MASK_KEYS = ("gts", "mask", "masks", "segmentation", "labels")


def add_export_args(parser):
    parser.add_argument("--input", type=Path, help="NPZ file or folder containing imgs/gts NPZ files")
    parser.add_argument("--image", type=Path, help="Numpy image array")
    parser.add_argument("--mask", type=Path, help="Numpy label mask array")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--val-count", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--axes", nargs="+", type=int, choices=(0, 1, 2), default=[0])
    parser.add_argument("--window", nargs=2, type=float, metavar=("MIN", "MAX"))
    parser.add_argument("--percentile-window", nargs=2, type=float, default=(0.5, 99.5), metavar=("LOW", "HIGH"))


def export_dataset(args):
    output = args.output.resolve()
    groups = split_segmented_volumes(source_segmented_volumes(args), args)
    manifest = manifest_path(output)
    data = existing_manifest(manifest)
    for split, segmented_volumes in groups.items():
        data["splits"][split] = write_split(output, split, segmented_volumes, args)
    manifest.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    for split in groups:
        print(f"exported {len(data['splits'][split])} {split} segmented volume(s) to {output / f'{split}_npz'}")
    print(f"manifest: {manifest}")


def write_split(output, split, segmented_volumes, args):
    axes = export_axes(args)
    split_dir = output / f"{split}_npz"
    split_dir.mkdir(parents=True, exist_ok=True)
    for old in split_dir.glob("*.npz"):
        old.unlink()
    items = []
    for name, image, mask in segmented_volumes:
        image_u8, window = to_uint8(image, args.window, args.percentile_window)
        labels = labels_array(mask)
        if image_u8.shape != labels.shape:
            raise ValueError(f"image and mask shapes differ for {name}: {image_u8.shape} != {labels.shape}")
        for axis in axes:
            item_name = axis_segmented_volume_name(name, axis, axes)
            image_axis = axis_volume(image_u8, axis)
            labels_axis = axis_volume(labels, axis)
            target = split_dir / f"{item_name}.npz"
            np.savez_compressed(target, imgs=image_axis, gts=labels_axis)
            items.append({
                "name": item_name,
                "source_name": safe_name(name),
                "path": str(target.relative_to(output)),
                "axis": axis,
                "shape": list(image_axis.shape),
                "labels": [int(value) for value in np.unique(labels_axis) if value != 0],
                "window": window,
            })
    return items


def export_axes(args):
    axes = tuple(getattr(args, "axes", [0]) or [0])
    if len(set(axes)) != len(axes):
        raise ValueError("--axes contains duplicates")
    if any(axis not in (0, 1, 2) for axis in axes):
        raise ValueError("--axes must contain only 0, 1, or 2")
    return axes


def axis_segmented_volume_name(name, axis, axes):
    base = safe_name(name)
    return base if axes == (0,) else f"{base}_axis{axis}"


def axis_volume(array, axis):
    return np.ascontiguousarray(np.moveaxis(array, axis, 0))


def split_segmented_volumes(segmented_volumes, args):
    val_count = args.val_count or 0
    val_fraction = args.val_fraction or 0.0
    if val_count and val_fraction:
        raise ValueError("use --val-count or --val-fraction, not both")
    if val_fraction:
        if not 0 < val_fraction < 1:
            raise ValueError("--val-fraction must be between 0 and 1")
        val_count = math.ceil(len(segmented_volumes) * val_fraction)
    if not val_count:
        return {args.split: segmented_volumes}
    if val_count < 0 or val_count >= len(segmented_volumes):
        raise ValueError("--val-count must leave at least one training segmented volume")
    val_ids = set(np.random.default_rng(args.seed).permutation(len(segmented_volumes))[:val_count])
    return {
        "train": [item for index, item in enumerate(segmented_volumes) if index not in val_ids],
        "val": [item for index, item in enumerate(segmented_volumes) if index in val_ids],
    }


def source_segmented_volumes(args):
    if args.input and (args.image or args.mask):
        raise ValueError("use either --input or --image/--mask")
    if args.input:
        return input_segmented_volumes(args.input, args.name)
    if args.image and args.mask:
        name = args.name or args.image.stem
        return [(name, load_array(args.image, IMAGE_KEYS), load_array(args.mask, MASK_KEYS))]
    raise ValueError("provide --input or --image and --mask")


def input_segmented_volumes(path, name):
    path = path.resolve()
    if path.is_dir():
        return [(item.stem, *load_npz_pair(item)) for item in sorted(path.rglob("*.npz"))]
    if path.suffix != ".npz":
        raise ValueError("--input must be an NPZ file or a folder of NPZ files")
    return [(name or path.stem, *load_npz_pair(path))]


def load_npz_pair(path):
    data = np.load(path)
    return first_key(data, IMAGE_KEYS, path), first_key(data, MASK_KEYS, path)


def load_array(path, keys):
    if path.suffix == ".npz":
        return first_key(np.load(path), keys, path)
    return np.load(path)


def first_key(data, keys, path):
    for key in keys:
        if key in data:
            return data[key]
    raise ValueError(f"{path} does not contain any of {', '.join(keys)}")


def to_uint8(image, window, percentile_window):
    image = volume_array(image, "image")
    if image.dtype == np.uint8 and window is None:
        return image, {"type": "uint8"}
    image = image.astype(np.float32, copy=False)
    low, high = window or np.percentile(image, percentile_window)
    if high <= low:
        raise ValueError(f"invalid image window: {low}, {high}")
    scaled = np.clip((image - low) / (high - low), 0, 1)
    return np.rint(scaled * 255).astype(np.uint8), {"type": "linear", "min": float(low), "max": float(high)}


def labels_array(mask):
    labels = volume_array(mask, "mask")
    if labels.dtype.kind not in "biu":
        raise ValueError(f"mask must contain integer labels, got {labels.dtype}")
    return labels.astype(np.uint16, copy=False)


def volume_array(array, label):
    array = np.asarray(array)
    if array.ndim == 2:
        return array[None, :, :]
    if array.ndim == 3:
        return array
    if array.ndim == 4 and array.shape[-1] == 1:
        return array[..., 0]
    if array.ndim == 4 and array.shape[-1] == 3 and label == "image":
        return array.mean(axis=-1)
    raise ValueError(f"{label} must have shape HxW, DxHxW, or DxHxWxC; got {array.shape}")


def safe_name(name):
    value = "".join(char if char.isalnum() or char in "-_" else "_" for char in name)
    return value.strip("_") or "segmented_volume"


def manifest_path(output):
    output.mkdir(parents=True, exist_ok=True)
    return output / "dataset.json"


def existing_manifest(path):
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"format": "samm-finetune-dataset-v1", "splits": {"train": [], "val": []}}
