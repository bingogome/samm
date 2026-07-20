from argparse import Namespace
from pathlib import Path
import tarfile
from urllib.request import urlretrieve

import numpy as np

from .dataset import export_dataset, safe_name


MSD_TASKS = {
    "Task09_Spleen": {
        "url": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar",
        "name": "msd_spleen",
        "window": (-100.0, 300.0),
    },
    "Task03_Liver": {
        "url": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task03_Liver.tar",
        "name": "msd_liver",
        "window": (-100.0, 300.0),
    },
}


def add_example_args(parser):
    parser.add_argument("--task", choices=tuple(MSD_TASKS), default="Task09_Spleen")
    parser.add_argument("--segmented-volumes", type=int, default=4)
    parser.add_argument("--val-segmented-volumes", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--axes", nargs="+", type=int, choices=(0, 1, 2), default=[0])
    parser.add_argument("--raw-root", type=Path, default=Path("raw_data"))
    parser.add_argument("--segmented-volumes-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--window", nargs=2, type=float, metavar=("MIN", "MAX"))
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--skip-download", action="store_true")


def setup_example_data(args):
    if args.segmented_volumes < 1:
        raise ValueError("--segmented-volumes must be at least 1")

    task = MSD_TASKS[args.task]
    raw_root = args.raw_root.resolve()
    task_dir = raw_root / args.task
    archive = raw_root / f"{args.task}.tar"
    segmented_volumes_root = (args.segmented_volumes_root or Path("segmented_volumes") / task["name"]).resolve()
    output = (args.output or Path("datasets") / task["name"]).resolve()

    if not args.skip_download:
        download(task["url"], archive, args.force_download)
    if not task_dir.is_dir():
        extract(archive, raw_root)

    pairs = msd_pairs(task_dir, args.segmented_volumes)
    write_segmented_volume_npzs(pairs, segmented_volumes_root)
    val_segmented_volumes = args.val_segmented_volumes if args.val_segmented_volumes is not None else int(args.segmented_volumes > 1)
    export_dataset(Namespace(
        input=segmented_volumes_root,
        image=None,
        mask=None,
        output=output,
        name=None,
        split="train",
        val_count=val_segmented_volumes,
        val_fraction=0.0,
        seed=args.seed,
        axes=args.axes,
        window=args.window or task["window"],
        percentile_window=(0.5, 99.5),
    ))
    print(f"example segmented_volumes: {segmented_volumes_root}")
    print(f"training dataset: {output}")


def download(url, target, force=False):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and not force:
        print(f"using existing archive: {target}")
        return
    print(f"downloading {url}")
    urlretrieve(url, target)


def extract(archive, output):
    if not archive.is_file():
        raise FileNotFoundError(archive)
    print(f"extracting {archive}")
    with tarfile.open(archive) as tar:
        tar.extractall(output)


def msd_pairs(task_dir, count):
    images = sorted(path for path in (task_dir / "imagesTr").glob("*.nii.gz") if is_image_file(path))
    if not images:
        raise FileNotFoundError(task_dir / "imagesTr")
    pairs = [(image, task_dir / "labelsTr" / image.name) for image in images[:count]]
    missing = [label for _, label in pairs if not label.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    return pairs


def is_image_file(path):
    return not path.name.startswith(".")


def write_segmented_volume_npzs(pairs, output, read_image=None):
    reader = read_image or read_nifti
    output.mkdir(parents=True, exist_ok=True)
    for image_path, label_path in pairs:
        image = reader(image_path).astype(np.float32, copy=False)
        label = reader(label_path).astype(np.uint16, copy=False)
        np.savez_compressed(output / f"{safe_name(segmented_volume_name(image_path))}.npz", imgs=image, gts=label)
    print(f"converted {len(pairs)} segmented volume(s) to {output}")


def read_nifti(path):
    import SimpleITK as sitk

    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def segmented_volume_name(path):
    name = path.name
    return name[:-7] if name.endswith(".nii.gz") else path.stem
