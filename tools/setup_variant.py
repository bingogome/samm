#!/usr/bin/env python3
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
import os
import shutil
import subprocess
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
RED = "\033[31m"
RESET = "\033[0m"


@dataclass(frozen=True)
class CheckpointLink:
    source: Path
    target: Path
    required: bool = True
    page: str = ""
    preferred: Path | None = None
    hints: tuple[str, ...] = ()


@dataclass(frozen=True)
class CheckpointDownload:
    url: str
    target: Path


@dataclass(frozen=True)
class HuggingFaceCheckpoint:
    repo_id: str
    filename: str
    target: Path
    page: str
    gated: bool = True


@dataclass(frozen=True)
class Variant:
    key: str
    repo_url: str
    repo_dir: Path
    environment: str
    smoke: str
    editable_installs: tuple[Path, ...] = ()
    checkpoint_downloads: tuple[CheckpointDownload, ...] = ()
    checkpoint_links: tuple[CheckpointLink, ...] = ()
    hf_checkpoints: tuple[HuggingFaceCheckpoint, ...] = ()


VARIANTS = {
    "sam1": Variant(
        "sam1",
        "https://github.com/facebookresearch/segment-anything.git",
        ROOT / "sam_variants" / "segment-anything",
        "sam1",
        "import torch, segment_anything; print('sam1', torch.__version__, torch.cuda.is_available())",
        (ROOT / "sam_variants" / "segment-anything",),
        (
            CheckpointDownload(
                "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
                ROOT / "checkpoints" / "sam_vit_b_01ec64.pth",
            ),
            CheckpointDownload(
                "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
                ROOT / "checkpoints" / "sam_vit_l_0b3195.pth",
            ),
            CheckpointDownload(
                "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                ROOT / "checkpoints" / "sam_vit_h_4b8939.pth",
            ),
        ),
    ),
    "sam2": Variant(
        "sam2",
        "https://github.com/facebookresearch/sam2.git",
        ROOT / "sam_variants" / "sam2",
        "sam2",
        "import torch, sam2; from sam2.sam2_image_predictor import SAM2ImagePredictor; print('sam2', torch.__version__, torch.cuda.is_available())",
        (ROOT / "sam_variants" / "sam2",),
        (
            CheckpointDownload(
                "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
                ROOT / "checkpoints" / "sam2.1_hiera_tiny.pt",
            ),
            CheckpointDownload(
                "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
                ROOT / "checkpoints" / "sam2.1_hiera_small.pt",
            ),
            CheckpointDownload(
                "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
                ROOT / "checkpoints" / "sam2.1_hiera_base_plus.pt",
            ),
            CheckpointDownload(
                "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
                ROOT / "checkpoints" / "sam2.1_hiera_large.pt",
            ),
        ),
    ),
    "mobile-sam": Variant(
        "mobile-sam",
        "https://github.com/ChaoningZhang/MobileSAM.git",
        ROOT / "sam_variants" / "MobileSAM",
        "mobile-sam",
        "import torch, mobile_sam; print('mobile-sam', torch.__version__, torch.cuda.is_available())",
        (ROOT / "sam_variants" / "MobileSAM",),
        (),
        (
            CheckpointLink(
                ROOT / "sam_variants" / "MobileSAM" / "weights" / "mobile_sam.pt",
                ROOT / "checkpoints" / "mobile_sam.pt",
            ),
        ),
    ),
    "medsam": Variant(
        "medsam",
        "https://github.com/bowang-lab/MedSAM.git",
        ROOT / "sam_variants" / "MedSAM",
        "medsam",
        "import sys, pathlib, torch, transformers; sys.path.insert(0, str(pathlib.Path('sam_variants/MedSAM').resolve())); import segment_anything; from transformers import CLIPTokenizer, CLIPTextModel; print('medsam', torch.__version__, torch.cuda.is_available())",
        (),
        (),
        (
            CheckpointLink(
                ROOT / "sam_variants" / "MedSAM" / "work_dir" / "MedSAM" / "medsam_vit_b.pth",
                ROOT / "checkpoints" / "medsam_vit_b.pth",
                False,
                "https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN?usp=drive_link",
                ROOT / "checkpoints" / "medsam_vit_b.pth",
            ),
            CheckpointLink(
                ROOT / "sam_variants" / "MedSAM" / "extensions" / "text_prompt" / "medsam_text_prompt_flare22.pth",
                ROOT / "checkpoints" / "medsam_text_prompt_flare22.pth",
                False,
                "https://drive.google.com/file/d/12YH-N6PAKayulhS99MBURVNpuQtVj98S/view?usp=sharing",
                ROOT / "checkpoints" / "medsam_text_prompt_flare22.pth",
                ("download the MedSAM text prompt FLARE22 checkpoint",),
            ),
        ),
    ),
    "medsam2": Variant(
        "medsam2",
        "https://github.com/bowang-lab/MedSAM2.git",
        ROOT / "sam_variants" / "MedSAM2",
        "medsam2",
        "import torch, sam2; from sam2.sam2_image_predictor import SAM2ImagePredictor; print('medsam2', torch.__version__, torch.cuda.is_available())",
        (ROOT / "sam_variants" / "MedSAM2",),
        (
            CheckpointDownload(
                "https://huggingface.co/wanglab/MedSAM2/resolve/main/MedSAM2_latest.pt",
                ROOT / "checkpoints" / "MedSAM2_latest.pt",
            ),
            CheckpointDownload(
                "https://huggingface.co/wanglab/MedSAM2/resolve/main/MedSAM2_2411.pt",
                ROOT / "checkpoints" / "MedSAM2_2411.pt",
            ),
            CheckpointDownload(
                "https://huggingface.co/wanglab/MedSAM2/resolve/main/MedSAM2_CTLesion.pt",
                ROOT / "checkpoints" / "MedSAM2_CTLesion.pt",
            ),
            CheckpointDownload(
                "https://huggingface.co/wanglab/MedSAM2/resolve/main/MedSAM2_MRI_LiverLesion.pt",
                ROOT / "checkpoints" / "MedSAM2_MRI_LiverLesion.pt",
            ),
            CheckpointDownload(
                "https://huggingface.co/wanglab/MedSAM2/resolve/main/MedSAM2_US_Heart.pt",
                ROOT / "checkpoints" / "MedSAM2_US_Heart.pt",
            ),
        ),
    ),
    "sam3": Variant(
        key="sam3",
        repo_url="https://github.com/facebookresearch/sam3.git",
        repo_dir=ROOT / "sam_variants" / "sam3",
        environment="sam3",
        smoke="import torch, sam3; from sam3.model.sam3_image_processor import Sam3Processor; print('sam3', torch.__version__, torch.cuda.is_available())",
        editable_installs=(ROOT / "sam_variants" / "sam3",),
        hf_checkpoints=(
            HuggingFaceCheckpoint(
                "facebook/sam3",
                "sam3.pt",
                ROOT / "checkpoints" / "sam3.pt",
                "https://huggingface.co/facebook/sam3",
            ),
        ),
    ),
    "medical-sam3": Variant(
        key="medical-sam3",
        repo_url="https://github.com/AIM-Research-Lab/Medical-SAM3.git",
        repo_dir=ROOT / "sam_variants" / "Medical-SAM3",
        environment="medical-sam3",
        smoke="import torch, sam3; from sam3.model.sam3_image_processor import Sam3Processor; from sam3.model_builder import build_sam3_image_model; print('medical-sam3', torch.__version__, torch.cuda.is_available())",
        editable_installs=(ROOT / "sam_variants" / "Medical-SAM3",),
        hf_checkpoints=(
            HuggingFaceCheckpoint(
                "ChongCong/Medical-SAM3",
                "checkpoint_2D.pt",
                ROOT / "checkpoints" / "medical_sam3.pt",
                "https://huggingface.co/ChongCong/Medical-SAM3",
                False,
            ),
        ),
    ),
    "fastsam": Variant(
        "fastsam",
        "https://github.com/CASIA-IVA-Lab/FastSAM.git",
        ROOT / "sam_variants" / "FastSAM",
        "fastsam",
        "import os, sys, pathlib, torch; pathlib.Path('/tmp/samm-fastsam-home/.config').mkdir(parents=True, exist_ok=True); os.environ['HOME'] = '/tmp/samm-fastsam-home'; os.environ.setdefault('MPLCONFIGDIR', '/tmp/samm-matplotlib'); os.environ.setdefault('YOLO_CONFIG_DIR', '/tmp/samm-ultralytics'); sys.path.insert(0, str(pathlib.Path('sam_variants/FastSAM').resolve())); import fastsam, clip; from fastsam import FastSAM, FastSAMPrompt; print('fastsam', torch.__version__, torch.cuda.is_available())",
        (),
        (),
        (
            CheckpointLink(
                ROOT / "checkpoints" / "FastSAM-x.pt",
                ROOT / "checkpoints" / "FastSAM-x.pt",
                False,
                "https://drive.google.com/file/d/1m1sjY4ihXBU1fZXdQ-Xdj-mDltW-2Rqv/view?usp=sharing",
                ROOT / "checkpoints" / "FastSAM-x.pt",
                ("download the default/FastSAM-x checkpoint from the FastSAM model zoo",),
            ),
            CheckpointLink(
                ROOT / "checkpoints" / "FastSAM-s.pt",
                ROOT / "checkpoints" / "FastSAM-s.pt",
                False,
                "https://drive.google.com/file/d/10XmSj6mmpmRb8NhXbtiuO9cTTBwR_9SV/view?usp=sharing",
                ROOT / "checkpoints" / "FastSAM-s.pt",
                ("download the FastSAM-s checkpoint from the FastSAM model zoo",),
            ),
        ),
    ),
}


def main():
    args = parser().parse_args()
    for variant in selected_variants(args.variant):
        setup_variant(variant, args)


def parser():
    parser = ArgumentParser(description="Set up SAMM model variant sources, checkpoints, and Pixi environments.")
    parser.add_argument(
        "variant",
        choices=("sam1", "sam2", "mobile-sam", "medsam", "medsam2", "sam3", "medical-sam3", "fastsam", "all"),
    )
    parser.add_argument("--skip-clone", action="store_true")
    parser.add_argument("--skip-pixi", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    return parser


def selected_variants(key):
    return VARIANTS.values() if key == "all" else (VARIANTS[key],)


def setup_variant(variant, args):
    say(f"== {variant.key} ==")
    if not args.skip_clone:
        clone_source(variant)
    for download in variant.checkpoint_downloads:
        ensure_download(download)
    for link in variant.checkpoint_links:
        ensure_link(link)
    if not args.skip_pixi:
        run("pixi", "install", "--environment", variant.environment)
        install_editable_sources(variant)
        pip_check(variant)
    for checkpoint in variant.hf_checkpoints:
        ensure_hf_checkpoint(variant, checkpoint, args)
    if not args.skip_smoke:
        run("pixi", "run", "-e", variant.environment, "python", "-W", "ignore::FutureWarning", "-W", "ignore::UserWarning", "-c", variant.smoke)


def clone_source(variant):
    if variant.repo_dir.exists():
        say(f"source: {variant.repo_dir}")
        return
    variant.repo_dir.parent.mkdir(parents=True, exist_ok=True)
    run("git", "clone", variant.repo_url, str(variant.repo_dir))


def install_editable_sources(variant):
    for path in variant.editable_installs:
        if not path.is_dir():
            raise SystemExit(f"source missing for editable install: {path}")
        say(f"editable install: {path}")
        run("pixi", "run", "-e", variant.environment, "python", "-m", "pip", "install", "--no-deps", "-e", str(path))


def pip_check(variant):
    say(f"pip check: {variant.environment}")
    run("pixi", "run", "-e", variant.environment, "python", "-m", "pip", "check")


def ensure_link(link):
    if link.target.is_file():
        ensure_reverse_link(link)
        say(f"checkpoint: {link.target}")
        return
    if link.target.exists() or link.target.is_symlink():
        raise SystemExit(f"checkpoint path exists but is not a file: {link.target}")
    if not link.source.is_file():
        if not link.required:
            warn(f"checkpoint missing: {link.target}")
            warn(f"place file at: {link.preferred or link.source}")
            if link.page:
                warn(f"download page: {link.page}")
            for hint in link.hints:
                warn(hint)
            return
        raise SystemExit(f"checkpoint source missing: {link.source}")
    link.target.parent.mkdir(parents=True, exist_ok=True)
    link.target.symlink_to(os.path.relpath(link.source, link.target.parent))
    say(f"checkpoint: {link.target} -> {link.source}")


def ensure_reverse_link(link):
    if link.source.is_file():
        return
    if link.source.exists() or link.source.is_symlink():
        raise SystemExit(f"checkpoint path exists but is not a file: {link.source}")
    link.source.parent.mkdir(parents=True, exist_ok=True)
    link.source.symlink_to(os.path.relpath(link.target, link.source.parent))
    say(f"checkpoint: {link.source} -> {link.target}")


def ensure_download(download):
    if download.target.is_file():
        say(f"checkpoint: {download.target}")
        return
    if download.target.exists() or download.target.is_symlink():
        raise SystemExit(f"checkpoint path exists but is not a file: {download.target}")
    download.target.parent.mkdir(parents=True, exist_ok=True)
    temp = download.target.with_name(f"{download.target.name}.download")
    say(f"download: {download.url}")
    with urlopen(download.url) as response, temp.open("wb") as output:
        shutil.copyfileobj(response, output)
    temp.replace(download.target)
    say(f"checkpoint: {download.target}")


def ensure_hf_checkpoint(variant, checkpoint, args):
    if checkpoint.target.is_file():
        say(f"checkpoint: {checkpoint.target}")
        return
    if checkpoint.target.exists() or checkpoint.target.is_symlink():
        raise SystemExit(f"checkpoint path exists but is not a file: {checkpoint.target}")
    downloaded = checkpoint.target.parent / checkpoint.filename
    if downloaded.is_file():
        move_hf_checkpoint(downloaded, checkpoint.target)
        return
    if downloaded.exists() or downloaded.is_symlink():
        raise SystemExit(f"checkpoint path exists but is not a file: {downloaded}")
    if args.skip_pixi:
        warn_hf_missing(variant, checkpoint)
        warn("skipping Hugging Face checkpoint download because --skip-pixi was set")
        return
    warn_hf_missing(variant, checkpoint)
    checkpoint.target.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint.gated:
        if not ask_approved():
            warn("skipping gated checkpoint download")
            return
        say(f"login: pixi run -e {variant.environment} hf auth login")
        run("pixi", "run", "-e", variant.environment, "hf", "auth", "login")
    say(f"download: hf://{checkpoint.repo_id}/{checkpoint.filename}")
    run(
        "pixi",
        "run",
        "-e",
        variant.environment,
        "hf",
        "download",
        checkpoint.repo_id,
        checkpoint.filename,
        "--local-dir",
        str(checkpoint.target.parent),
    )
    if downloaded.is_file() and downloaded != checkpoint.target:
        move_hf_checkpoint(downloaded, checkpoint.target)
        return
    if not checkpoint.target.is_file():
        raise SystemExit(f"huggingface download completed but checkpoint is missing: {checkpoint.target}")
    say(f"checkpoint: {checkpoint.target}")


def move_hf_checkpoint(source, target):
    if target.exists() or target.is_symlink():
        raise SystemExit(f"checkpoint path exists but is not a file: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    source.replace(target)
    say(f"checkpoint: {target}")


def warn_hf_missing(variant, checkpoint):
    warn(f"checkpoint missing: {checkpoint.target}")
    warn(f"place file at: {checkpoint.target}")
    if checkpoint.gated:
        warn(f"get model approval first: {checkpoint.page}")
        warn(f"setup will authenticate and download after approval: pixi run setup-{variant.key}")
        return
    warn(f"download page: {checkpoint.page}")
    warn(f"setup will download with Hugging Face CLI: pixi run setup-{variant.key}")


def ask_approved():
    answer = input("Have you received Hugging Face approval for this model? [y/N]: ")
    return answer.strip().lower() in ("y", "yes")


def say(message):
    print(message, flush=True)


def warn(message):
    print(f"{RED}{message}{RESET}", flush=True)


def run(*command):
    program = shutil.which(command[0])
    if not program:
        raise SystemExit(f"program not found: {command[0]}")
    subprocess.run((program, *command[1:]), cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
