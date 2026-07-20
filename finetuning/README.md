# SAMM Finetuning

SAMM 1.2.0 supports MedSAM2 finetuning from Slicer and from the command line.
For normal interactive use, run finetuning from the Slicer `Finetune` panel. The
Slicer extension starts the SAMM server, sends build/train/eval requests, polls
job state, and stops the server process it started. Use the command line for
initial setup, example data, scripted/batch runs, reproducing a Slicer job, or
debugging.

| Task | Run from |
| --- | --- |
| Add segmented volumes from Slicer scene data | Slicer `Finetune` panel |
| Build a dataset interactively | Slicer `Finetune` panel |
| Start, monitor, cancel, evaluate, and report an interactive run | Slicer `Finetune` panel |
| Install checkpoints | Terminal setup command |
| Run standalone batch training/eval/report without Slicer | Terminal CLI |

## Concepts

A **segmented volume** is one image volume paired with one integer label mask. Following a MedSAM workflow, SAMM stores it as an `.npz` file:

```text
imgs: D x H x W image volume
gts:  D x H x W integer label mask, with 0 as background
```

Slicer exports source segmented volumes to:

```text
segmented_volumes/<dataset>/<volume>.npz
```

Those files include `imgs`, `gts`, and extra segment metadata such as segment
ids, names, labels, spacing, origin, and directions. The CLI exporter only needs
the image and mask arrays and ignores extra arrays.

A **dataset** is a named collection of exported training NPZs plus a
manifest:

```text
datasets/<dataset>/train_npz/     training split
datasets/<dataset>/val_npz/       validation split
datasets/<dataset>/dataset.json   export manifest
```

Source segmented volumes can keep original image intensities. The dataset
exporter windows them to uint8 images for MedSAM2 training. If no explicit
window is provided, it uses percentile windowing with the 0.5 and 99.5
percentiles.

Dataset and run names use letters, numbers, dots, dashes, and underscores. A
single source segmented volume may contain multiple non-empty segments; SAMM
exports each as a distinct non-zero integer label. Segments must not overlap,
because one output voxel cannot represent two class labels in the stored mask.

## Prerequisites

Set up MedSAM2 ([Cite](#citation)) from a terminal before using the Slicer finetuning panel or the
CLI tools. This installs the environment and makes sure the base SAM2.1
checkpoint exists:

```bash
pixi run setup-medsam2
```

The Slicer finetuning panel currently exposes one base model:

```text
SAM2.1 Hiera Tiny -> checkpoints/sam2.1_hiera_tiny.pt
```

The CLI can use any checkpoint path accepted by `finetune-medsam2`.

## Slicer Workflow

This is the user-facing workflow. Start here when working interactively in
Slicer.

1. Open the `Finetune` panel.
2. Type a dataset name and click `New dataset`, or select an existing dataset. `Refresh` discovers datasets created or changed outside the current module session.
3. Select a volume and segmentation. The selectors stay synchronized with the Data panel.
4. Click `Add segmented volume`.
5. Choose the number of validation source volumes, one or more slice axes, and an optional explicit intensity window.
6. Click `Build dataset`.
7. Set the run name, base MedSAM2 model, epochs, batch size, frames, and dataloader workers.
8. Click `Run training`.
9. Set the optional evaluation volume limit and click `Run eval`.
10. Click `Show report`, or cancel the active train/eval job when needed.

Training and eval run on the SAMM server that the Slicer extension started. The
panel shows job status, the log path, the latest log tail, and progress.
Training progress is parsed by epoch; eval shows a busy state while it runs.

The dataset status shows source, train, and validation segmented-volume counts.
Train/validation splitting happens per source volume before slice-axis
augmentation, preventing alternate views of the same volume from leaking into
both splits. A run name left empty defaults to `<dataset>_v1`; evaluation uses
that run's checkpoint unless another checkpoint is supplied through the API or
CLI.

When Slicer exports a segmented volume, every non-empty segment becomes a
non-zero integer label. Overlapping exported segments are rejected.

Dataset, build, training, and evaluation settings are stored in SAMM's MRML
parameter node and survive module switches and scene save/reload. Active job
IDs and progress are runtime state and are not serialized into the medical
scene.

## Example Data

Use this command-line helper when you want a small public dataset for testing
before opening Slicer, or when you want to exercise the CLI workflow without
Slicer.

Download and convert a small Medical Segmentation Decathlon example:

```bash
pixi run setup-example-data --segmented-volumes 4 --val-segmented-volumes 1
```

This creates:

```text
raw_data/Task09_Spleen/
segmented_volumes/msd_spleen/
datasets/msd_spleen/train_npz/
datasets/msd_spleen/val_npz/
datasets/msd_spleen/dataset.json
```

Useful options:

```bash
pixi run setup-example-data --task Task03_Liver --segmented-volumes 4 --val-segmented-volumes 1
pixi run setup-example-data --axes 0 1 2
pixi run setup-example-data --window -100 300
pixi run setup-example-data --skip-download
pixi run setup-example-data --force-download
```

Supported example tasks are `Task09_Spleen` and `Task03_Liver`.

## Build A Dataset

In normal use, click `Build dataset` in Slicer. Use the CLI exporter when you
are preparing data outside Slicer or scripting a dataset build.

Export a folder of source segmented volumes and reserve validation volumes:

```bash
pixi run export-finetune-dataset \
  --input segmented_volumes/my_task \
  --output datasets/my_task \
  --val-count 2
```

Export all three axes as slice augmentation:

```bash
pixi run export-finetune-dataset \
  --input segmented_volumes/my_task \
  --output datasets/my_task \
  --val-count 2 \
  --axes 0 1 2
```

Train/validation splitting happens before axis augmentation, so different views
of the same source segmented volume stay in the same split.

Use a validation fraction instead of a fixed validation count:

```bash
pixi run export-finetune-dataset \
  --input segmented_volumes/my_task \
  --output datasets/my_task \
  --val-fraction 0.2 \
  --seed 7
```

Export one source `.npz` file:

```bash
pixi run export-finetune-dataset \
  --input segmented_volumes/segmented_volume001.npz \
  --output datasets/my_task \
  --name segmented_volume001
```

Export separate NumPy arrays:

```bash
pixi run export-finetune-dataset \
  --image segmented_volumes/segmented_volume001_image.npy \
  --mask segmented_volumes/segmented_volume001_mask.npy \
  --output datasets/my_task \
  --name segmented_volume001
```

Use a manual split when needed:

```bash
pixi run export-finetune-dataset \
  --input segmented_volumes/my_task \
  --output datasets/my_task \
  --split val
```

Use an explicit intensity window for non-uint8 volumes when you know the desired
range:

```bash
pixi run export-finetune-dataset \
  --input segmented_volumes/my_task \
  --output datasets/my_task \
  --window -100 300
```

Input `.npz` files can use `imgs`, `image`, `images`, or `volume` for image
arrays and `gts`, `mask`, `masks`, `segmentation`, or `labels` for masks. The
exporter accepts 2D, 3D, single-channel 4D, and RGB image arrays; RGB images are
converted to grayscale.

## Train

In normal use, click `Run training` in the Slicer `Finetune` panel. Use the CLI
when you want a standalone training run without starting Slicer or the SAMM
server.

Launch a MedSAM2 training run:

```bash
pixi run -e medsam2 finetune-medsam2 \
  --dataset datasets/my_task \
  --name my_task_v1 \
  --checkpoint checkpoints/sam2.1_hiera_tiny.pt \
  --epochs 25 \
  --batch-size 1 \
  --num-workers 0 \
  --num-frames 4
```

Useful training options:

```bash
pixi run -e medsam2 finetune-medsam2 \
  --dataset datasets/my_task \
  --name my_task_v1 \
  --checkpoint checkpoints/sam2.1_hiera_tiny.pt \
  --max-objects 3 \
  --num-gpus 1 \
  --num-nodes 1 \
  --weight-id medsam2_my_task_v1 \
  --label my_task_v1
```

Dry-run the generated MedSAM2 command:

```bash
pixi run -e medsam2 finetune-medsam2 \
  --dataset datasets/my_task \
  --name my_task_v1 \
  --dry-run
```

Training writes:

```text
finetuning_runs/<run>/config.yaml
finetuning_runs/<run>/samm_run.json
finetuning_runs/<run>/samm_model.json
finetuning_runs/<run>/checkpoints/checkpoint.pt
```

`samm_model.json` registers the run as a MedSAM2 weight. By default the
registered weight id is `medsam2_<run>`, the label is the run name, and the
registered checkpoint is:

```text
finetuning_runs/<run>/checkpoints/checkpoint.pt
```

The service scans `finetuning_runs/*/samm_model.json` and adds valid registered
weights to the MedSAM2 weight list. Availability is still based on whether the
registered checkpoint file exists.

After a Slicer-started training job completes, the model list refreshes
automatically. Select the new MedSAM2 weight and prepare it to use the trained
checkpoint for the same prediction, embedding, and supported temporal
workflows as other MedSAM2 weights.

## Evaluate

In normal use, click `Run eval` in Slicer. Use the CLI evaluator when you want
to evaluate a checkpoint outside the Slicer/server workflow.

Evaluate a checkpoint on the validation split:

```bash
pixi run -e medsam2 eval-finetuned \
  --dataset datasets/my_task \
  --checkpoint finetuning_runs/my_task_v1/checkpoints/checkpoint.pt \
  --prompt box point box-point \
  --output finetuning_runs/my_task_v1/eval.json
```

The evaluator creates prompts from the ground truth mask and reports Dice for
each prompt mode. Prompt modes are `box`, `point`, and `box-point`. CLI eval
defaults to `--prompt box`; the Slicer/service eval path sends all three prompt
modes.

The evaluator uses `val_npz` by default. Use `--split train` only for smoke
checks:

```bash
pixi run -e medsam2 eval-finetuned \
  --dataset datasets/my_task \
  --checkpoint finetuning_runs/my_task_v1/checkpoints/checkpoint.pt \
  --split train \
  --max-segmented-volumes 2
```

Eval writes a JSON payload with `split`, `prompts`, `summary`, and per-slice
`items` when `--output` is provided.

## Report

In normal use, click `Show report` in Slicer. Use the CLI report command when
inspecting a run from the terminal.

Summarize a run:

```bash
pixi run -e medsam2 finetune-report --run finetuning_runs/my_task_v1
```

The report reads run metadata, checkpoint status, `logs/train_stats.json` or
`train_stats.json`, `eval*.json` files, configured epochs, final loss, best
loss, and Dice summaries.

The SAMM service exposes the same text report at:

```text
GET /finetuning/reports/<run>
```

## Server Jobs

The service exposes finetuning jobs so Slicer can stay responsive. These
endpoints are called by the Slicer `Finetune` panel; call them directly only for
API testing or automation around the service.

```text
POST /finetuning/datasets/<name>/build
POST /finetuning/train
POST /finetuning/eval
GET  /finetuning/jobs/<job_id>
POST /finetuning/jobs/<job_id>/cancel
```

Training logs are written to:

```text
logs/finetuning_jobs/<run>_train.log
logs/finetuning_jobs/<run>_eval.log
```

Training jobs default to `checkpoints/sam2.1_hiera_tiny.pt`, 25 epochs, batch
size 1, 0 workers, and 4 frames through the service. Eval jobs default to
`finetuning_runs/<run>/checkpoints/checkpoint.pt` and prompts `box`, `point`,
and `box-point`.

## Tests

```bash
pixi run test-finetuning
```

`pixi run check` runs both the finetuning and service suites. The normal GitHub
Actions workflow exercises dataset/export/report logic without downloading
MedSAM2, checkpoints, or requiring a GPU; end-to-end CUDA training and
evaluation remain local GPU workflows.

## Related Documentation

- [../README.md](../README.md): complete Slicer setup and feature reference.
- [../service/README.md](../service/README.md): finetuning HTTP endpoints and job payloads.
- [../sam_variants/README.md](../sam_variants/README.md): MedSAM2 environment and checkpoint setup.


## Citation 

The finetuning workflow uses MedSAM2

```bibtex
@article{MedSAM2,
    title={MedSAM2: Segment Anything in 3D Medical Images and Videos},
    author={Ma, Jun and Yang, Zongxin and Kim, Sumin and Chen, Bihui and Baharoon, Mohammed and Fallahpour, Adibvafa and Asakereh, Reza and Lyu, Hongwei and Wang, Bo},
    journal={arXiv preprint arXiv:2504.03600},
    year={2025}
}
```

If you use SAMM in your research, please consider use the following BibTeX entry.

```bibtex
@article{liu2024segment,
  title={Segment Any Medical Model Extended},
  author={Liu, Yihao and Zhang, Jiaming and Diaz-Pinto, Andres and Li, Haowei and Martin-Gomez, Alejandro and Kheradmand, Amir and Armand, Mehran},
  journal={arXiv preprint arXiv:2403.18114},
  year={2024}
}
@article{liu2023samm,
  title={SAMM (Segment Any Medical Model): A 3D Slicer Integration to SAM},
  author={Liu, Yihao and Zhang, Jiaming and She, Zhangcong and Kheradmand, Amir and Armand, Mehran},
  journal={arXiv preprint arXiv:2304.05622},
  year={2023}
}
```