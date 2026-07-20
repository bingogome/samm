# SAMM Service

The SAMM service is the local inference gateway used by the 3D Slicer
extension. It exposes one HTTP API, lists model weights, starts the correct Pixi
worker environment, and routes prediction, embedding, bulk prediction,
temporal propagation, and finetuning jobs.

## Normal Slicer Use

For normal SAMM use, do not start the service from a terminal. Open the Slicer
extension and click `Start server`. The extension launches `pixi run server`,
records the managed process and log under `logs/`, and sends session heartbeats.
`Stop` terminates the process started by the module. When Slicer quits, a
six-second prompt closes that process by default or lets you leave it running. `Show log` displays the log path and its most recent output.

Use the Slicer UI for user-facing workflows:

- Prepare models and run prediction.
- Embed views and save or load embedding folders.
- Run 3D box prediction and auto prediction.
- Build finetuning datasets, launch training/eval jobs, cancel jobs, and show reports.

| Task | Run from |
| --- | --- |
| Start or stop the SAMM server for interactive use | Slicer extension |
| Prepare a model, predict, embed, auto predict, or run 3D box prediction | Slicer extension |
| Build finetuning datasets, train, eval, cancel, or show reports interactively | Slicer extension |
| Inspect the managed gateway log and recent output | Slicer extension |
| Install model variants, environments, and checkpoints | Terminal setup commands |
| Exercise HTTP endpoints, run headless automation, or debug service behavior | Terminal/API client |
| Start a worker directly | Terminal, development only |

The command examples below are for setup, development, API testing,
troubleshooting, or running finetuning tools outside Slicer.

## Manual Gateway Run

Run the gateway manually only when you want to exercise the service without the
Slicer extension. From the project root:

```bash
pixi run server
```

The Pixi task listens on `127.0.0.1:8799`, starts workers on
`127.0.0.1:8801`, reads weights from `checkpoints/`, and uses CUDA:

```bash
python -m samm_server.app \
  --host 127.0.0.1 \
  --port 8799 \
  --worker-port 8801 \
  --model-dir checkpoints \
  --device cuda
```

If you run `python -m samm_server.app` directly without `--device`, the CLI
default is CPU. A manually launched gateway is not owned by Slicer; stop it from
the terminal where you started it.

The service is designed for local use and binds to loopback by default. It does
not provide authentication, TLS, or a multi-user scheduling boundary. Add those
outside the service before exposing it beyond the local workstation.

Useful gateway options:

```bash
python -m samm_server.app --help
python -m samm_server.app --model-dir checkpoints --device cpu
python -m samm_server.app --idle-timeout-seconds 600 --idle-check-seconds 10
```

## Workers

The gateway starts one worker process for the selected backend when a weight is
prepared. It builds the command through `RuntimeRegistry`:

```bash
pixi run -e <environment> python -m samm_server.worker \
  --host 127.0.0.1 \
  --port 8801 \
  --model-dir checkpoints \
  --device cuda
```

Backend-to-environment mapping:

| Backend | Pixi environment |
| --- | --- |
| `sam1` | `sam1` |
| `sam2` | `sam2` |
| `mobile_sam` | `mobile-sam` |
| `medsam` | `medsam` |
| `medsam_text` | `medsam` |
| `medsam2` | `medsam2` |
| `sam3` | `sam3` |
| `medical_sam3` | `medical-sam3` |
| `fastsam` | `fastsam` |

`SAMM_PIXI` can override the Pixi executable used by the gateway. Otherwise the
service uses `pixi` from `PATH`, then `~/.pixi/bin/pixi`.

Equivalent manual worker tasks also exist:

```bash
pixi run -e sam1 worker-sam1
pixi run -e sam2 worker-sam2
pixi run -e mobile-sam worker-mobile-sam
pixi run -e medsam worker-medsam
pixi run -e medsam2 worker-medsam2
pixi run -e sam3 worker-sam3
pixi run -e medical-sam3 worker-medical-sam3
pixi run -e fastsam worker-fastsam
```

These worker commands are mainly useful for development. In normal use, the
gateway launched by Slicer starts and stops workers for you. The worker owns the
prepared model and cached embeddings. `POST /offload` stops the active worker
and releases model memory while keeping the gateway alive.

Only one backend worker and prepared weight are active at a time. Preparing a
weight from another backend replaces the worker. Finish or cancel worker-local
prediction, embedding, and temporal jobs before preparing another backend or
offloading; finetuning jobs run as separate server-managed subprocesses and
keep the otherwise-idle gateway alive.

## Setup

Run setup commands from a terminal before using a backend in Slicer. Setup
installs upstream sources, environments, and checkpoints; it does not start the
Slicer-facing service:

```bash
pixi run setup-variants
```

Or install one backend at a time:

```bash
pixi run setup-sam1
pixi run setup-sam2
pixi run setup-mobile-sam
pixi run setup-medsam
pixi run setup-medsam2
pixi run setup-sam3
pixi run setup-medical-sam3
pixi run setup-fastsam
```

SAM 3 requires Hugging Face approval before setup can download the gated
checkpoint. Medical-SAM3 downloads its public 2D Hugging Face checkpoint with
`hf download`. FastSAM and MedSAM user-managed checkpoints are reported with red
placement instructions when missing.

## Model Capabilities

`GET /models` returns model families, weights, checkpoint availability, and
prompt capabilities. Slicer uses this payload to enable or gray out prompt
controls. Each model reports the union of its weight capabilities, and each
weight reports its own backend and checkpoint state.

You normally see this through the Slicer model and weight selectors. Query the
endpoint directly only for API testing or debugging.

```json
{
  "models": [
    {
      "id": "sam1",
      "label": "SAM 1",
      "capabilities": {
        "auto_predict_2d": true,
        "box": true,
        "box_3d": true,
        "embeddings": true,
        "mask": true,
        "points": true,
        "text": false,
        "video_propagation": false
      },
      "weights": [
        {
          "id": "sam_vit_b",
          "model_id": "sam1",
          "label": "ViT-B",
          "checkpoint": "sam_vit_b_01ec64.pth",
          "backend": "sam1",
          "model_type": "vit_b",
          "available": true,
          "capabilities": {
            "points": true,
            "box": true,
            "box_3d": true,
            "mask": true,
            "text": false,
            "auto_predict_2d": true,
            "embeddings": true,
            "video_propagation": false
          }
        }
      ]
    }
  ]
}
```

Prompt support by backend:

| Backend | Environment | Prompt support |
| --- | --- | --- |
| `sam1` | `sam1` | Points, box, 3D box, mask input, auto 2D, embeddings |
| `sam2` | `sam2` | Points, box, 3D box, mask input, auto 2D, 3D box and auto slices-as-video, embeddings |
| `mobile_sam` | `mobile-sam` | Points, box, 3D box, mask input, auto 2D, embeddings |
| `medsam` | `medsam` | Box, 3D box, auto 2D, embeddings |
| `medsam_text` | `medsam` | Text, auto 2D, embeddings |
| `medsam2` | `medsam2` | Points, box, 3D box, mask input, auto 2D, 3D box and auto slices-as-video, embeddings |
| `sam3` | `sam3` | Text, text+box, points, box, 3D box, 3D box and auto slices-as-video, mask input for 2D prediction, auto 2D, embeddings |
| `medical_sam3` | `medical-sam3` | Text, text+box, box, 3D box, auto 2D, embeddings |
| `fastsam` | `fastsam` | Text, points, box, 3D box, auto 2D without embeddings |

The `video_propagation` capability is currently `true` for the `sam2`,
`medsam2`, and `sam3` backends. It reports availability of the protocol 1.2 temporal job
API used by the additional Slicer 3D-box and auto slices-as-video controls. The
original 3D-box and scrolling auto-predict controls retain their independent 2D
behavior.

SAM 3 also reports `video_text: true` and `video_mask: false`. Its upstream
video API accepts text, box, text+box, and point refinement, but does not expose
mask seeding. SAMM can refine a box-selected object with points and unions any
multiple tracked SAM 3 objects into the single binary mask returned by the
temporal job. The video model's detector is reused for SAMM image prediction,
so preparing SAM 3 does not load a second copy of the 3.3 GB checkpoint.

Medical-SAM3's upstream 2D evaluation exercises box and text prompts separately.
SAMM also supports the processor's text+box combination. Point and mask-input
prompts are disabled because the 2D checkpoint does not include the separate
SAM3 interactive-head weights they require. The listed 3D-box support is SAMM's
per-slice 2D-box workflow and does not use Medical-SAM3's upstream 3D checkpoint.

## Endpoints

Core endpoints:

```text
GET  /health
GET  /models
GET  /models/<weight_id>
GET  /prepared
POST /session
POST /models/<weight_id>/prepare
POST /offload
POST /predict
```

Embedding endpoints:

```text
POST /embeddings
POST /embedding-files/save
POST /embedding-files/load
POST /embedding-jobs
POST /embedding-jobs/<job_id>/items
GET  /embedding-jobs/<job_id>
```

Bulk prediction endpoints:

```text
POST /prediction-jobs
POST /prediction-jobs/<job_id>/items
GET  /prediction-jobs/<job_id>
```

Temporal prediction endpoints:

```text
POST /video-prediction-jobs
POST /video-prediction-jobs/<job_id>/frames
POST /video-prediction-jobs/<job_id>/run
GET  /video-prediction-jobs/<job_id>?cursor=<count>
POST /video-prediction-jobs/<job_id>/cancel
```

Finetuning endpoints:

```text
GET  /finetuning/datasets
GET  /finetuning/datasets/<name>
POST /finetuning/datasets/<name>/build
GET  /finetuning/reports/<run>
POST /finetuning/train
POST /finetuning/eval
GET  /finetuning/jobs/<job_id>
POST /finetuning/jobs/<job_id>/cancel
```

Dataset and run names must be non-empty and use only letters, numbers, dots,
dashes, and underscores.

## Health And Sessions

`GET /health` returns service and protocol versions:

```json
{
  "name": "SAMM service",
  "version": "1.2.0",
  "status": "ready",
  "protocol_version": "1.2.0"
}
```

Slicer posts a heartbeat to `POST /session` with a stable `session_id` and
current autosave paths:

```json
{
  "session_id": "slicer-session",
  "autosave": {
    "segmentation": "autosave/volume_segmentation.seg.nrrd",
    "embeddings": "autosave/volume_sam_vit_b_embeddings"
  }
}
```

The response includes `status`, `session_id`, `active_sessions`, and
`timeout_seconds`. The gateway exits after 10 minutes without a session
heartbeat unless a finetuning job is still queued or running.

## Prepare A Model

Prepare by weight id:

```text
POST /models/sam_vit_b/prepare
```

In normal use, click `Prepare model` in Slicer. That calls this endpoint. The
gateway checks checkpoint availability, launches the matching worker
environment, loads the model, and returns prepared model status:

```json
{
  "model_id": "sam1",
  "weight_id": "sam_vit_b",
  "status": "prepared",
  "checkpoint": "sam_vit_b_01ec64.pth",
  "backend": "sam1",
  "model_type": "vit_b",
  "device": "cuda",
  "capabilities": {
    "points": true,
    "box": true,
    "box_3d": true,
    "mask": true,
    "text": false,
    "auto_predict_2d": true,
    "embeddings": true,
    "video_propagation": false
  }
}
```

Use `GET /prepared` to read the current prepared state. When no worker is
prepared it returns `{"prepared": null}`.

## Prediction

Prediction requests require at least one point, box, mask, or text prompt.
Requests can use either a direct image payload or a cached embedding id.

In normal use, Slicer builds these payloads from the selected volume, prompts,
segment, and view. Send requests manually only when testing the service API.

Direct image prediction:

```json
{
  "image": {"shape": [512, 512, 3], "data": "base64 uint8 RGB bytes"},
  "points": [[120, 140]],
  "labels": [1],
  "box": [80, 90, 180, 210]
}
```

Cached embedding prediction:

```json
{
  "embedding_id": "abc123",
  "points": [[120, 140]],
  "labels": [1]
}
```

Mask prompt:

```json
{
  "image": {"shape": [512, 512, 3], "data": "base64 uint8 RGB bytes"},
  "mask": {"shape": [512, 512], "data": "base64 uint8 mask bytes"}
}
```

SAM 3 and Medical-SAM3 text grounding:

```json
{
  "image": {"shape": [512, 512, 3], "data": "base64 uint8 RGB bytes"},
  "text": "kidney",
  "box": [70, 80, 220, 260]
}
```

Response:

```json
{"shape": [512, 512], "data": "base64 uint8 mask bytes"}
```

Prompt notes:

- `box` uses `[x0, y0, x1, y1]` image coordinates.
- Point `labels` use `1` for positive and `0` for negative.
- Mask prompt shape must match the direct image shape or cached embedding shape.
- SAM 3 text can be used alone or with a box; it cannot be combined with points or mask input.
- Medical-SAM3 accepts box-only, text-only, or text+box grounding. Its released 2D checkpoint does not support point or mask prompts.
- MedSAM base requires a box prompt.
- MedSAM Text accepts text-only FLARE22 organ prompts.
- FastSAM accepts exactly one prompt mode: text, points, or box.

## Embeddings

Embedding requests require a prepared embedding-capable model:

```json
{
  "image": {"shape": [512, 512, 3], "data": "base64 uint8 RGB bytes"}
}
```

Response:

```json
{"embedding_id": "abc123", "shape": [512, 512]}
```

Embeddings are cached in the active worker by image shape and digest. Saved
embedding folders contain `metadata.json` and backend tensor data such as
`embeddings.pt`.

In normal use, use `Embed view`, `Embed all axes`, `Save embeddings`, and `Load
embeddings` in Slicer.

Save embeddings:

```json
{
  "path": "embeddings/volume_sam_vit_b",
  "items": [
    {
      "embedding_id": "abc123",
      "slice_spec": {"view": "Red", "axis": 0, "index": 12},
      "image_shape": [512, 512, 3],
      "image_digest": "sha256"
    }
  ]
}
```

Load embeddings:

```json
{"path": "embeddings/volume_sam_vit_b"}
```

Bulk embedding starts a job:

```json
{"total": 12}
```

Then submit items:

```json
{
  "items": [
    {
      "key": "red-12",
      "slice_spec": {"view": "Red", "axis": 0, "index": 12},
      "image_shape": [512, 512, 3],
      "image_digest": "sha256",
      "image": {"shape": [512, 512, 3], "data": "base64 uint8 RGB bytes"}
    }
  ]
}
```

Jobs move through `queued`, `running`, `complete`, or `failed`. A job closes
automatically once `submitted == total`.

## Bulk Prediction

Slicer uses prediction jobs for 3D ROI box prediction. In normal use, click
`Predict 3D box`; the API sequence below is for service testing. Start a job:

```json
{"total": 12}
```

Submit slice items:

```json
{
  "items": [
    {
      "key": "red-12",
      "slice_spec": {"view": "Red", "axis": 0, "index": 12},
      "image": {"shape": [512, 512, 3], "data": "base64 uint8 RGB bytes"},
      "points": [],
      "labels": [],
      "box": [80, 90, 180, 210]
    }
  ]
}
```

Poll `GET /prediction-jobs/<job_id>`. Results include `key`, `slice_spec`,
mask `shape`, and base64 mask `data`.

## Temporal Prediction Jobs

Protocol 1.2 provides a separate temporal job lifecycle for SAM 2.1, MedSAM2,
and SAM 3.
All frames are uploaded before inference because the predictors initialize the
complete ordered sequence, then mask results become available incrementally.
The service keeps the existing independent-slice prediction jobs unchanged.

Start an uploading job:

```json
{"total": 12}
```

Upload one or more numbered frames. Frame indices are zero-based, unique, and
must cover the complete range from `0` through `total - 1`; all RGB images must
have the same shape.

```json
{
  "frames": [
    {
      "frame_index": 0,
      "key": "red-12",
      "slice_spec": {"view": "Red", "axis": 0, "index": 12},
      "image": {"shape": [512, 512, 3], "data": "base64 uint8 RGB bytes"}
    }
  ]
}
```

After every frame is uploaded, explicitly start propagation. The prompt frame
uses the temporal frame index, not the Slicer array index.

```json
{
  "prompt": {
    "frame_index": 5,
    "points": [],
    "labels": [],
    "box": [80, 90, 180, 210]
  },
  "direction": "both",
  "offload_video_to_cpu": true,
  "offload_state_to_cpu": false
}
```

`direction` may be `both`, `forward`, or `backward`. A point, box, mask, or text
prompt is required. A mask prompt uses the same `{shape, data}` representation
as direct prediction and must match the frame shape. Text is accepted only when
the prepared model reports `video_text: true`; mask input is accepted only when
`video_mask` is not false. For SAM 3, text may be combined with a box but not
with points, and mask seeds are unavailable.

Poll with a consumed-result cursor:

```text
GET /video-prediction-jobs/<job_id>?cursor=0
```

The response returns only `results[cursor:]` and sets `next_cursor` to the
number of results produced so far. Each result contains `sequence`,
`frame_index`, `key`, `slice_spec`, mask `shape`, and base64 mask `data`.
Jobs move through `uploading`, `queued`, `running`, and `complete`, or terminate
as `failed` or `cancelled`. Cancel with:

```text
POST /video-prediction-jobs/<job_id>/cancel
```

Cancellation is checked between propagated frames; inference already running
for the current frame is allowed to finish. Uploaded frames and temporal state
remain private to the worker job and are not returned by polling.

Temporal jobs are worker-local. Wait for a job to reach a terminal state before
preparing a model from another backend or calling `/offload`; either operation
restarts/stops the worker and invalidates its job IDs.

The worker accepts at most four temporal jobs, 4096 frames per job, and 1 GiB
of encoded RGB frame data per job. Terminal jobs and abandoned uploads are
retained for ten minutes; starting a new job may evict the oldest terminal job
when the job limit is reached. Worker-held input frame bytes are released when
inference starts (or when an upload is cancelled). Preprocessing resizes and normalizes frames in batches of eight to avoid
materializing both the full source and normalized stacks at once; the complete
normalized tensor remains resident because the video predictor requires it.

The Slicer `Predict 3D box (Slices as video)` workflow uses this API for SAM 2.1,
MedSAM2, and SAM 3. It uploads only the ROI slice range, seeds the middle frame with the
projected box, propagates in both directions, polls by cursor, and clips masks
to the original ROI before writing them into the selected segment.

`Auto predict (Slices as video)` uploads the complete selected view and seeds
the current slice with its enabled supported prompt. It propagates in both
directions and writes cursor-polled results without ROI clipping. SAM 2.1 and
MedSAM2 accept points, a 2D box, or an input mask; mask seeds cannot be combined
with point or box seeds. SAM 3 accepts text, a 2D box, text+box, points, or
points+box, but not an input-mask seed.

## Finetuning Datasets

List datasets:

```text
GET /finetuning/datasets
```

In normal use, the Slicer `Finetune` panel exports source segmented volumes and
calls the dataset build endpoint. Use the endpoint directly only for API testing
or automation around the service.

Dataset status includes the source and exported split counts:

```json
{
  "name": "my_task",
  "source": "segmented_volumes/my_task",
  "dataset": "datasets/my_task",
  "source_segmented_volumes": 4,
  "train_segmented_volumes": 3,
  "val_segmented_volumes": 1
}
```

Build a dataset from `segmented_volumes/<name>`:

```json
{
  "val_count": 1,
  "seed": 0,
  "axes": [0, 1, 2],
  "window": [-100, 300]
}
```

`window` can be omitted to use percentile windowing in the exporter.

## Finetuning Jobs

Training and eval jobs run server-side so Slicer can stay responsive. In normal
use, click `Run training`, `Run eval`, `Cancel job`, and `Show report` in the
Slicer `Finetune` panel. The JSON examples below are for API testing or
automation. Start training:

```json
{
  "dataset": "my_task",
  "run": "my_task_v1",
  "checkpoint": "checkpoints/sam2.1_hiera_tiny.pt",
  "epochs": 25,
  "batch_size": 1,
  "num_workers": 0,
  "num_frames": 4,
  "max_objects": 3
}
```

Start eval:

```json
{
  "dataset": "my_task",
  "run": "my_task_v1",
  "checkpoint": "finetuning_runs/my_task_v1/checkpoints/checkpoint.pt",
  "prompts": ["box", "point", "box-point"],
  "max_segmented_volumes": 2
}
```

`checkpoint`, `run`, `prompts`, `max_objects`, and `max_segmented_volumes` are
optional where defaults are useful. Training defaults to run `<dataset>_v1`,
`checkpoints/sam2.1_hiera_tiny.pt`, 25 epochs, batch size 1, 0 workers, and 4
frames. Eval defaults to the run checkpoint and all three prompt modes.

Job state includes status, command, log path, latest log tail, return code,
error, and progress:

```json
{
  "job_id": "job123",
  "kind": "train",
  "status": "running",
  "log": "logs/finetuning_jobs/my_task_v1_train.log",
  "progress": {
    "mode": "determinate",
    "current": 12000,
    "total": 25000,
    "label": "12 / 25 epochs, batch 1 / 1"
  }
}
```

Eval jobs report a busy progress mode while queued or running. Jobs can be
`queued`, `running`, `complete`, `failed`, or `canceled`.

`GET /finetuning/reports/<run>` returns the same text summary produced by the
`finetune-report` CLI task.

## Tests

Run the service unit and integration-safe tests from the project root:

```bash
pixi run test-service
```

Run the complete CPU-safe suite, including finetuning tests:

```bash
pixi run check
```

Run the Slicer-side protocol and widget self-tests in an installed Slicer:

```bash
pixi run test-slicer
```

GitHub Actions runs `pixi run check` and a checksum-pinned Slicer 5.12.2 native
integration job. Ordinary CI does not download model checkpoints, install
variant GPU environments, start the gateway, or run CUDA inference.

## Related Documentation

- [../README.md](../README.md): installation and complete Slicer workflow.
- [../sam_variants/README.md](../sam_variants/README.md): backend setup, checkpoints, and capabilities.
- [../finetuning/README.md](../finetuning/README.md): dataset, training, evaluation, and report workflows.
