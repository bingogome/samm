from pathlib import Path
import json
import tempfile
import time
import unittest

import numpy as np

from samm_server.finetuning import FinetuningService, job_progress


class FinetuningServiceTest(unittest.TestCase):
    def test_status_and_build_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_source_segmented_volume(root, "tumor", "segmented_volume001")
            service = self.service(root)

            status = service.dataset_status("tumor")
            datasets = service.datasets()
            build = service.build_dataset("tumor", {"val_count": 0, "axes": [0, 1], "window": [0, 7]})

        self.assertEqual(status.status_code, 200)
        self.assertEqual(datasets.payload["datasets"][0]["name"], "tumor")
        self.assertEqual(status.payload["source_segmented_volumes"], 1)
        self.assertEqual(build.status_code, 200)
        self.assertEqual(build.payload["status"]["train_segmented_volumes"], 2)
        self.assertIn("exported 2 train segmented volume(s)", build.payload["output"])

    def test_build_reports_missing_source_segmented_volumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.service(Path(tmp)).build_dataset("missing", {"axes": [0]})

        self.assertEqual(result.status_code, 400)
        self.assertIn("No source segmented volumes", result.payload["error"])

    def test_lists_already_built_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "datasets" / "tumor" / "train_npz").mkdir(parents=True)
            (root / "datasets" / "tumor" / "train_npz" / "segmented_volume001.npz").touch()
            result = self.service(root).datasets()

        self.assertEqual(result.payload["datasets"][0]["name"], "tumor")
        self.assertEqual(result.payload["datasets"][0]["source_segmented_volumes"], 0)
        self.assertEqual(result.payload["datasets"][0]["train_segmented_volumes"], 1)

    def test_training_progress_parses_epoch_log(self):
        progress = job_progress(
            {"kind": "train", "status": "running", "command": ["--epochs", "25"]},
            "INFO Train Epoch: [24][0/1] | Losses/train_all_loss: 1.64e+00",
        )

        self.assertEqual(progress["mode"], "determinate")
        self.assertEqual(progress["current"], 25000)
        self.assertEqual(progress["total"], 25000)
        self.assertEqual(progress["label"], "25 / 25 epochs, batch 1 / 1")

    def test_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_run(root, "tumor_v1")
            result = self.service(root).report("tumor_v1")

        self.assertEqual(result.status_code, 200)
        self.assertIn("Run: tumor_v1", result.payload["output"])
        self.assertIn("Weight: medsam2_tumor_v1", result.payload["output"])

    def test_report_rejects_missing_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.service(Path(tmp)).report("missing")

        self.assertEqual(result.status_code, 404)
        self.assertIn("finetuning run not found", result.payload["error"])

    def service(self, root):
        model_dir = root / "checkpoints"
        model_dir.mkdir(exist_ok=True)
        return FinetuningService(model_dir, pixi="/bin/true")

    def test_training_job_runs_server_side_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = self.service(Path(tmp))
            result = service.start_train({
                "dataset": "tumor",
                "run": "tumor_v1",
                "checkpoint": "checkpoints/sam2.1_hiera_tiny.pt",
                "epochs": 1,
                "batch_size": 1,
                "num_workers": 0,
                "num_frames": 2,
            })
            state = self.wait_for_terminal_state(service, result.payload["job_id"])

        self.assertEqual(result.status_code, 202)
        self.assertEqual(state.payload["status"], "complete")
        self.assertEqual(state.payload["kind"], "train")
        self.assertIn("finetune-medsam2", state.payload["command"])
        self.assertEqual(state.payload["progress"]["current"], state.payload["progress"]["total"])

    def test_eval_job_runs_server_side_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = self.service(Path(tmp))
            result = service.start_eval({"dataset": "tumor", "run": "tumor_v1", "prompts": ["box", "point"]})
            state = self.wait_for_terminal_state(service, result.payload["job_id"])

        self.assertEqual(result.status_code, 202)
        self.assertEqual(state.payload["status"], "complete")
        self.assertEqual(state.payload["kind"], "eval")
        self.assertIn("eval-finetuned", state.payload["command"])

    def wait_for_terminal_state(self, service, job_id):
        state = None
        for _ in range(50):
            state = service.job(job_id)
            if state.payload["status"] in ("complete", "failed", "canceled"):
                return state
            time.sleep(0.01)
        return state

    def write_source_segmented_volume(self, root, dataset, name):
        path = root / "segmented_volumes" / dataset / f"{name}.npz"
        path.parent.mkdir(parents=True)
        image = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
        labels = np.zeros((2, 2, 2), dtype=np.uint16)
        labels[:, 0, 0] = 1
        np.savez_compressed(path, imgs=image, gts=labels)

    def write_run(self, root, name):
        run = root / "finetuning_runs" / name
        dataset = root / "datasets" / "tumor" / "train_npz"
        checkpoint = run / "checkpoints" / "checkpoint.pt"
        (run / "logs").mkdir(parents=True)
        dataset.mkdir(parents=True)
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()
        (dataset / "segmented_volume001.npz").touch()
        (run / "samm_model.json").write_text(json.dumps({
            "id": "medsam2_tumor_v1",
            "label": "tumor_v1",
            "model_id": "medsam2",
            "backend": "medsam2",
            "checkpoint": "finetuning_runs/tumor_v1/checkpoints/checkpoint.pt",
            "model_type": "configs/sam2.1_hiera_t512.yaml",
        }), encoding="utf-8")
        (run / "samm_run.json").write_text(json.dumps({
            "variant": "medsam2",
            "dataset": str(dataset),
            "checkpoint": str(checkpoint),
            "config": "configs/samm/tumor_v1.yaml",
            "command": [],
        }), encoding="utf-8")
        (run / "config.yaml").write_text("trainer:\n  num_epochs: 2\n", encoding="utf-8")
        (run / "logs" / "train_stats.json").write_text(
            json.dumps({"Trainer/epoch": 0, "Trainer/steps_train": 1, "Losses/train_all_loss": 1.5}) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
