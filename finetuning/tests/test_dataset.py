from argparse import Namespace
from pathlib import Path
import json
import tempfile
import unittest

import numpy as np

from finetuning.dataset import export_dataset
from finetuning.eval_medsam2 import segmented_volume_paths, dice, eval_segmented_volume, mask_point, summarize
from finetuning.example_data import segmented_volume_name, msd_pairs, write_segmented_volume_npzs
from finetuning.report import collect_report, format_report
from finetuning.train_medsam2 import dataset_folder, model_registration
from samm.SegmentAnyMedicalModel.samm_lib.finetune_export import merge_label_masks


class FinetuningDatasetTest(unittest.TestCase):
    def test_export_npz_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "segmented_volume.npz"
            output = root / "dataset"
            np.savez(source, imgs=np.array([[[-1, 0], [1, 2]]], dtype=np.float32), gts=np.array([[[0, 1], [1, 0]]], dtype=np.uint8))

            export_dataset(Namespace(
                input=source,
                image=None,
                mask=None,
                output=output,
                name="segmented volume one",
                split="train",
                val_count=0,
                val_fraction=0.0,
                seed=0,
                window=[-1, 2],
                percentile_window=(0.5, 99.5),
            ))

            exported = np.load(output / "train_npz" / "segmented_volume_one.npz")
            manifest = json.loads((output / "dataset.json").read_text(encoding="utf-8"))
            self.assertEqual(exported["imgs"].dtype, np.uint8)
            self.assertEqual(exported["imgs"].shape, (1, 2, 2))
            self.assertEqual(exported["gts"].tolist(), [[[0, 1], [1, 0]]])
            self.assertEqual(manifest["splits"]["train"][0]["name"], "segmented_volume_one")

    def test_export_npz_dataset_with_validation_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "segmented_volumes"
            output = root / "dataset"
            source.mkdir()
            for index in range(4):
                np.savez(
                    source / f"segmented_volume_{index}.npz",
                    imgs=np.full((1, 2, 2), index, dtype=np.uint8),
                    gts=np.array([[[0, 1], [1, 0]]], dtype=np.uint8),
                )

            export_dataset(Namespace(
                input=source,
                image=None,
                mask=None,
                output=output,
                name=None,
                split="train",
                val_count=1,
                val_fraction=0.0,
                seed=0,
                window=None,
                percentile_window=(0.5, 99.5),
            ))

            manifest = json.loads((output / "dataset.json").read_text(encoding="utf-8"))
            self.assertEqual(len(list((output / "train_npz").glob("*.npz"))), 3)
            self.assertEqual(len(list((output / "val_npz").glob("*.npz"))), 1)
            self.assertEqual(len(manifest["splits"]["train"]), 3)
            self.assertEqual(len(manifest["splits"]["val"]), 1)

    def test_export_npz_dataset_with_axis_augmentation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "segmented_volume.npz"
            output = root / "dataset"
            image = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
            mask = np.zeros((2, 3, 4), dtype=np.uint8)
            mask[1, 2, 3] = 1
            np.savez(source, imgs=image, gts=mask)

            export_dataset(Namespace(
                input=source,
                image=None,
                mask=None,
                output=output,
                name="segmented_volume",
                split="train",
                val_count=0,
                val_fraction=0.0,
                seed=0,
                axes=[0, 1, 2],
                window=[0, 23],
                percentile_window=(0.5, 99.5),
            ))

            axis0 = np.load(output / "train_npz" / "segmented_volume_axis0.npz")
            axis1 = np.load(output / "train_npz" / "segmented_volume_axis1.npz")
            axis2 = np.load(output / "train_npz" / "segmented_volume_axis2.npz")
            manifest = json.loads((output / "dataset.json").read_text(encoding="utf-8"))

            self.assertEqual(axis0["imgs"].shape, (2, 3, 4))
            self.assertEqual(axis1["imgs"].shape, (3, 2, 4))
            self.assertEqual(axis2["imgs"].shape, (4, 2, 3))
            self.assertEqual(axis1["gts"].tolist(), np.moveaxis(mask, 1, 0).tolist())
            self.assertEqual(axis2["gts"].tolist(), np.moveaxis(mask, 2, 0).tolist())
            self.assertEqual([item["axis"] for item in manifest["splits"]["train"]], [0, 1, 2])

    def test_dataset_folder_prefers_train_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train_npz").mkdir()
            self.assertEqual(dataset_folder(root), root / "train_npz")

    def test_msd_pairs_match_images_and_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "Task09_Spleen"
            (root / "imagesTr").mkdir(parents=True)
            (root / "labelsTr").mkdir()
            image = root / "imagesTr" / "spleen_1.nii.gz"
            label = root / "labelsTr" / "spleen_1.nii.gz"
            image.touch()
            label.touch()
            (root / "imagesTr" / "._spleen_10.nii.gz").touch()
            (root / "labelsTr" / "._spleen_10.nii.gz").touch()

            self.assertEqual(msd_pairs(root, 1), [(image, label)])
            self.assertEqual(segmented_volume_name(image), "spleen_1")

    def test_write_example_segmented_volume_npz(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "imagesTr" / "spleen_1.nii.gz"
            label = root / "labelsTr" / "spleen_1.nii.gz"
            output = root / "segmented_volumes"
            image.parent.mkdir()
            label.parent.mkdir()

            def read(path):
                if "labelsTr" in path.parts:
                    return np.array([[[0, 1], [1, 0]]], dtype=np.uint8)
                return np.array([[[10, 20], [30, 40]]], dtype=np.float32)

            write_segmented_volume_npzs([(image, label)], output, read)
            segmented_volume = np.load(output / "spleen_1.npz")

        self.assertEqual(segmented_volume["imgs"].dtype, np.float32)
        self.assertEqual(segmented_volume["gts"].dtype, np.uint16)
        self.assertEqual(segmented_volume["gts"].tolist(), [[[0, 1], [1, 0]]])

    def test_medsam2_registration_metadata_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "My_Task"
            metadata = model_registration(Namespace(
                name="My Task",
                weight_id=None,
                label=None,
                registered_checkpoint=None,
                inference_config="configs/custom.yaml",
            ), output)

        self.assertEqual(metadata["id"], "medsam2_my_task")
        self.assertEqual(metadata["label"], "My Task")
        self.assertEqual(metadata["model_id"], "medsam2")
        self.assertEqual(metadata["backend"], "medsam2")
        self.assertEqual(metadata["checkpoint"], str((output / "checkpoints" / "checkpoint.pt").resolve()))
        self.assertEqual(metadata["model_type"], "configs/custom.yaml")

    def test_dice(self):
        pred = np.array([[True, False], [True, False]])
        gt = np.array([[True, True], [False, False]])
        self.assertEqual(dice(pred, gt), 0.5)

    def test_eval_segmented_volume_supports_prompt_modes(self):
        class Predictor:
            def __init__(self):
                self.calls = []
                self.shape = None

            def set_image(self, image):
                self.shape = image.shape[:2]

            def predict(self, **kwargs):
                self.calls.append(kwargs)
                return np.zeros((1, *self.shape), dtype=bool), np.array([1.0]), None

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "segmented_volume.npz"
            image = np.zeros((1, 4, 5), dtype=np.uint8)
            mask = np.zeros((1, 4, 5), dtype=np.uint8)
            mask[0, 1:3, 2:4] = 1
            np.savez(path, imgs=image, gts=mask)

            predictor = Predictor()
            items = eval_segmented_volume(path, predictor, np, ("box", "point", "box-point"))

        self.assertEqual([item["prompt"] for item in items], ["box", "point", "box-point"])
        self.assertEqual(items[0]["box"], [2, 1, 4, 3])
        self.assertIn("box", predictor.calls[0])
        self.assertNotIn("point_coords", predictor.calls[0])
        self.assertIn("point_coords", predictor.calls[1])
        self.assertNotIn("box", predictor.calls[1])
        self.assertIn("box", predictor.calls[2])
        self.assertIn("point_coords", predictor.calls[2])

    def test_mask_point_is_foreground_centroid(self):
        mask = np.zeros((4, 5), dtype=bool)
        mask[1:3, 2:4] = True

        self.assertIn(mask_point(mask, np), ([2, 1], [3, 1], [2, 2], [3, 2]))

    def test_summarize_groups_by_prompt(self):
        summary = summarize([
            {"prompt": "box", "dice": 1.0},
            {"prompt": "box", "dice": 0.0},
            {"prompt": "point", "dice": 0.25},
        ])

        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["prompts"]["box"], {"count": 2, "mean_dice": 0.5})
        self.assertEqual(summary["prompts"]["point"], {"count": 1, "mean_dice": 0.25})

    def test_segmented_volume_paths_requires_requested_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root / "dataset" / "train_npz"
            train.mkdir(parents=True)
            segmented_volume = train / "segmented_volume.npz"
            segmented_volume.touch()

            self.assertEqual(segmented_volume_paths(root / "dataset", 0, "train"), [segmented_volume])
            with self.assertRaises(FileNotFoundError):
                segmented_volume_paths(root / "dataset", 0, "val")

    def test_merge_label_masks(self):
        labels, segments = merge_label_masks((1, 2, 3), (
            ("a", "A", np.array([[[0, 1, 0], [0, 0, 0]]])),
            ("empty", "Empty", np.zeros((1, 2, 3), dtype=np.uint8)),
            ("b", "B", np.array([[[0, 0, 0], [1, 0, 1]]])),
        ))

        self.assertEqual(labels.tolist(), [[[0, 1, 0], [2, 0, 2]]])
        self.assertEqual([segment["name"] for segment in segments], ["A", "B"])

    def test_merge_label_masks_rejects_overlap(self):
        masks = (
            ("a", "A", np.array([[[1, 0]]])),
            ("b", "B", np.array([[[1, 0]]])),
        )

        with self.assertRaises(ValueError):
            merge_label_masks((1, 1, 2), masks)

    def test_finetuning_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "finetuning_runs" / "tumor_v1"
            dataset = root / "datasets" / "tumor" / "train_npz"
            checkpoint = run / "checkpoints" / "checkpoint.pt"
            (run / "logs").mkdir(parents=True)
            dataset.mkdir(parents=True)
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"checkpoint")
            (dataset / "segmented_volume.npz").touch()
            (run / "samm_model.json").write_text(json.dumps({
                "id": "medsam2_tumor_v1",
                "label": "tumor_v1",
                "checkpoint": str(checkpoint),
            }), encoding="utf-8")
            (run / "samm_run.json").write_text(json.dumps({
                "dataset": str(dataset),
                "checkpoint": "base.pt",
            }), encoding="utf-8")
            (run / "config.yaml").write_text("  num_epochs: 25\n", encoding="utf-8")
            (run / "logs" / "train_stats.json").write_text(
                json.dumps({"Trainer/epoch": 0, "Trainer/steps_train": 1, "Losses/train_all_loss": 2.0}) + "\n"
                + json.dumps({"Trainer/epoch": 1, "Trainer/steps_train": 2, "Losses/train_all_loss": 1.0}) + "\n",
                encoding="utf-8",
            )
            (run / "eval.json").write_text(json.dumps({
                "split": "val",
                "summary": {
                    "count": 2,
                    "mean_dice": 0.75,
                    "prompts": {"box": {"count": 1, "mean_dice": 0.8}},
                },
            }), encoding="utf-8")

            text = format_report(collect_report(run))

        self.assertIn("Run: tumor_v1", text)
        self.assertIn("Weight: medsam2_tumor_v1 (tumor_v1)", text)
        self.assertIn("Checkpoint: present", text)
        self.assertIn("Configured epochs: 25", text)
        self.assertIn("final loss: 1.0000", text)
        self.assertIn("eval.json [val]", text)
        self.assertIn("box: n=1 mean_dice=0.8000", text)


if __name__ == "__main__":
    unittest.main()
