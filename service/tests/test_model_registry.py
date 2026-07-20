from pathlib import Path
import json
import tempfile
import unittest

from samm_server.model_registry import MODEL_DEFINITIONS
from samm_server.protocol import models_payload, weight_payload


class ModelRegistryTest(unittest.TestCase):
    def test_empty_model_dir_returns_all_unavailable(self):
        with tempfile.TemporaryDirectory() as model_dir:
            models = models_payload(model_dir)["models"]

        self.assertEqual(len(models), len(MODEL_DEFINITIONS))
        weights = [weight for model in models for weight in model["weights"]]
        self.assertEqual([weight["available"] for weight in weights], [False] * len(weights))

    def test_checkpoint_file_marks_model_available(self):
        model = MODEL_DEFINITIONS[0]
        weight = model.weights[0]
        with tempfile.TemporaryDirectory() as model_dir:
            Path(model_dir, weight.checkpoint).touch()
            models = models_payload(model_dir)["models"]

        weights = [weight for model in models for weight in model["weights"]]
        self.assertEqual(weights[0]["id"], weight.id)
        self.assertEqual(weights[0]["checkpoint"], weight.checkpoint)
        self.assertEqual(weights[0]["backend"], weight.backend)
        self.assertEqual(weights[0]["model_type"], weight.model_type)
        self.assertEqual(weights[0]["capabilities"]["points"], True)
        self.assertEqual(weights[0]["capabilities"]["box"], True)
        self.assertEqual(weights[0]["capabilities"]["box_3d"], True)
        self.assertEqual(weights[0]["capabilities"]["mask"], True)
        self.assertEqual(weights[0]["capabilities"]["text"], False)
        self.assertEqual(weights[0]["capabilities"]["auto_predict_2d"], True)
        self.assertEqual(weights[0]["capabilities"]["embeddings"], True)
        self.assertEqual(weights[0]["capabilities"]["video_propagation"], False)
        self.assertTrue(weights[0]["available"])
        self.assertEqual([item["available"] for item in weights[1:]], [False] * (len(weights) - 1))

    def test_selected_weight_payload(self):
        model = MODEL_DEFINITIONS[0]
        weight = model.weights[0]
        with tempfile.TemporaryDirectory() as model_dir:
            Path(model_dir, weight.checkpoint).touch()
            payload = weight_payload(model_dir, weight.id)

        self.assertEqual(payload["model_id"], model.id)
        self.assertEqual(payload["id"], weight.id)
        self.assertEqual(payload["backend"], "sam1")
        self.assertEqual(payload["model_type"], "vit_b")
        self.assertEqual(payload["capabilities"]["points"], True)
        self.assertEqual(payload["capabilities"]["box"], True)
        self.assertTrue(payload["available"])

    def test_model_payload_reports_union_capabilities(self):
        with tempfile.TemporaryDirectory() as model_dir:
            models = models_payload(model_dir)["models"]

        sam1 = models[0]
        sam2 = models[1]
        medsam = models[3]
        medsam2 = models[4]
        sam3 = models[5]
        medical_sam3 = models[6]
        fastsam = models[7]
        self.assertEqual(sam1["capabilities"]["points"], True)
        self.assertEqual(sam1["capabilities"]["box"], True)
        self.assertEqual(sam2["capabilities"]["points"], True)
        self.assertEqual(sam2["capabilities"]["box"], True)
        self.assertEqual(sam2["capabilities"]["box_3d"], True)
        self.assertEqual(sam2["capabilities"]["mask"], True)
        self.assertEqual(sam2["capabilities"]["text"], False)
        self.assertEqual(sam2["capabilities"]["auto_predict_2d"], True)
        self.assertEqual(sam2["capabilities"]["video_propagation"], True)
        self.assertEqual(medsam["capabilities"]["points"], False)
        self.assertEqual(medsam["capabilities"]["box"], True)
        self.assertEqual(medsam["capabilities"]["box_3d"], True)
        self.assertEqual(medsam["capabilities"]["mask"], False)
        self.assertEqual(medsam["capabilities"]["text"], True)
        self.assertEqual(medsam["capabilities"]["auto_predict_2d"], True)
        self.assertEqual(medsam2["capabilities"]["points"], True)
        self.assertEqual(medsam2["capabilities"]["box"], True)
        self.assertEqual(medsam2["capabilities"]["box_3d"], True)
        self.assertEqual(medsam2["capabilities"]["mask"], True)
        self.assertEqual(medsam2["capabilities"]["text"], False)
        self.assertEqual(medsam2["capabilities"]["auto_predict_2d"], True)
        self.assertEqual(medsam2["capabilities"]["embeddings"], True)
        self.assertEqual(medsam2["capabilities"]["video_propagation"], True)
        self.assertEqual(sam3["capabilities"]["points"], True)
        self.assertEqual(sam3["capabilities"]["box"], True)
        self.assertEqual(sam3["capabilities"]["box_3d"], True)
        self.assertEqual(sam3["capabilities"]["mask"], True)
        self.assertEqual(sam3["capabilities"]["text"], True)
        self.assertEqual(sam3["capabilities"]["auto_predict_2d"], True)
        self.assertEqual(sam3["capabilities"]["embeddings"], True)
        self.assertEqual(sam3["capabilities"]["video_propagation"], True)
        self.assertEqual(sam3["capabilities"]["video_mask"], False)
        self.assertEqual(sam3["capabilities"]["video_text"], True)
        self.assertEqual(medical_sam3["capabilities"]["points"], False)
        self.assertEqual(medical_sam3["capabilities"]["box"], True)
        self.assertEqual(medical_sam3["capabilities"]["box_3d"], True)
        self.assertEqual(medical_sam3["capabilities"]["mask"], False)
        self.assertEqual(medical_sam3["capabilities"]["text"], True)
        self.assertEqual(medical_sam3["capabilities"]["auto_predict_2d"], True)
        self.assertEqual(medical_sam3["capabilities"]["embeddings"], True)
        self.assertEqual(medical_sam3["capabilities"]["video_propagation"], False)
        self.assertEqual(fastsam["capabilities"]["points"], True)
        self.assertEqual(fastsam["capabilities"]["box"], True)
        self.assertEqual(fastsam["capabilities"]["box_3d"], True)
        self.assertEqual(fastsam["capabilities"]["mask"], False)
        self.assertEqual(fastsam["capabilities"]["text"], True)
        self.assertEqual(fastsam["capabilities"]["auto_predict_2d"], True)
        self.assertEqual(fastsam["capabilities"]["embeddings"], False)

    def test_unknown_weight_returns_none(self):
        with tempfile.TemporaryDirectory() as model_dir:
            self.assertIsNone(weight_payload(model_dir, "unknown"))

    def test_sam1_has_three_weights(self):
        with tempfile.TemporaryDirectory() as model_dir:
            sam1 = models_payload(model_dir)["models"][0]

        self.assertEqual(sam1["id"], "sam1")
        self.assertEqual([weight["id"] for weight in sam1["weights"]], ["sam_vit_b", "sam_vit_l", "sam_vit_h"])

    def test_sam2_has_four_weights(self):
        with tempfile.TemporaryDirectory() as model_dir:
            sam2 = models_payload(model_dir)["models"][1]

        self.assertEqual(sam2["id"], "sam2")
        self.assertEqual(
            [weight["id"] for weight in sam2["weights"]],
            ["sam2_1_hiera_tiny", "sam2_1_hiera_small", "sam2_1_hiera_base_plus", "sam2_1_hiera_large"],
        )
        self.assertEqual(sam2["weights"][0]["checkpoint"], "sam2.1_hiera_tiny.pt")
        self.assertEqual(sam2["weights"][0]["backend"], "sam2")
        self.assertEqual(sam2["weights"][0]["capabilities"]["points"], True)
        self.assertEqual(sam2["weights"][0]["capabilities"]["mask"], True)
        self.assertEqual(sam2["weights"][0]["capabilities"]["text"], False)
        self.assertEqual(sam2["weights"][0]["capabilities"]["video_propagation"], True)

    def test_sam3_has_text_weight(self):
        with tempfile.TemporaryDirectory() as model_dir:
            sam3 = models_payload(model_dir)["models"][5]

        self.assertEqual(sam3["id"], "sam3")
        self.assertEqual([weight["id"] for weight in sam3["weights"]], ["sam3"])
        self.assertEqual(sam3["weights"][0]["checkpoint"], "sam3.pt")
        self.assertEqual(sam3["weights"][0]["capabilities"]["points"], True)
        self.assertEqual(sam3["weights"][0]["capabilities"]["mask"], True)
        self.assertEqual(sam3["weights"][0]["capabilities"]["text"], True)
        self.assertEqual(sam3["weights"][0]["capabilities"]["video_propagation"], True)
        self.assertEqual(sam3["weights"][0]["capabilities"]["video_mask"], False)
        self.assertEqual(sam3["weights"][0]["capabilities"]["video_text"], True)

    def test_medical_sam3_has_text_weight(self):
        with tempfile.TemporaryDirectory() as model_dir:
            medical_sam3 = models_payload(model_dir)["models"][6]

        self.assertEqual(medical_sam3["id"], "medical_sam3")
        self.assertEqual([weight["id"] for weight in medical_sam3["weights"]], ["medical_sam3"])
        self.assertEqual(medical_sam3["weights"][0]["checkpoint"], "medical_sam3.pt")
        self.assertEqual(medical_sam3["weights"][0]["backend"], "medical_sam3")
        self.assertEqual(medical_sam3["weights"][0]["capabilities"]["points"], False)
        self.assertEqual(medical_sam3["weights"][0]["capabilities"]["mask"], False)
        self.assertEqual(medical_sam3["weights"][0]["capabilities"]["text"], True)

    def test_medsam_has_base_and_text_weights(self):
        with tempfile.TemporaryDirectory() as model_dir:
            medsam = models_payload(model_dir)["models"][3]

        self.assertEqual(medsam["id"], "medsam")
        self.assertEqual([weight["id"] for weight in medsam["weights"]], ["medsam_vit_b", "medsam_text_flare22"])
        self.assertEqual(medsam["weights"][0]["backend"], "medsam")
        self.assertEqual(medsam["weights"][0]["capabilities"]["box"], True)
        self.assertEqual(medsam["weights"][0]["capabilities"]["text"], False)
        self.assertEqual(medsam["weights"][1]["checkpoint"], "medsam_text_prompt_flare22.pth")
        self.assertEqual(medsam["weights"][1]["backend"], "medsam_text")
        self.assertEqual(medsam["weights"][1]["capabilities"]["box"], False)
        self.assertEqual(medsam["weights"][1]["capabilities"]["box_3d"], False)
        self.assertEqual(medsam["weights"][1]["capabilities"]["text"], True)
        self.assertEqual(medsam["weights"][1]["capabilities"]["embeddings"], True)

    def test_medsam2_has_five_weights(self):
        with tempfile.TemporaryDirectory() as model_dir:
            medsam2 = models_payload(model_dir)["models"][4]

        self.assertEqual(medsam2["id"], "medsam2")
        self.assertEqual(
            [weight["id"] for weight in medsam2["weights"]],
            [
                "medsam2_latest",
                "medsam2_2411",
                "medsam2_ct_lesion",
                "medsam2_mri_liver_lesion",
                "medsam2_us_heart",
            ],
        )
        self.assertEqual(medsam2["weights"][0]["checkpoint"], "MedSAM2_latest.pt")
        self.assertEqual(medsam2["weights"][0]["backend"], "medsam2")
        self.assertEqual(medsam2["weights"][0]["model_type"], "configs/sam2.1_hiera_t512.yaml")
        self.assertEqual(medsam2["weights"][0]["capabilities"]["points"], True)
        self.assertEqual(medsam2["weights"][0]["capabilities"]["mask"], True)
        self.assertEqual(medsam2["weights"][0]["capabilities"]["text"], False)
        self.assertEqual(medsam2["weights"][0]["capabilities"]["video_propagation"], True)

    def test_finetuned_medsam2_weight_is_discovered(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "checkpoints"
            run_dir = root / "finetuning_runs" / "liver"
            checkpoint = run_dir / "checkpoints" / "checkpoint.pt"
            model_dir.mkdir()
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()
            (run_dir / "samm_model.json").write_text(json.dumps({
                "id": "medsam2_liver",
                "label": "Liver",
                "model_id": "medsam2",
                "backend": "medsam2",
                "checkpoint": "finetuning_runs/liver/checkpoints/checkpoint.pt",
                "model_type": "configs/sam2.1_hiera_t512.yaml",
            }), encoding="utf-8")

            medsam2 = models_payload(model_dir)["models"][4]
            payload = weight_payload(model_dir, "medsam2_liver")

        self.assertEqual(medsam2["weights"][-1]["id"], "medsam2_liver")
        self.assertEqual(payload["model_id"], "medsam2")
        self.assertEqual(payload["backend"], "medsam2")
        self.assertEqual(payload["checkpoint"], str(checkpoint.resolve()))
        self.assertTrue(payload["available"])

    def test_fastsam_has_two_weights(self):
        with tempfile.TemporaryDirectory() as model_dir:
            fastsam = models_payload(model_dir)["models"][7]

        self.assertEqual(fastsam["id"], "fastsam")
        self.assertEqual([weight["id"] for weight in fastsam["weights"]], ["fastsam_x", "fastsam_s"])
        self.assertEqual(fastsam["weights"][0]["checkpoint"], "FastSAM-x.pt")
        self.assertEqual(fastsam["weights"][1]["checkpoint"], "FastSAM-s.pt")
        self.assertEqual(fastsam["weights"][0]["backend"], "fastsam")
        self.assertEqual(fastsam["weights"][0]["capabilities"]["points"], True)
        self.assertEqual(fastsam["weights"][0]["capabilities"]["mask"], False)
        self.assertEqual(fastsam["weights"][0]["capabilities"]["text"], True)
        self.assertEqual(fastsam["weights"][0]["capabilities"]["auto_predict_2d"], True)
        self.assertEqual(fastsam["weights"][0]["capabilities"]["embeddings"], False)


if __name__ == "__main__":
    unittest.main()
