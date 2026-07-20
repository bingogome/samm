from base64 import b64encode
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import slicer
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleTest

from .logic import SegmentAnyMedicalModelLogic
from .client import ServiceClient
from . import logic as logic_module, prediction_debug
from .segmentation_writer import clip_mask


class SegmentAnyMedicalModelTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_defaults()
        self.test_video_prediction_client_paths()
        self.test_video_box_cursor_filters_already_applied_results()
        self.test_video_box_roi_clipping()
        self.test_auto_video_uses_current_slice_prompt_and_full_view()
        self.test_prediction_debug_saves_original_input()

    def test_defaults(self):
        logic = SegmentAnyMedicalModelLogic()
        node = logic.getParameterNode()
        self.assertEqual(node.modelFamily, "sam1")
        self.assertEqual(node.weightName, "sam_vit_b")

        client = ServiceClient()
        self.assertEqual(client.health_url, "http://127.0.0.1:8799/health")
        self.assertEqual(client.url("/models"), "http://127.0.0.1:8799/models")

    def test_video_prediction_client_paths(self):
        class RecordingClient(ServiceClient):
            def __init__(self):
                super().__init__()
                self.calls = []

            def post_json(self, path, payload=None, timeout=None):
                self.calls.append(("POST", path, payload))
                return {"path": path}

            def get_json(self, path, timeout=None):
                self.calls.append(("GET", path, None))
                return {"path": path}

        client = RecordingClient()
        job_id = "job/id ?#"
        client.start_video_prediction_job({"total": 2})
        client.add_video_prediction_job_frames(job_id, {"frames": [{"frame_index": 0}]})
        client.run_video_prediction_job(
            job_id,
            {"prompt": {"frame_index": 0, "points": [], "labels": [], "box": [1, 2, 3, 4]}},
        )
        client.video_prediction_job(job_id, cursor="2&unexpected=true")
        client.cancel_video_prediction_job(job_id)

        job_path = "/video-prediction-jobs/job%2Fid%20%3F%23"
        self.assertEqual(
            client.calls,
            [
                ("POST", "/video-prediction-jobs", {"total": 2}),
                ("POST", f"{job_path}/frames", {"frames": [{"frame_index": 0}]}),
                (
                    "POST",
                    f"{job_path}/run",
                    {"prompt": {"frame_index": 0, "points": [], "labels": [], "box": [1, 2, 3, 4]}},
                ),
                ("GET", f"{job_path}?cursor=2%26unexpected%3Dtrue", None),
                ("POST", f"{job_path}/cancel", None),
            ],
        )

    def test_video_box_roi_clipping(self):
        mask = np.ones((5, 6), dtype=np.uint8)

        clipped = clip_mask(mask, [2, 1, 4, 3])

        expected = np.zeros_like(mask)
        expected[1:4, 2:5] = 1
        np.testing.assert_array_equal(clipped, expected)

    def test_video_box_cursor_filters_already_applied_results(self):
        logic = SegmentAnyMedicalModelLogic()
        logic.boxVolumeJob = {
            "status": "running",
            "total": 3,
            "submitted": 3,
            "completed": 1,
            "cursor": 1,
            "error": None,
        }
        merged = []

        def capture_results(_parameterNode, state):
            merged.extend(result["sequence"] for result in state["results"])
            return len(state["results"])

        logic.merge_video_box_volume_results = capture_results
        job = logic.receive_video_box_volume_state(
            None,
            {
                "status": "cancelled",
                "total": 3,
                "submitted": 3,
                "completed": 3,
                "next_cursor": 3,
                "error": None,
                "results": [
                    {"sequence": 0},
                    {"sequence": 1},
                    {"sequence": 2},
                ],
            },
        )

        self.assertEqual(merged, [1, 2])
        self.assertEqual(job["cursor"], 3)
        self.assertEqual(job["lastApplied"], 2)

    def test_auto_video_uses_current_slice_prompt_and_full_view(self):
        class RecordingVideoClient:
            def __init__(self):
                self.frames = None
                self.run_payload = None

            def state(self, status, submitted):
                return {
                    "job_id": "video-job",
                    "status": status,
                    "total": 2,
                    "submitted": submitted,
                    "completed": 0,
                    "results": [],
                    "next_cursor": 0,
                    "error": None,
                }

            def start_video_prediction_job(self, payload):
                self.total_payload = payload
                return self.state("uploading", 0)

            def add_video_prediction_job_frames(self, job_id, payload):
                self.frames = payload["frames"]
                return self.state("uploading", len(self.frames))

            def run_video_prediction_job(self, job_id, payload):
                self.run_payload = payload
                return self.state("queued", 2)

        volume = SimpleNamespace(GetID=lambda: "volume-id")
        segmentation = SimpleNamespace(GetID=lambda: "segmentation-id")
        parameterNode = SimpleNamespace(
            inputVolume=volume,
            segmentation=segmentation,
            maskSegmentation=None,
            positivePrompts=object(),
            negativePrompts=object(),
            boxPrompts=object(),
            weightName="sam2_1_hiera_tiny",
            sliceView="Red",
            selectedSegmentId="segment-id",
        )
        image = {"image": {"shape": [2, 2, 3], "data": b64encode(bytes(12)).decode("ascii")}}
        specs = [
            {"view": "Red", "axis": 0, "index": 0},
            {"view": "Red", "axis": 0, "index": 1},
        ]
        frames = iter([(dict(image), dict(spec)) for spec in specs])
        client = RecordingVideoClient()
        logic = SegmentAnyMedicalModelLogic()
        logic.client = client
        logic.ensure_segment = lambda _parameterNode: "segment-id"

        with (
            patch.object(logic_module.slicer.util, "arrayFromVolume", return_value=np.zeros((2, 2, 2))),
            patch.object(logic_module, "image_slice_payload", return_value=(dict(image), dict(specs[1]))),
            patch.object(logic_module, "image_slice_specs", return_value=specs),
            patch.object(logic_module, "image_slice_payloads", return_value=frames),
            patch.object(logic_module, "segment_mask_payload", return_value=None),
            patch.object(logic_module, "prompt_payload", return_value=([[1, 1]], [1], [0, 0, 1, 1], None)),
        ):
            job = logic.start_video_auto_prediction(parameterNode)
            job = logic.submit_video_volume_frames(parameterNode, 2)

        self.assertEqual(client.total_payload, {"total": 2})
        self.assertEqual([frame["frame_index"] for frame in client.frames], [0, 1])
        self.assertEqual(job["mode"], "auto_video")
        self.assertEqual(job["seedFrameIndex"], 1)
        self.assertEqual(
            client.run_payload["prompt"],
            {"frame_index": 1, "points": [[1, 1]], "labels": [1], "box": [0, 0, 1, 1]},
        )
        self.assertEqual(client.run_payload["direction"], "both")

    def test_prediction_debug_saves_original_input(self):
        image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
        mask = np.zeros((2, 3), dtype=np.uint8)
        segment = SimpleNamespace(GetName=lambda: "Segment 1")
        segmentation = SimpleNamespace(
            GetID=lambda: "segmentation-id",
            GetName=lambda: "Segmentation",
            GetSegmentation=lambda: SimpleNamespace(GetSegment=lambda segment_id: segment),
        )
        parameterNode = SimpleNamespace(
            inputVolume=SimpleNamespace(GetID=lambda: "volume-id", GetName=lambda: "Volume"),
            segmentation=segmentation,
            modelFamily="sam3",
            weightName="sam3",
        )
        imagePayload = {
            "image": {
                "shape": list(image.shape),
                "data": b64encode(image.tobytes()).decode("ascii"),
            }
        }
        response = {
            "shape": list(mask.shape),
            "data": b64encode(mask.tobytes()).decode("ascii"),
        }

        with tempfile.TemporaryDirectory() as root:
            folder = prediction_debug.save_prediction_debug(
                Path(root),
                parameterNode,
                "segment-id",
                imagePayload,
                {"view": "Red", "axis": 0, "index": 1},
                {"points": [[1, 1]], "labels": [1]},
                response,
            )
            self.assertEqual(
                sorted(path.name for path in folder.iterdir()),
                ["data.json", "input_image.png", "input_prompts_overlay.png", "output_overlay.png"],
            )
            reader = prediction_debug.vtk.vtkPNGReader()
            reader.SetFileName(str(folder / "input_image.png"))
            reader.Update()
            output = reader.GetOutput()
            width, height, _ = output.GetDimensions()
            saved = prediction_debug.numpy_support.vtk_to_numpy(output.GetPointData().GetScalars())
            saved = np.flipud(saved.reshape(height, width, 3))
            np.testing.assert_array_equal(saved, image)
