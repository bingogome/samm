from datetime import datetime
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import re
from shutil import which
import os
import shlex
import signal
import subprocess
import sys
from uuid import uuid4

import numpy as np
import slicer
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleLogic

from .client import ServiceClient
from .finetune_export import merge_label_masks
from .parameter_node import SegmentAnyMedicalModelParameterNode
from .prediction_debug import save_prediction_debug
from .prompt_sync import sync_2d_prompts_to_slice
from .protocol import SLICE_VIEWS
from .segmentation_writer import has_binary_labelmap, write_slice, write_slices
from .slice_payload import image_payload as slice_image_payload
from .slice_payload import (
    image_slice_count,
    image_slice_payload,
    image_slice_payloads,
    image_slice_specs,
    prompt_payload,
    segment_mask_payload,
    volume_box_payloads,
    volume_box_specs,
)


DEFAULT_SEGMENT_NAME = "SAMM"


class SegmentAnyMedicalModelLogic(ScriptedLoadableModuleLogic):
    def __init__(self):
        super().__init__()
        self.client = None
        self.maskVolumes = {}
        self.embeddings = {}
        self.embeddingJob = None
        self.embeddingPayloads = None
        self.boxVolumeJob = None
        self.boxVolumePayloads = None
        self.sessionId = uuid4().hex

    def getParameterNode(self):
        return SegmentAnyMedicalModelParameterNode(super().getParameterNode())

    def start_server(self):
        command = [self.pixi_program(), "run", "server"]
        log = self.new_server_log_path()
        with log.open("wb") as output:
            self.write_log(output, self.server_launch_log(command, log))
            process = subprocess.Popen(
                command,
                cwd=self.project_root(),
                stdout=output,
                stderr=subprocess.STDOUT,
                env=self.server_environment(),
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            self.write_log(output, self.server_process_log(process.pid))
        self.server_pid_path().write_text(f"{process.pid}\n")
        self.server_state_path().write_text(json.dumps({
            "pid": process.pid,
            "pgid": process.pid,
            "log": str(log),
            "started_at": self.log_time(),
        }, indent=2) + "\n")
        return process.pid

    def stop_server(self):
        self.disconnect_service()
        pid = self.server_pid()
        if pid is None:
            return None
        self.append_server_log(f"[{self.log_time()}] stopping SAMM server pid={pid} pgid={pid}\n")
        try:
            os.killpg(pid, signal.SIGTERM)
            self.append_server_log(f"[{self.log_time()}] sent SIGTERM to SAMM server pgid={pid}\n")
        except ProcessLookupError:
            self.append_server_log(f"[{self.log_time()}] SAMM server pid={pid} already stopped\n")
        self.clear_server_state()
        return pid

    def project_root(self):
        return Path(__file__).resolve().parents[3]

    def log_dir_path(self):
        path = self.project_root() / "logs"
        path.mkdir(exist_ok=True)
        return path

    def prediction_debug_dir_path(self):
        path = self.log_dir_path() / "predictions"
        path.mkdir(exist_ok=True)
        return path

    def new_server_log_path(self):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return self.log_dir_path() / f"samm-server-{stamp}.log"

    def server_log_path(self):
        state = self.server_state()
        if state and state.get("log"):
            return Path(state["log"])
        pid = self.server_pid_path().read_text().strip() if self.server_pid_path().exists() else "unknown"
        return self.log_dir_path() / f"samm-server-{pid}.log"

    def server_pid_path(self):
        return self.log_dir_path() / "samm-server.pid"

    def server_state_path(self):
        return self.log_dir_path() / "samm-server.json"

    def server_state(self):
        path = self.server_state_path()
        return json.loads(path.read_text()) if path.exists() else None

    def server_pid(self):
        state = self.server_state()
        if state:
            return int(state["pid"])
        path = self.server_pid_path()
        return int(path.read_text().strip()) if path.exists() else None

    def server_running(self):
        pid = self.server_pid()
        if pid is None:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            self.clear_server_state()
            return False
        return True

    def server_launch_log(self, command, log):
        return (
            f"[{self.log_time()}] launching SAMM server\n"
            f"cwd={self.project_root()}\n"
            f"command={shlex.join(command)}\n"
            f"log={log}\n"
            f"pid_file={self.server_pid_path()}\n"
            f"state_file={self.server_state_path()}\n"
        )

    def server_process_log(self, pid):
        return f"pid={pid}\npgid={pid}\nkill=kill -- -{pid}\n\n"

    def append_server_log(self, text):
        with self.server_log_path().open("a", encoding="utf-8") as output:
            output.write(text)

    def clear_server_state(self):
        self.server_pid_path().unlink(missing_ok=True)
        self.server_state_path().unlink(missing_ok=True)

    def write_log(self, output, text):
        output.write(text.encode("utf-8"))
        output.flush()

    def log_time(self):
        return datetime.now().isoformat(timespec="seconds")

    def pixi_program(self):
        return which("pixi") or str(Path.home() / ".pixi" / "bin" / "pixi")

    def server_environment(self):
        env = os.environ.copy()
        env.pop("PYTHONHOME", None)
        env.pop("PYTHONPATH", None)
        env["SAMM_PIXI"] = self.pixi_program()
        return env

    def ensure_scene_nodes(self, parameterNode):
        if not parameterNode.segmentation:
            self.add_segmentation(parameterNode)
        self.ensure_segment(parameterNode)
        if not parameterNode.maskSegmentation:
            parameterNode.maskSegmentation = parameterNode.segmentation
        self.selected_mask_segment_id(parameterNode)
        if not parameterNode.positivePrompts:
            parameterNode.positivePrompts = self.prompt_node("SAMM positive", (0, 1, 0))
        if not parameterNode.negativePrompts:
            parameterNode.negativePrompts = self.prompt_node("SAMM negative", (1, 0, 0))
        if not parameterNode.boxPrompts or parameterNode.boxPrompts.GetClassName() != "vtkMRMLMarkupsPlaneNode":
            if parameterNode.boxPrompts:
                slicer.mrmlScene.RemoveNode(parameterNode.boxPrompts)
            parameterNode.boxPrompts = self.plane_prompt_node("SAMM 2D box", (1, 0.85, 0))
        if not parameterNode.volumeBoxPrompt:
            parameterNode.volumeBoxPrompt = self.roi_prompt_node("SAMM 3D box", (1, 0.85, 0))

    def add_segmentation(self, parameterNode):
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "SAMM")
        node.CreateDefaultDisplayNodes()
        parameterNode.segmentation = node
        parameterNode.selectedSegmentId = ""
        if not parameterNode.maskSegmentation:
            parameterNode.maskSegmentation = node
            parameterNode.maskSegmentId = ""
        return node

    def remove_segmentation(self, parameterNode):
        node = parameterNode.segmentation
        if not node:
            return None
        nodeId = node.GetID()
        if parameterNode.maskSegmentation is node:
            parameterNode.maskSegmentation = None
            parameterNode.maskSegmentId = ""
        parameterNode.segmentation = None
        parameterNode.selectedSegmentId = ""
        slicer.mrmlScene.RemoveNode(node)
        self.maskVolumes = {key: value for key, value in self.maskVolumes.items() if key[1] != nodeId}
        return nodeId

    def ensure_segment(self, parameterNode):
        segmentId = self.selected_segment_id(parameterNode)
        if segmentId:
            return segmentId
        return self.add_segment(parameterNode, DEFAULT_SEGMENT_NAME)

    def selected_segment_id(self, parameterNode):
        segmentation = parameterNode.segmentation.GetSegmentation()
        segmentId = parameterNode.selectedSegmentId
        if segmentId and segmentation.GetSegment(segmentId):
            return segmentId
        segmentId = segmentation.GetNthSegmentID(0) if segmentation.GetNumberOfSegments() else ""
        parameterNode.selectedSegmentId = segmentId
        return segmentId

    def selected_mask_segment_id(self, parameterNode):
        if not parameterNode.maskSegmentation:
            parameterNode.maskSegmentId = ""
            return ""
        segmentation = parameterNode.maskSegmentation.GetSegmentation()
        segmentId = parameterNode.maskSegmentId
        if segmentId and segmentation.GetSegment(segmentId):
            return segmentId
        segmentId = segmentation.GetNthSegmentID(0) if segmentation.GetNumberOfSegments() else ""
        parameterNode.maskSegmentId = segmentId
        return segmentId

    def add_segment(self, parameterNode, name=None):
        segmentation = parameterNode.segmentation.GetSegmentation()
        segmentId = segmentation.AddEmptySegment(name or self.unique_segment_name(segmentation, DEFAULT_SEGMENT_NAME))
        parameterNode.selectedSegmentId = segmentId
        parameterNode.segmentation.Modified()
        return segmentId

    def remove_segment(self, parameterNode):
        segmentation = parameterNode.segmentation.GetSegmentation()
        segmentId = self.selected_segment_id(parameterNode)
        if not segmentId:
            return None
        segmentation.RemoveSegment(segmentId)
        parameterNode.selectedSegmentId = segmentation.GetNthSegmentID(0) if segmentation.GetNumberOfSegments() else ""
        if parameterNode.maskSegmentation is parameterNode.segmentation and parameterNode.maskSegmentId == segmentId:
            parameterNode.maskSegmentId = parameterNode.selectedSegmentId
        self.maskVolumes = {key: value for key, value in self.maskVolumes.items() if key[2] != segmentId}
        parameterNode.segmentation.Modified()
        return segmentId

    def unique_segment_name(self, segmentation, base):
        names = {
            segmentation.GetSegment(segmentation.GetNthSegmentID(index)).GetName()
            for index in range(segmentation.GetNumberOfSegments())
        }
        if base not in names:
            return base
        index = 2
        while f"{base} {index}" in names:
            index += 1
        return f"{base} {index}"

    def segment_items(self, segmentationNode):
        segmentation = segmentationNode.GetSegmentation()
        items = []
        for index in range(segmentation.GetNumberOfSegments()):
            segmentId = segmentation.GetNthSegmentID(index)
            items.append((segmentId, segmentation.GetSegment(segmentId).GetName()))
        return items

    def save_segmentation(self, parameterNode, path):
        if not parameterNode or not parameterNode.segmentation:
            raise ValueError("No segmentation to save.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not slicer.util.saveNode(parameterNode.segmentation, str(path)):
            raise RuntimeError(f"Failed to save segmentation to {path}")
        return {"path": str(path), "segments": self.segment_count(parameterNode.segmentation)}

    def load_segmentation(self, parameterNode, path):
        path = Path(path)
        node = slicer.util.loadSegmentation(str(path))
        if not node:
            raise RuntimeError(f"Failed to load segmentation from {path}")
        node.CreateDefaultDisplayNodes()
        parameterNode.segmentation = node
        parameterNode.selectedSegmentId = self.ensure_segment(parameterNode)
        if not parameterNode.maskSegmentation:
            parameterNode.maskSegmentation = node
            parameterNode.maskSegmentId = parameterNode.selectedSegmentId
        return {"path": str(path), "name": node.GetName(), "segments": self.segment_count(node)}

    def segment_count(self, segmentationNode):
        return segmentationNode.GetSegmentation().GetNumberOfSegments()

    def add_segmented_volume(self, parameterNode):
        return self.write_segmented_volume(parameterNode, self.default_segmented_volume_path(parameterNode))

    def write_segmented_volume(self, parameterNode, path):
        if not parameterNode or not parameterNode.inputVolume or not parameterNode.segmentation:
            raise ValueError("Select an input volume and segmentation first.")
        path = Path(path)
        path = path if path.suffix == ".npz" else path.with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        volume = slicer.util.arrayFromVolume(parameterNode.inputVolume).astype(np.float32, copy=False)
        labels, segments = merge_label_masks(volume.shape, self.segment_masks(parameterNode))
        np.savez_compressed(
            path,
            imgs=volume,
            gts=labels,
            segment_ids=np.array([segment["id"] for segment in segments], dtype=str),
            segment_names=np.array([segment["name"] for segment in segments], dtype=str),
            segment_labels=np.array([segment["label"] for segment in segments], dtype=np.uint16),
            spacing=np.array(parameterNode.inputVolume.GetSpacing(), dtype=np.float32),
            origin=np.array(parameterNode.inputVolume.GetOrigin(), dtype=np.float32),
            directions=self.volume_directions(parameterNode.inputVolume),
        )
        return {"path": str(path), "volume": path.stem, "segments": segments}

    def finetune_dataset_status(self, parameterNode):
        name = self.finetune_dataset_name(parameterNode)
        return self.client.finetune_dataset_status(name)

    def local_finetune_dataset_status(self, parameterNode):
        return self.finetune_dataset_status_from_name(self.finetune_dataset_name(parameterNode))

    def new_finetune_dataset(self, parameterNode):
        name = self.unique_finetune_dataset_name(self.finetune_dataset_name(parameterNode))
        (self.finetune_segmented_volumes_dir_path() / name).mkdir(parents=True, exist_ok=False)
        return self.finetune_dataset_status_from_name(name)

    def unique_finetune_dataset_name(self, base):
        names = {status["name"] for status in self.local_finetune_datasets()}
        if base not in names:
            return base
        index = 2
        while f"{base}_{index}" in names:
            index += 1
        return f"{base}_{index}"

    def local_finetune_datasets(self):
        names = set()
        for root in (self.finetune_segmented_volumes_dir_path(), self.finetune_datasets_dir_path()):
            if root.is_dir():
                names.update(path.name for path in root.iterdir() if path.is_dir())
        return [self.finetune_dataset_status_from_name(name) for name in sorted(names)]

    def finetune_dataset_status_from_name(self, name):
        source = self.finetune_segmented_volumes_dir_path() / name
        dataset = self.finetune_datasets_dir_path() / name
        return {
            "name": name,
            "source": str(source),
            "dataset": str(dataset),
            "source_segmented_volumes": self.npz_count(source),
            "train_segmented_volumes": self.npz_count(dataset / "train_npz"),
            "val_segmented_volumes": self.npz_count(dataset / "val_npz"),
        }

    def npz_count(self, path):
        return len(list(path.glob("*.npz"))) if path.is_dir() else 0

    def list_finetune_datasets(self):
        return self.client.list_finetune_datasets()

    def build_finetune_dataset(self, parameterNode, val_count, axes, window=None):
        payload = {"val_count": int(val_count), "axes": axes, "window": list(window) if window else None}
        return self.client.build_finetune_dataset(self.finetune_dataset_name(parameterNode), payload)

    def finetune_report(self, parameterNode, run_name=None):
        return self.client.finetune_report(self.finetune_run_name(parameterNode, run_name))

    def start_finetune_training(self, parameterNode, checkpoint, epochs, batch_size, num_workers, num_frames):
        return self.client.start_finetune_training({
            "dataset": self.finetune_dataset_name(parameterNode),
            "run": self.finetune_run_name(parameterNode),
            "checkpoint": checkpoint,
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "num_workers": int(num_workers),
            "num_frames": int(num_frames),
        })

    def start_finetune_eval(self, parameterNode, max_segmented_volumes=0):
        payload = {
            "dataset": self.finetune_dataset_name(parameterNode),
            "run": self.finetune_run_name(parameterNode),
            "prompts": ["box", "point", "box-point"],
        }
        if max_segmented_volumes:
            payload["max_segmented_volumes"] = int(max_segmented_volumes)
        return self.client.start_finetune_eval(payload)

    def finetune_job(self, job_id):
        return self.client.finetune_job(job_id)

    def cancel_finetune_job(self, job_id):
        return self.client.cancel_finetune_job(job_id)

    def finetune_dataset_name(self, parameterNode):
        return self.clean_file_part(parameterNode.finetuneDatasetName or "my_task")

    def finetune_run_name(self, parameterNode, run_name=None):
        return self.clean_file_part(run_name or parameterNode.finetuneRunName or f"{self.finetune_dataset_name(parameterNode)}_v1")

    def finetune_run_path(self, parameterNode, run_name=None):
        return self.project_root() / "finetuning_runs" / self.finetune_run_name(parameterNode, run_name)

    def segment_masks(self, parameterNode):
        volumeShape = slicer.util.arrayFromVolume(parameterNode.inputVolume).shape
        for segmentId, name in self.segment_items(parameterNode.segmentation):
            if not has_binary_labelmap(parameterNode.segmentation, segmentId):
                continue
            mask = slicer.util.arrayFromSegmentBinaryLabelmap(parameterNode.segmentation, segmentId, parameterNode.inputVolume)
            if mask.shape != volumeShape:
                raise ValueError(f"Segment {name} shape {mask.shape} does not match volume shape {volumeShape}")
            yield segmentId, name, mask

    def volume_directions(self, volumeNode):
        directions = np.zeros((3, 3), dtype=np.float32)
        volumeNode.GetIJKToRASDirections(directions)
        return directions

    def prompt_node(self, name, color):
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", name)
        node.CreateDefaultDisplayNodes()
        node.GetDisplayNode().SetColor(*color)
        node.GetDisplayNode().SetSelectedColor(*color)
        node.GetDisplayNode().SetGlyphScale(1.2)
        node.GetDisplayNode().SetTextScale(0)
        return node

    def roi_prompt_node(self, name, color):
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode", name)
        node.CreateDefaultDisplayNodes()
        node.SetLocked(False)
        displayNode = node.GetDisplayNode()
        displayNode.SetColor(*color)
        displayNode.SetSelectedColor(*color)
        displayNode.SetInteractionHandleScale(1)
        return node

    def plane_prompt_node(self, name, color):
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsPlaneNode", name)
        node.CreateDefaultDisplayNodes()
        displayNode = node.GetDisplayNode()
        displayNode.SetColor(*color)
        displayNode.SetSelectedColor(*color)
        displayNode.SetGlyphScale(0.5)
        displayNode.SetInteractionHandleScale(1)
        return node

    def place_prompt(self, node, persistent=True):
        selectionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLSelectionNodeSingleton")
        interactionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton")
        selectionNode.SetReferenceActivePlaceNodeClassName(node.GetClassName())
        selectionNode.SetReferenceActivePlaceNodeID(node.GetID())
        interactionNode.SetPlaceModePersistence(1 if persistent else 0)
        interactionNode.SetCurrentInteractionMode(1)

    def clear_prompts(self, parameterNode):
        self.clear_2d_prompts(parameterNode)
        self.clear_3d_prompts(parameterNode)

    def clear_2d_prompts(self, parameterNode, points=True, box=True, mask=False):
        if points:
            parameterNode.positivePrompts.RemoveAllControlPoints()
            parameterNode.negativePrompts.RemoveAllControlPoints()
        if box:
            parameterNode.boxPrompts.RemoveAllControlPoints()
        if mask:
            parameterNode.sendMaskPrompt = False

    def clear_3d_prompts(self, parameterNode):
        parameterNode.volumeBoxPrompt.RemoveAllControlPoints()

    def prompt_count(self, parameterNode):
        return self.slice_prompt_count(parameterNode) + self.volume_prompt_count(parameterNode)

    def slice_prompt_count(self, parameterNode):
        return (
            parameterNode.positivePrompts.GetNumberOfControlPoints()
            + parameterNode.negativePrompts.GetNumberOfControlPoints()
            + parameterNode.boxPrompts.GetNumberOfControlPoints()
        )

    def volume_prompt_count(self, parameterNode):
        return parameterNode.volumeBoxPrompt.GetNumberOfControlPoints()

    def has_prompt(self, parameterNode):
        return (
            self.has_point_prompt(parameterNode)
            or self.has_box_prompt(parameterNode)
            or self.has_mask_prompt(parameterNode)
            or self.has_text_prompt(parameterNode)
        )

    def has_point_prompt(self, parameterNode):
        return parameterNode.positivePrompts.GetNumberOfControlPoints() + parameterNode.negativePrompts.GetNumberOfControlPoints() > 0

    def has_box_prompt(self, parameterNode):
        return parameterNode.boxPrompts.GetNumberOfControlPoints() > 0

    def has_mask_prompt(self, parameterNode):
        if not parameterNode.maskSegmentation:
            return False
        segmentId = self.selected_mask_segment_id(parameterNode)
        return bool(segmentId and has_binary_labelmap(parameterNode.maskSegmentation, segmentId))

    def has_text_prompt(self, parameterNode):
        return bool(parameterNode.textPrompt.strip())

    def has_volume_box_prompt(self, parameterNode):
        if parameterNode.volumeBoxPrompt.GetNumberOfControlPoints() == 0:
            return False
        bounds = [0.0] * 6
        parameterNode.volumeBoxPrompt.GetRASBounds(bounds)
        return bounds[0] < bounds[1] and bounds[2] < bounds[3] and bounds[4] < bounds[5]

    def sync_2d_prompts_to_slice(self, parameterNode):
        sync_2d_prompts_to_slice(
            parameterNode.inputVolume,
            parameterNode.sliceView,
            parameterNode.positivePrompts,
            parameterNode.negativePrompts,
            parameterNode.boxPrompts,
        )

    def predict_slice(
        self,
        parameterNode,
        cached_embedding=False,
        floating_prompts=False,
        use_points=True,
        use_box=True,
        use_mask=True,
        use_text=True,
        use_embeddings=True,
        debug_log=False,
    ):
        segmentId = self.ensure_segment(parameterNode)
        key = (parameterNode.inputVolume.GetID(), parameterNode.segmentation.GetID(), segmentId)
        imagePayload, sliceSpec = image_slice_payload(parameterNode.inputVolume, parameterNode.sliceView)
        if use_embeddings and cached_embedding:
            embedding = self.cached_embedding_for_slice(parameterNode, imagePayload, sliceSpec)
            created = False
        elif use_embeddings:
            embedding, created = self.embedding_for_slice(parameterNode, imagePayload, sliceSpec)
        else:
            embedding, created = None, False
        mask = segment_mask_payload(
            parameterNode.maskSegmentation,
            self.selected_mask_segment_id(parameterNode),
            parameterNode.inputVolume,
            sliceSpec,
            imagePayload["image"]["shape"][:2],
        ) if use_mask else None
        text = parameterNode.textPrompt.strip() if use_text else ""
        points, labels, box, mask = prompt_payload(
            parameterNode.inputVolume,
            sliceSpec,
            parameterNode.positivePrompts,
            parameterNode.negativePrompts,
            parameterNode.boxPrompts,
            imagePayload["image"]["shape"][:2],
            require_slice=not floating_prompts,
            use_points=use_points,
            use_box=use_box,
            mask=mask,
            text=text,
        )
        payload = {"points": points, "labels": labels}
        if embedding:
            payload["embedding_id"] = embedding["id"]
        else:
            payload.update(imagePayload)
        if box:
            payload["box"] = box
        if mask:
            payload["mask"] = mask
        if text:
            payload["text"] = text
        response = self.client.predict(payload)
        debugPath = save_prediction_debug(
            self.prediction_debug_dir_path(),
            parameterNode,
            segmentId,
            imagePayload,
            sliceSpec,
            payload,
            response,
        ) if debug_log else None
        self.maskVolumes[key], segmentId = write_slice(
            response,
            self.maskVolumes.get(key),
            parameterNode.segmentation,
            segmentId,
            parameterNode.inputVolume,
            sliceSpec,
        )
        segmentName = parameterNode.segmentation.GetSegmentation().GetSegment(segmentId).GetName()
        action = "Embedded and predicted" if created else "Predicted"
        message = f"{action} {sliceSpec['view']} slice {sliceSpec['index']} into {segmentName}"
        return f"{message}\nDebug log: {debugPath}" if debugPath else message

    def predict_auto_slice(
        self,
        parameterNode,
        use_points=True,
        use_box=True,
        use_mask=True,
        use_text=True,
        use_embeddings=True,
        debug_log=False,
    ):
        return self.predict_slice(
            parameterNode,
            cached_embedding=True,
            floating_prompts=True,
            use_points=use_points,
            use_box=use_box,
            use_mask=use_mask,
            use_text=use_text,
            use_embeddings=use_embeddings,
            debug_log=debug_log,
        )

    def start_box_volume_prediction(self, parameterNode):
        segmentId = self.ensure_segment(parameterNode)
        volume = slicer.util.arrayFromVolume(parameterNode.inputVolume)
        specs = volume_box_specs(parameterNode.inputVolume, volume.shape, parameterNode.sliceView, parameterNode.volumeBoxPrompt)
        total = len(specs)
        state = self.client.start_prediction_job({"total": total})
        self.boxVolumeJob = {
            "job_id": state["job_id"],
            "mode": "slices",
            "weightName": parameterNode.weightName,
            "volumeId": parameterNode.inputVolume.GetID(),
            "segmentationId": parameterNode.segmentation.GetID(),
            "segmentId": segmentId,
            "view": parameterNode.sliceView,
            "status": state["status"],
            "total": state["total"],
            "submitted": state.get("submitted", 0),
            "completed": state["completed"],
            "applied": 0,
            "lastApplied": 0,
            "error": state.get("error"),
        }
        self.boxVolumePayloads = volume_box_payloads(parameterNode.inputVolume, parameterNode.sliceView, parameterNode.volumeBoxPrompt, specs)
        return self.receive_box_volume_state(parameterNode, state)

    def start_video_box_volume_prediction(self, parameterNode):
        segmentId = self.ensure_segment(parameterNode)
        volume = slicer.util.arrayFromVolume(parameterNode.inputVolume)
        specs = volume_box_specs(parameterNode.inputVolume, volume.shape, parameterNode.sliceView, parameterNode.volumeBoxPrompt)
        total = len(specs)
        seedFrameIndex = total // 2
        state = self.client.start_video_prediction_job({"total": total})
        self.boxVolumeJob = {
            "job_id": state["job_id"],
            "mode": "video",
            "weightName": parameterNode.weightName,
            "volumeId": parameterNode.inputVolume.GetID(),
            "segmentationId": parameterNode.segmentation.GetID(),
            "segmentId": segmentId,
            "view": parameterNode.sliceView,
            "status": state["status"],
            "total": state["total"],
            "submitted": state.get("submitted", 0),
            "completed": state["completed"],
            "applied": 0,
            "lastApplied": 0,
            "cursor": state.get("next_cursor", 0),
            "seedFrameIndex": seedFrameIndex,
            "prompt": {
                "frame_index": seedFrameIndex,
                "points": [],
                "labels": [],
                "box": list(specs[seedFrameIndex][1]),
            },
            "clipBox": list(specs[seedFrameIndex][1]),
            "error": state.get("error"),
        }
        self.boxVolumePayloads = volume_box_payloads(
            parameterNode.inputVolume,
            parameterNode.sliceView,
            parameterNode.volumeBoxPrompt,
            specs,
        )
        return self.receive_video_box_volume_state(parameterNode, state)

    def start_video_auto_prediction(
        self,
        parameterNode,
        use_points=True,
        use_box=True,
        use_mask=True,
        use_text=False,
    ):
        volume = slicer.util.arrayFromVolume(parameterNode.inputVolume)
        seedImagePayload, seedSliceSpec = image_slice_payload(parameterNode.inputVolume, parameterNode.sliceView)
        specs = image_slice_specs(parameterNode.inputVolume, parameterNode.sliceView, volume.shape)
        seedFrameIndex = next(
            index
            for index, sliceSpec in enumerate(specs)
            if sliceSpec["axis"] == seedSliceSpec["axis"] and sliceSpec["index"] == seedSliceSpec["index"]
        )
        mask = segment_mask_payload(
            parameterNode.maskSegmentation,
            self.selected_mask_segment_id(parameterNode),
            parameterNode.inputVolume,
            seedSliceSpec,
            seedImagePayload["image"]["shape"][:2],
        ) if use_mask else None
        text = parameterNode.textPrompt.strip() if use_text else ""
        points, labels, box, mask = prompt_payload(
            parameterNode.inputVolume,
            seedSliceSpec,
            parameterNode.positivePrompts,
            parameterNode.negativePrompts,
            parameterNode.boxPrompts,
            seedImagePayload["image"]["shape"][:2],
            require_slice=True,
            use_points=use_points,
            use_box=use_box,
            mask=mask,
            text=text,
        )
        if mask and (points or box or text):
            raise ValueError("Slices-as-video mask prompts cannot be combined with points, a 2D box, or text.")
        prompt = {
            "frame_index": seedFrameIndex,
            "points": points,
            "labels": labels,
        }
        if box:
            prompt["box"] = box
        if mask:
            prompt["mask"] = mask
        if text:
            prompt["text"] = text

        segmentId = self.ensure_segment(parameterNode)
        state = self.client.start_video_prediction_job({"total": len(specs)})
        self.boxVolumeJob = {
            "job_id": state["job_id"],
            "mode": "auto_video",
            "weightName": parameterNode.weightName,
            "volumeId": parameterNode.inputVolume.GetID(),
            "segmentationId": parameterNode.segmentation.GetID(),
            "segmentId": segmentId,
            "view": parameterNode.sliceView,
            "status": state["status"],
            "total": state["total"],
            "submitted": state.get("submitted", 0),
            "completed": state["completed"],
            "applied": 0,
            "lastApplied": 0,
            "cursor": state.get("next_cursor", 0),
            "seedFrameIndex": seedFrameIndex,
            "seedSliceIndex": seedSliceSpec["index"],
            "prompt": prompt,
            "error": state.get("error"),
        }
        self.boxVolumePayloads = image_slice_payloads(parameterNode.inputVolume, parameterNode.sliceView)
        return self.receive_video_box_volume_state(parameterNode, state)

    def submit_box_volume_slices(self, parameterNode, count):
        if not self.boxVolumeJob or not self.boxVolumePayloads:
            return None
        items = []
        for _ in range(count):
            try:
                payload, sliceSpec, box = next(self.boxVolumePayloads)
            except StopIteration:
                self.boxVolumePayloads = None
                break
            payload.update({
                "key": self.box_volume_slice_key(sliceSpec),
                "slice_spec": dict(sliceSpec),
                "points": [],
                "labels": [],
                "box": list(box),
            })
            text = parameterNode.textPrompt.strip() if parameterNode.sendTextPrompt else ""
            if text:
                payload["text"] = text
            items.append(payload)
        if not items:
            return self.refresh_box_volume_job(parameterNode)
        state = self.client.add_prediction_job_items(self.boxVolumeJob["job_id"], {"items": items})
        return self.receive_box_volume_state(parameterNode, state)

    def submit_video_volume_frames(self, parameterNode, count):
        if (
            not self.boxVolumeJob
            or self.boxVolumeJob.get("mode") not in ("video", "auto_video")
            or not self.boxVolumePayloads
        ):
            return None
        frames = []
        frameIndex = self.boxVolumeJob.get("submitted", 0)
        for _ in range(count):
            try:
                item = next(self.boxVolumePayloads)
            except StopIteration:
                self.boxVolumePayloads = None
                break
            payload, sliceSpec = item[:2]
            payload.update({
                "frame_index": frameIndex,
                "key": self.box_volume_slice_key(sliceSpec),
                "slice_spec": dict(sliceSpec),
            })
            frames.append(payload)
            frameIndex += 1
        if frames:
            state = self.client.add_video_prediction_job_frames(self.boxVolumeJob["job_id"], {"frames": frames})
            self.receive_video_box_volume_state(parameterNode, state)
        if self.boxVolumeJob.get("submitted", 0) >= self.boxVolumeJob["total"]:
            self.boxVolumePayloads = None
            state = self.client.run_video_prediction_job(
                self.boxVolumeJob["job_id"],
                {
                    "prompt": dict(self.boxVolumeJob["prompt"]),
                    "direction": "both",
                    "offload_video_to_cpu": True,
                    "offload_state_to_cpu": False,
                },
            )
            return self.receive_video_box_volume_state(parameterNode, state)
        return self.boxVolumeJob

    def submit_video_box_volume_frames(self, parameterNode, count):
        return self.submit_video_volume_frames(parameterNode, count)

    def refresh_box_volume_job(self, parameterNode):
        if not self.boxVolumeJob:
            return None
        if self.boxVolumeJob.get("mode") in ("video", "auto_video"):
            if not self.box_volume_job_matches(parameterNode):
                self.clear_box_volume_prediction()
                return None
            state = self.client.video_prediction_job(
                self.boxVolumeJob["job_id"],
                cursor=self.boxVolumeJob.get("cursor", 0),
            )
            return self.receive_video_box_volume_state(parameterNode, state)
        state = self.client.prediction_job(self.boxVolumeJob["job_id"])
        return self.receive_box_volume_state(parameterNode, state)

    def receive_box_volume_state(self, parameterNode, state):
        self.update_box_volume_job(state)
        self.boxVolumeJob["lastApplied"] = self.merge_box_volume_results(parameterNode, state)
        return self.boxVolumeJob

    def receive_video_box_volume_state(self, parameterNode, state):
        self.update_box_volume_job(state)
        cursor = self.boxVolumeJob.get("cursor", 0)
        newState = dict(state)
        newState["results"] = [
            result
            for result in state.get("results", [])
            if result.get("sequence", cursor) >= cursor
        ]
        self.boxVolumeJob["lastApplied"] = self.merge_video_box_volume_results(parameterNode, newState)
        self.boxVolumeJob["cursor"] = state.get("next_cursor", self.boxVolumeJob.get("cursor", 0))
        return self.boxVolumeJob

    def update_box_volume_job(self, state):
        self.boxVolumeJob["status"] = state["status"]
        self.boxVolumeJob["total"] = state["total"]
        self.boxVolumeJob["submitted"] = state.get("submitted", self.boxVolumeJob.get("submitted", 0))
        self.boxVolumeJob["completed"] = state["completed"]
        self.boxVolumeJob["error"] = state.get("error")
        submitted = self.boxVolumeJob["submitted"]
        total = self.boxVolumeJob["total"]
        if self.boxVolumeJob["status"] in ("complete", "failed", "cancelled") or submitted >= total:
            self.boxVolumePayloads = None

    def merge_box_volume_results(self, parameterNode, state):
        results = state.get("results", [])
        applied = self.boxVolumeJob.get("applied", 0)
        key = (self.boxVolumeJob["volumeId"], self.boxVolumeJob["segmentationId"], self.boxVolumeJob["segmentId"])
        for result in results[applied:]:
            self.maskVolumes[key], segmentId = write_slice(
                result,
                self.maskVolumes.get(key),
                parameterNode.segmentation,
                self.boxVolumeJob["segmentId"],
                parameterNode.inputVolume,
                result["slice_spec"],
            )
            self.boxVolumeJob["segmentId"] = segmentId
            applied += 1
        newResults = applied - self.boxVolumeJob.get("applied", 0)
        self.boxVolumeJob["applied"] = applied
        return newResults

    def merge_video_box_volume_results(self, parameterNode, state):
        results = state.get("results", [])
        if not results:
            return 0
        if not self.box_volume_job_matches(parameterNode):
            return 0
        key = (self.boxVolumeJob["volumeId"], self.boxVolumeJob["segmentationId"], self.boxVolumeJob["segmentId"])
        items = [(result, result["slice_spec"]) for result in results]
        self.maskVolumes[key], segmentId = write_slices(
            items,
            self.maskVolumes.get(key),
            parameterNode.segmentation,
            self.boxVolumeJob["segmentId"],
            parameterNode.inputVolume,
            clip_box=self.boxVolumeJob.get("clipBox"),
        )
        self.boxVolumeJob["segmentId"] = segmentId
        self.boxVolumeJob["applied"] = self.boxVolumeJob.get("applied", 0) + len(results)
        return len(results)

    def box_volume_slice_key(self, sliceSpec):
        return f"{sliceSpec['view']}|{sliceSpec['axis']}|{sliceSpec['index']}"

    def box_volume_prediction_running(self):
        return bool(self.boxVolumeJob and self.boxVolumeJob.get("status") in ("uploading", "queued", "running"))

    def video_box_volume_prediction_running(self):
        return bool(
            self.boxVolumeJob
            and self.boxVolumeJob.get("mode") == "video"
            and self.boxVolumeJob.get("status") in ("uploading", "queued", "running")
        )

    def auto_video_prediction_running(self):
        return bool(
            self.boxVolumeJob
            and self.boxVolumeJob.get("mode") == "auto_video"
            and self.boxVolumeJob.get("status") in ("uploading", "queued", "running")
        )

    def video_volume_prediction_running(self):
        return self.video_box_volume_prediction_running() or self.auto_video_prediction_running()

    def box_volume_submission_running(self):
        if not self.boxVolumeJob or not self.boxVolumePayloads:
            return False
        if self.boxVolumeJob.get("mode") in ("video", "auto_video"):
            return self.boxVolumeJob.get("status") == "uploading"
        return self.boxVolumeJob.get("status") in ("queued", "running")

    def box_volume_job_for(self, parameterNode):
        return self.boxVolumeJob if self.box_volume_job_matches(parameterNode) else None

    def box_volume_job_matches(self, parameterNode):
        if not self.boxVolumeJob or not parameterNode or not parameterNode.inputVolume or not parameterNode.segmentation:
            return False
        return bool(
            self.boxVolumeJob["weightName"] == parameterNode.weightName
            and self.boxVolumeJob["volumeId"] == parameterNode.inputVolume.GetID()
            and self.boxVolumeJob["segmentationId"] == parameterNode.segmentation.GetID()
            and self.boxVolumeJob["segmentId"] == parameterNode.selectedSegmentId
            and self.boxVolumeJob["view"] == parameterNode.sliceView
        )

    def cancel_video_box_volume_prediction(self, parameterNode):
        if not self.boxVolumeJob or self.boxVolumeJob.get("mode") != "video":
            return self.boxVolumeJob
        return self.cancel_video_volume_prediction(parameterNode)

    def cancel_video_auto_prediction(self, parameterNode):
        if not self.boxVolumeJob or self.boxVolumeJob.get("mode") != "auto_video":
            return self.boxVolumeJob
        return self.cancel_video_volume_prediction(parameterNode)

    def cancel_video_volume_prediction(self, parameterNode):
        if not self.boxVolumeJob or self.boxVolumeJob.get("mode") not in ("video", "auto_video"):
            return self.boxVolumeJob
        if self.boxVolumeJob.get("status") not in ("uploading", "queued", "running"):
            return self.boxVolumeJob
        self.boxVolumePayloads = None
        state = self.client.cancel_video_prediction_job(self.boxVolumeJob["job_id"])
        return self.receive_video_box_volume_state(parameterNode, state)

    def clear_box_volume_prediction(self):
        job = self.boxVolumeJob
        if (
            job
            and job.get("mode") in ("video", "auto_video")
            and job.get("status") in ("uploading", "queued", "running")
            and self.client
        ):
            try:
                self.client.cancel_video_prediction_job(job["job_id"])
            except (ConnectionError, RuntimeError):
                pass
        self.boxVolumeJob = None
        self.boxVolumePayloads = None

    def embed_all_slices(self, parameterNode):
        return self.start_embedding_job(parameterNode, (parameterNode.sliceView,))

    def embed_all_axes(self, parameterNode):
        return self.start_embedding_job(parameterNode, SLICE_VIEWS)

    def start_embedding_job(self, parameterNode, viewNames):
        specs = self.embedding_slice_specs(parameterNode.inputVolume, viewNames)
        total = len(specs)
        state = self.client.start_embedding_job({"total": total})
        self.embeddingJob = {
            "job_id": state["job_id"],
            "weightName": parameterNode.weightName,
            "volumeId": parameterNode.inputVolume.GetID(),
            "status": state["status"],
            "total": state["total"],
            "submitted": state.get("submitted", 0),
            "completed": state["completed"],
            "error": state.get("error"),
        }
        self.embeddingPayloads = self.embedding_payloads(parameterNode.inputVolume, specs)
        self.merge_embedding_job(state)
        return state

    def embedding_slice_specs(self, volumeNode, viewNames):
        return [spec for viewName in viewNames for spec in image_slice_specs(volumeNode, viewName)]

    def embedding_payloads(self, volumeNode, sliceSpecs):
        volume = slicer.util.arrayFromVolume(volumeNode)
        displayNode = volumeNode.GetDisplayNode()
        for sliceSpec in sliceSpecs:
            yield slice_image_payload(volume, sliceSpec, displayNode), sliceSpec

    def submit_embedding_slices(self, count, jobId=None):
        if not self.embeddingJob or not self.embeddingPayloads:
            return None
        if jobId and self.embeddingJob["job_id"] != jobId:
            return None
        remaining = self.embeddingJob["total"] - self.embeddingJob.get("submitted", 0)
        if remaining <= 0:
            self.embeddingPayloads = None
            return self.refresh_embedding_job()
        items = []
        for _ in range(min(count, remaining)):
            try:
                payload, sliceSpec = next(self.embeddingPayloads)
            except StopIteration:
                self.embeddingPayloads = None
                break
            imageShape, imageDigest = self.embedding_signature(payload)
            key = self.embedding_key_from_signature(
                self.embeddingJob["weightName"],
                self.embeddingJob["volumeId"],
                sliceSpec,
                imageShape,
                imageDigest,
            )
            items.append({
                "key": key,
                "slice_spec": dict(sliceSpec),
                "image_shape": list(imageShape),
                "image_digest": imageDigest,
                "image": payload["image"],
            })
        if not items:
            return self.refresh_embedding_job()
        state = self.client.add_embedding_job_items(self.embeddingJob["job_id"], {"items": items})
        self.update_embedding_job(state)
        self.merge_embedding_job(state)
        return state

    def refresh_embedding_job(self):
        if not self.embeddingJob:
            return None
        state = self.client.embedding_job(self.embeddingJob["job_id"])
        self.update_embedding_job(state)
        self.merge_embedding_job(state)
        return state

    def update_embedding_job(self, state):
        self.embeddingJob["status"] = state["status"]
        self.embeddingJob["total"] = state["total"]
        self.embeddingJob["submitted"] = state.get("submitted", self.embeddingJob.get("submitted", 0))
        self.embeddingJob["completed"] = state["completed"]
        self.embeddingJob["error"] = state.get("error")
        submitted = self.embeddingJob["submitted"]
        total = self.embeddingJob["total"]
        if self.embeddingJob["status"] in ("complete", "failed") or submitted >= total:
            self.embeddingPayloads = None

    def merge_embedding_job(self, state):
        weightName = self.embeddingJob["weightName"]
        volumeId = self.embeddingJob["volumeId"]
        for result in state.get("results", []):
            self.embeddings[result["key"]] = {
                "id": result["embedding_id"],
                "weightName": weightName,
                "volumeId": volumeId,
                "sliceSpec": dict(result["slice_spec"]),
                "imageShape": list(result["image_shape"]),
                "imageDigest": result["image_digest"],
            }

    def embedding_job_running(self, jobId=None):
        if not self.embeddingJob:
            return False
        if jobId and self.embeddingJob["job_id"] != jobId:
            return False
        return self.embeddingJob.get("status") in ("queued", "running")

    def embedding_submission_running(self, jobId=None):
        if not self.embeddingJob or not self.embeddingPayloads:
            return False
        if jobId and self.embeddingJob["job_id"] != jobId:
            return False
        return self.embeddingJob.get("status") in ("queued", "running")

    def embedding_job_for(self, parameterNode):
        if not self.embeddingJob or not parameterNode or not parameterNode.inputVolume:
            return None
        if self.embeddingJob["weightName"] != parameterNode.weightName:
            return None
        if self.embeddingJob["volumeId"] != parameterNode.inputVolume.GetID():
            return None
        return self.embeddingJob

    def view_embedding_count(self, parameterNode, viewName=None):
        if not parameterNode or not parameterNode.inputVolume:
            return 0, 0
        viewName = viewName or parameterNode.sliceView
        total = image_slice_count(parameterNode.inputVolume, viewName)
        indices = {
            embedding["sliceSpec"]["index"]
            for embedding in self.current_embeddings(parameterNode).values()
            if embedding["sliceSpec"]["view"] == viewName
        }
        return len(indices), total

    def view_fully_embedded(self, parameterNode, viewName=None):
        embedded, total = self.view_embedding_count(parameterNode, viewName)
        return total > 0 and embedded == total

    def embedding_for_slice(self, parameterNode, payload, sliceSpec):
        key = self.embedding_key(parameterNode, payload, sliceSpec)
        embedding = self.embeddings.get(key)
        if embedding:
            return embedding, False
        response = self.client.embed(payload)
        imageShape, imageDigest = self.embedding_signature(payload)
        embedding = {
            "id": response["embedding_id"],
            "weightName": parameterNode.weightName,
            "volumeId": parameterNode.inputVolume.GetID(),
            "sliceSpec": dict(sliceSpec),
            "imageShape": list(imageShape),
            "imageDigest": imageDigest,
        }
        self.embeddings[key] = embedding
        return embedding, True

    def cached_embedding_for_slice(self, parameterNode, payload, sliceSpec):
        embedding = self.embeddings.get(self.embedding_key(parameterNode, payload, sliceSpec))
        if not embedding:
            raise ValueError(f"Embed {sliceSpec['view']} view before auto predict.")
        return embedding

    def save_embeddings(self, parameterNode, path):
        items = [
            {
                "embedding_id": embedding["id"],
                "slice_spec": embedding["sliceSpec"],
                "image_shape": embedding["imageShape"],
                "image_digest": embedding["imageDigest"],
            }
            for embedding in self.current_embeddings(parameterNode).values()
        ]
        if not items:
            raise ValueError("No embeddings to save for the current volume and weight.")
        return self.client.save_embeddings({"path": str(path), "items": items})

    def load_embeddings(self, parameterNode, path):
        result = self.client.load_embeddings({"path": str(path)})
        volume = slicer.util.arrayFromVolume(parameterNode.inputVolume)
        displayNode = parameterNode.inputVolume.GetDisplayNode()
        volumeId = parameterNode.inputVolume.GetID()
        for item in result["items"]:
            payload = slice_image_payload(volume, item["slice_spec"], displayNode)
            imageShape, imageDigest = self.embedding_signature(payload)
            if list(imageShape) != item["image_shape"] or imageDigest != item["image_digest"]:
                spec = item["slice_spec"]
                raise ValueError(f"Embedding folder does not match {spec['view']} slice {spec['index']}.")
            key = self.embedding_key_from_signature(
                parameterNode.weightName,
                volumeId,
                item["slice_spec"],
                imageShape,
                imageDigest,
            )
            self.embeddings[key] = {
                "id": item["embedding_id"],
                "weightName": parameterNode.weightName,
                "volumeId": volumeId,
                "sliceSpec": dict(item["slice_spec"]),
                "imageShape": list(imageShape),
                "imageDigest": imageDigest,
            }
        return result

    def current_embeddings(self, parameterNode):
        if not parameterNode or not parameterNode.inputVolume:
            return {}
        return {
            key: embedding
            for key, embedding in self.embeddings.items()
            if embedding["weightName"] == parameterNode.weightName
            and embedding["volumeId"] == parameterNode.inputVolume.GetID()
        }

    def default_embedding_path(self, parameterNode):
        self.embedding_dir_path().mkdir(exist_ok=True)
        volumeName = self.clean_file_part(parameterNode.inputVolume.GetName())
        weightName = self.clean_file_part(parameterNode.weightName)
        count = len(self.current_embeddings(parameterNode))
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.embedding_dir_path() / f"{volumeName}_{weightName}_{count}_embeddings_{stamp}"

    def default_segmentation_path(self, parameterNode):
        self.segmentation_dir_path().mkdir(exist_ok=True)
        volumeName = self.clean_file_part(parameterNode.inputVolume.GetName() if parameterNode.inputVolume else "volume")
        segmentationName = self.clean_file_part(parameterNode.segmentation.GetName())
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.segmentation_dir_path() / f"{volumeName}_{segmentationName}_{stamp}.seg.nrrd"

    def default_segmented_volume_path(self, parameterNode):
        datasetName = self.clean_file_part(parameterNode.finetuneDatasetName or "my_task")
        volumeName = self.clean_file_part(parameterNode.inputVolume.GetName() if parameterNode.inputVolume else "volume")
        return self.unique_path(self.finetune_segmented_volumes_dir_path() / datasetName / f"{volumeName}.npz")

    def unique_path(self, path):
        if not path.exists():
            return path
        index = 2
        while path.with_name(f"{path.stem}_{index}{path.suffix}").exists():
            index += 1
        return path.with_name(f"{path.stem}_{index}{path.suffix}")

    def auto_embedding_path(self, parameterNode):
        self.autosave_dir_path().mkdir(exist_ok=True)
        volumeName = self.clean_file_part(parameterNode.inputVolume.GetName())
        weightName = self.clean_file_part(parameterNode.weightName)
        return self.autosave_dir_path() / f"{volumeName}_{weightName}_embeddings"

    def auto_segmentation_path(self, parameterNode):
        self.autosave_dir_path().mkdir(exist_ok=True)
        volumeName = self.clean_file_part(parameterNode.inputVolume.GetName() if parameterNode.inputVolume else "volume")
        return self.autosave_dir_path() / f"{volumeName}_segmentation.seg.nrrd"

    def embedding_dir_path(self):
        return self.project_root() / "embeddings"

    def segmentation_dir_path(self):
        return self.project_root() / "segmentations"

    def finetune_segmented_volumes_dir_path(self):
        return self.project_root() / "segmented_volumes"

    def finetune_datasets_dir_path(self):
        return self.project_root() / "datasets"

    def autosave_dir_path(self):
        return self.project_root() / "autosave"

    def clean_file_part(self, text):
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "unnamed"

    def clear_embedding(self):
        self.embeddings.clear()
        self.embeddingJob = None
        self.embeddingPayloads = None

    def embedding_key(self, parameterNode, payload, sliceSpec):
        imageShape, imageDigest = self.embedding_signature(payload)
        return self.embedding_key_from_signature(
            parameterNode.weightName,
            parameterNode.inputVolume.GetID(),
            sliceSpec,
            imageShape,
            imageDigest,
        )

    def embedding_signature(self, payload):
        image = payload["image"]
        return tuple(image["shape"]), sha256(image["data"].encode("ascii")).hexdigest()

    def embedding_key_from_signature(self, weightName, volumeId, sliceSpec, imageShape, imageDigest):
        return "|".join(str(item) for item in (
            weightName,
            volumeId,
            sliceSpec["view"],
            sliceSpec["axis"],
            sliceSpec["index"],
            tuple(imageShape),
            imageDigest,
        ))

    def embedded_slice_labels(self, parameterNode):
        if not parameterNode or not parameterNode.inputVolume:
            return []
        labels = []
        viewOrder = {view: index for index, view in enumerate(SLICE_VIEWS)}
        for embedding in self.current_embeddings(parameterNode).values():
            spec = embedding["sliceSpec"]
            labels.append((
                viewOrder.get(spec["view"], len(viewOrder)),
                spec["index"],
                f"{spec['view']} slice {spec['index']}",
            ))
        return [label for view, index, label in sorted(labels)]

    def connect_to_service(self):
        client = ServiceClient()
        status = client.connect()
        self.client = client
        return status

    def touch_session(self, autosave=None):
        return self.client.touch_session(self.sessionId, autosave)

    def disconnect_service(self):
        self.clear_embedding()
        self.clear_box_volume_prediction()
        self.client = None

    def list_models(self):
        return self.client.list_models()

    def local_models(self):
        model_dir = self.project_root() / "checkpoints"
        model_dir.mkdir(exist_ok=True)
        module_path = self.project_root() / "service" / "samm_server" / "model_registry.py"
        spec = importlib.util.spec_from_file_location("_samm_model_registry", module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module.model_payloads(model_dir)

    def get_weight_status(self, weight_id: str):
        return self.client.get_weight(weight_id)

    def prepare_weight(self, weight_id: str):
        return self.client.prepare_weight(weight_id)

    def offload_model(self):
        return self.client.offload_model()

    def prepared_status(self):
        return self.client.prepared()
