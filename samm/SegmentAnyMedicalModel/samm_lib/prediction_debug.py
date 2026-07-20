from base64 import b64decode
from datetime import datetime
import json
import re

import numpy as np
import vtk
from vtk.util import numpy_support

from .segmentation_writer import decode_mask


def save_prediction_debug(root, parameterNode, segmentId, imagePayload, sliceSpec, payload, response):
    folder = root / folder_name(parameterNode, sliceSpec)
    folder.mkdir(parents=True)
    image = decode_rgb(imagePayload["image"])
    data = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "volume": node_info(parameterNode.inputVolume),
        "segmentation": node_info(parameterNode.segmentation),
        "segment": segment_info(parameterNode.segmentation, segmentId),
        "model": {"family": parameterNode.modelFamily, "weight": parameterNode.weightName},
        "slice": dict(sliceSpec),
        "embedding_id": payload.get("embedding_id"),
        "prompts": {
            "points": payload.get("points", []),
            "labels": payload.get("labels", []),
            "box": payload.get("box"),
            "mask": payload.get("mask"),
            "text": payload.get("text"),
        },
        "response": {"shape": response["shape"]},
    }
    (folder / "data.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    save_rgb_png(folder / "input_image.png", image)
    save_rgb_png(folder / "input_prompts_overlay.png", input_overlay(image, data["prompts"]))
    save_rgb_png(folder / "output_overlay.png", mask_overlay(image, decode_mask(response), (255, 48, 48), 0.45))
    return folder


def folder_name(parameterNode, sliceSpec):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    volume = clean_file_part(parameterNode.inputVolume.GetName())
    weight = clean_file_part(parameterNode.weightName)
    return f"{stamp}_{volume}_{weight}_{sliceSpec['view']}_{sliceSpec['index']}"


def clean_file_part(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "unnamed"


def node_info(node):
    return {"id": node.GetID(), "name": node.GetName()}


def segment_info(segmentationNode, segmentId):
    segment = segmentationNode.GetSegmentation().GetSegment(segmentId)
    return {"id": segmentId, "name": segment.GetName()}


def decode_rgb(payload):
    return np.frombuffer(b64decode(payload["data"]), dtype=np.uint8).reshape(tuple(payload["shape"])).copy()


def decode_prompt_mask(mask):
    return np.frombuffer(b64decode(mask["data"]), dtype=np.uint8).reshape(tuple(mask["shape"]))


def input_overlay(image, prompts):
    result = image.copy()
    if prompts.get("mask"):
        result = mask_overlay(result, decode_prompt_mask(prompts["mask"]), (32, 220, 255), 0.35)
    if prompts.get("box"):
        draw_rect(result, prompts["box"], (255, 216, 0))
    for point, label in zip(prompts.get("points", []), prompts.get("labels", [])):
        draw_point(result, point, (42, 255, 96) if label else (255, 64, 64))
    return result


def mask_overlay(image, mask, color, alpha):
    result = image.copy()
    active = mask > 0
    if active.any():
        color = np.array(color, dtype=np.float32)
        result[active] = (result[active].astype(np.float32) * (1.0 - alpha) + color * alpha).astype(np.uint8)
    return result


def draw_rect(image, box, color):
    height, width = image.shape[:2]
    x0, y0, x1, y1 = [int(round(value)) for value in box]
    x0, x1 = sorted((clip(x0, 0, width - 1), clip(x1, 0, width - 1)))
    y0, y1 = sorted((clip(y0, 0, height - 1), clip(y1, 0, height - 1)))
    for offset in range(2):
        image[clip(y0 + offset, 0, height - 1), x0:x1 + 1] = color
        image[clip(y1 - offset, 0, height - 1), x0:x1 + 1] = color
        image[y0:y1 + 1, clip(x0 + offset, 0, width - 1)] = color
        image[y0:y1 + 1, clip(x1 - offset, 0, width - 1)] = color


def draw_point(image, point, color):
    height, width = image.shape[:2]
    x, y = [int(round(value)) for value in point]
    radius = 5
    for row in range(max(0, y - radius), min(height, y + radius + 1)):
        for column in range(max(0, x - radius), min(width, x + radius + 1)):
            if (row - y) ** 2 + (column - x) ** 2 <= radius ** 2:
                image[row, column] = color


def clip(value, low, high):
    return max(low, min(high, value))


def save_rgb_png(path, image):
    height, width, _ = image.shape
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(width, height, 1)
    scalars = numpy_support.numpy_to_vtk(
        np.ascontiguousarray(np.flipud(image)).reshape(-1, 3),
        deep=True,
        array_type=vtk.VTK_UNSIGNED_CHAR,
    )
    scalars.SetNumberOfComponents(3)
    vtk_image.GetPointData().SetScalars(scalars)
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(vtk_image)
    writer.Write()
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"Failed to write debug image to {path}")
