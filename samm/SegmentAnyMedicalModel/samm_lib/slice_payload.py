from base64 import b64encode
from math import ceil, floor

import numpy as np
import slicer
import vtk

from .segmentation_writer import has_binary_labelmap


def slice_payload(volumeNode, positiveNode, negativeNode, boxNode, viewName):
    payload, sliceSpec = image_slice_payload(volumeNode, viewName)
    points, labels, box, mask = prompt_payload(volumeNode, sliceSpec, positiveNode, negativeNode, boxNode, payload["image"]["shape"][:2])
    payload.update({"points": points, "labels": labels})
    if box:
        payload["box"] = box
    if mask:
        payload["mask"] = mask
    return payload, sliceSpec


def image_slice_payload(volumeNode, viewName):
    volume = slicer.util.arrayFromVolume(volumeNode)
    sliceSpec = slice_spec(volumeNode, volume.shape, viewName)
    return image_payload(volume, sliceSpec, volumeNode.GetDisplayNode()), sliceSpec


def image_slice_payloads(volumeNode, viewName):
    volume = slicer.util.arrayFromVolume(volumeNode)
    displayNode = volumeNode.GetDisplayNode()
    for spec in image_slice_specs(volumeNode, viewName, volume.shape):
        yield image_payload(volume, spec, displayNode), spec


def image_slice_specs(volumeNode, viewName, volumeShape=None):
    volumeShape = volumeShape or slicer.util.arrayFromVolume(volumeNode).shape
    _, axis, _ = slice_axes(volumeNode, viewName)
    return [{"view": viewName, "axis": axis, "index": index} for index in range(volumeShape[axis])]


def volume_box_payloads(volumeNode, viewName, roiNode, specs=None):
    volume = slicer.util.arrayFromVolume(volumeNode)
    specs = specs or volume_box_specs(volumeNode, volume.shape, viewName, roiNode)
    displayNode = volumeNode.GetDisplayNode()
    for sliceSpec, box in specs:
        yield image_payload(volume, sliceSpec, displayNode), sliceSpec, box


def volume_box_specs(volumeNode, volumeShape, viewName, roiNode):
    _, axis, _ = slice_axes(volumeNode, viewName)
    low, high = roi_array_bounds(volumeNode, roiNode, volumeShape)
    start, end = low[axis], high[axis]
    planeAxes = [item for item in range(3) if item != axis]
    rowAxis, columnAxis = planeAxes
    box = [low[columnAxis], low[rowAxis], high[columnAxis], high[rowAxis]]
    if start > end or box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError("3D box does not overlap the selected view.")
    return [({"view": viewName, "axis": axis, "index": index}, box) for index in range(start, end + 1)]


def roi_array_bounds(volumeNode, roiNode, volumeShape):
    bounds = [0.0] * 6
    roiNode.GetRASBounds(bounds)
    corners = [
        [x, y, z, 1.0]
        for x in bounds[0:2]
        for y in bounds[2:4]
        for z in bounds[4:6]
    ]
    points = [array_position_from_ras(volumeNode, ras) for ras in corners]
    low = [max(0, floor(min(point[axis] for point in points))) for axis in range(3)]
    high = [min(volumeShape[axis] - 1, ceil(max(point[axis] for point in points))) for axis in range(3)]
    if any(low[axis] > high[axis] for axis in range(3)):
        raise ValueError("3D box does not overlap the input volume.")
    return low, high


def image_slice_count(volumeNode, viewName):
    return len(image_slice_specs(volumeNode, viewName))


def image_payload(volume, sliceSpec, displayNode):
    image = rgb_slice(volume, sliceSpec, displayNode)
    return {"image": {"shape": list(image.shape), "data": b64encode(image.tobytes()).decode("ascii")}}


def prompt_payload(volumeNode, sliceSpec, positiveNode, negativeNode, boxNode, shape, require_slice=True, use_points=True, use_box=True, mask=None, text=None):
    points, labels = prompt_points(volumeNode, sliceSpec, positiveNode, negativeNode, shape, require_slice) if use_points else ([], [])
    box = prompt_box(volumeNode, sliceSpec, boxNode, shape) if use_box else None
    if not points and not box and not mask and not text:
        raise ValueError(f"Enable and place prompt points on the {sliceSpec['view']} slice, a 2D box, a mask, or text.")
    return points, labels, box, mask


def segment_mask_payload(segmentationNode, segmentId, volumeNode, sliceSpec, shape):
    if not segmentationNode or not segmentId or not has_binary_labelmap(segmentationNode, segmentId):
        return None
    volumeShape = slicer.util.arrayFromVolume(volumeNode).shape
    maskVolume = slicer.util.arrayFromSegmentBinaryLabelmap(segmentationNode, segmentId, volumeNode)
    if maskVolume.shape != volumeShape:
        raise ValueError(f"Existing segment shape {maskVolume.shape} does not match volume shape {volumeShape}")
    target = [slice(None), slice(None), slice(None)]
    target[sliceSpec["axis"]] = sliceSpec["index"]
    mask = (maskVolume[tuple(target)] > 0).astype(np.uint8)
    if mask.shape != tuple(shape):
        raise ValueError(f"Mask prompt shape {mask.shape} does not match selected slice shape {tuple(shape)}")
    return {"shape": list(mask.shape), "data": b64encode(mask.tobytes()).decode("ascii")}


def prompt_box(volumeNode, sliceSpec, boxNode, shape):
    count = boxNode.GetNumberOfControlPoints()
    if count == 0:
        return None
    height, width = shape
    corners = box_corner_points(volumeNode, sliceSpec, boxNode)
    rows = [row for row, column in corners]
    columns = [column for row, column in corners]
    x0, x1 = max(0, min(columns)), min(width - 1, max(columns))
    y0, y1 = max(0, min(rows)), min(height - 1, max(rows))
    if x0 >= x1 or y0 >= y1:
        raise ValueError("Box prompt does not cover the selected view.")
    return [x0, y0, x1, y1]


def box_corner_points(volumeNode, sliceSpec, boxNode):
    origin = [0.0] * 3
    xAxis = [0.0] * 3
    yAxis = [0.0] * 3
    normal = [0.0] * 3
    boxNode.GetOrigin(origin)
    boxNode.GetAxes(xAxis, yAxis, normal)
    bounds = boxNode.GetPlaneBounds()
    corners = []
    for x, y in ((bounds[0], bounds[2]), (bounds[1], bounds[3])):
        ras = [origin[axis] + x * xAxis[axis] + y * yAxis[axis] for axis in range(3)] + [1.0]
        arrayPoint = [round(value) for value in array_position_from_ras(volumeNode, ras)]
        corners.append(plane_point(arrayPoint, sliceSpec))
    return corners


def slice_spec(volumeNode, volumeShape, viewName, index=None):
    sliceNode, axis, ijkAxis = slice_axes(volumeNode, viewName)
    ras = sliceNode.GetXYToRAS().MultiplyPoint([0.0, 0.0, 0.0, 1.0])
    index = round(ras_to_ijk(volumeNode, ras)[ijkAxis]) if index is None else index
    if index < 0 or index >= volumeShape[axis]:
        raise IndexError(f"{viewName} slice index {index} is outside axis {axis} with size {volumeShape[axis]}")
    return {"view": viewName, "axis": axis, "index": index}


def slice_axes(volumeNode, viewName):
    sliceNode = slicer.app.layoutManager().sliceWidget(viewName).sliceController().mrmlSliceNode()
    ijkAxis = slice_ijk_axis(volumeNode, sliceNode)
    return sliceNode, 2 - ijkAxis, ijkAxis


def slice_ijk_axis(volumeNode, sliceNode):
    normal = np.array([sliceNode.GetXYToRAS().GetElement(row, 2) for row in range(3)])
    directions = np.zeros((3, 3))
    volumeNode.GetIJKToRASDirections(directions)
    return int(np.argmax(np.abs(directions.T @ normal)))


def rgb_slice(volume, sliceSpec, displayNode):
    gray = normalize(np.take(volume, sliceSpec["index"], axis=sliceSpec["axis"]), displayNode)
    return np.repeat(gray[:, :, None], 3, axis=2)


def normalize(image, displayNode):
    values = image.astype(np.float32)
    low, high = value_range(values, displayNode)
    if high <= low:
        return np.zeros(values.shape, dtype=np.uint8)
    values = np.clip(values, low, high)
    return ((values - low) * (255.0 / (high - low))).astype(np.uint8)


def value_range(values, displayNode):
    if displayNode and displayNode.GetWindow() > 0:
        window = displayNode.GetWindow()
        level = displayNode.GetLevel()
        return level - window / 2.0, level + window / 2.0
    return float(values.min()), float(values.max())


def prompt_points(volumeNode, sliceSpec, positiveNode, negativeNode, shape, require_slice=True):
    points, labels = [], []
    add_prompt_points(points, labels, volumeNode, sliceSpec, positiveNode, 1, shape, require_slice)
    add_prompt_points(points, labels, volumeNode, sliceSpec, negativeNode, 0, shape, require_slice)
    return points, labels


def add_prompt_points(points, labels, volumeNode, sliceSpec, node, label, shape, require_slice=True):
    height, width = shape
    for index in range(node.GetNumberOfControlPoints()):
        arrayPoint = control_point_array_position(volumeNode, node, index)
        row, column = plane_point(arrayPoint, sliceSpec)
        if (not require_slice or arrayPoint[sliceSpec["axis"]] == sliceSpec["index"]) and 0 <= column < width and 0 <= row < height:
            points.append([column, row])
            labels.append(label)


def control_point_array_position(volumeNode, node, index):
    ras = vtk.vtkVector3d(0.0, 0.0, 0.0)
    node.GetNthControlPointPosition(index, ras)
    return [round(value) for value in array_position_from_ras(volumeNode, [ras[0], ras[1], ras[2], 1.0])]


def array_position_from_ras(volumeNode, ras):
    i, j, k = ras_to_ijk(volumeNode, ras)[:3]
    return [k, j, i]


def plane_point(arrayPoint, sliceSpec):
    return [arrayPoint[axis] for axis in range(3) if axis != sliceSpec["axis"]]


def ras_to_ijk(volumeNode, ras):
    matrix = vtk.vtkMatrix4x4()
    volumeNode.GetRASToIJKMatrix(matrix)
    return matrix.MultiplyPoint(ras)
