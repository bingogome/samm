from base64 import b64decode
from math import ceil, floor

import numpy as np
import slicer


def write_slice(response, maskVolume, segmentationNode, segmentId, referenceVolumeNode, sliceSpec, clip_box=None):
    return write_slices(
        [(response, sliceSpec)],
        maskVolume,
        segmentationNode,
        segmentId,
        referenceVolumeNode,
        clip_box=clip_box,
    )


def write_slices(items, maskVolume, segmentationNode, segmentId, referenceVolumeNode, clip_box=None):
    volumeShape = slicer.util.arrayFromVolume(referenceVolumeNode).shape
    maskVolume = current_mask_volume(maskVolume, segmentationNode, segmentId, referenceVolumeNode, volumeShape)
    for response, sliceSpec in items:
        mask = clip_mask(decode_mask(response), clip_box) if clip_box else decode_mask(response)
        target = [slice(None), slice(None), slice(None)]
        target[sliceSpec["axis"]] = sliceSpec["index"]
        if maskVolume[tuple(target)].shape != mask.shape:
            raise ValueError(f"Mask shape {mask.shape} does not match target slice shape {maskVolume[tuple(target)].shape}")
        maskVolume[tuple(target)] = mask
    segmentationNode.CreateDefaultDisplayNodes()
    segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(referenceVolumeNode)
    slicer.util.updateSegmentBinaryLabelmapFromArray(maskVolume, segmentationNode, segmentId, referenceVolumeNode)
    return maskVolume, segmentId


def decode_mask(response):
    height, width = response["shape"]
    return np.frombuffer(b64decode(response["data"]), dtype=np.uint8).reshape((height, width))


def clip_mask(mask, box):
    height, width = mask.shape
    x0 = max(0, min(width - 1, floor(box[0])))
    y0 = max(0, min(height - 1, floor(box[1])))
    x1 = max(0, min(width - 1, ceil(box[2])))
    y1 = max(0, min(height - 1, ceil(box[3])))
    clipped = np.zeros_like(mask)
    if x0 <= x1 and y0 <= y1:
        clipped[y0:y1 + 1, x0:x1 + 1] = mask[y0:y1 + 1, x0:x1 + 1]
    return clipped


def current_mask_volume(maskVolume, segmentationNode, segmentId, referenceVolumeNode, volumeShape):
    if maskVolume is not None and maskVolume.shape == volumeShape:
        return maskVolume
    if has_binary_labelmap(segmentationNode, segmentId):
        existing = slicer.util.arrayFromSegmentBinaryLabelmap(segmentationNode, segmentId, referenceVolumeNode)
        if existing.shape != volumeShape:
            raise ValueError(f"Existing segment shape {existing.shape} does not match volume shape {volumeShape}")
        return existing.astype(np.uint8, copy=True)
    return np.zeros(volumeShape, dtype=np.uint8)


def has_binary_labelmap(segmentationNode, segmentId):
    import vtkSegmentationCorePython as vtkSegmentationCore

    name = vtkSegmentationCore.vtkSegmentationConverter.GetSegmentationBinaryLabelmapRepresentationName()
    return bool(segmentationNode.GetSegmentation().GetSegment(segmentId).GetRepresentation(name))
