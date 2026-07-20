import numpy as np


def merge_label_masks(volume_shape, masks):
    labels = np.zeros(volume_shape, dtype=np.uint16)
    segments = []
    for segment_id, name, mask in masks:
        mask = np.asarray(mask) > 0
        if mask.shape != volume_shape:
            raise ValueError(f"Segment {name} shape {mask.shape} does not match volume shape {volume_shape}")
        voxels = int(mask.sum())
        if not voxels:
            continue
        if np.any((labels > 0) & mask):
            raise ValueError(f"Segment {name} overlaps another exported segment")
        label = len(segments) + 1
        labels[mask] = label
        segments.append({"label": label, "id": segment_id, "name": name, "voxels": voxels})
    if not segments:
        raise ValueError("No non-empty segments to export.")
    return labels, segments
