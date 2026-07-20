from .fastsam import FastSamBackend
from .medsam import MedSamBackend
from .medsam_text import MedSamTextBackend
from .mobile_sam import MobileSamBackend
from .sam1 import Sam1Backend
from .sam2 import Sam2Backend
from .sam3 import Sam3Backend


def default_backends():
    return {
        "fastsam": FastSamBackend(),
        "sam1": Sam1Backend(),
        "sam2": Sam2Backend(),
        "medsam2": Sam2Backend("MedSAM2", "samm-medsam2-embedding-data"),
        "mobile_sam": MobileSamBackend(),
        "medsam": MedSamBackend(),
        "medsam_text": MedSamTextBackend(),
        "sam3": Sam3Backend(enable_video_propagation=True),
        "medical_sam3": Sam3Backend(
            enable_inst_interactivity=False,
            confidence_threshold=0.1,
            grounding_mask_mode="best",
        ),
    }


__all__ = [
    "FastSamBackend",
    "MedSamBackend",
    "MedSamTextBackend",
    "MobileSamBackend",
    "Sam1Backend",
    "Sam2Backend",
    "Sam3Backend",
    "default_backends",
]
