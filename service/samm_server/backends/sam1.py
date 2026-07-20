from .sam_predictor import SamPredictorBackend


DATA_FORMAT = "samm-sam1-embedding-data"


class Sam1Backend(SamPredictorBackend):
    def __init__(self):
        super().__init__("segment_anything", "segment-anything", DATA_FORMAT)
