from .sam_predictor import SamPredictorBackend


DATA_FORMAT = "samm-mobile-sam-embedding-data"


class MobileSamBackend(SamPredictorBackend):
    def __init__(self):
        super().__init__("mobile_sam", "mobile_sam", DATA_FORMAT)
