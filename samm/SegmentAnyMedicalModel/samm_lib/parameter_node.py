from typing import Annotated

from slicer import vtkMRMLMarkupsFiducialNode, vtkMRMLMarkupsPlaneNode, vtkMRMLMarkupsROINode, vtkMRMLScalarVolumeNode, vtkMRMLSegmentationNode
from slicer.parameterNodeWrapper import Choice, parameterNodeWrapper

from .finetune_models import FINETUNE_BASE_MODEL_LABELS
from .protocol import MODEL_IDS, SLICE_VIEWS, WEIGHT_IDS


@parameterNodeWrapper
class SegmentAnyMedicalModelParameterNode:
    inputVolume: vtkMRMLScalarVolumeNode
    segmentation: vtkMRMLSegmentationNode
    maskSegmentation: vtkMRMLSegmentationNode
    positivePrompts: vtkMRMLMarkupsFiducialNode
    negativePrompts: vtkMRMLMarkupsFiducialNode
    boxPrompts: vtkMRMLMarkupsPlaneNode
    volumeBoxPrompt: vtkMRMLMarkupsROINode
    modelFamily: Annotated[str, Choice(MODEL_IDS)] = MODEL_IDS[0]
    weightName: str = WEIGHT_IDS[0]
    sliceView: Annotated[str, Choice(SLICE_VIEWS)] = SLICE_VIEWS[0]
    selectedSegmentId: str = ""
    maskSegmentId: str = ""
    autoPredict2d: bool = False
    sendPointPrompts: bool = True
    sendBoxPrompt: bool = True
    sendMaskPrompt: bool = False
    sendTextPrompt: bool = True
    textPrompt: str = ""
    finetuneDatasetName: str = "my_task"
    finetuneValCount: int = 1
    finetuneAxis0: bool = True
    finetuneAxis1: bool = False
    finetuneAxis2: bool = False
    finetuneUseWindow: bool = False
    finetuneWindowMin: float = -100.0
    finetuneWindowMax: float = 300.0
    finetuneRunName: str = ""
    finetuneBaseModel: Annotated[str, Choice(FINETUNE_BASE_MODEL_LABELS)] = FINETUNE_BASE_MODEL_LABELS[0]
    finetuneEpochs: int = 25
    finetuneBatchSize: int = 1
    finetuneNumWorkers: int = 0
    finetuneNumFrames: int = 4
    finetuneEvalMaxSegmentedVolumes: int = 0
    clearPointPrompts: bool = True
    clearBoxPrompt: bool = True
    clearMaskPrompt: bool = False
    debugPredictionLog: bool = False
