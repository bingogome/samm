import vtk

from pathlib import Path
import qt
import slicer
import time
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleWidget
from slicer.util import VTKObservationMixin

from .finetune_models import FINETUNE_BASE_MODEL_LABELS, finetune_base_checkpoint
from .logic import SegmentAnyMedicalModelLogic
from .parameter_node import SegmentAnyMedicalModelParameterNode
from .tooltips import ToolTipManager


class SegmentAnyMedicalModelWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    def __init__(self, parent=None):
        super().__init__(parent)
        VTKObservationMixin.__init__(self)
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None
        self._models = {}
        self._weights = {}
        self._modelIdsByLabel = {}
        self._weightIdsByLabel = {}
        self._prepared = None
        self._promptObserverNodes = []
        self._autoPredictSliceNode = None
        self._segmentationObserverNode = None
        self._maskSegmentationObserverNode = None
        self._segmentIdsByLabel = {}
        self._maskSegmentIdsByLabel = {}
        self._segmentState = None
        self._maskSegmentState = None
        self._updatingSegmentCombo = False
        self._updatingMaskSegmentCombo = False
        self._updatingSliceViewControls = False
        self._syncingDataSelectors = False
        self._startingServer = False
        self.serverStartAttempts = 0
        self.embeddingJobPollMs = 1000
        self.embeddingSubmitMs = 20
        self.embeddingSubmitBatchSize = 1
        self.boxVolumeJobPollMs = 1000
        self.boxVolumeSubmitMs = 20
        self.boxVolumeSubmitBatchSize = 1
        self.autoPredictDelayMs = 150
        self.heartbeatMs = 3000
        self.autosaveMs = 60000
        self._heartbeatActive = False
        self._slicerClosing = False
        self._segmentationDirty = False
        self._embeddingDirty = False
        self._lastAutosave = 0
        self._lastAutosaveManifest = {}
        self._autoPredictPending = False
        self._autoPredictRunning = False
        self._syncingAutoPredictPrompts = False
        self._finetuneJob = None
        self._updatingFinetuneDatasetCombo = False
        self._finetuneDatasetNamesByLabel = {}
        self.finetuneJobPollMs = 2000
        self.toolTips = None
        self._sectionInfoLabels = []

    def setup(self):
        super().setup()
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/SegmentAnyMedicalModel.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        uiWidget.setMRMLScene(slicer.mrmlScene)
        uiWidget.layout().removeWidget(self.ui.serviceCollapsibleButton)
        uiWidget.layout().insertWidget(1, self.ui.serviceCollapsibleButton)
        self.addSectionExplanations()
        self.setComboLabels(self.ui.finetuneBaseModelComboBox, FINETUNE_BASE_MODEL_LABELS)
        self.configureWrappingLabels()
        self.toolTips = ToolTipManager(self)
        self.toolTips.configure()

        self.logic = SegmentAnyMedicalModelLogic()
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)
        self.ui.startServerButton.connect("clicked(bool)", self.onStartServer)
        self.ui.stopServerButton.connect("clicked(bool)", self.onStopServer)
        self.ui.showServerLogButton.connect("clicked(bool)", self.onShowServerLog)
        self.ui.prepareModelButton.connect("clicked(bool)", self.onPrepareModel)
        self.ui.offloadModelButton.connect("clicked(bool)", self.onOffloadModel)
        self.ui.positivePromptButton.connect("clicked(bool)", self.onPlacePositivePrompt)
        self.ui.negativePromptButton.connect("clicked(bool)", self.onPlaceNegativePrompt)
        self.ui.boxPromptButton.connect("clicked(bool)", self.onPlaceBoxPrompt)
        self.ui.volumeBoxPromptButton.connect("clicked(bool)", self.onPlaceVolumeBoxPrompt)
        self.ui.clear2dPromptsButton.connect("clicked(bool)", self.onClear2dPrompts)
        self.ui.clear3dPromptsButton.connect("clicked(bool)", self.onClear3dPrompts)
        self.ui.sendPointPromptsCheckBox.connect("toggled(bool)", self.onPromptSendToggled)
        self.ui.sendBoxPromptCheckBox.connect("toggled(bool)", self.onPromptSendToggled)
        self.ui.sendMaskPromptCheckBox.connect("toggled(bool)", self.onPromptSendToggled)
        self.ui.sendTextPromptCheckBox.connect("toggled(bool)", self.onPromptSendToggled)
        self.ui.textPromptLineEdit.connect("textChanged(QString)", self.onTextPromptChanged)
        self.ui.clearPointPromptsCheckBox.connect("toggled(bool)", self.onClearOptionsToggled)
        self.ui.clearBoxPromptCheckBox.connect("toggled(bool)", self.onClearOptionsToggled)
        self.ui.clearMaskPromptCheckBox.connect("toggled(bool)", self.onClearOptionsToggled)
        self.ui.debugPredictionLogCheckBox.connect("toggled(bool)", self.onDebugPredictionLogToggled)
        self.ui.embedAllButton.connect("clicked(bool)", self.onEmbedAll)
        self.ui.embedAllAxesButton.connect("clicked(bool)", self.onEmbedAllAxes)
        self.ui.saveEmbeddingsButton.connect("clicked(bool)", self.onSaveEmbeddings)
        self.ui.loadEmbeddingsButton.connect("clicked(bool)", self.onLoadEmbeddings)
        self.ui.predictSliceButton.connect("clicked(bool)", self.onPredictSlice)
        self.ui.predictBoxVolumeButton.connect("clicked(bool)", self.onPredictBoxVolume)
        self.ui.predictBoxVolumeVideoButton.connect("clicked(bool)", self.onPredictBoxVolumeVideo)
        self.ui.cancelBoxVolumeVideoButton.connect("clicked(bool)", self.onCancelBoxVolumeVideo)
        self.ui.autoPredictCheckBox.connect("toggled(bool)", self.onAutoPredictToggled)
        self.ui.predictAutoVideoButton.connect("clicked(bool)", self.onPredictAutoVideo)
        self.ui.cancelAutoVideoButton.connect("clicked(bool)", self.onCancelAutoVideo)
        self.ui.addSegmentationButton.connect("clicked(bool)", self.onAddSegmentation)
        self.ui.removeSegmentationButton.connect("clicked(bool)", self.onRemoveSegmentation)
        self.ui.saveSegmentationButton.connect("clicked(bool)", self.onSaveSegmentation)
        self.ui.loadSegmentationButton.connect("clicked(bool)", self.onLoadSegmentation)
        self.ui.addSegmentButton.connect("clicked(bool)", self.onAddSegment)
        self.ui.removeSegmentButton.connect("clicked(bool)", self.onRemoveSegment)
        self.ui.inputVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onInputVolumeNodeChanged)
        self.ui.finetuneVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onInputVolumeNodeChanged)
        self.ui.segmentationSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSegmentationNodeChanged)
        self.ui.finetuneSegmentationSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSegmentationNodeChanged)
        self.ui.addSegmentedVolumeButton.connect("clicked(bool)", self.onAddSegmentedVolume)
        self.ui.buildFinetuneDatasetButton.connect("clicked(bool)", self.onBuildFinetuneDataset)
        self.ui.showFinetuneReportButton.connect("clicked(bool)", self.onShowFinetuneReport)
        self.ui.startFinetuneTrainingButton.connect("clicked(bool)", self.onStartFinetuneTraining)
        self.ui.startFinetuneEvalButton.connect("clicked(bool)", self.onStartFinetuneEval)
        self.ui.cancelFinetuneJobButton.connect("clicked(bool)", self.onCancelFinetuneJob)
        self.ui.newFinetuneDatasetButton.connect("clicked(bool)", self.onNewFinetuneDataset)
        self.ui.refreshFinetuneDatasetsButton.connect("clicked(bool)", self.refreshFinetuneDatasets)
        self.ui.finetuneDatasetComboBox.connect("currentIndexChanged(int)", self.onFinetuneDatasetChoiceChanged)
        self.ui.finetuneDatasetLineEdit.connect("textChanged(QString)", self.onFinetuneSettingsChanged)
        self.ui.finetuneValCountSpinBox.connect("valueChanged(int)", self.onFinetuneSettingsChanged)
        self.ui.finetuneAxis0CheckBox.connect("toggled(bool)", self.onFinetuneSettingsChanged)
        self.ui.finetuneAxis1CheckBox.connect("toggled(bool)", self.onFinetuneSettingsChanged)
        self.ui.finetuneAxis2CheckBox.connect("toggled(bool)", self.onFinetuneSettingsChanged)
        self.ui.finetuneUseWindowCheckBox.connect("toggled(bool)", self.onFinetuneSettingsChanged)
        self.ui.finetuneBaseModelComboBox.connect("currentIndexChanged(int)", self.onFinetuneSettingsChanged)
        self.ui.modelFamilyComboBox.connect("currentIndexChanged(int)", self.onModelFamilyChanged)
        self.ui.weightComboBox.connect("currentIndexChanged(int)", self.onWeightChanged)
        self.ui.sliceViewComboBox.connect("currentIndexChanged(int)", self.onDataSliceViewChanged)
        self.ui.predictionSliceViewComboBox.connect("currentIndexChanged(int)", self.onPredictionSliceViewChanged)
        self.ui.embeddingSliceViewComboBox.connect("currentIndexChanged(int)", self.onEmbeddingSliceViewChanged)
        self.ui.segmentComboBox.connect("currentIndexChanged(int)", self.onSegmentChanged)
        self.ui.maskSegmentComboBox.connect("currentIndexChanged(int)", self.onMaskSegmentChanged)
        slicer.app.connect("aboutToQuit()", self.onSlicerAboutToQuit)
        self.initializeParameterNode()
        self.refreshFinetuneDatasets()
        self.updateControls()
        self._heartbeatActive = True
        qt.QTimer.singleShot(self.heartbeatMs, self.heartbeat)

    def addSectionExplanations(self):
        sections = (
            (
                self.ui.serviceCollapsibleButton,
                "serviceInfoLabel",
                "Start the local gateway, choose a model weight, and prepare it on the GPU. Model workers start on demand.",
            ),
            (
                self.ui.dataCollapsibleButton,
                "dataInfoLabel",
                "Choose the input volume and output segment, then select the slice view shared by prompts and prediction.",
            ),
            (
                self.ui.predictionCollapsibleButton,
                "predictionInfoLabel",
                "Place prompts supported by the prepared model, then predict the current slice, an ROI volume, or slices as video.",
            ),
            (
                self.ui.embeddingCollapsibleButton,
                "embeddingInfoLabel",
                "Precompute slice embeddings for the selected volume, weight, and view to make compatible predictions faster.",
            ),
            (
                self.ui.finetuneCollapsibleButton,
                "finetuneInfoLabel",
                "Build labeled train/validation data from Slicer segmentations, then run, monitor, evaluate, and register fine-tuned weights.",
            ),
        )
        style = (
            "QLabel { background-color: palette(alternate-base); "
            "border: 1px solid palette(mid); border-radius: 3px; padding: 6px; }"
        )
        self._sectionInfoLabels = []
        for section, objectName, text in sections:
            label = qt.QLabel(text, section)
            label.objectName = objectName
            label.wordWrap = True
            label.styleSheet = style
            label.toolTip = text
            label.setAttribute(qt.Qt.WA_AlwaysShowToolTips, True)
            label.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Preferred)
            section.layout().insertRow(0, label)
            self._sectionInfoLabels.append(label)

    def configureWrappingLabels(self):
        labels = [
            self.ui.statusLabel,
            self.ui.maskSourceLabel,
            self.ui.clear2dOptionsLabel,
            self.ui.embeddingStatusLabel,
            self.ui.boxVolumeStatusLabel,
            self.ui.modelAvailabilityLabel,
            self.ui.finetuneDatasetStatusLabel,
            self.ui.finetuneJobStatusLabel,
        ] + self._sectionInfoLabels
        for label in labels:
            label.wordWrap = True
            label.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Preferred)

    def cleanup(self):
        self._heartbeatActive = False
        self.removeObservers()

    def enter(self):
        self.initializeParameterNode()

    def exit(self):
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateControls)
            self._parameterNodeGuiTag = None
            self.removePromptObservers()
            self.removeAutoPredictObserver()
            self.removeSegmentationObserver()
            self.removeMaskSegmentationObserver()

    def onSceneStartClose(self, caller, event):
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event):
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self):
        parameterNode = self.logic.getParameterNode()
        if not parameterNode.inputVolume:
            parameterNode.inputVolume = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
        self.logic.ensure_scene_nodes(parameterNode)
        self.setParameterNode(parameterNode)
        if not self._parameterNode.inputVolume:
            self._parameterNode.inputVolume = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            self.syncDataSelectors()

    def setParameterNode(self, inputParameterNode: SegmentAnyMedicalModelParameterNode | None):
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateControls)
            self.removePromptObservers()
            self.removeAutoPredictObserver()
            self.removeSegmentationObserver()
            self.removeMaskSegmentationObserver()

        self._parameterNode = inputParameterNode

        if self._parameterNode:
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateControls)
            self.addPromptObservers()
            if not self._models:
                self.setLocalModels()
            self.updateSegmentationObserver()
            self.updateMaskSegmentationObserver()
            self.syncDataSelectors()
            self.syncSliceViewControls()
            self.refreshSegments()
            self.refreshMaskSegments()
            self.updateControls()
            self.updateModelAvailability()
        else:
            self._segmentState = None
            self._maskSegmentState = None
            self.setComboLabels(self.ui.segmentComboBox, [])
            self.setComboLabels(self.ui.maskSegmentComboBox, [])
            self.updateEmbeddingView()
            self.updateControls()

    def addPromptObservers(self):
        self._promptObserverNodes = [
            self._parameterNode.positivePrompts,
            self._parameterNode.negativePrompts,
            self._parameterNode.boxPrompts,
            self._parameterNode.volumeBoxPrompt,
        ]
        for node in self._promptObserverNodes:
            self.addObserver(node, vtk.vtkCommand.ModifiedEvent, self.onPromptModified)

    def removePromptObservers(self):
        for node in self._promptObserverNodes:
            self.removeObserver(node, vtk.vtkCommand.ModifiedEvent, self.onPromptModified)
        self._promptObserverNodes = []

    def onPromptModified(self, caller=None, event=None):
        if self._parameterNode and self.logic.auto_video_prediction_running() and caller in (
            self._parameterNode.positivePrompts,
            self._parameterNode.negativePrompts,
            self._parameterNode.boxPrompts,
        ):
            self.invalidateVideoVolumePrediction("seed prompt changed")
        elif (
            self._parameterNode
            and caller is self._parameterNode.volumeBoxPrompt
            and self.logic.video_box_volume_prediction_running()
        ):
            self.invalidateVideoVolumePrediction("3D box ROI changed")
        self.updateControls()
        if not self._syncingAutoPredictPrompts:
            self.scheduleAutoPredict()

    def updateAutoPredictObserver(self):
        node = self.autoPredictSliceNode() if self._parameterNode and self.ui.autoPredictCheckBox.checked else None
        if node is self._autoPredictSliceNode:
            return
        self.removeAutoPredictObserver()
        self._autoPredictSliceNode = node
        if node:
            self.addObserver(node, vtk.vtkCommand.ModifiedEvent, self.onAutoPredictSliceModified)

    def removeAutoPredictObserver(self):
        if self._autoPredictSliceNode:
            self.removeObserver(self._autoPredictSliceNode, vtk.vtkCommand.ModifiedEvent, self.onAutoPredictSliceModified)
            self._autoPredictSliceNode = None

    def autoPredictSliceNode(self):
        return slicer.app.layoutManager().sliceWidget(self._parameterNode.sliceView).sliceController().mrmlSliceNode()

    def onAutoPredictSliceModified(self, caller=None, event=None):
        self.syncAutoPredictPrompts()
        self.scheduleAutoPredict()

    def updateSegmentationObserver(self):
        node = self._parameterNode.segmentation if self._parameterNode else None
        if node is self._segmentationObserverNode:
            return
        self.removeSegmentationObserver()
        self._segmentationObserverNode = node
        if node:
            self.addObserver(node, vtk.vtkCommand.ModifiedEvent, self.onSegmentationModified)

    def removeSegmentationObserver(self):
        if self._segmentationObserverNode:
            self.removeObserver(self._segmentationObserverNode, vtk.vtkCommand.ModifiedEvent, self.onSegmentationModified)
            self._segmentationObserverNode = None

    def onSegmentationModified(self, caller=None, event=None):
        if self._updatingSegmentCombo:
            return
        self._segmentationDirty = True
        self._segmentState = None
        self.refreshSegments()
        if self._parameterNode and caller is self._parameterNode.maskSegmentation:
            self._maskSegmentState = None
            self.refreshMaskSegments()
        self.updateControls()

    def updateMaskSegmentationObserver(self):
        node = self._parameterNode.maskSegmentation if self._parameterNode else None
        if node is self._maskSegmentationObserverNode:
            return
        self.removeMaskSegmentationObserver()
        self._maskSegmentationObserverNode = node
        if node:
            self.addObserver(node, vtk.vtkCommand.ModifiedEvent, self.onMaskSegmentationModified)

    def removeMaskSegmentationObserver(self):
        if self._maskSegmentationObserverNode:
            self.removeObserver(self._maskSegmentationObserverNode, vtk.vtkCommand.ModifiedEvent, self.onMaskSegmentationModified)
            self._maskSegmentationObserverNode = None

    def onMaskSegmentationModified(self, caller=None, event=None):
        if self._updatingMaskSegmentCombo:
            return
        self._maskSegmentState = None
        self.refreshMaskSegments()
        self.updateControls()

    def sliceViewComboBoxes(self):
        return (
            self.ui.sliceViewComboBox,
            self.ui.predictionSliceViewComboBox,
            self.ui.embeddingSliceViewComboBox,
        )

    def setSliceViewControlsEnabled(self, enabled):
        for comboBox in self.sliceViewComboBoxes():
            comboBox.enabled = enabled

    def dataVolumeSelectors(self):
        return self.ui.inputVolumeSelector, self.ui.finetuneVolumeSelector

    def dataSegmentationSelectors(self):
        return self.ui.segmentationSelector, self.ui.finetuneSegmentationSelector

    def syncDataSelectors(self):
        if not self._parameterNode or self._syncingDataSelectors:
            return
        self._syncingDataSelectors = True
        try:
            for selector in self.dataVolumeSelectors():
                self.setCurrentNode(selector, self._parameterNode.inputVolume)
            for selector in self.dataSegmentationSelectors():
                self.setCurrentNode(selector, self._parameterNode.segmentation)
        finally:
            self._syncingDataSelectors = False

    def setCurrentNode(self, selector, node):
        if selector.currentNode() is not node:
            selector.setCurrentNode(node)

    def onInputVolumeNodeChanged(self, node):
        if self._syncingDataSelectors or not self._parameterNode:
            return
        self.invalidateVideoVolumePrediction("input volume changed")
        self._parameterNode.inputVolume = node
        self.syncDataSelectors()
        self.updateControls()

    def onSegmentationNodeChanged(self, node):
        if self._syncingDataSelectors or not self._parameterNode:
            return
        self.invalidateVideoVolumePrediction("output segmentation changed")
        self._parameterNode.segmentation = node
        self._segmentState = None
        self.syncDataSelectors()
        self.refreshSegments()
        self.updateControls()

    def setFinetuneBuildControlsEnabled(self, enabled):
        for widget in (
            self.ui.finetuneValCountSpinBox,
            self.ui.finetuneAxis0CheckBox,
            self.ui.finetuneAxis1CheckBox,
            self.ui.finetuneAxis2CheckBox,
            self.ui.finetuneUseWindowCheckBox,
            self.ui.finetuneWindowMinSpinBox,
            self.ui.finetuneWindowMaxSpinBox,
            self.ui.finetuneRunLineEdit,
            self.ui.finetuneBaseModelComboBox,
            self.ui.finetuneEpochsSpinBox,
            self.ui.finetuneBatchSizeSpinBox,
            self.ui.finetuneNumFramesSpinBox,
            self.ui.finetuneNumWorkersSpinBox,
            self.ui.finetuneEvalMaxSegmentedVolumesSpinBox,
        ):
            widget.enabled = enabled

    def updateFinetuneDatasetStatus(self):
        if not self.logic.client:
            status = self.logic.local_finetune_dataset_status(self._parameterNode)
        else:
            status = self.logic.finetune_dataset_status(self._parameterNode)
        self.ui.finetuneDatasetStatusLabel.text = (
            f"Source {status['source_segmented_volumes']} volume(s); "
            f"train {status['train_segmented_volumes']}; val {status['val_segmented_volumes']}"
        )
        return status

    def finetuneBuildReady(self, status):
        if not status:
            return False
        valCount = self.ui.finetuneValCountSpinBox.value
        return bool(
            status["source_segmented_volumes"] > 0
            and self.selectedFinetuneAxes()
            and (valCount == 0 or valCount < status["source_segmented_volumes"])
        )

    def selectedFinetuneAxes(self):
        axes = []
        if self.ui.finetuneAxis0CheckBox.checked:
            axes.append(0)
        if self.ui.finetuneAxis1CheckBox.checked:
            axes.append(1)
        if self.ui.finetuneAxis2CheckBox.checked:
            axes.append(2)
        return axes

    def finetuneWindow(self):
        if not self.ui.finetuneUseWindowCheckBox.checked:
            return None
        low = float(self.ui.finetuneWindowMinSpinBox.value)
        high = float(self.ui.finetuneWindowMaxSpinBox.value)
        if high <= low:
            raise ValueError("Finetuning window max must be greater than min.")
        return low, high

    def syncSliceViewControls(self):
        if not self._parameterNode:
            return
        self._updatingSliceViewControls = True
        try:
            for comboBox in self.sliceViewComboBoxes():
                comboBox.setCurrentText(self._parameterNode.sliceView)
        finally:
            self._updatingSliceViewControls = False

    def onDataSliceViewChanged(self, index=None):
        self.onSliceViewChanged(self.ui.sliceViewComboBox)

    def onPredictionSliceViewChanged(self, index=None):
        self.onSliceViewChanged(self.ui.predictionSliceViewComboBox)

    def onEmbeddingSliceViewChanged(self, index=None):
        self.onSliceViewChanged(self.ui.embeddingSliceViewComboBox)

    def onSliceViewChanged(self, comboBox):
        if self._updatingSliceViewControls or not self._parameterNode:
            return
        viewName = comboBox.currentText
        if not viewName:
            return
        if viewName != self._parameterNode.sliceView:
            self.invalidateVideoVolumePrediction("slice view changed")
        self._parameterNode.sliceView = viewName
        self.syncSliceViewControls()
        self.updateEmbeddingView()
        self.updateBoxVolumeView()
        self.updateControls()

    def updateControls(self, caller=None, event=None):
        if not self._parameterNode:
            self.ui.startServerButton.enabled = False
            self.ui.stopServerButton.enabled = False
            logPath = self.logic.server_log_path() if self.logic else None
            self.ui.showServerLogButton.enabled = bool(logPath and Path(logPath).exists())
            self.ui.prepareModelButton.enabled = False
            self.ui.offloadModelButton.enabled = False
            self.ui.modelFamilyComboBox.enabled = False
            self.ui.weightComboBox.enabled = False
            self.ui.positivePromptButton.enabled = False
            self.ui.negativePromptButton.enabled = False
            self.ui.boxPromptButton.enabled = False
            self.ui.volumeBoxPromptButton.enabled = False
            self.ui.clear2dPromptsButton.enabled = False
            self.ui.clear3dPromptsButton.enabled = False
            self.ui.sendPointPromptsCheckBox.enabled = False
            self.ui.sendBoxPromptCheckBox.enabled = False
            self.ui.sendMaskPromptCheckBox.enabled = False
            self.ui.sendTextPromptCheckBox.enabled = False
            self.ui.textPromptLabel.enabled = False
            self.ui.textPromptLineEdit.enabled = False
            self.ui.clear2dOptionsLabel.enabled = False
            self.ui.clearPointPromptsCheckBox.enabled = False
            self.ui.clearBoxPromptCheckBox.enabled = False
            self.ui.clearMaskPromptCheckBox.enabled = False
            self.ui.maskSourceLabel.enabled = False
            self.ui.maskSegmentationSelector.enabled = False
            self.ui.maskSegmentComboBox.enabled = False
            self.ui.debugPredictionLogCheckBox.enabled = False
            self.ui.embedAllButton.enabled = False
            self.ui.embedAllAxesButton.enabled = False
            self.ui.saveEmbeddingsButton.enabled = False
            self.ui.loadEmbeddingsButton.enabled = False
            self.ui.predictSliceButton.enabled = False
            self.ui.predictBoxVolumeButton.enabled = False
            self.ui.predictBoxVolumeVideoButton.enabled = False
            self.ui.cancelBoxVolumeVideoButton.enabled = False
            self.ui.autoPredictCheckBox.enabled = False
            self.ui.predictAutoVideoButton.enabled = False
            self.ui.cancelAutoVideoButton.enabled = False
            self.setAutoPredictChecked(False)
            self.ui.segmentComboBox.enabled = False
            self.ui.addSegmentationButton.enabled = False
            self.ui.removeSegmentationButton.enabled = False
            self.ui.saveSegmentationButton.enabled = False
            self.ui.loadSegmentationButton.enabled = False
            self.ui.addSegmentButton.enabled = False
            self.ui.removeSegmentButton.enabled = False
            self.ui.finetuneDatasetLineEdit.enabled = False
            self.ui.newFinetuneDatasetButton.enabled = False
            self.ui.finetuneDatasetComboBox.enabled = False
            self.ui.refreshFinetuneDatasetsButton.enabled = False
            self.ui.finetuneVolumeSelector.enabled = False
            self.ui.finetuneSegmentationSelector.enabled = False
            self.ui.addSegmentedVolumeButton.enabled = False
            self.ui.buildFinetuneDatasetButton.enabled = False
            self.ui.showFinetuneReportButton.enabled = False
            self.ui.startFinetuneTrainingButton.enabled = False
            self.ui.startFinetuneEvalButton.enabled = False
            self.ui.cancelFinetuneJobButton.enabled = False
            self.ui.finetuneDatasetStatusLabel.text = "No finetuning dataset selected"
            self.ui.finetuneJobStatusLabel.text = "No finetuning job"
            self.resetFinetuneJobProgress()
            self.setFinetuneBuildControlsEnabled(False)
            self.setSliceViewControlsEnabled(False)
            self.updateEmbeddingView()
            self.updateBoxVolumeView()
            self.updateAutoPredictObserver()
            self.toolTips.update()
            return
        self.updateSegmentationObserver()
        self.updateMaskSegmentationObserver()
        self.syncDataSelectors()
        self.syncSliceViewControls()
        self.syncSegmentCombo()
        self.syncMaskSegmentCombo()
        self.updateEmbeddingView()
        self.updateBoxVolumeView()
        connected = self.logic.client is not None
        started = self.logic.server_running()
        boxRunning = self.logic.box_volume_prediction_running()
        weight = self._weights.get(self.currentWeightId())
        capabilities = weight.get("capabilities", {}) if weight else {}
        textBlocksInteractive = self.textBlocksInteractivePrompts(weight)
        supportsPoints = capabilities.get("points", False) and not textBlocksInteractive
        supportsBox = capabilities.get("box", False)
        supportsMask = capabilities.get("mask", False) and not textBlocksInteractive
        supportsText = capabilities.get("text", False)
        supportsVolumeBox = capabilities.get("box_3d", False)
        supportsVideoPropagation = capabilities.get("video_propagation", False)
        supportsAutoPredict = capabilities.get("auto_predict_2d", False)
        supportsEmbeddings = capabilities.get("embeddings", False)
        sendMaskPrompt = self._parameterNode.sendMaskPrompt
        sendTextPrompt = self._parameterNode.sendTextPrompt
        hasSegmentation = bool(self._parameterNode.segmentation)
        hasSegment = bool(self._parameterNode.selectedSegmentId)
        hasMaskSegmentation = bool(self._parameterNode.maskSegmentation)
        hasMaskSegment = bool(self._parameterNode.maskSegmentId)
        hasVolume = bool(self._parameterNode.inputVolume)
        hasData = bool(self._parameterNode.inputVolume and self._parameterNode.segmentation and hasSegment)
        volumePromptCount = self.logic.volume_prompt_count(self._parameterNode)
        sendOptions = self.promptSendOptions(capabilities)
        videoSendOptions = self.videoPromptSendOptions(capabilities)
        hasSlicePrompt = self.hasEnabledSlicePrompt(sendOptions)
        hasVideoPoints = bool(videoSendOptions["use_points"] and self.logic.has_point_prompt(self._parameterNode))
        hasVideoBox = bool(videoSendOptions["use_box"] and self.logic.has_box_prompt(self._parameterNode))
        hasVideoMask = bool(videoSendOptions["use_mask"] and self.logic.has_mask_prompt(self._parameterNode))
        hasVideoText = bool(videoSendOptions["use_text"] and self.logic.has_text_prompt(self._parameterNode))
        hasVideoPrompt = hasVideoPoints or hasVideoBox or hasVideoMask or hasVideoText
        videoPromptCompatible = not (hasVideoMask and (hasVideoPoints or hasVideoBox or hasVideoText))
        hasVolumeBoxPrompt = self.logic.has_volume_box_prompt(self._parameterNode)
        viewEmbedded = bool(hasVolume and self.logic.view_fully_embedded(self._parameterNode))
        autoPredictDataReady = viewEmbedded if supportsEmbeddings else hasVolume
        self.ui.startServerButton.enabled = not connected and not self._startingServer and not started
        self.ui.stopServerButton.enabled = connected or self._startingServer or started
        logPath = self.logic.server_log_path()
        self.ui.showServerLogButton.enabled = bool(logPath and Path(logPath).exists())
        self.ui.modelFamilyComboBox.enabled = not boxRunning
        self.ui.weightComboBox.enabled = not boxRunning
        self.ui.addSegmentationButton.enabled = True
        self.ui.removeSegmentationButton.enabled = hasSegmentation
        self.ui.saveSegmentationButton.enabled = hasSegmentation
        self.ui.loadSegmentationButton.enabled = True
        self.setSliceViewControlsEnabled(not boxRunning)
        for selector in self.dataVolumeSelectors() + self.dataSegmentationSelectors():
            selector.enabled = not boxRunning
        self.ui.segmentComboBox.enabled = hasSegment
        self.ui.addSegmentButton.enabled = hasSegmentation
        self.ui.removeSegmentButton.enabled = hasSegment
        finetuneJobRunning = self.finetuneJobRunning()
        self.ui.finetuneDatasetLineEdit.enabled = True
        self.ui.newFinetuneDatasetButton.enabled = not finetuneJobRunning
        self.ui.finetuneDatasetComboBox.enabled = not finetuneJobRunning and bool(self._finetuneDatasetNamesByLabel)
        self.ui.refreshFinetuneDatasetsButton.enabled = not finetuneJobRunning
        self.ui.finetuneVolumeSelector.enabled = not finetuneJobRunning
        self.ui.finetuneSegmentationSelector.enabled = not finetuneJobRunning
        self.ui.addSegmentedVolumeButton.enabled = bool(hasVolume and hasSegmentation and not finetuneJobRunning)
        finetuneStatus = self.updateFinetuneDatasetStatus()
        self.setFinetuneBuildControlsEnabled(connected and not finetuneJobRunning)
        self.ui.finetuneWindowMinSpinBox.enabled = connected and not finetuneJobRunning and self.ui.finetuneUseWindowCheckBox.checked
        self.ui.finetuneWindowMaxSpinBox.enabled = connected and not finetuneJobRunning and self.ui.finetuneUseWindowCheckBox.checked
        self.ui.buildFinetuneDatasetButton.enabled = connected and not finetuneJobRunning and self.finetuneBuildReady(finetuneStatus)
        self.ui.showFinetuneReportButton.enabled = connected
        self.ui.startFinetuneTrainingButton.enabled = bool(connected and not finetuneJobRunning and finetuneStatus and finetuneStatus["train_segmented_volumes"] > 0)
        self.ui.startFinetuneEvalButton.enabled = bool(connected and not finetuneJobRunning)
        self.ui.cancelFinetuneJobButton.enabled = connected and finetuneJobRunning
        self.ui.positivePromptButton.enabled = hasData and supportsPoints and not boxRunning
        self.ui.negativePromptButton.enabled = hasData and supportsPoints and not boxRunning
        self.ui.boxPromptButton.enabled = hasData and supportsBox and not boxRunning
        self.ui.volumeBoxPromptButton.enabled = hasData and supportsVolumeBox and not boxRunning
        self.ui.clear2dOptionsLabel.enabled = not boxRunning
        self.ui.clearPointPromptsCheckBox.enabled = not boxRunning
        self.ui.clearBoxPromptCheckBox.enabled = not boxRunning
        self.ui.clearMaskPromptCheckBox.enabled = supportsMask and not boxRunning
        self.ui.clear2dPromptsButton.enabled = self.hasClearable2dPrompt() and not boxRunning
        self.ui.clear3dPromptsButton.enabled = volumePromptCount > 0 and not boxRunning
        self.ui.sendPointPromptsCheckBox.enabled = supportsPoints and not boxRunning
        self.ui.sendBoxPromptCheckBox.enabled = supportsBox and not boxRunning
        self.ui.sendMaskPromptCheckBox.enabled = supportsMask and not boxRunning
        self.ui.sendTextPromptCheckBox.enabled = supportsText and not boxRunning
        self.ui.textPromptLabel.enabled = supportsText and sendTextPrompt and not boxRunning
        self.ui.textPromptLineEdit.enabled = supportsText and sendTextPrompt and not boxRunning
        self.ui.maskSourceLabel.enabled = supportsMask and sendMaskPrompt and not boxRunning
        self.ui.maskSegmentationSelector.enabled = supportsMask and sendMaskPrompt and not boxRunning
        self.ui.maskSegmentComboBox.enabled = supportsMask and sendMaskPrompt and hasMaskSegmentation and hasMaskSegment and not boxRunning
        self.ui.debugPredictionLogCheckBox.enabled = not boxRunning
        canEmbed = bool(connected and self._prepared and hasVolume and supportsEmbeddings and not self.logic.embedding_job_running() and not boxRunning)
        self.ui.embedAllButton.enabled = canEmbed
        self.ui.embedAllAxesButton.enabled = canEmbed
        self.ui.saveEmbeddingsButton.enabled = bool(canEmbed and self.logic.current_embeddings(self._parameterNode))
        self.ui.loadEmbeddingsButton.enabled = canEmbed
        self.ui.predictSliceButton.enabled = bool(connected and self._prepared and hasData and hasSlicePrompt and not boxRunning)
        self.ui.predictBoxVolumeButton.enabled = bool(
            connected
            and self._prepared
            and hasData
            and supportsVolumeBox
            and hasVolumeBoxPrompt
            and not boxRunning
            and not self.logic.embedding_job_running()
        )
        self.ui.predictBoxVolumeVideoButton.enabled = bool(
            connected
            and self._prepared
            and hasData
            and supportsVolumeBox
            and supportsVideoPropagation
            and hasVolumeBoxPrompt
            and not boxRunning
            and not self.logic.embedding_job_running()
        )
        self.ui.cancelBoxVolumeVideoButton.enabled = bool(connected and self.logic.video_box_volume_prediction_running())
        self.ui.predictAutoVideoButton.enabled = bool(
            connected
            and self._prepared
            and hasData
            and supportsVideoPropagation
            and hasVideoPrompt
            and videoPromptCompatible
            and not boxRunning
            and not self.logic.embedding_job_running()
        )
        self.ui.cancelAutoVideoButton.enabled = bool(connected and self.logic.auto_video_prediction_running())
        canAutoPredict = bool(
            connected
            and self._prepared
            and hasData
            and supportsAutoPredict
            and hasSlicePrompt
            and autoPredictDataReady
            and not boxRunning
            and not self.logic.embedding_job_running()
        )
        self.ui.autoPredictCheckBox.enabled = canAutoPredict
        if self.ui.autoPredictCheckBox.checked and not canAutoPredict:
            self.setAutoPredictChecked(False)
        preparedWeightId = self._prepared["weight_id"] if self._prepared else None
        alreadyPrepared = self.currentWeightId() == preparedWeightId
        self.ui.prepareModelButton.enabled = bool(connected and weight and weight["available"] and not alreadyPrepared and not boxRunning)
        self.ui.offloadModelButton.enabled = bool(connected and self._prepared and not boxRunning)
        self.toolTips.update()
        self.updateAutoPredictObserver()

    def promptSendOptions(self, capabilities=None):
        if not self._parameterNode:
            return {"use_points": False, "use_box": False, "use_mask": False, "use_text": False}
        if capabilities is None:
            weight = self._weights.get(self.currentWeightId())
            capabilities = weight.get("capabilities", {}) if weight else {}
        else:
            weight = self._weights.get(self.currentWeightId())
        textBlocksInteractive = self.textBlocksInteractivePrompts(weight)
        return {
            "use_points": bool(capabilities.get("points", False) and self._parameterNode.sendPointPrompts and not textBlocksInteractive),
            "use_box": bool(capabilities.get("box", False) and self._parameterNode.sendBoxPrompt),
            "use_mask": bool(capabilities.get("mask", False) and self._parameterNode.sendMaskPrompt and not textBlocksInteractive),
            "use_text": bool(capabilities.get("text", False) and self._parameterNode.sendTextPrompt),
        }

    def videoPromptSendOptions(self, capabilities=None):
        if capabilities is None:
            weight = self._weights.get(self.currentWeightId())
            capabilities = weight.get("capabilities", {}) if weight else {}
        options = self.promptSendOptions(capabilities)
        options["use_mask"] = bool(
            options["use_mask"]
            and capabilities.get("video_mask", capabilities.get("mask", False))
        )
        options["use_text"] = bool(
            options["use_text"] and capabilities.get("video_text", False)
        )
        return options

    def textBlocksInteractivePrompts(self, weight=None):
        return bool(
            self._parameterNode
            and weight
            and weight.get("backend") in ("sam3", "medical_sam3")
            and self._parameterNode.sendTextPrompt
            and self.logic.has_text_prompt(self._parameterNode)
        )

    def hasEnabledSlicePrompt(self, sendOptions):
        return (
            (sendOptions["use_points"] and self.logic.has_point_prompt(self._parameterNode))
            or (sendOptions["use_box"] and self.logic.has_box_prompt(self._parameterNode))
            or (sendOptions["use_mask"] and self.logic.has_mask_prompt(self._parameterNode))
            or (sendOptions["use_text"] and self.logic.has_text_prompt(self._parameterNode))
        )

    def hasClearable2dPrompt(self):
        return (
            (self._parameterNode.clearPointPrompts and self.logic.has_point_prompt(self._parameterNode))
            or (self._parameterNode.clearBoxPrompt and self.logic.has_box_prompt(self._parameterNode))
            or (self._parameterNode.clearMaskPrompt and self._parameterNode.sendMaskPrompt)
        )

    def onStartServer(self):
        self._startingServer = True
        pid = self.logic.start_server()
        self.ui.statusLabel.text = "Starting SAMM server"
        self.serverStartAttempts = 0
        slicer.util.infoDisplay(f"SAMM server started in process {pid}\n{self.logic.server_log_path()}", "SAMM")
        self.updateControls()
        qt.QTimer.singleShot(1000, self.connectAfterServerStart)

    def connectAfterServerStart(self):
        if not self._startingServer:
            return
        try:
            status = self.connectServiceUi()
            self._startingServer = False
            self.ui.statusLabel.text = status.message
            slicer.util.infoDisplay(status.message, "SAMM")
            self.updateControls()
        except Exception:
            self.serverStartAttempts += 1
            if self.serverStartAttempts >= 30:
                self._startingServer = False
                self.ui.statusLabel.text = "Server start timed out"
                self.updateControls()
                raise
            qt.QTimer.singleShot(1000, self.connectAfterServerStart)

    def heartbeat(self):
        if not self._heartbeatActive:
            return
        self.checkServiceHeartbeat()
        if self._heartbeatActive:
            qt.QTimer.singleShot(self.heartbeatMs, self.heartbeat)

    def checkServiceHeartbeat(self):
        if self._slicerClosing or not self.logic or not self._parameterNode or self._startingServer:
            return
        if self.logic.client and self.serviceStillConnected():
            return
        running = self.logic.server_running()
        try:
            status = self.connectServiceUi()
        except ConnectionError:
            if running:
                self.ui.statusLabel.text = "SAMM server process found; waiting for service"
            self.updateControls()
            return
        except RuntimeError as exc:
            self.ui.statusLabel.text = str(exc)
            self.updateControls()
            return
        self.ui.statusLabel.text = status.message
        self.updateControls()

    def serviceStillConnected(self):
        try:
            self.logic.client.connect()
        except ConnectionError:
            self.logic.disconnect_service()
            self.clearModels()
            self.ui.statusLabel.text = "Disconnected from SAMM service"
            self.updateControls()
            return False
        except RuntimeError as exc:
            self.logic.disconnect_service()
            self.clearModels()
            self.ui.statusLabel.text = str(exc)
            self.updateControls()
            return False
        self.touchServiceSession()
        return True

    def connectServiceUi(self):
        status = self.logic.connect_to_service()
        self.touchServiceSession()
        self.setModels(self.logic.list_models())
        self.refreshFinetuneDatasets()
        self._prepared = self.logic.prepared_status()["prepared"]
        return status

    def touchServiceSession(self, force=False):
        autosave = self.autosaveSessionState(force)
        self.logic.touch_session(autosave)

    def onStopServer(self):
        if self.logic.client and self._parameterNode:
            self.touchServiceSession(force=True)
        pid = self.logic.stop_server()
        self._startingServer = False
        self._prepared = None
        self._finetuneJob = None
        self.clearModels()
        message = f"Stopped SAMM server process {pid}" if pid else "Disconnected from SAMM service"
        self.ui.statusLabel.text = message
        self.updateModelAvailability()
        self.updateControls()
        slicer.util.infoDisplay(message, "SAMM")

    def onShowServerLog(self):
        path = self.logic.server_log_path()
        if not path or not Path(path).exists():
            slicer.util.infoDisplay("No managed server log is available yet.", "SAMM")
            return
        try:
            lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
            tail = "\n".join(lines[-80:])
        except OSError as exc:
            tail = f"Could not read log: {exc}"
        box = qt.QMessageBox(slicer.util.mainWindow())
        box.setWindowTitle("SAMM server log")
        box.setText(str(path))
        box.setInformativeText("The most recent server output is available under Details.")
        box.setDetailedText(tail or "The log is empty.")
        box.exec_()

    def onSlicerAboutToQuit(self):
        self._slicerClosing = True
        self._heartbeatActive = False
        if self.logic and self.logic.client and self._parameterNode:
            self.touchServiceSession(force=True)
        if not self.logic or not self.logic.server_running():
            return
        pid = self.logic.server_pid()
        if self.shouldCloseServerOnSlicerQuit(pid):
            self.logic.stop_server()

    def shouldCloseServerOnSlicerQuit(self, pid):
        box = qt.QMessageBox(slicer.util.mainWindow())
        box.setWindowTitle("SAMM server left running")
        box.setText("SAMM server is still running.")
        closeButton = box.addButton("", qt.QMessageBox.AcceptRole)
        leaveButton = box.addButton("Leave running", qt.QMessageBox.RejectRole)
        box.setDefaultButton(closeButton)
        remaining = {"seconds": 6}

        def updatePrompt():
            closeButton.setText(f"Close server ({remaining['seconds']})")
            box.setInformativeText(
                "Gateway: http://127.0.0.1:8799\n"
                f"Process ID: {pid}\n"
                f"Log: {self.logic.server_log_path()}\n"
                f"Close command: kill -- -{pid}\n\n"
                f"Closing server automatically in {remaining['seconds']} seconds."
            )

        def tick():
            remaining["seconds"] -= 1
            if remaining["seconds"] <= 0:
                timer.stop()
                box.accept()
            else:
                updatePrompt()

        updatePrompt()
        timer = qt.QTimer(box)
        timer.setInterval(1000)
        timer.connect("timeout()", tick)
        timer.start()
        box.exec_()
        timer.stop()
        return box.clickedButton() != leaveButton

    def clearModels(self):
        self._prepared = None
        self.logic.clear_embedding()
        self.logic.clear_box_volume_prediction()
        self.setAutoPredictChecked(False)
        self.updateEmbeddingView()
        self.updateBoxVolumeView()
        self.setLocalModels()

    def setLocalModels(self):
        if self._parameterNode:
            self.setModels(self.logic.local_models())
            return
        self._models = {}
        self._weights = {}
        self._modelIdsByLabel = {}
        self._weightIdsByLabel = {}
        self.setComboLabels(self.ui.modelFamilyComboBox, [])
        self.setComboLabels(self.ui.weightComboBox, [])

    def setModels(self, models):
        self._models = {model["id"]: model for model in models}
        self._weights = {weight["id"]: weight for model in models for weight in model["weights"]}
        self._modelIdsByLabel = {model["label"]: model["id"] for model in models}
        self.setComboLabels(self.ui.modelFamilyComboBox, [model["label"] for model in models])
        self.setModelFamily(self._parameterNode.modelFamily)
        self.setWeights(self.currentModelFamilyId(), self._parameterNode.weightName)
        self.updateModelAvailability()

    def onModelFamilyChanged(self, index=None):
        modelId = self.currentModelFamilyId()
        if not self._parameterNode or modelId not in self._models:
            return
        self.logic.clear_embedding()
        self.logic.clear_box_volume_prediction()
        self.setAutoPredictChecked(False)
        self.updateEmbeddingView()
        self.updateBoxVolumeView()
        self._parameterNode.modelFamily = modelId
        self.setWeights(modelId, self._parameterNode.weightName)

    def onWeightChanged(self, index=None):
        weightId = self.currentWeightId()
        if not self._parameterNode or weightId not in self._weights:
            return
        with slicer.util.tryWithErrorDisplay("Failed to refresh model status."):
            self.logic.clear_embedding()
            self.logic.clear_box_volume_prediction()
            self.setAutoPredictChecked(False)
            self.updateEmbeddingView()
            self.updateBoxVolumeView()
            self._parameterNode.weightName = weightId
            if self.logic.client:
                self._weights[weightId] = self.logic.get_weight_status(weightId)
            self.updateModelAvailability()

    def onPrepareModel(self):
        with slicer.util.tryWithErrorDisplay("Failed to prepare model."):
            self._prepared = None
            self.logic.clear_embedding()
            self.logic.clear_box_volume_prediction()
            self.setAutoPredictChecked(False)
            self.updateEmbeddingView()
            self.updateBoxVolumeView()
            self.updateControls()
            self.logic.prepare_weight(self.currentWeightId())
            prepared = self.logic.prepared_status()["prepared"]
            self._prepared = prepared
            message = f"Prepared {prepared['model_id']} / {prepared['weight_id']} on {prepared['device']}"
            self.ui.statusLabel.text = message
            slicer.util.infoDisplay(message, "SAMM")
            if self.logic.client:
                self.refreshFinetuneDatasets()
            self.updateControls()

    def onOffloadModel(self):
        with slicer.util.tryWithErrorDisplay("Failed to offload model."):
            self.logic.offload_model()
            self._prepared = None
            self.logic.clear_embedding()
            self.logic.clear_box_volume_prediction()
            self.setAutoPredictChecked(False)
            self.updateEmbeddingView()
            self.updateBoxVolumeView()
            self.ui.statusLabel.text = "Model offloaded"
            slicer.util.infoDisplay("Model offloaded", "SAMM")
            self.updateControls()

    def onPlacePositivePrompt(self):
        self.logic.place_prompt(self._parameterNode.positivePrompts)

    def onPlaceNegativePrompt(self):
        self.logic.place_prompt(self._parameterNode.negativePrompts)

    def onPlaceBoxPrompt(self):
        if self._parameterNode.boxPrompts.GetNumberOfControlPoints() > 0:
            self._parameterNode.boxPrompts.RemoveAllControlPoints()
        self.logic.place_prompt(self._parameterNode.boxPrompts, persistent=False)
        self.ui.statusLabel.text = "Place and resize the 2D box"

    def onPlaceVolumeBoxPrompt(self):
        if self._parameterNode.volumeBoxPrompt.GetNumberOfControlPoints() > 0:
            self._parameterNode.volumeBoxPrompt.RemoveAllControlPoints()
        self.logic.place_prompt(self._parameterNode.volumeBoxPrompt, persistent=False)
        self.ui.statusLabel.text = "Place and resize the 3D box ROI"

    def onClear2dPrompts(self):
        self.logic.clear_2d_prompts(
            self._parameterNode,
            points=self._parameterNode.clearPointPrompts,
            box=self._parameterNode.clearBoxPrompt,
            mask=self._parameterNode.clearMaskPrompt,
        )
        self.ui.sendMaskPromptCheckBox.blockSignals(True)
        self.ui.sendMaskPromptCheckBox.checked = self._parameterNode.sendMaskPrompt
        self.ui.sendMaskPromptCheckBox.blockSignals(False)
        self.updateControls()
        self.scheduleAutoPredict()

    def onClear3dPrompts(self):
        self.logic.clear_3d_prompts(self._parameterNode)
        self.updateControls()

    def onEmbedAll(self):
        with slicer.util.tryWithErrorDisplay("Failed to embed slices."):
            state = self.logic.embed_all_slices(self._parameterNode)
            self.startEmbeddingTimers(state)

    def onEmbedAllAxes(self):
        with slicer.util.tryWithErrorDisplay("Failed to embed slices."):
            state = self.logic.embed_all_axes(self._parameterNode)
            self.startEmbeddingTimers(state)

    def onSaveEmbeddings(self):
        with slicer.util.tryWithErrorDisplay("Failed to save embeddings."):
            path = qt.QFileDialog.getSaveFileName(
                slicer.util.mainWindow(),
                "Save SAMM embedding folder",
                str(self.logic.default_embedding_path(self._parameterNode)),
                "SAMM embedding folder (*)",
            )
            if not path:
                return
            result = self.logic.save_embeddings(self._parameterNode, path)
            self._embeddingDirty = False
            self._lastAutosaveManifest["embeddings"] = result["path"]
            message = f"Saved {result['count']} embeddings\n{result['path']}"
            self.ui.statusLabel.text = message
            slicer.util.infoDisplay(message, "SAMM")
            self.updateControls()

    def onLoadEmbeddings(self):
        with slicer.util.tryWithErrorDisplay("Failed to load embeddings."):
            self.logic.embedding_dir_path().mkdir(exist_ok=True)
            path = qt.QFileDialog.getExistingDirectory(
                slicer.util.mainWindow(),
                "Load SAMM embedding folder",
                str(self.logic.embedding_dir_path()),
            )
            if not path:
                return
            result = self.logic.load_embeddings(self._parameterNode, path)
            self._embeddingDirty = False
            self.updateEmbeddingView()
            message = f"Loaded {result['count']} embeddings\n{result['path']}"
            self.ui.statusLabel.text = message
            slicer.util.infoDisplay(message, "SAMM")
            self.updateControls()

    def startEmbeddingTimers(self, state):
        self.updateEmbeddingView()
        message = self.embeddingJobMessage(state)
        self.ui.statusLabel.text = message
        self.updateControls()
        if state.get("status") in ("queued", "running"):
            jobId = state["job_id"]
            qt.QTimer.singleShot(0, lambda jobId=jobId: self.submitEmbeddingSlices(jobId))
            qt.QTimer.singleShot(self.embeddingJobPollMs, lambda jobId=jobId: self.pollEmbeddingJob(jobId))

    def submitEmbeddingSlices(self, jobId=None):
        if not self._parameterNode or not self.logic.embedding_submission_running(jobId):
            return
        with slicer.util.tryWithErrorDisplay("Failed to submit embedding slices."):
            state = self.logic.submit_embedding_slices(self.embeddingSubmitBatchSize, jobId)
            self.updateEmbeddingView()
            if state:
                if state.get("results"):
                    self._embeddingDirty = True
                self.ui.statusLabel.text = self.embeddingJobMessage(state)
            self.updateControls()
            if self.logic.embedding_submission_running(jobId):
                qt.QTimer.singleShot(self.embeddingSubmitMs, lambda jobId=jobId: self.submitEmbeddingSlices(jobId))

    def pollEmbeddingJob(self, jobId=None):
        if not self._parameterNode or not self.logic.embedding_job_running(jobId):
            return
        with slicer.util.tryWithErrorDisplay("Failed to refresh embedding job."):
            state = self.logic.refresh_embedding_job()
            self.updateEmbeddingView()
            if state and state.get("results"):
                self._embeddingDirty = True
            self.ui.statusLabel.text = self.embeddingJobMessage(state)
            self.updateControls()
            if state and state.get("status") in ("queued", "running"):
                qt.QTimer.singleShot(self.embeddingJobPollMs, lambda jobId=jobId: self.pollEmbeddingJob(jobId))

    def onAddSegmentation(self):
        node = self.logic.add_segmentation(self._parameterNode)
        self.logic.ensure_segment(self._parameterNode)
        self._segmentationDirty = True
        self._segmentState = None
        self._maskSegmentState = None
        self.refreshSegments()
        self.refreshMaskSegments()
        self.ui.statusLabel.text = f"Added segmentation {node.GetName()}"
        self.updateControls()

    def onRemoveSegmentation(self):
        node = self._parameterNode.segmentation
        if not node:
            return
        name = node.GetName()
        self.removeSegmentationObserver()
        self.logic.remove_segmentation(self._parameterNode)
        self._segmentationDirty = True
        self._segmentState = None
        self._maskSegmentState = None
        self.refreshSegments()
        self.refreshMaskSegments()
        self.ui.statusLabel.text = f"Removed segmentation {name}"
        self.updateControls()

    def onSaveSegmentation(self):
        with slicer.util.tryWithErrorDisplay("Failed to save segmentation."):
            path = qt.QFileDialog.getSaveFileName(
                slicer.util.mainWindow(),
                "Save SAMM segmentation",
                str(self.logic.default_segmentation_path(self._parameterNode)),
                "Slicer segmentation (*.seg.nrrd)",
            )
            if not path:
                return
            result = self.logic.save_segmentation(self._parameterNode, path)
            self._segmentationDirty = False
            self._lastAutosaveManifest["segmentation"] = result["path"]
            message = f"Saved {result['segments']} segments\n{result['path']}"
            self.ui.statusLabel.text = message
            slicer.util.infoDisplay(message, "SAMM")
            self.updateControls()

    def onLoadSegmentation(self):
        with slicer.util.tryWithErrorDisplay("Failed to load segmentation."):
            self.logic.segmentation_dir_path().mkdir(exist_ok=True)
            path = qt.QFileDialog.getOpenFileName(
                slicer.util.mainWindow(),
                "Load SAMM segmentation",
                str(self.logic.segmentation_dir_path()),
                "Slicer segmentation (*.seg.nrrd)",
            )
            if not path:
                return
            result = self.logic.load_segmentation(self._parameterNode, path)
            self._segmentationDirty = False
            self.updateSegmentationObserver()
            self._segmentState = None
            self._maskSegmentState = None
            self.refreshSegments()
            self.refreshMaskSegments()
            message = f"Loaded {result['segments']} segments from {result['name']}\n{result['path']}"
            self.ui.statusLabel.text = message
            slicer.util.infoDisplay(message, "SAMM")
            self.updateControls()

    def onAddSegmentedVolume(self):
        with slicer.util.tryWithErrorDisplay("Failed to add segmented volume."):
            result = self.logic.add_segmented_volume(self._parameterNode)
            segmentCount = len(result["segments"])
            voxelCount = sum(segment["voxels"] for segment in result["segments"])
            dataset = self.logic.finetune_dataset_name(self._parameterNode)
            message = f"Added segmented volume {result['volume']} to {dataset} with {segmentCount} label(s), {voxelCount} voxel(s)\n{result['path']}"
            self.ui.statusLabel.text = message
            slicer.util.infoDisplay(message, "SAMM")
            self.refreshFinetuneDatasets()
            self.updateControls()

    def onBuildFinetuneDataset(self):
        with slicer.util.tryWithErrorDisplay("Failed to build finetuning dataset."):
            axes = self.selectedFinetuneAxes()
            if not axes:
                raise ValueError("Select at least one finetuning axis.")
            result = self.logic.build_finetune_dataset(
                self._parameterNode,
                self.ui.finetuneValCountSpinBox.value,
                axes,
                self.finetuneWindow(),
            )
            status = result["status"]
            message = f"Built {status['name']}: train {status['train_segmented_volumes']}, val {status['val_segmented_volumes']}"
            self.ui.statusLabel.text = message
            slicer.util.infoDisplay(f"{message}\n\n{result['output']}", "SAMM")
            self.refreshFinetuneDatasets()
            self.updateControls()

    def onShowFinetuneReport(self):
        with slicer.util.tryWithErrorDisplay("Failed to show finetuning report."):
            result = self.logic.finetune_report(self._parameterNode, self.ui.finetuneRunLineEdit.text)
            self.ui.statusLabel.text = f"Loaded report for {Path(result['run']).name}"
            self.showTextDialog("SAMM finetuning report", result["output"])
            self.updateControls()

    def onStartFinetuneTraining(self):
        with slicer.util.tryWithErrorDisplay("Failed to start finetuning training."):
            self._finetuneJob = self.logic.start_finetune_training(
                self._parameterNode,
                finetune_base_checkpoint(self.ui.finetuneBaseModelComboBox.currentText),
                self.ui.finetuneEpochsSpinBox.value,
                self.ui.finetuneBatchSizeSpinBox.value,
                self.ui.finetuneNumWorkersSpinBox.value,
                self.ui.finetuneNumFramesSpinBox.value,
            )
            self.updateFinetuneJobStatus()
            self.updateControls()
            qt.QTimer.singleShot(self.finetuneJobPollMs, self.pollFinetuneJob)

    def onStartFinetuneEval(self):
        with slicer.util.tryWithErrorDisplay("Failed to start finetuning evaluation."):
            self._finetuneJob = self.logic.start_finetune_eval(self._parameterNode, self.ui.finetuneEvalMaxSegmentedVolumesSpinBox.value)
            self.updateFinetuneJobStatus()
            self.updateControls()
            qt.QTimer.singleShot(self.finetuneJobPollMs, self.pollFinetuneJob)

    def onCancelFinetuneJob(self):
        if not self._finetuneJob:
            return
        with slicer.util.tryWithErrorDisplay("Failed to cancel finetuning job."):
            self._finetuneJob = self.logic.cancel_finetune_job(self._finetuneJob["job_id"])
            self.updateFinetuneJobStatus()
            self.updateControls()

    def pollFinetuneJob(self):
        if not self._parameterNode or not self.finetuneJobRunning():
            return
        with slicer.util.tryWithErrorDisplay("Failed to refresh finetuning job."):
            self._finetuneJob = self.logic.finetune_job(self._finetuneJob["job_id"])
            self.updateFinetuneJobStatus()
            if self.finetuneJobRunning():
                qt.QTimer.singleShot(self.finetuneJobPollMs, self.pollFinetuneJob)
            elif self._finetuneJob["kind"] == "train" and self._finetuneJob["status"] == "complete":
                self.setModels(self.logic.list_models())
            self.updateControls()

    def finetuneJobRunning(self):
        return bool(self._finetuneJob and self._finetuneJob["status"] in ("queued", "running"))

    def updateFinetuneJobStatus(self):
        if not self._finetuneJob:
            self.ui.finetuneJobStatusLabel.text = "No finetuning job"
            self.resetFinetuneJobProgress()
            return
        job = self._finetuneJob
        tail = self.lastLogLine(job.get("log_tail", ""))
        message = f"{job['kind']} {job['status']} ({job['job_id'][:8]})\nLog: {self.displayPath(job['log'])}"
        if job.get("error"):
            message += f"\n{job['error']}"
        if tail:
            message += f"\n{tail}"
        self.ui.finetuneJobStatusLabel.text = message
        self.updateFinetuneJobProgress(job)

    def resetFinetuneJobProgress(self):
        self.ui.finetuneJobProgressBar.setRange(0, 1)
        self.ui.finetuneJobProgressBar.value = 0
        self.ui.finetuneJobProgressBar.format = "No job"

    def updateFinetuneJobProgress(self, job):
        progress = job["progress"]
        self.ui.finetuneJobProgressBar.format = progress["label"]
        if progress["mode"] == "busy":
            self.ui.finetuneJobProgressBar.setRange(0, 0)
            return
        self.ui.finetuneJobProgressBar.setRange(0, progress["total"])
        self.ui.finetuneJobProgressBar.value = progress["current"]

    def displayPath(self, path):
        root = str(self.logic.project_root())
        value = str(path)
        return value[len(root) + 1:] if value.startswith(f"{root}/") else value

    def lastLogLine(self, text):
        lines = [line for line in text.splitlines() if line.strip()]
        return lines[-1] if lines else ""

    def onFinetuneSettingsChanged(self, value=None):
        if self._parameterNode:
            self.syncFinetuneDatasetCombo()
            self.updateControls()

    def onNewFinetuneDataset(self, value=None):
        with slicer.util.tryWithErrorDisplay("Failed to create finetuning dataset."):
            status = self.logic.new_finetune_dataset(self._parameterNode)
            self.ui.finetuneDatasetLineEdit.text = status["name"]
            self._parameterNode.finetuneDatasetName = status["name"]
            self.refreshFinetuneDatasets()
            message = f"Created finetuning dataset {status['name']}\n{self.displayPath(status['source'])}"
            self.ui.statusLabel.text = message
            slicer.util.infoDisplay(message, "SAMM")
            self.updateControls()

    def refreshFinetuneDatasets(self, value=None):
        if not self.logic.client:
            datasets = self.logic.local_finetune_datasets()
        else:
            datasets = self.logic.list_finetune_datasets()
        labels = [self.finetuneDatasetChoiceLabel(item) for item in datasets]
        self._finetuneDatasetNamesByLabel = dict(zip(labels, [item["name"] for item in datasets]))
        self.setComboLabels(self.ui.finetuneDatasetComboBox, labels)
        self.syncFinetuneDatasetCombo()

    def finetuneDatasetChoiceLabel(self, item):
        return f"{item['name']}  source:{item['source_segmented_volumes']} train:{item['train_segmented_volumes']} val:{item['val_segmented_volumes']}"

    def syncFinetuneDatasetCombo(self):
        if self._updatingFinetuneDatasetCombo:
            return
        name = self.logic.finetune_dataset_name(self._parameterNode) if self._parameterNode else ""
        label = next((label for label, itemName in self._finetuneDatasetNamesByLabel.items() if itemName == name), "")
        self._updatingFinetuneDatasetCombo = True
        try:
            self.ui.finetuneDatasetComboBox.setCurrentText(label)
        finally:
            self._updatingFinetuneDatasetCombo = False

    def onFinetuneDatasetChoiceChanged(self, index=None):
        if self._updatingFinetuneDatasetCombo or not self._parameterNode:
            return
        name = self._finetuneDatasetNamesByLabel.get(self.ui.finetuneDatasetComboBox.currentText)
        if not name:
            return
        self.ui.finetuneDatasetLineEdit.text = name
        self._parameterNode.finetuneDatasetName = name
        self.updateControls()

    def showTextDialog(self, title, text):
        dialog = qt.QDialog(slicer.util.mainWindow())
        dialog.setWindowTitle(title)
        layout = qt.QVBoxLayout(dialog)
        edit = qt.QPlainTextEdit(dialog)
        edit.setReadOnly(True)
        edit.setPlainText(text)
        edit.minimumWidth = 720
        edit.minimumHeight = 420
        layout.addWidget(edit)
        buttonBox = qt.QDialogButtonBox(qt.QDialogButtonBox.Ok)
        buttonBox.connect("accepted()", dialog.accept)
        layout.addWidget(buttonBox)
        dialog.exec_()

    def onAddSegment(self):
        segmentId = self.logic.add_segment(self._parameterNode)
        name = self._parameterNode.segmentation.GetSegmentation().GetSegment(segmentId).GetName()
        self._segmentationDirty = True
        self._segmentState = None
        self._maskSegmentState = None
        self.refreshSegments()
        self.refreshMaskSegments()
        self.ui.statusLabel.text = f"Added segment {name}"
        self.updateControls()

    def onRemoveSegment(self):
        segmentation = self._parameterNode.segmentation.GetSegmentation()
        segmentId = self.logic.selected_segment_id(self._parameterNode)
        if not segmentId:
            return
        name = segmentation.GetSegment(segmentId).GetName()
        self.logic.remove_segment(self._parameterNode)
        self._segmentationDirty = True
        self._segmentState = None
        self._maskSegmentState = None
        self.refreshSegments()
        self.refreshMaskSegments()
        self.ui.statusLabel.text = f"Removed segment {name}"
        self.updateControls()

    def onSegmentChanged(self, index=None):
        segmentId = self._segmentIdsByLabel.get(self.ui.segmentComboBox.currentText)
        if self._updatingSegmentCombo or not self._parameterNode or not segmentId:
            return
        if segmentId != self._parameterNode.selectedSegmentId:
            self.invalidateVideoVolumePrediction("output segment changed")
        self._parameterNode.selectedSegmentId = segmentId
        self.updateControls()

    def onPredictSlice(self):
        with slicer.util.tryWithErrorDisplay("Failed to predict mask."):
            usesEmbeddings = self.currentWeightSupportsEmbeddings()
            message = self.logic.predict_slice(
                self._parameterNode,
                use_embeddings=usesEmbeddings,
                debug_log=self._parameterNode.debugPredictionLog,
                **self.promptSendOptions(),
            )
            self._segmentationDirty = True
            self._embeddingDirty = self._embeddingDirty or usesEmbeddings
            self.updateEmbeddingView()
            self.ui.statusLabel.text = message
            slicer.util.infoDisplay(message, "SAMM")
            self.refreshSegments()
            self.updateControls()

    def onPromptSendToggled(self, checked=None):
        if self.logic.auto_video_prediction_running():
            self.invalidateVideoVolumePrediction("seed prompt options changed")
        if self._parameterNode:
            self._parameterNode.sendPointPrompts = self.ui.sendPointPromptsCheckBox.checked
            self._parameterNode.sendBoxPrompt = self.ui.sendBoxPromptCheckBox.checked
            self._parameterNode.sendMaskPrompt = self.ui.sendMaskPromptCheckBox.checked
            self._parameterNode.sendTextPrompt = self.ui.sendTextPromptCheckBox.checked
        self.updateControls()
        self.scheduleAutoPredict()

    def onTextPromptChanged(self, text):
        if self._parameterNode:
            self._parameterNode.textPrompt = text
        self.updateControls()
        self.scheduleAutoPredict()

    def onClearOptionsToggled(self, checked=None):
        if self._parameterNode:
            self._parameterNode.clearPointPrompts = self.ui.clearPointPromptsCheckBox.checked
            self._parameterNode.clearBoxPrompt = self.ui.clearBoxPromptCheckBox.checked
            self._parameterNode.clearMaskPrompt = self.ui.clearMaskPromptCheckBox.checked
        self.updateControls()

    def onDebugPredictionLogToggled(self, checked):
        if self._parameterNode:
            self._parameterNode.debugPredictionLog = checked
        self.toolTips.update()

    def onAutoPredictToggled(self, checked):
        if self._parameterNode:
            self._parameterNode.autoPredict2d = checked
        if not checked:
            self._autoPredictPending = False
            self.removeAutoPredictObserver()
            self.updateControls()
            return
        ready, message = self.autoPredictReady()
        if not ready:
            self.setAutoPredictChecked(False)
            self.ui.statusLabel.text = message
            self.updateControls()
            return
        self.updateAutoPredictObserver()
        self.syncAutoPredictPrompts()
        self.scheduleAutoPredict()
        self.updateControls()

    def setAutoPredictChecked(self, checked):
        self.ui.autoPredictCheckBox.blockSignals(True)
        self.ui.autoPredictCheckBox.checked = checked
        self.ui.autoPredictCheckBox.blockSignals(False)
        if self._parameterNode:
            self._parameterNode.autoPredict2d = checked
        if not checked:
            self._autoPredictPending = False
            self.removeAutoPredictObserver()

    def autoPredictReady(self):
        if not self._parameterNode or not self._prepared:
            return False, "Prepare a model before auto predict."
        weight = self._weights.get(self.currentWeightId())
        capabilities = weight.get("capabilities", {}) if weight else {}
        supportsPrompt = self.hasEnabledSlicePrompt(self.promptSendOptions(capabilities))
        if not capabilities.get("auto_predict_2d", False):
            return False, "Selected weight does not support auto predict."
        if not supportsPrompt:
            return False, "Add a supported 2D prompt before auto predict."
        if capabilities.get("embeddings", False):
            embedded, total = self.logic.view_embedding_count(self._parameterNode)
            if embedded != total:
                return False, f"Embed {self._parameterNode.sliceView} view before auto predict ({embedded} of {total})."
        if self.logic.box_volume_prediction_running() or self.logic.embedding_job_running():
            return False, "Wait for the running job to finish."
        return True, "Auto predict ready"

    def syncAutoPredictPrompts(self):
        if not self._parameterNode or not self.ui.autoPredictCheckBox.checked:
            return
        self._syncingAutoPredictPrompts = True
        try:
            self.logic.sync_2d_prompts_to_slice(self._parameterNode)
        finally:
            self._syncingAutoPredictPrompts = False

    def scheduleAutoPredict(self):
        if not self._parameterNode or not self.ui.autoPredictCheckBox.checked:
            return
        if self._autoPredictPending:
            return
        self._autoPredictPending = True
        qt.QTimer.singleShot(self.autoPredictDelayMs, self.runAutoPredict)

    def runAutoPredict(self):
        if self._autoPredictRunning:
            self._autoPredictPending = True
            return
        self._autoPredictPending = False
        if not self._parameterNode or not self.ui.autoPredictCheckBox.checked:
            return
        ready, message = self.autoPredictReady()
        if not ready:
            self.setAutoPredictChecked(False)
            self.ui.statusLabel.text = message
            self.updateControls()
            return
        self._autoPredictRunning = True
        try:
            with slicer.util.tryWithErrorDisplay("Failed to auto predict mask."):
                message = self.logic.predict_auto_slice(
                    self._parameterNode,
                    use_embeddings=self.currentWeightSupportsEmbeddings(),
                    debug_log=self._parameterNode.debugPredictionLog,
                    **self.promptSendOptions(),
                )
                self._segmentationDirty = True
                self.ui.statusLabel.text = message
                self.refreshSegments()
                self.updateControls()
        finally:
            self._autoPredictRunning = False
        if self._autoPredictPending:
            self.scheduleAutoPredict()

    def currentWeightSupportsEmbeddings(self):
        weight = self._weights.get(self.currentWeightId())
        capabilities = weight.get("capabilities", {}) if weight else {}
        return capabilities.get("embeddings", False)

    def onPredictBoxVolume(self):
        with slicer.util.tryWithErrorDisplay("Failed to start box volume prediction."):
            job = self.logic.start_box_volume_prediction(self._parameterNode)
            self.startBoxVolumeTimers(job)

    def onPredictBoxVolumeVideo(self):
        with slicer.util.tryWithErrorDisplay("Failed to start slices-as-video box prediction."):
            job = self.logic.start_video_box_volume_prediction(self._parameterNode)
            self.startBoxVolumeTimers(job)

    def onPredictAutoVideo(self):
        self.setAutoPredictChecked(False)
        with slicer.util.tryWithErrorDisplay("Failed to start auto slices-as-video prediction."):
            options = self.videoPromptSendOptions()
            job = self.logic.start_video_auto_prediction(
                self._parameterNode,
                use_points=options["use_points"],
                use_box=options["use_box"],
                use_mask=options["use_mask"],
                use_text=options["use_text"],
            )
            self.startBoxVolumeTimers(job)

    def onCancelBoxVolumeVideo(self):
        if not self._parameterNode or not self.logic.video_box_volume_prediction_running():
            return
        with slicer.util.tryWithErrorDisplay("Failed to cancel slices-as-video box prediction."):
            job = self.logic.cancel_video_box_volume_prediction(self._parameterNode)
            self.updateBoxVolumeAfterJob(job)

    def onCancelAutoVideo(self):
        if not self._parameterNode or not self.logic.auto_video_prediction_running():
            return
        with slicer.util.tryWithErrorDisplay("Failed to cancel auto slices-as-video prediction."):
            job = self.logic.cancel_video_auto_prediction(self._parameterNode)
            self.updateBoxVolumeAfterJob(job)

    def invalidateVideoVolumePrediction(self, reason):
        if not self.logic or not self.logic.video_volume_prediction_running():
            return
        mode = self.logic.boxVolumeJob.get("mode")
        self.logic.clear_box_volume_prediction()
        if hasattr(self, "ui"):
            self.updateBoxVolumeView()
            label = "auto slices-as-video" if mode == "auto_video" else "3D-box slices-as-video"
            self.ui.statusLabel.text = f"Cancelled {label} prediction: {reason}"

    def startBoxVolumeTimers(self, job):
        self.updateBoxVolumeView()
        self.ui.statusLabel.text = self.boxVolumeJobMessage(job)
        self.updateControls()
        if job.get("status") in ("uploading", "queued", "running"):
            qt.QTimer.singleShot(0, self.submitBoxVolumeSlices)
            qt.QTimer.singleShot(self.boxVolumeJobPollMs, self.pollBoxVolumeJob)

    def submitBoxVolumeSlices(self):
        if not self._parameterNode or not self.logic.box_volume_submission_running():
            return
        with slicer.util.tryWithErrorDisplay("Failed to submit volume prediction slices."):
            if self.logic.boxVolumeJob.get("mode") in ("video", "auto_video"):
                job = self.logic.submit_video_volume_frames(self._parameterNode, self.boxVolumeSubmitBatchSize)
            else:
                job = self.logic.submit_box_volume_slices(self._parameterNode, self.boxVolumeSubmitBatchSize)
            self.updateBoxVolumeAfterJob(job)
            if self.logic.box_volume_submission_running():
                qt.QTimer.singleShot(self.boxVolumeSubmitMs, self.submitBoxVolumeSlices)

    def pollBoxVolumeJob(self):
        if not self._parameterNode or not self.logic.box_volume_prediction_running():
            return
        with slicer.util.tryWithErrorDisplay("Failed to refresh volume prediction."):
            job = self.logic.refresh_box_volume_job(self._parameterNode)
            self.updateBoxVolumeAfterJob(job)
            if job and job.get("status") in ("uploading", "queued", "running"):
                qt.QTimer.singleShot(self.boxVolumeJobPollMs, self.pollBoxVolumeJob)

    def updateBoxVolumeAfterJob(self, job):
        self.updateBoxVolumeView()
        if not job:
            return
        if job.get("lastApplied", 0):
            self._segmentationDirty = True
            self.refreshSegments()
        self.ui.statusLabel.text = self.boxVolumeJobMessage(job)
        self.updateControls()

    def updateModelAvailability(self):
        weight = self._weights.get(self.currentWeightId())
        model = self._models.get(self.currentModelFamilyId())
        if not model or not weight:
            self.ui.modelAvailabilityLabel.text = "Not checked"
            self.updateControls()
            return
        availableCount = len([item for item in self._weights.values() if item["available"]])
        if weight["available"]:
            self.ui.modelAvailabilityLabel.text = (
                f'{model["label"]} / {weight["label"]} available '
                f"({availableCount} of {len(self._weights)} weights available)"
            )
        else:
            self.ui.modelAvailabilityLabel.text = (
                f'{model["label"]} / {weight["label"]} unavailable; missing {weight["checkpoint"]} '
                f"({availableCount} of {len(self._weights)} weights available)"
            )
        self.updateControls()

    def setComboLabels(self, comboBox, labels):
        comboBox.blockSignals(True)
        comboBox.clear()
        comboBox.addItems(labels)
        comboBox.blockSignals(False)

    def updateEmbeddingView(self):
        labels = self.logic.embedded_slice_labels(self._parameterNode) if self.logic else []
        self.ui.embeddedSlicesListWidget.clear()
        self.ui.embeddedSlicesListWidget.addItems(labels)
        if self.currentWeightIsFastSamWithoutCache():
            self.ui.embeddingProgressBar.setRange(0, 1)
            self.ui.embeddingProgressBar.value = 0
            self.ui.embeddingStatusLabel.text = "FastSAM does not support cache"
            return
        job = self.logic.embedding_job_for(self._parameterNode) if self.logic else None
        if job:
            self.updateEmbeddingJobProgress(job)
            return
        self.ui.embeddingProgressBar.setRange(0, max(len(labels), 1))
        self.ui.embeddingProgressBar.value = len(labels)
        self.ui.embeddingStatusLabel.text = f"{len(labels)} embedded" if labels else "Not embedded"

    def currentWeightIsFastSamWithoutCache(self):
        weight = self._weights.get(self.currentWeightId()) if self._weights else None
        capabilities = weight.get("capabilities", {}) if weight else {}
        return bool(weight and weight.get("backend") == "fastsam" and not capabilities.get("embeddings", False))

    def updateEmbeddingJobProgress(self, job):
        total = max(job.get("total", 0), 1)
        completed = job.get("completed", 0)
        self.ui.embeddingProgressBar.setRange(0, total)
        self.ui.embeddingProgressBar.value = completed
        self.ui.embeddingStatusLabel.text = self.embeddingJobMessage(job)

    def updateBoxVolumeView(self):
        job = self.logic.box_volume_job_for(self._parameterNode) if self.logic else None
        if job:
            self.updateBoxVolumeJobProgress(job)
            return
        self.ui.boxVolumeProgressBar.setRange(0, 1)
        self.ui.boxVolumeProgressBar.value = 0
        self.ui.boxVolumeStatusLabel.text = "Not running"

    def updateBoxVolumeJobProgress(self, job):
        total = max(job.get("total", 0), 1)
        self.ui.boxVolumeProgressBar.setRange(0, total)
        self.ui.boxVolumeProgressBar.value = job.get("applied", job.get("completed", 0))
        self.ui.boxVolumeStatusLabel.text = self.boxVolumeJobMessage(job)

    def embeddingJobMessage(self, job):
        if not job:
            return "Not embedded"
        status = job.get("status")
        completed = job.get("completed", 0)
        submitted = job.get("submitted", completed)
        total = job.get("total", 0)
        if status == "failed":
            return f"Embedding failed: {job.get('error')}"
        if status == "complete":
            return f"Embedded {completed} of {total} slices"
        if status in ("queued", "running"):
            if submitted < total:
                return f"Submitting {submitted} of {total}; embedded {completed}"
            return f"Embedding {completed} of {total} slices"
        return f"{completed} embedded"

    def boxVolumeJobMessage(self, job):
        if not job:
            return "Not running"
        status = job.get("status")
        applied = job.get("applied", job.get("completed", 0))
        submitted = job.get("submitted", applied)
        total = job.get("total", 0)
        view = job.get("view", "view")
        mode = job.get("mode")
        if status == "failed":
            if mode == "auto_video":
                prefix = "Auto slices-as-video prediction"
            else:
                prefix = "Slices-as-video prediction" if mode == "video" else "3D box prediction"
            return f"{prefix} failed: {job.get('error')}"
        if status == "cancelled":
            label = "auto video prediction" if mode == "auto_video" else ("3D-box video prediction" if mode == "video" else "3D box prediction")
            return f"Cancelled {label} after {applied} of {total} {view} slices"
        if status == "complete":
            prefix = "Auto-video predicted" if mode == "auto_video" else ("Video-predicted" if mode == "video" else "Predicted")
            return f"{prefix} {applied} of {total} {view} slices"
        if status == "uploading":
            label = "auto video prediction" if mode == "auto_video" else "video prediction"
            return f"Uploading {submitted} of {total} {view} slices for {label}"
        if status in ("queued", "running"):
            if submitted < total:
                return f"Submitting {submitted} of {total}; predicted {applied}"
            prefix = "Auto-video predicting" if mode == "auto_video" else ("Video-predicting" if mode == "video" else "Predicting")
            return f"{prefix} {applied} of {total} {view} slices"
        return f"{applied} predicted"

    def autosaveSessionState(self, force=False):
        if not self._parameterNode:
            return self._lastAutosaveManifest
        now = time.monotonic()
        if not force and now - self._lastAutosave < self.autosaveMs / 1000:
            return self._lastAutosaveManifest
        self._lastAutosave = now
        manifest = {}
        if self._segmentationDirty and self._parameterNode.segmentation:
            result = self.logic.save_segmentation(self._parameterNode, self.logic.auto_segmentation_path(self._parameterNode))
            self._segmentationDirty = False
            manifest["segmentation"] = result["path"]
        canSaveEmbeddings = bool(
            self.logic.client
            and self._prepared
            and self._embeddingDirty
            and not self.logic.embedding_job_running()
            and self.logic.current_embeddings(self._parameterNode)
        )
        if canSaveEmbeddings:
            result = self.logic.save_embeddings(self._parameterNode, self.logic.auto_embedding_path(self._parameterNode))
            self._embeddingDirty = False
            manifest["embeddings"] = result["path"]
        self._lastAutosaveManifest.update(manifest)
        return self._lastAutosaveManifest

    def syncSegmentCombo(self):
        if self._updatingSegmentCombo:
            return
        state = self.segmentState()
        if state != self._segmentState:
            self.refreshSegments()

    def refreshSegments(self):
        self._updatingSegmentCombo = True
        try:
            if not self._parameterNode or not self._parameterNode.segmentation:
                if self._parameterNode:
                    self._parameterNode.selectedSegmentId = ""
                self._segmentIdsByLabel = {}
                self.setComboLabels(self.ui.segmentComboBox, [])
                self._segmentState = None
                return
            segmentId = self.logic.selected_segment_id(self._parameterNode)
            choices = self.segmentChoices(self.logic.segment_items(self._parameterNode.segmentation))
            self._segmentIdsByLabel = {label: segmentId for label, segmentId in choices}
            self.setComboLabels(self.ui.segmentComboBox, [label for label, segmentId in choices])
            if segmentId:
                currentLabel = next(label for label, itemSegmentId in choices if itemSegmentId == segmentId)
                self.ui.segmentComboBox.setCurrentText(currentLabel)
            self._segmentState = self.segmentState()
        finally:
            self._updatingSegmentCombo = False

    def syncMaskSegmentCombo(self):
        if self._updatingMaskSegmentCombo:
            return
        state = self.maskSegmentState()
        if state != self._maskSegmentState:
            self.refreshMaskSegments()

    def refreshMaskSegments(self):
        self._updatingMaskSegmentCombo = True
        try:
            if not self._parameterNode or not self._parameterNode.maskSegmentation:
                if self._parameterNode:
                    self._parameterNode.maskSegmentId = ""
                self._maskSegmentIdsByLabel = {}
                self.setComboLabels(self.ui.maskSegmentComboBox, [])
                self._maskSegmentState = None
                return
            segmentId = self.logic.selected_mask_segment_id(self._parameterNode)
            choices = self.segmentChoices(self.logic.segment_items(self._parameterNode.maskSegmentation))
            self._maskSegmentIdsByLabel = {label: segmentId for label, segmentId in choices}
            self.setComboLabels(self.ui.maskSegmentComboBox, [label for label, segmentId in choices])
            if segmentId:
                currentLabel = next(label for label, itemSegmentId in choices if itemSegmentId == segmentId)
                self.ui.maskSegmentComboBox.setCurrentText(currentLabel)
            self._maskSegmentState = self.maskSegmentState()
        finally:
            self._updatingMaskSegmentCombo = False

    def segmentState(self):
        if not self._parameterNode or not self._parameterNode.segmentation:
            return None
        return (self._parameterNode.segmentation.GetID(), tuple(self.logic.segment_items(self._parameterNode.segmentation)))

    def maskSegmentState(self):
        if not self._parameterNode or not self._parameterNode.maskSegmentation:
            return None
        return (self._parameterNode.maskSegmentation.GetID(), tuple(self.logic.segment_items(self._parameterNode.maskSegmentation)))

    def segmentChoices(self, items):
        names = [name or segmentId for segmentId, name in items]
        duplicates = {name for name in names if names.count(name) > 1}
        return [
            (f"{label} ({segmentId})" if label in duplicates else label, segmentId)
            for (segmentId, _), label in zip(items, names)
        ]

    def setModelFamily(self, modelId):
        modelIds = list(self._models)
        modelId = modelId if modelId in self._models else modelIds[0]
        self._parameterNode.modelFamily = modelId
        self.ui.modelFamilyComboBox.setCurrentText(self._models[modelId]["label"])

    def setWeights(self, modelId, weightId):
        weights = self._models[modelId]["weights"]
        self._weightIdsByLabel = {weight["label"]: weight["id"] for weight in weights}
        weightIds = [weight["id"] for weight in weights]
        weightId = weightId if weightId in weightIds else weightIds[0]
        self.setComboLabels(self.ui.weightComboBox, [weight["label"] for weight in weights])
        self._parameterNode.weightName = weightId
        self.ui.weightComboBox.setCurrentText(self._weights[weightId]["label"])
        self.updateModelAvailability()

    def currentModelFamilyId(self):
        return self._modelIdsByLabel.get(self.ui.modelFamilyComboBox.currentText, self._parameterNode.modelFamily)

    def currentWeightId(self):
        return self._weightIdsByLabel.get(self.ui.weightComboBox.currentText, self._parameterNode.weightName)

    def onMaskSegmentChanged(self, index=None):
        segmentId = self._maskSegmentIdsByLabel.get(self.ui.maskSegmentComboBox.currentText)
        if self._updatingMaskSegmentCombo or not self._parameterNode or not segmentId:
            return
        self._parameterNode.maskSegmentId = segmentId
        self.updateControls()
