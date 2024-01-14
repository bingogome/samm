"""
MIT License
Copyright (c) 2023 Yihao Liu
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import slicer, qt, os
from SammBaseLib.WidgetSamm import SammWidgetBase
from slicer.util import VTKObservationMixin
from vtk.util.numpy_support import vtk_to_numpy

class SammBaseWidget(SammWidgetBase):

    def __init__(self, parent=None):

        super().__init__(parent)

    def setup(self):

        super().setup()
        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # UI
        self.ui.radioWorkOnRed.connect("toggled(bool)", self.onRadioWorkOnOptions)
        self.ui.radioWorkOnGreen.connect("toggled(bool)", self.onRadioWorkOnOptions)
        self.ui.radioWorkOnYellow.connect("toggled(bool)", self.onRadioWorkOnOptions)
        self.ui.radioDataVolume.connect("toggled(bool)", self.onRadioDataOptions)
        self.ui.radioData2D.connect("toggled(bool)", self.onRadioDataOptions)

        self.ui.checkSaveToLocal.connect("toggled(bool)", self.updateParameterNodeFromGUI)

        self.ui.pushUseLocalEmb.connect("clicked(bool)", self.onPushUseLocalEmb)
        self.ui.pushComputePredictor.connect('clicked(bool)', self.onPushComputePredictor)
        self.ui.pushStartMaskSync.connect('clicked(bool)', self.onPushStartMaskSync)
        self.ui.pushStopMaskSync.connect('clicked(bool)', self.onPushStopMaskSync)
        self.ui.pushFreezeSlice.connect('clicked(bool)', self.onPushFreezeSlice)
        self.ui.pushUnfreezeSlice.connect('clicked(bool)', self.onPushUnfreezeSlice)
        self.ui.pushModuleSeg.connect('clicked(bool)', self.onPushModuleSeg)
        self.ui.pushModuleSegEditor.connect('clicked(bool)', self.onPushModuleSegEditor)

        self.ui.comboVolumeNode.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.comboSegmentationNode.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.comboSegmentNode.connect("currentIndexChanged(int)", self.updateParameterNodeFromGUI)
        self.ui.comboModel.connect("currentIndexChanged(int)", self.updateParameterNodeFromGUI)
        self.ui.comboModel.connect("currentIndexChanged(int)", self.onUpdateComboModel)
        comboModelItems = ['vit_b', 'vit_l', 'vit_h', 'mobile_vit_t', 'medsam_vit_b']
        for item in comboModelItems:
            self.ui.comboModel.addItem(item)

        self.ui.markupsAdd.connect("markupsNodeChanged()", self.updateParameterNodeFromGUI)
        self.ui.markupsRemove.connect("markupsNodeChanged()", self.updateParameterNodeFromGUI)
        self.ui.markups2DBox.connect("markupsNodeChanged()", self.updateParameterNodeFromGUI)
        self.ui.pushMarkups2DBox.connect("clicked(bool)", self.onPushMarkups2DBox)
        self.ui.markupsAdd.markupsPlaceWidget().setPlaceModePersistency(True)
        self.ui.markupsRemove.markupsPlaceWidget().setPlaceModePersistency(True)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

        # Attach to logic
        self.logic.ui = self.ui

    def updateGUIFromParameterNode(self, caller=None, event=None):
        """
        This method is called whenever parameter node is changed.
        The module GUI is updated to show the current state of the parameter node.
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
        self._updatingGUIFromParameterNode = True

        # Update node selectors and sliders
        self.ui.comboVolumeNode.setCurrentNode(self._parameterNode.GetNodeReference("sammInputVolume"))
        self.ui.markupsAdd.setCurrentNode(self._parameterNode.GetNodeReference("sammPromptAdd"))
        self.ui.markupsRemove.setCurrentNode(self._parameterNode.GetNodeReference("sammPromptRemove"))
        self.ui.markups2DBox.setCurrentNode(self._parameterNode.GetNodeReference("sammPrompt2DBox"))

        self.ui.comboSegmentationNode.setCurrentNode(self._parameterNode.GetNodeReference("sammSegmentation"))

        if self._parameterNode.GetNodeReferenceID("sammSegmentation"):
            self._parameterNode.GetNodeReference("sammSegmentation").SetReferenceImageGeometryParameterFromVolumeNode(
                self._parameterNode.GetNodeReference("sammInputVolume"))
        
        if self._parameterNode.GetNodeReferenceID("sammSegmentation"):
            segmentationNode = self._parameterNode.GetNodeReference("sammSegmentation")
            nOfSegments = segmentationNode.GetSegmentation().GetNumberOfSegments()
            self.ui.comboSegmentNode.clear()
            for i in range(nOfSegments):
                self.ui.comboSegmentNode.addItem(segmentationNode.GetSegmentation().GetNthSegmentID(i))

        self.ui.comboModel.setCurrentText(self._parameterNode.GetParameter("sammModelSelection"))

        self.ui.comboSegmentNode.setCurrentText(self._parameterNode.GetParameter("sammCurrentSegment"))
        self.ui.checkSaveToLocal.checked = (self._parameterNode.GetParameter("sammSaveEmbToLocal") == "true")
        
        if self._parameterNode.GetNodeReference("sammPromptAdd"):
            self._parameterNode.GetNodeReference("sammPromptAdd").GetDisplayNode().SetGlyphScale(1)
        if self._parameterNode.GetNodeReference("sammPromptRemove"):
            self._parameterNode.GetNodeReference("sammPromptRemove").GetDisplayNode().SetGlyphScale(1)

        # All the GUI updates are done
        self._updatingGUIFromParameterNode = False

    def updateParameterNodeFromGUI(self, caller=None, event=None):
        """
        This method is called when the user makes any change in the GUI.
        The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

        self._parameterNode.SetNodeReferenceID("sammInputVolume", self.ui.comboVolumeNode.currentNodeID)
        self._parameterNode.SetNodeReferenceID("sammPromptAdd", self.ui.markupsAdd.currentNode().GetID())
        self._parameterNode.SetNodeReferenceID("sammPromptRemove", self.ui.markupsRemove.currentNode().GetID())
        if self.ui.markups2DBox.currentNode():
            self._parameterNode.SetNodeReferenceID("sammPrompt2DBox", self.ui.markups2DBox.currentNode().GetID())
        self._parameterNode._workspace = os.path.dirname(os.path.abspath(self.ui.pathWorkSpace.currentPath.strip()))
        self._parameterNode.SetNodeReferenceID("sammSegmentation", self.ui.comboSegmentationNode.currentNodeID)
        self._parameterNode.GetNodeReference("sammSegmentation").SetReferenceImageGeometryParameterFromVolumeNode(
            self._parameterNode.GetNodeReference("sammInputVolume"))
        self._parameterNode.SetParameter("sammModelSelection", self.ui.comboModel.currentText)
        self._parameterNode.SetParameter("sammCurrentSegment", self.ui.comboSegmentNode.currentText)
        self._parameterNode.SetParameter("sammSaveEmbToLocal", "true" if self.ui.checkSaveToLocal.checked else "false")
        self.onRadioWorkOnOptions()
        self.onRadioDataOptions()

        self._parameterNode.EndModify(wasModified)

    def onRadioDataOptions(self):
        if self.ui.radioDataVolume.checked:
            self._parameterNode.SetParameter("sammDataOptions", "Volume")
        if self.ui.radioData2D.checked:
            self._parameterNode.SetParameter("sammDataOptions", "2D")

    def onRadioWorkOnOptions(self):
        if self.ui.radioWorkOnRed.checked:
            self._parameterNode.SetParameter("sammViewOptions", "RED")
            self.logic._slider = \
                slicer.app.layoutManager().sliceWidget('Red').sliceController().sliceOffsetSlider()
            self.logic._viewController = \
                slicer.app.layoutManager().sliceWidget('Red').sliceController().mrmlSliceNode()
        if self.ui.radioWorkOnGreen.checked:
            self._parameterNode.SetParameter("sammViewOptions", "GREEN")
            self.logic._slider = \
                slicer.app.layoutManager().sliceWidget('Green').sliceController().sliceOffsetSlider()
            self.logic._viewController = \
                slicer.app.layoutManager().sliceWidget('Green').sliceController().mrmlSliceNode()
        if self.ui.radioWorkOnYellow.checked:
            self._parameterNode.SetParameter("sammViewOptions", "YELLOW")
            self.logic._slider = \
                slicer.app.layoutManager().sliceWidget('Yellow').sliceController().sliceOffsetSlider()
            self.logic._viewController = \
                slicer.app.layoutManager().sliceWidget('Yellow').sliceController().mrmlSliceNode()

    def onUpdateComboModel(self):
        self.logic.processSelectModel()

    def onPushComputePredictor(self):
        self.logic.processComputeEmbeddings()

    def onPushUseLocalEmb(self):
        print("test")
        self.logic.processLoadEmbeddings()

    def onPushStartMaskSync(self):
        self.logic._flag_prompt_sync = True
        self.logic.processInitPromptSync()
        self.logic.processStartPromptSync()

        if self._parameterNode.GetParameter("sammDataOptions") == "Volume":
            self.logic._flag_promptpts_sync = True
            self.logic.processPromptPointsSync()
        
    def onPushStopMaskSync(self):
        self.logic._flag_promptpts_sync = False
        self.logic._flag_prompt_sync = False

    def onPushFreezeSlice(self):
        curslc, view, _ = self.logic.utilGetCurrentSliceIndex()
        if curslc not in self.logic._frozenSlice[view[0]]:
            self.logic._frozenSlice[view[0]].append(curslc)

    def onPushUnfreezeSlice(self):
        curslc, view, _ = self.logic.utilGetCurrentSliceIndex()
        if curslc in self.logic._frozenSlice[view[0]]:
            self.logic._frozenSlice[view[0]].remove(curslc)        

    def onPushModuleSeg(self):
        slicer.util.selectModule("Segmentations")

    def onPushModuleSegEditor(self):
        slicer.util.selectModule("SegmentEditor")

    def onPushMarkups2DBox(self):
        planeNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsPlaneNode').GetID()
        selectionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLSelectionNodeSingleton")
        selectionNode.SetReferenceActivePlaceNodeID(planeNode)
        interactionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton")
        placeModePersistence = 0
        interactionNode.SetPlaceModePersistence(placeModePersistence)
        # mode 1 is Place, can also be accessed via slicer.vtkMRMLInteractionNode().Place
        interactionNode.SetCurrentInteractionMode(1)

        self._parameterNode.SetNodeReferenceID("sammPrompt2DBox", planeNode)
        slicer.mrmlScene.GetNodeByID(planeNode).GetDisplayNode().SetGlyphScale(0.5)
        slicer.mrmlScene.GetNodeByID(planeNode).GetDisplayNode().SetInteractionHandleScale(1)