"""
MIT License
Copyright (c) 2022 [Insert copyright holders]
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

import slicer, mmap, qt, vtk, os
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
        self.ui.pushComputePredictor.connect('clicked(bool)', self.onPushComputePredictor)
        self.ui.comboVolumeNode.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

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

        self._parameterNode.EndModify(wasModified)

    def onPushComputePredictor(self):
        """
        Sync the images to Meta SAM
        """

        # checkers
        if not self.ui.pathWorkSpace.currentPath:
            slicer.util.errorDisplay("Please select workspace path first!")
            return
        
        # get workspaces (optimize this!)
        workspacepath_arr = self.ui.pathWorkSpace.currentPath.strip().split("/")
        workspacepath_arr.pop()
        workspacepath = ""
        for i in workspacepath_arr:
            workspacepath = workspacepath + i + "/"

        if not self._parameterNode.GetNodeReference("sammInputVolume"):
            slicer.util.errorDisplay("Please select a volume first!")
            return

        # load in volume meta data (need to optimize here)
        inModel = self._parameterNode.GetNodeReference("sammInputVolume")
        imageData = slicer.util.arrayFromVolume(inModel)
        imageSliceNum = imageData.shape
        del imageData

        sliceController = slicer.app.layoutManager().sliceWidget("Red").sliceController()
        minSliceVal = sliceController.sliceOffsetSlider().minimum
        maxSliceVal = sliceController.sliceOffsetSlider().maximum
        spacingSlice = (maxSliceVal - minSliceVal) / imageSliceNum[2]

        # iterate through the slice (RED view)
        for slc in [0]:
        # for slc in range(imageSliceNum[2]):

            # set current slice offset

            lm = slicer.app.layoutManager()
            redWidget = lm.sliceWidget('Red')
            redWidget.sliceController().sliceOffsetSlider().value = minSliceVal + slc * spacingSlice
            slicer.app.processEvents()
            redView = redWidget.sliceView()
            wti = vtk.vtkWindowToImageFilter()
            wti.SetInput(redView.renderWindow())
            wti.Update()

            vtk_image = wti.GetOutput()

            width, height, _ = vtk_image.GetDimensions()
            vtk_array = vtk_image.GetPointData().GetScalars()
            components = vtk_array.GetNumberOfComponents()
            img = vtk_to_numpy(vtk_array).reshape(height, width, components)

            input_bytes = img.tobytes()

            SHARED_MEMORY_SIZE = len(input_bytes)
            fd = os.open(workspacepath + "slices/slc" + str(slc), os.O_CREAT | os.O_TRUNC | os.O_RDWR)
            os.truncate(fd, SHARED_MEMORY_SIZE)  # resize file

            map = mmap.mmap(fd, SHARED_MEMORY_SIZE)
            map.write(input_bytes)

        f = open(self.ui.pathWorkSpace.currentPath.strip(), "w")
        f.write("IMAGE_WIDTH: " + str(img.shape[0]) + "\n" + "IMAGE_HEIGHT: " + str(img.shape[1]) + "\n" )
        f.close()
