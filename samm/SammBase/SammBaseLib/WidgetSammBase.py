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
        self.ui.pushSyncImage.connect('clicked(bool)', self.onPushSyncImage)
        self.ui.comboVolumeNode.connect(
            "currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)

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
        # self.ui.inputSelector.setCurrentNode(self._parameterNode.GetNodeReference("InputVolume"))
        
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

        # self._parameterNode.SetNodeReferenceID("InputVolume", self.ui.inputSelector.currentNodeID)

        self._parameterNode.EndModify(wasModified)

    def onPushSyncImage(self):
        """
        Sync the image to Meta SAM
        """

        if not self.ui.pathWorkSpace.currentPath:
            slicer.util.errorDisplay("Please select workspace path first!")
            return
        
        lm = slicer.app.layoutManager()
        redWidget = lm.sliceWidget('Red')
        redView = redWidget.sliceView()
        wti = vtk.vtkWindowToImageFilter()
        wti.SetInput(redView.renderWindow())
        wti.Update()

        vtk_image = wti.GetOutput()

        width, height, _ = vtk_image.GetDimensions()
        vtk_array = vtk_image.GetPointData().GetScalars()
        components = vtk_array.GetNumberOfComponents()
        img = vtk_to_numpy(vtk_array).reshape(height, width, components)
        
        print(img.shape)

        input_bytes = img.tobytes()

        SHARED_MEMORY_SIZE = len(input_bytes)
        fd = os.open('/home/yl/software/mmaptest/testtemp', os.O_CREAT | os.O_TRUNC | os.O_RDWR)
        #os.write(fd, b'\x00' * n)  # resize file
        os.truncate(fd, SHARED_MEMORY_SIZE)  # resize file

        print(type(mmap.ACCESS_WRITE))

        map = mmap.mmap(fd, SHARED_MEMORY_SIZE)
        map.write(input_bytes)


