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

from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
import slicer, mmap, qt, json, os

#
# SammBaseLogic
#

class SammBaseLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self):
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)
        self._parameterNode = self.getParameterNode()
        self._connections = None
        self._flag_mask_sync = False
        self._flag_prompt_sync = False

    def setDefaultParameters(self, parameterNode):
        """
        Initialize parameter node with default settings.
        """
        # if not parameterNode.GetParameter("Threshold"):
        #     parameterNode.SetParameter("Threshold", "100.0")
        # if not parameterNode.GetParameter("Invert"):
        #     parameterNode.SetParameter("Invert", "false")

    def processComputePredictor(self):
        # checkers
        if not self.ui.pathWorkSpace.currentPath:
            slicer.util.errorDisplay("Please select workspace path first!")
            return

        if not self._parameterNode.GetNodeReference("sammInputVolume"):
            slicer.util.errorDisplay("Please select a volume first!")
            return
        
        # get workspaces (optimize this!)
        workspacepath_arr = self.ui.pathWorkSpace.currentPath.strip().split("/")
        workspacepath_arr.pop()
        workspacepath = ""
        for i in workspacepath_arr:
            workspacepath = workspacepath + i + "/"

        # load in volume meta data (need to optimize here)
        inModel = self._parameterNode.GetNodeReference("sammInputVolume")
        imageData = slicer.util.arrayFromVolume(inModel)
        imageSliceNum = imageData.shape

        sliceController = slicer.app.layoutManager().sliceWidget("Red").sliceController()
        minSliceVal = sliceController.sliceOffsetSlider().minimum
        maxSliceVal = sliceController.sliceOffsetSlider().maximum
        spacingSlice = (maxSliceVal - minSliceVal) / imageSliceNum[2]

        for slc in range(imageSliceNum[2]):

            # set current slice offset
            lm = slicer.app.layoutManager()
            redWidget = lm.sliceWidget('Red')
            redWidget.sliceController().sliceOffsetSlider().value = minSliceVal + slc * spacingSlice
            slicer.app.processEvents()

            img = imageData[:,slc,:]
            input_bytes = img.tobytes()

            SHARED_MEMORY_SIZE = len(input_bytes)
            fd = os.open(workspacepath + "slices/slc" + str(slc), os.O_CREAT | os.O_TRUNC | os.O_RDWR)
            os.truncate(fd, SHARED_MEMORY_SIZE)  # resize file

            map = mmap.mmap(fd, SHARED_MEMORY_SIZE)
            map.write(input_bytes)

        f = open(self.ui.pathWorkSpace.currentPath.strip(), "w")
        f.write("IMAGE_WIDTH: " + str(img.shape[0]) + "\n" + "IMAGE_HEIGHT: " + str(img.shape[1]) + "\n" )
        f.close()

        msg = {
            "command": "COMPUTE_EMBEDDING",
            "parameters": []
        }
        msg = json.dumps(msg)
        self._connections.sendCmd(msg)

    def processStartMaskSync(self):
        """
        Receives updated masks TODO
        """
        if self._flag_mask_sync:
            qt.QTimer.singleShot(250, self.processStartMaskSync)

    def processStartPromptSync(self):
        """
        Sends updated prompts TODO
        """
        if self._flag_prompt_sync:
            qt.QTimer.singleShot(250, self.processStartPromptSync)
