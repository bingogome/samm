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
import slicer, mmap, qt, json, os, vtk, numpy
import SimpleITK as sitk

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

    def processGetVolumeMetaData(self, imageDataShape):
        """
        Get the spacing, min, max of the view slider
        """
        def getViewData(strview, numview):
            sliceController = slicer.app.layoutManager().sliceWidget(strview).sliceController()
            minSliceVal = sliceController.sliceOffsetSlider().minimum
            maxSliceVal = sliceController.sliceOffsetSlider().maximum
            spacingSlice = (maxSliceVal - minSliceVal) / imageDataShape[numview]
            return [minSliceVal, maxSliceVal, spacingSlice]
        
        return [getViewData("Red", 2), getViewData("Green", 1), getViewData("Yellow", 0)]

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
        metadata = self.processGetVolumeMetaData(imageSliceNum)
        minSliceVal, maxSliceVal, spacingSlice = metadata[0][0], metadata[0][1], metadata[0][2]
        self._parameterNode._volMetaData = metadata

        for slc in range(imageSliceNum[2]):

            # set current slice offset
            lm = slicer.app.layoutManager()
            redWidget = lm.sliceWidget('Red')
            redWidget.sliceController().sliceOffsetSlider().value = maxSliceVal - slc * spacingSlice
            slicer.app.processEvents()

            img = imageData[:,slc,:]
            input_bytes = img.tobytes()

            SHARED_MEMORY_SIZE = len(input_bytes)
            fd = os.open(workspacepath + "slices/slc" + str(slc), os.O_CREAT | os.O_TRUNC | os.O_RDWR)
            os.truncate(fd, SHARED_MEMORY_SIZE)  # resize file

            # Use numpy memmap instead TODO
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
        print("Sent Embedding Computing Command.")

    def processInitMaskSync(self):
        # load in volume meta data (need to optimize here)
        inModel = self._parameterNode.GetNodeReference("sammInputVolume")
        imageData = slicer.util.arrayFromVolume(inModel)
        imageSliceNum = imageData.shape
        self._imageSliceNum = imageSliceNum
        self._workspace = "/home/yl/software/mmaptest"

    def processStartMaskSync(self):
        """
        Receives updated masks 
        """
        if self._flag_mask_sync:

            self.processInitMaskSync()

            memmap = numpy.memmap(self._workspace + '/mask.memmap', dtype='bool', mode='r+', shape=self._imageSliceNum)
            image = sitk.GetImageFromArray(memmap.astype(int))
            mask = sitk.BinaryThreshold(image, lowerThreshold=0, upperThreshold=1)
            slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode( \
                mask, self._parameterNode.GetNodeReferenceID("sammMask"))

            # qt.QTimer.singleShot(250, self.processStartMaskSync)

    def processInitPromptSync(self):
        # Init
        self._prompt_add = self._parameterNode.GetNodeReference("sammPromptAdd")
        self._prompt_remove = self._parameterNode.GetNodeReference("sammPromptRemove")
        # load in volume meta data (need to optimize here)
        inModel = self._parameterNode.GetNodeReference("sammInputVolume")
        imageData = slicer.util.arrayFromVolume(inModel)
        imageSliceNum = imageData.shape
        metadata = self.processGetVolumeMetaData(imageSliceNum)
        self._parameterNode._volMetaData = metadata
        self._slider = slicer.app.layoutManager().sliceWidget('Red').sliceController().sliceOffsetSlider()

    def processStartPromptSync(self):
        """
        Sends updated prompts TODO
        """
        self.processInitPromptSync()
        volumeRasToIjk = vtk.vtkMatrix4x4()
        self._parameterNode.GetNodeReference("sammInputVolume").GetRASToIJKMatrix(volumeRasToIjk)
        prompt_add_point, prompt_remove_point = [], []

        if self._flag_prompt_sync:

            curslc = round((self._parameterNode._volMetaData[0][1]-self._slider.value)/self._parameterNode._volMetaData[0][2])

            numControlPoints = self._prompt_add.GetNumberOfControlPoints()
            for i in range(numControlPoints):
                ras = vtk.vtkVector3d(0,0,0)
                self._prompt_add.GetNthControlPointPosition(i,ras)
                temp = volumeRasToIjk.MultiplyPoint([ras[0],ras[1],ras[2],1])
                prompt_add_point.append([temp[0], temp[2]])

            numControlPoints = self._prompt_remove.GetNumberOfControlPoints()
            for i in range(numControlPoints):
                ras = vtk.vtkVector3d(0,0,0)
                self._prompt_remove.GetNthControlPointPosition(i,ras)
                temp = volumeRasToIjk.MultiplyPoint([ras[0],ras[1],ras[2],1])
                prompt_remove_point.append([temp[0], temp[2]])

            msg = {
                "command": "INFER_IMAGE",
                "parameters": {
                    "point": prompt_add_point + prompt_remove_point,
                    "label": [1] * len(prompt_add_point) + [0] * len(prompt_remove_point),
                    "name": "slc"+str(curslc)
                }
            }

            msg = json.dumps(msg)
            self._connections.sendCmd(msg)

            qt.QTimer.singleShot(250, self.processStartPromptSync)
