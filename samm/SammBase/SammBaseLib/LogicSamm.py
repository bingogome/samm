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

from SammBaseLib.UtilLatencyLogger import LatencyLogger
from SammBaseLib.UtilMsgFactory import *
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
import slicer, qt, json, os, vtk, numpy, copy, pickle, shutil
from tqdm import tqdm


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
        self._parameterNode         = self.getParameterNode()
        self._connections           = None
        self._flag_mask_sync        = False
        self._flag_prompt_sync      = False
        self._flag_promptpts_sync   = False
        self._frozenSlice           = []
        self._latlogger             = LatencyLogger()

    def setDefaultParameters(self, parameterNode):
        """
        Initialize parameter node with default settings.
        """
        if not parameterNode.GetParameter("sammDataOptions"):
            parameterNode.SetParameter("sammDataOptions", "Volume")

    def processGetVolumeMetaData(self):

        def getViewData(strview, numview):
            sliceController = slicer.app.layoutManager().sliceWidget(strview).sliceController()
            minSliceVal     = sliceController.sliceOffsetSlider().minimum
            maxSliceVal     = sliceController.sliceOffsetSlider().maximum
            spacingSlice    = (maxSliceVal - minSliceVal) / imageDataShape[numview]
            return [minSliceVal, maxSliceVal, spacingSlice]
        
        inModel         = self._parameterNode.GetNodeReference("sammInputVolume")
        imageData       = slicer.util.arrayFromVolume(inModel)
        imageDataShape  = imageData.shape

        # get axis directions aligning RGY views (need to optimize here)
        IjkToRasDir = numpy.array([[0,0,0],[0,0,0],[0,0,0]])
        self._parameterNode.GetNodeReference("sammInputVolume").GetIJKToRASDirections(IjkToRasDir)
        self._parameterNode.RGYNpArrOrder = [0, 0, 0]
        for i in range(3):
            self._parameterNode.RGYNpArrOrder[i] = numpy.argmax(numpy.abs(IjkToRasDir.transpose()[i]))
        metadata        = \
                [getViewData("Red", self._parameterNode.RGYNpArrOrder[0]), \
                getViewData("Green", self._parameterNode.RGYNpArrOrder[1]), \
                getViewData("Yellow", self._parameterNode.RGYNpArrOrder[2]), \
                self._parameterNode.RGYNpArrOrder]
        self._parameterNode._volMetaData = metadata
        return 

                
    def processSlicePreProcess(self):
        """
        Takes in slices and pre process
            1. Get the intensity changed images
            2. Apply a "conversion" from original axis to a axis-agnostic format (order: RGY)
        """
        self.processGetVolumeMetaData()

        volumeNodeDataPointer = slicer.util.arrayFromVolume(
            self._parameterNode.GetNodeReference("sammInputVolume"))
        volumeDisplayNode = self._parameterNode.GetNodeReference("sammInputVolume").GetDisplayNode()
        window = volumeDisplayNode.GetWindow()
        level = volumeDisplayNode.GetLevel()

        imageMin = level - window / 2
        imageMax = level + window / 2

        imageNormalized = (((copy.deepcopy(volumeNodeDataPointer) - imageMin).astype("float32") / (imageMax - imageMin)) * 256).astype(numpy.uint8)
        imageNormalized = imageNormalized.transpose(self._parameterNode.RGYNpArrOrder)

        return imageNormalized
    

    def processComputeEmbeddings(self):
        # checkers
        if not self.ui.pathWorkSpace.currentPath:
            slicer.util.errorDisplay("Please select workspace path first!")
            return

        if not self._parameterNode.GetNodeReferenceID("sammInputVolume"):
            slicer.util.errorDisplay("Please select a volume first!")
            return

        # get meta data
        self.processGetVolumeMetaData()

        # get image slices
        imageNormalized = self.processSlicePreProcess()

        # send sizes to server
        self._connections.pushRequest(SammMsgType.SET_IMAGE_SIZE, {
            "r" : imageNormalized.shape[0],
            "g" : imageNormalized.shape[1],
            "y" : imageNormalized.shape[2]
        })
        print("[SAMM INFO] Sent Size Command.")

        # send volume, slice by slice on R view
        for i in tqdm(range(imageNormalized.shape[0])):
            self._connections.pushRequest(SammMsgType.SET_NTH_IMAGE, {
                "n"     : i, 
                "N"     : {
                    "R": imageNormalized.shape[0], 
                    "G": imageNormalized.shape[1], 
                    "Y": imageNormalized.shape[2]
                },
                "image" : imageNormalized[i,:,:]
            })
        print("[SAMM INFO] Sent Image.")

        # send embedding request    
        self._connections.pushRequest(SammMsgType.CALCULATE_EMBEDING, {})
        print("[SAMM INFO] Sent Embedding Computing Command.")

        # f = open(self._parameterNode._workspace + "/imgsize", "w+")
        # f.write("IMAGE_WIDTH: " + str(img.shape[0]) + "\n" + "IMAGE_HEIGHT: " + str(img.shape[1]) + "\n" )
        # f.close()
        
    def processInitMaskSync(self):
        # load in volume meta data (need to optimize here)
        inModel         = self._parameterNode.GetNodeReference("sammInputVolume")
        imageData       = slicer.util.arrayFromVolume(inModel)
        imageSliceNum   = imageData.shape
        self._imageSliceNum = imageSliceNum
        self._segNumpy  = numpy.zeros(imageSliceNum)

    def processStartMaskSync(self):
        """
        Receives updated masks 
        """
        
        if self._flag_mask_sync:
            
            # assume red TODO (expand to different view)
            if self._parameterNode.RGYNpArrOrder[0] == 0:
                curslc = round((self._slider.value-self._parameterNode._volMetaData[0][0])/self._parameterNode._volMetaData[0][2])
                sliceshape = (self._imageSliceNum[0], self._imageSliceNum[1])
            elif self._parameterNode.RGYNpArrOrder[0] == 1:
                curslc = round((self._parameterNode._volMetaData[0][1]-self._slider.value)/self._parameterNode._volMetaData[0][2])
                sliceshape = (self._imageSliceNum[0], self._imageSliceNum[2])
            elif self._parameterNode.RGYNpArrOrder[0] == 2:
                curslc = round((self._slider.value-self._parameterNode._volMetaData[0][0])/self._parameterNode._volMetaData[0][2])
                sliceshape = (self._imageSliceNum[1], self._imageSliceNum[2])
            
            if curslc not in self._frozenSlice:
                memmap = numpy.memmap(self._parameterNode._workspace + '/mask.memmap', \
                    dtype='bool', mode='r+', shape=sliceshape) 
                # assume red TODO (expand to different view)
                if self._parameterNode.RGYNpArrOrder[0] == 0:
                    self._segNumpy[:,:,curslc] = memmap.astype(int)
                elif self._parameterNode.RGYNpArrOrder[0] == 1:
                    self._segNumpy[:,curslc,:] = memmap.astype(int)
                elif self._parameterNode.RGYNpArrOrder[0] == 2:
                    self._segNumpy[curslc,:,:] = memmap.astype(int)
                self._latlogger.event_receive_mask()
                del memmap
                slicer.util.updateSegmentBinaryLabelmapFromArray( \
                    self._segNumpy, \
                    self._parameterNode.GetNodeReference("sammSegmentation"), \
                    self._parameterNode.GetParameter("sammCurrentSegment"), \
                    self._parameterNode.GetNodeReference("sammInputVolume") )
                
            self._latlogger.event_apply_mask()
            qt.QTimer.singleShot(60, self.processStartMaskSync)

    def processInitPromptSync(self):
        
        # Init
        self._prompt_add    = self._parameterNode.GetNodeReference("sammPromptAdd")
        self._prompt_remove = self._parameterNode.GetNodeReference("sammPromptRemove")

        # get meta data
        self.processGetVolumeMetaData()
        
        # assume red TODO (expand to different view)
        self._slider    = slicer.app.layoutManager().sliceWidget('Red').sliceController().sliceOffsetSlider()
        volumeRasToIjk  = vtk.vtkMatrix4x4()
        self._parameterNode.GetNodeReference("sammInputVolume").GetRASToIJKMatrix(volumeRasToIjk)
        self._volumeRasToIjk = volumeRasToIjk
        volumeIjkToRas  = vtk.vtkMatrix4x4()
        self._parameterNode.GetNodeReference("sammInputVolume").GetIJKToRASMatrix(volumeIjkToRas)
        self._volumeIjkToRas = volumeIjkToRas

    def processStartPromptSync(self):
        """
        Sends updated prompts
        """

        prompt_add_point, prompt_remove_point = [], []

        if self._flag_prompt_sync:

            # assume red TODO (expand to different view)
            if self._parameterNode.RGYNpArrOrder[0] == 0:
                curslc = round((self._slider.value-self._parameterNode._volMetaData[0][0])/self._parameterNode._volMetaData[0][2])
            elif self._parameterNode.RGYNpArrOrder[0] == 1:
                curslc = round((self._parameterNode._volMetaData[0][1]-self._slider.value)/self._parameterNode._volMetaData[0][2])
            elif self._parameterNode.RGYNpArrOrder[0] == 2:
                curslc = round((self._slider.value-self._parameterNode._volMetaData[0][0])/self._parameterNode._volMetaData[0][2])

            if curslc not in self._frozenSlice:

                numControlPoints = self._prompt_add.GetNumberOfControlPoints()
                for i in range(numControlPoints):
                    ras = vtk.vtkVector3d(0,0,0)
                    self._prompt_add.GetNthControlPointPosition(i,ras)
                    temp = self._volumeRasToIjk.MultiplyPoint([ras[0],ras[1],ras[2],1])
                    if self._parameterNode.RGYNpArrOrder[0] == 0: # assume red TODO (expand to different view)
                        prompt_add_point.append([temp[1], temp[2]])
                    elif self._parameterNode.RGYNpArrOrder[0] == 1:
                        prompt_add_point.append([temp[0], temp[2]])
                    elif self._parameterNode.RGYNpArrOrder[0] == 2:
                        prompt_add_point.append([temp[0], temp[1]])

                numControlPoints = self._prompt_remove.GetNumberOfControlPoints()
                for i in range(numControlPoints):
                    ras = vtk.vtkVector3d(0,0,0)
                    self._prompt_remove.GetNthControlPointPosition(i,ras)
                    temp = self._volumeRasToIjk.MultiplyPoint([ras[0],ras[1],ras[2],1])
                    if self._parameterNode.RGYNpArrOrder[0] == 0: # assume red TODO (expand to different view)
                        prompt_remove_point.append([temp[1], temp[2]])
                    elif self._parameterNode.RGYNpArrOrder[0] == 1:
                        prompt_remove_point.append([temp[0], temp[2]])
                    elif self._parameterNode.RGYNpArrOrder[0] == 2:
                        prompt_remove_point.append([temp[0], temp[1]])

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

            self._latlogger.event_send_inferencerequest()
            qt.QTimer.singleShot(60, self.processStartPromptSync)

    def processPromptPointsSync(self):
        if self._flag_promptpts_sync:

            mode = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton").GetInteractionModeAsString()            
            numControlPoints = self._prompt_add.GetNumberOfControlPoints()
            if mode == "Place":
                numControlPoints = numControlPoints - 1
            for i in range(numControlPoints):
                ras = vtk.vtkVector3d(0,0,0)
                self._prompt_add.GetNthControlPointPosition(i,ras)
                temp = self._volumeRasToIjk.MultiplyPoint([ras[0],ras[1],ras[2],1])
                if self._parameterNode.RGYNpArrOrder[0] == 0: # assume red TODO (expand to different view)
                    curslc = (self._slider.value-self._parameterNode._volMetaData[0][0])/self._parameterNode._volMetaData[0][2]
                    ras = self._volumeIjkToRas.MultiplyPoint([curslc,temp[1],temp[2],1])
                elif self._parameterNode.RGYNpArrOrder[0] == 1:
                    curslc = (self._parameterNode._volMetaData[0][1]-self._slider.value)/self._parameterNode._volMetaData[0][2]
                    ras = self._volumeIjkToRas.MultiplyPoint([temp[0],curslc,temp[2],1])
                elif self._parameterNode.RGYNpArrOrder[0] == 2:
                    curslc = (self._slider.value-self._parameterNode._volMetaData[0][0])/self._parameterNode._volMetaData[0][2]
                    ras = self._volumeIjkToRas.MultiplyPoint([temp[0],temp[1],curslc, 1])
                self._prompt_add.SetNthControlPointPosition(i,ras[0],ras[1],ras[2])

            numControlPoints = self._prompt_remove.GetNumberOfControlPoints()
            if mode == "Place":
                numControlPoints = numControlPoints - 1
            for i in range(numControlPoints):
                ras = vtk.vtkVector3d(0,0,0)
                self._prompt_remove.GetNthControlPointPosition(i,ras)
                temp = self._volumeRasToIjk.MultiplyPoint([ras[0],ras[1],ras[2],1])
                if self._parameterNode.RGYNpArrOrder[0] == 0: # assume red TODO (expand to different view)
                    ras = self._volumeIjkToRas.MultiplyPoint([curslc,temp[1],temp[2],1])
                elif self._parameterNode.RGYNpArrOrder[0] == 1:
                    ras = self._volumeIjkToRas.MultiplyPoint([temp[0],curslc,temp[2],1])
                elif self._parameterNode.RGYNpArrOrder[0] == 2:
                    ras = self._volumeIjkToRas.MultiplyPoint([temp[0],temp[1],curslc, 1])
                self._prompt_remove.SetNthControlPointPosition(i,ras[0],ras[1],ras[2])
                
            qt.QTimer.singleShot(60, self.processPromptPointsSync)