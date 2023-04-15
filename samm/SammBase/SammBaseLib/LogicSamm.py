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
import slicer, qt, json, os, vtk, numpy, copy, pickle, shutil
from datetime import datetime

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

        # Latency logging
        # log latency?
        self.flag_loglat            = False
        if self.flag_loglat:
            now                     = datetime.now()
            self.logctrmax          = 300
            self.timearr_SND_INF    = [now for idx in range(self.logctrmax)]
            self.timearr_RCV_MSK    = [now for idx in range(self.logctrmax)]
            self.timearr_APL_MSK    = [now for idx in range(self.logctrmax)]
            self.ctr_SND_INF        = 0
            self.ctr_RCV_MSK        = 0
            self.ctr_APL_MSK        = 0

    def setDefaultParameters(self, parameterNode):
        """
        Initialize parameter node with default settings.
        """

    def processGetVolumeMetaData(self, imageDataShape):
        """
        Get the spacing, min, max of the view slider
        """
        def getViewData(strview, numview):
            sliceController = slicer.app.layoutManager().sliceWidget(strview).sliceController()
            minSliceVal     = sliceController.sliceOffsetSlider().minimum
            maxSliceVal     = sliceController.sliceOffsetSlider().maximum
            spacingSlice    = (maxSliceVal - minSliceVal) / imageDataShape[numview]
            return [minSliceVal, maxSliceVal, spacingSlice]
        
        return [getViewData("Red", 2-self._parameterNode.RGYNpArrOrder[0]), \
                getViewData("Green", 2-self._parameterNode.RGYNpArrOrder[1]), \
                getViewData("Yellow", 2-self._parameterNode.RGYNpArrOrder[2])]

    def processComputePredictor(self):
        # checkers
        if not self.ui.pathWorkSpace.currentPath:
            slicer.util.errorDisplay("Please select workspace path first!")
            return

        if not self._parameterNode.GetNodeReferenceID("sammInputVolume"):
            slicer.util.errorDisplay("Please select a volume first!")
            return

        # load in volume meta data (need to optimize here)
        IjkToRasDir = numpy.array([[0,0,0],[0,0,0],[0,0,0]])
        self._parameterNode.GetNodeReference("sammInputVolume").GetIJKToRASDirections(IjkToRasDir)
        self._parameterNode.RGYNpArrOrder = [0, 0, 0]
        for i in range(3):
            self._parameterNode.RGYNpArrOrder[i] = 2-numpy.argmax(numpy.abs(IjkToRasDir.transpose()[i]))
        inModel         = self._parameterNode.GetNodeReference("sammInputVolume")
        imageData       = slicer.util.arrayFromVolume(inModel)
        imageSliceNum   = imageData.shape
        metadata        = self.processGetVolumeMetaData(imageSliceNum)
        minSliceVal, maxSliceVal, spacingSlice = metadata[0][0], metadata[0][1], metadata[0][2]
        self._parameterNode._volMetaData = metadata

        # create a folder to store slices
        output_folder = os.path.join(self._parameterNode._workspace, 'slices')
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # clear previous slices
        for filename in os.listdir(output_folder):
            file_path = os.path.join(output_folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

        # assume red TODO (expand to different view)
        for slc in range(imageSliceNum[2-self._parameterNode.RGYNpArrOrder[0]]):

            # set current slice offset
            lm = slicer.app.layoutManager()
            redWidget = lm.sliceWidget('Red') # assume red TODO (expand to different view)
            # send to server temp files # assume red TODO (expand to different view)
            if self._parameterNode.RGYNpArrOrder[0] == 0:
                redWidget.sliceController().sliceOffsetSlider().value = minSliceVal + slc * spacingSlice
            elif self._parameterNode.RGYNpArrOrder[1] == 1:
                redWidget.sliceController().sliceOffsetSlider().value = maxSliceVal - slc * spacingSlice
            elif self._parameterNode.RGYNpArrOrder[2] == 2:
                redWidget.sliceController().sliceOffsetSlider().value = minSliceVal + slc * spacingSlice

            slicer.app.processEvents()

            # send to server temp files # assume red TODO (expand to different view)
            if self._parameterNode.RGYNpArrOrder[0] == 0:
                img = imageData[:,:,slc]
            elif self._parameterNode.RGYNpArrOrder[0] == 1:
                img = imageData[:,slc,:]
            elif self._parameterNode.RGYNpArrOrder[0] == 2:
                img = imageData[slc,:,:]
            memmap = numpy.memmap(self._parameterNode._workspace + "/slices/slc" + str(slc), dtype='float64', mode='w+', shape=img.shape)
            memmap[:] = img[:]
            memmap.flush()

        f = open(self._parameterNode._workspace + "/imgsize", "w+")
        f.write("IMAGE_WIDTH: " + str(img.shape[0]) + "\n" + "IMAGE_HEIGHT: " + str(img.shape[1]) + "\n" )
        f.close()

        msg = {
            "command": "COMPUTE_EMBEDDING",
            "parameters": {
            }
        }
        msg = json.dumps(msg)
        self._connections.sendCmd(msg)
        print("Sent Embedding Computing Command.")

    def processInitMaskSync(self):
        # load in volume meta data (need to optimize here)
        inModel         = self._parameterNode.GetNodeReference("sammInputVolume")
        imageData       = slicer.util.arrayFromVolume(inModel)
        imageSliceNum   = imageData.shape
        self._imageSliceNum = imageSliceNum
        self._segNumpy  = numpy.zeros(imageSliceNum)

    def processSaveLatencyLog(self):

        file_name = self._parameterNode._workspace + "/timearr_SND_INF.pkl"
        with open(file_name, 'wb') as file:
            pickle.dump(self.timearr_SND_INF, file)

        file_name = self._parameterNode._workspace + "/timearr_RCV_MSK.pkl"
        with open(file_name, 'wb') as file:
            pickle.dump(self.timearr_RCV_MSK, file)

        file_name = self._parameterNode._workspace + "/timearr_APL_MSK.pkl"
        with open(file_name, 'wb') as file:
            pickle.dump(self.timearr_APL_MSK, file)

        print("Time for inference is saved.")

    def processStartMaskSync(self):
        """
        Receives updated masks 
        """
        
        if self._flag_mask_sync:
            
            # assume red TODO (expand to different view)
            if self._parameterNode.RGYNpArrOrder[0] == 0:
                curslc = round((self._parameterNode._volMetaData[0][0]+self._slider.value)/self._parameterNode._volMetaData[0][2])
                sliceshape = (self._imageSliceNum[0], self._imageSliceNum[1])
            elif self._parameterNode.RGYNpArrOrder[0] == 1:
                curslc = round((self._parameterNode._volMetaData[0][1]-self._slider.value)/self._parameterNode._volMetaData[0][2])
                sliceshape = (self._imageSliceNum[0], self._imageSliceNum[2])
            elif self._parameterNode.RGYNpArrOrder[0] == 2:
                curslc = round((self._parameterNode._volMetaData[0][0]+self._slider.value)/self._parameterNode._volMetaData[0][2])
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

                if self.flag_loglat:
                    self.timearr_RCV_MSK[self.ctr_RCV_MSK] = datetime.now()
                    self.ctr_RCV_MSK = self.ctr_RCV_MSK + 1
                    if self.ctr_RCV_MSK >= self.logctrmax - 1:
                        self.processSaveLatencyLog()
                        self.flag_loglat = False

                del memmap
                slicer.util.updateSegmentBinaryLabelmapFromArray( \
                    self._segNumpy, \
                    self._parameterNode.GetNodeReference("sammMask"), \
                    "current", \
                    self._parameterNode.GetNodeReference("sammInputVolume") )
                
            if self.flag_loglat:
                self.timearr_APL_MSK[self.ctr_APL_MSK] = datetime.now()
                self.ctr_APL_MSK = self.ctr_APL_MSK + 1
                if self.ctr_APL_MSK >= self.logctrmax - 1:
                    self.processSaveLatencyLog()
                    self.flag_loglat = False
                    
            qt.QTimer.singleShot(60, self.processStartMaskSync)

    def processInitPromptSync(self):
        # Init
        self._prompt_add    = self._parameterNode.GetNodeReference("sammPromptAdd")
        self._prompt_remove = self._parameterNode.GetNodeReference("sammPromptRemove")
        # load in volume meta data (need to optimize here)
        IjkToRasDir = numpy.array([[0,0,0],[0,0,0],[0,0,0]])
        self._parameterNode.GetNodeReference("sammInputVolume").GetIJKToRASDirections(IjkToRasDir)
        self._parameterNode.RGYNpArrOrder = [0, 0, 0]
        for i in range(3):
            self._parameterNode.RGYNpArrOrder[i] = 2-numpy.argmax(numpy.abs(IjkToRasDir.transpose()[i]))
        inModel         = self._parameterNode.GetNodeReference("sammInputVolume")
        imageData       = slicer.util.arrayFromVolume(inModel)
        imageSliceNum   = imageData.shape
        metadata        = self.processGetVolumeMetaData(imageSliceNum)
        self._parameterNode._volMetaData = metadata
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
                curslc = round((self._parameterNode._volMetaData[0][0]+self._slider.value)/self._parameterNode._volMetaData[0][2])
            elif self._parameterNode.RGYNpArrOrder[0] == 1:
                curslc = round((self._parameterNode._volMetaData[0][1]-self._slider.value)/self._parameterNode._volMetaData[0][2])
            elif self._parameterNode.RGYNpArrOrder[0] == 2:
                curslc = round((self._parameterNode._volMetaData[0][0]+self._slider.value)/self._parameterNode._volMetaData[0][2])

            if curslc not in self._frozenSlice:

                numControlPoints = self._prompt_add.GetNumberOfControlPoints()
                for i in range(numControlPoints):
                    ras = vtk.vtkVector3d(0,0,0)
                    self._prompt_add.GetNthControlPointPosition(i,ras)
                    temp = self._volumeRasToIjk.MultiplyPoint([ras[0],ras[1],ras[2],1])
                    if self._parameterNode.RGYNpArrOrder[0] == 0: # assume red TODO (expand to different view)
                        prompt_add_point.append([temp[0], temp[1]])
                    elif self._parameterNode.RGYNpArrOrder[0] == 1:
                        prompt_add_point.append([temp[0], temp[2]])
                    elif self._parameterNode.RGYNpArrOrder[0] == 2:
                        prompt_add_point.append([temp[1], temp[2]])

                numControlPoints = self._prompt_remove.GetNumberOfControlPoints()
                for i in range(numControlPoints):
                    ras = vtk.vtkVector3d(0,0,0)
                    self._prompt_remove.GetNthControlPointPosition(i,ras)
                    temp = self._volumeRasToIjk.MultiplyPoint([ras[0],ras[1],ras[2],1])
                    if self._parameterNode.RGYNpArrOrder[0] == 0: # assume red TODO (expand to different view)
                        prompt_remove_point.append([temp[0], temp[1]])
                    elif self._parameterNode.RGYNpArrOrder[0] == 1:
                        prompt_remove_point.append([temp[0], temp[2]])
                    elif self._parameterNode.RGYNpArrOrder[0] == 2:
                        prompt_remove_point.append([temp[1], temp[2]])

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

            if self.flag_loglat:
                self.timearr_SND_INF[self.ctr_SND_INF] = datetime.now()
                self.ctr_SND_INF = self.ctr_SND_INF + 1
                if self.ctr_SND_INF >= self.logctrmax - 1:
                    self.processSaveLatencyLog()
                    self.flag_loglat = False

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
                    curslc = (self._parameterNode._volMetaData[0][0]+self._slider.value)/self._parameterNode._volMetaData[0][2]
                    ras = self._volumeIjkToRas.MultiplyPoint([temp[0],temp[1],curslc,1])
                elif self._parameterNode.RGYNpArrOrder[0] == 1:
                    curslc = (self._parameterNode._volMetaData[0][1]-self._slider.value)/self._parameterNode._volMetaData[0][2]
                    ras = self._volumeIjkToRas.MultiplyPoint([temp[0],curslc,temp[2],1])
                elif self._parameterNode.RGYNpArrOrder[0] == 2:
                    curslc = (self._parameterNode._volMetaData[0][0]+self._slider.value)/self._parameterNode._volMetaData[0][2]
                    ras = self._volumeIjkToRas.MultiplyPoint([curslc,temp[1],temp[2],1])
                self._prompt_add.SetNthControlPointPosition(i,ras[0],ras[1],ras[2])

            numControlPoints = self._prompt_remove.GetNumberOfControlPoints()
            if mode == "Place":
                numControlPoints = numControlPoints - 1
            for i in range(numControlPoints):
                ras = vtk.vtkVector3d(0,0,0)
                self._prompt_remove.GetNthControlPointPosition(i,ras)
                temp = self._volumeRasToIjk.MultiplyPoint([ras[0],ras[1],ras[2],1])
                if self._parameterNode.RGYNpArrOrder[0] == 0: # assume red TODO (expand to different view)
                    ras = self._volumeIjkToRas.MultiplyPoint([temp[0],temp[1],curslc,1])
                elif self._parameterNode.RGYNpArrOrder[0] == 1:
                    ras = self._volumeIjkToRas.MultiplyPoint([temp[0],curslc,temp[2],1])
                elif self._parameterNode.RGYNpArrOrder[0] == 2:
                    ras = self._volumeIjkToRas.MultiplyPoint([curslc,temp[1],temp[2],1])
                self._prompt_remove.SetNthControlPointPosition(i,ras[0],ras[1],ras[2])
                
            qt.QTimer.singleShot(60, self.processPromptPointsSync)