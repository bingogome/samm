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
        self._flag_prompt_sync      = False
        self._flag_promptpts_sync   = False
        self._frozenSlice           = {"R": [], "G": [], "Y": []}
        self._latlogger             = LatencyLogger()

    def setDefaultParameters(self, parameterNode):
        """
        Initialize parameter node with default settings.
        """
        if not parameterNode.GetParameter("sammDataOptions"):
            parameterNode.SetParameter("sammDataOptions", "Volume")

    def processGetVolumeMetaData(self):
        
        inModel         = self._parameterNode.GetNodeReference("sammInputVolume")
        imageData       = slicer.util.arrayFromVolume(inModel)
        imageDataShape  = imageData.shape
        self._segNumpy  = numpy.zeros(imageDataShape)

        # get axis directions aligning RGY views (need to optimize here)
        IjkToRasDir = numpy.array([[0,0,0],[0,0,0],[0,0,0]])
        self._parameterNode.GetNodeReference("sammInputVolume").GetIJKToRASDirections(IjkToRasDir)
        self._parameterNode.RGYNpArrOrder = [0, 0, 0]
        for i in range(3):
            self._parameterNode.RGYNpArrOrder[i] = 2-numpy.argmax(numpy.abs(IjkToRasDir.transpose()[i]))
        metadata = [ \
            [], \
            self._parameterNode.RGYNpArrOrder
        ]
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
        imageNormalized = imageNormalized.transpose(2-np.array(self._parameterNode.RGYNpArrOrder))

        self._parameterNode._volMetaData[0] = imageNormalized.shape

        return imageNormalized
    

    def processComputeEmbeddings(self):
        # checkers
        if not self.ui.pathWorkSpace.currentPath:
            slicer.util.errorDisplay("Please select workspace path first!")
            return

        if not self._parameterNode.GetNodeReferenceID("sammInputVolume"):
            slicer.util.errorDisplay("Please select a volume first!")
            return

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
        self._connections.pushRequest(SammMsgType.CALCULATE_EMBEDDINGS, {})
        print("[SAMM INFO] Sent Embedding Computing Command.")
        

    def processMaskSync(self, curslc, mask, view, shape):
        """
        Receives updated masks 
        """
        
        if curslc not in self._frozenSlice[view[0]]:
            mask = np.frombuffer(mask, dtype="uint8").reshape(shape)
            self._segNumpy = self._segNumpy.transpose(
                2-np.array(self._parameterNode.RGYNpArrOrder)
            )

            if view == "RED":
                self._segNumpy[curslc,:,:] = mask
            if view == "GREEN":
                self._segNumpy[:,curslc,:] = mask
            if view == "YELLOW":
                self._segNumpy[:,:,curslc] = mask
            
            self._latlogger.event_receive_mask()

            self._segNumpy = self._segNumpy.transpose(
                (2-np.array(self._parameterNode.RGYNpArrOrder)).argsort()
            )
            slicer.util.updateSegmentBinaryLabelmapFromArray( \
                self._segNumpy, \
                self._parameterNode.GetNodeReference("sammSegmentation"), \
                self._parameterNode.GetParameter("sammCurrentSegment"), \
                self._parameterNode.GetNodeReference("sammInputVolume") )
                
            self._latlogger.event_apply_mask()

    def processInitPromptSync(self):
        
        # Init
        self._prompt_add    = self._parameterNode.GetNodeReference("sammPromptAdd")
        self._prompt_remove = self._parameterNode.GetNodeReference("sammPromptRemove")

        if self._parameterNode.GetParameter("sammViewOptions") == "RED":
            self._slider = \
                slicer.app.layoutManager().sliceWidget('Red').sliceController().sliceOffsetSlider()
            self._viewController = \
                slicer.app.layoutManager().sliceWidget('Red').sliceController().mrmlSliceNode()
        if self._parameterNode.GetParameter("sammViewOptions") == "GREEN":
            self._slider = \
                slicer.app.layoutManager().sliceWidget('Green').sliceController().sliceOffsetSlider()
            self._viewController = \
                slicer.app.layoutManager().sliceWidget('Green').sliceController().mrmlSliceNode()
        if self._parameterNode.GetParameter("sammViewOptions") == "YELLOW":
            self._slider = \
                slicer.app.layoutManager().sliceWidget('Yellow').sliceController().sliceOffsetSlider()
            self._viewController = \
                slicer.app.layoutManager().sliceWidget('Yellow').sliceController().mrmlSliceNode()
        
        # get ijk ras
        volumeRasToIjk  = vtk.vtkMatrix4x4()
        self._parameterNode.GetNodeReference("sammInputVolume").GetRASToIJKMatrix(volumeRasToIjk)
        self._volumeRasToIjk = volumeRasToIjk
        volumeIjkToRas  = vtk.vtkMatrix4x4()
        self._parameterNode.GetNodeReference("sammInputVolume").GetIJKToRASMatrix(volumeIjkToRas)
        self._volumeIjkToRas = volumeIjkToRas

    def utilGetCurrentSliceIndex(self):
        
        mat = self._viewController.GetXYToRAS()
        temp = mat.MultiplyPoint([0,0,0,1])
        ras2ijk = vtk.vtkMatrix4x4()
        self._parameterNode.GetNodeReference("sammInputVolume").GetRASToIJKMatrix(ras2ijk)
        temp = ras2ijk.MultiplyPoint(temp)
        temp = np.array([temp[0],temp[1],temp[2]])[self._parameterNode.RGYNpArrOrder]
        view = self._parameterNode.GetParameter("sammViewOptions")
        if view == "RED":
            curslc = temp[0]
            imshape = (self._parameterNode._volMetaData[0][1], self._parameterNode._volMetaData[0][2])
        if view == "GREEN":
            curslc = temp[1]
            imshape = (self._parameterNode._volMetaData[0][0], self._parameterNode._volMetaData[0][2])
        if view == "YELLOW":
            curslc = temp[2] 
            imshape = (self._parameterNode._volMetaData[0][0], self._parameterNode._volMetaData[0][1])
        return curslc, view, imshape
    
    def utilGetPositionOnSlicer(self, temp, view):
        if view == "RED":
            return [round(temp[1]),round(temp[2])]
        if view == "GREEN":
            return [round(temp[0]),round(temp[2])]
        if view == "YELLOW":
            return [round(temp[0]),round(temp[1])]

    def processStartPromptSync(self):
        """
        Sends updated prompts
        """

        prompt_add_point, prompt_remove_point = [], []

        if self._flag_prompt_sync:

            curslc, view, imshape = self.utilGetCurrentSliceIndex()
            curslc = int(curslc)

            mask = None

            if curslc not in self._frozenSlice[view[0]]:

                numControlPoints = self._prompt_add.GetNumberOfControlPoints()
                for i in range(numControlPoints):
                    ras = vtk.vtkVector3d(0,0,0)
                    self._prompt_add.GetNthControlPointPosition(i,ras)
                    temp = self._volumeRasToIjk.MultiplyPoint([ras[0],ras[1],ras[2],1])
                    temp = np.array([temp[0], temp[1], temp[2]])[self._parameterNode.RGYNpArrOrder]
                    prompt_add_point.append(self.utilGetPositionOnSlicer(temp, view))

                numControlPoints = self._prompt_remove.GetNumberOfControlPoints()
                for i in range(numControlPoints):
                    ras = vtk.vtkVector3d(0,0,0)
                    self._prompt_remove.GetNthControlPointPosition(i,ras)
                    temp = self._volumeRasToIjk.MultiplyPoint([ras[0],ras[1],ras[2],1])
                    temp = np.array([temp[0], temp[1], temp[2]])[self._parameterNode.RGYNpArrOrder]
                    prompt_remove_point.append(self.utilGetPositionOnSlicer(temp, view))

                mask = self._connections.pushRequest(SammMsgType.INFERENCE, {
                    "n" : curslc,
                    "view" : self._parameterNode.GetParameter("sammViewOptions")[0],
                    "positivePrompts" : np.array(prompt_add_point),
                    "negativePrompts" : np.array(prompt_remove_point)
                })

            self._latlogger.event_send_inferencerequest()

            self.processMaskSync(curslc, mask, view, imshape)

            qt.QTimer.singleShot(60, self.processStartPromptSync)

    def processPromptPointsSync(self):

        pass
        # if self._flag_promptpts_sync:

        #     mode = slicer.mrmlScene.GetNodeByID("vtkMRMLInteractionNodeSingleton").GetInteractionModeAsString()            
        #     numControlPoints = self._prompt_add.GetNumberOfControlPoints()
        #     if mode == "Place":
        #         numControlPoints = numControlPoints - 1
        #     for i in range(numControlPoints):
        #         ras = vtk.vtkVector3d(0,0,0)
        #         self._prompt_add.GetNthControlPointPosition(i,ras)
        #         temp = self._volumeRasToIjk.MultiplyPoint([ras[0],ras[1],ras[2],1])
        #         if self._parameterNode.RGYNpArrOrder[0] == 0: # assume red TODO (expand to different view)
        #             curslc = (self._slider.value-self._parameterNode._volMetaData[0][0])/self._parameterNode._volMetaData[0][2]
        #             ras = self._volumeIjkToRas.MultiplyPoint([curslc,temp[1],temp[2],1])
        #         elif self._parameterNode.RGYNpArrOrder[0] == 1:
        #             curslc = (self._parameterNode._volMetaData[0][1]-self._slider.value)/self._parameterNode._volMetaData[0][2]
        #             ras = self._volumeIjkToRas.MultiplyPoint([temp[0],curslc,temp[2],1])
        #         elif self._parameterNode.RGYNpArrOrder[0] == 2:
        #             curslc = (self._slider.value-self._parameterNode._volMetaData[0][0])/self._parameterNode._volMetaData[0][2]
        #             ras = self._volumeIjkToRas.MultiplyPoint([temp[0],temp[1],curslc, 1])
        #         self._prompt_add.SetNthControlPointPosition(i,ras[0],ras[1],ras[2])

        #     numControlPoints = self._prompt_remove.GetNumberOfControlPoints()
        #     if mode == "Place":
        #         numControlPoints = numControlPoints - 1
        #     for i in range(numControlPoints):
        #         ras = vtk.vtkVector3d(0,0,0)
        #         self._prompt_remove.GetNthControlPointPosition(i,ras)
        #         temp = self._volumeRasToIjk.MultiplyPoint([ras[0],ras[1],ras[2],1])
        #         if self._parameterNode.RGYNpArrOrder[0] == 0: # assume red TODO (expand to different view)
        #             ras = self._volumeIjkToRas.MultiplyPoint([curslc,temp[1],temp[2],1])
        #         elif self._parameterNode.RGYNpArrOrder[0] == 1:
        #             ras = self._volumeIjkToRas.MultiplyPoint([temp[0],curslc,temp[2],1])
        #         elif self._parameterNode.RGYNpArrOrder[0] == 2:
        #             ras = self._volumeIjkToRas.MultiplyPoint([temp[0],temp[1],curslc, 1])
        #         self._prompt_remove.SetNthControlPointPosition(i,ras[0],ras[1],ras[2])
                
        #     qt.QTimer.singleShot(60, self.processPromptPointsSync)