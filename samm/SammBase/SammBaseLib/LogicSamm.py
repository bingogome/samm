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
        if not parameterNode.GetParameter("PlanOnBrain"):
            parameterNode.SetParameter("sammSaveEmbToLocal", "false")

    def processGetVolumeMetaData(self):

        # checkers
        if not self.ui.pathWorkSpace.currentPath:
            slicer.util.errorDisplay("Please select workspace path first!")
            return

        if not self._parameterNode.GetNodeReferenceID("sammInputVolume"):
            slicer.util.errorDisplay("Please select a volume first!")
            return
        
        inModel         = self._parameterNode.GetNodeReference("sammInputVolume")
        imageData       = slicer.util.arrayFromVolume(inModel)
        imageDataShape  = imageData.shape
        self._segNumpy  = numpy.zeros(imageDataShape)

        # get axis directions aligning RGY views (need to optimize here)
        IjkToRasDir = numpy.array([[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]])
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

    def processSelectModel(self):
        # send sizes to server
        if self._parameterNode.GetParameter("sammModelSelection"):
            self._connections.pushRequest(SammMsgType.MODEL_SELECTION, {
                "model" : self._parameterNode.GetParameter("sammModelSelection")
            })
            print(f'[SAMM INFO] Model switched to: "{self._parameterNode.GetParameter("sammModelSelection")}"')
                
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

        # imageMin = np.amin(volumeNodeDataPointer)
        # imageMax = np.amax(volumeNodeDataPointer)

        imageNormalized = (((copy.deepcopy(volumeNodeDataPointer) - imageMin).astype("float32") / (imageMax - imageMin)) * 256)
        imageNormalized[imageNormalized<0] = 0
        imageNormalized[imageNormalized>255] = 255
        imageNormalized = imageNormalized.astype(numpy.uint8)
        
        imageNormalized = imageNormalized.transpose(2-np.array(self._parameterNode.RGYNpArrOrder))

        self._parameterNode._volMetaData[0] = imageNormalized.shape

        return imageNormalized
    
    def processPreEmbeddings(self):

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
    
    def processLoadEmbeddings(self):
        print("test1")
        self.processPreEmbeddings()
        # send embedding request    
        self._connections.pushRequest(SammMsgType.CALCULATE_EMBEDDINGS, {
            "saveToLocal" : int(self._parameterNode.GetParameter("sammSaveEmbToLocal") == "false"),
            "loadLocal" : 1
        })
        print("[SAMM INFO] Sent Embedding Command.")

    def processComputeEmbeddings(self):
        
        self.processPreEmbeddings()
        # send embedding request    
        self._connections.pushRequest(SammMsgType.CALCULATE_EMBEDDINGS, {
            "saveToLocal" : int(self._parameterNode.GetParameter("sammSaveEmbToLocal") == "true"),
            "loadLocal" : 0
        })
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
        self._prompt_2dbox = self._parameterNode.GetNodeReference("sammPrompt2DBox")

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
        temp = mat.MultiplyPoint([0.0, 0.0, 0.0, 1.0])
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
    
    def utilGetPositionOnSlice(self, ras, view):
        temp = self._volumeRasToIjk.MultiplyPoint([ras[0],ras[1],ras[2],1])
        temp = np.array([temp[0], temp[1], temp[2]])[self._parameterNode.RGYNpArrOrder]
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
            curslc = round(curslc)

            mask = None

            if curslc not in self._frozenSlice[view[0]]:

                numControlPoints = self._prompt_add.GetNumberOfControlPoints()
                for i in range(numControlPoints):
                    ras = vtk.vtkVector3d(0.0, 0.0, 0.0)
                    self._prompt_add.GetNthControlPointPosition(i,ras)
                    coord = self.utilGetPositionOnSlice(ras, view)
                    prompt_add_point.append([coord[1],coord[0]])

                numControlPoints = self._prompt_remove.GetNumberOfControlPoints()
                for i in range(numControlPoints):
                    ras = vtk.vtkVector3d(0.0, 0.0, 0.0)
                    self._prompt_remove.GetNthControlPointPosition(i,ras)
                    coord = self.utilGetPositionOnSlice(ras, view)
                    prompt_remove_point.append([coord[1],coord[0]])

                plane = self._parameterNode.GetNodeReference("sammPrompt2DBox")

                if plane:
                    points = vtk.vtkPoints() 
                    plane.GetPlaneCornerPoints(points)
                    ras = [points.GetPoint(0)[0],points.GetPoint(0)[1],points.GetPoint(0)[2]]
                    bbox1 = self.utilGetPositionOnSlice(ras, view)
                    ras = [points.GetPoint(2)[0],points.GetPoint(2)[1],points.GetPoint(2)[2]]
                    bbox2 = self.utilGetPositionOnSlice(ras, view)
                    bboxmin = [min(bbox1[1], bbox2[1]), min(bbox1[0], bbox2[0])]
                    bboxmax = [max(bbox1[1], bbox2[1]), max(bbox1[0], bbox2[0])]

                else:
                    bboxmin, bboxmax = [-404,-404], [-404,-404]

                mask = self._connections.pushRequest(SammMsgType.INFERENCE, {
                    "n" : curslc,
                    "view" : self._parameterNode.GetParameter("sammViewOptions")[0],
                    "bbox2D" : np.array([bboxmin[0], bboxmin[1], bboxmax[0], bboxmax[1]]),
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

    def processAutoSeg3D(self):

        self.processGetVolumeMetaData()
        self.processInitPromptSync()
        _ = self.processSlicePreProcess()

        bounds = [0, 0, 0, 0, 0, 0]
        slicer.mrmlScene.GetNodeByID(
            self._parameterNode.GetNodeReferenceID("sammPrompt3DBox")
        ).GetBounds(bounds)

        ras = [bounds[0],bounds[2],bounds[4]]
        bbox1 = self.utilGetPositionOnSlice(ras, "RED")
        ras = [bounds[1],bounds[3],bounds[5]]
        bbox2 = self.utilGetPositionOnSlice(ras, "RED")
        bboxmin_r = [min(bbox1[1], bbox2[1]), min(bbox1[0], bbox2[0])]
        bboxmax_r = [max(bbox1[1], bbox2[1]), max(bbox1[0], bbox2[0])]

        ras = [bounds[0],bounds[2],bounds[4]]
        bbox1 = self.utilGetPositionOnSlice(ras, "GREEN")
        ras = [bounds[1],bounds[3],bounds[5]]
        bbox2 = self.utilGetPositionOnSlice(ras, "GREEN")
        bboxmin_g = [min(bbox1[1], bbox2[1]), min(bbox1[0], bbox2[0])]
        bboxmax_g = [max(bbox1[1], bbox2[1]), max(bbox1[0], bbox2[0])]

        bboxmin = [bboxmin_r[0], bboxmin_r[1], bboxmin_g[1]]
        bboxmax = [bboxmax_r[0], bboxmax_r[1], bboxmax_g[1]]

        print(bboxmin, bboxmax)
        imshape = (self._parameterNode._volMetaData[0][1], self._parameterNode._volMetaData[0][2])

        # send inf request    
        for i in range(bboxmin_g[1], bboxmax_g[1]+1):
            mask = self._connections.pushRequest(SammMsgType.AUTO_SEG, {
                "segRangeMin" : bboxmin_r,
                "segRangeMax" : bboxmax_r,
                "segSlice" : i
            })
            print("[SAMM INFO] Sent Auto Segmentation Command.")
            self.processMaskSync(i, mask, "RED", imshape)