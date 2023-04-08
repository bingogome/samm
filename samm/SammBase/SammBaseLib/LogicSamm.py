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

from XRSlicerBaseLib.UtilSlicerFuncs import QImageToCvMat
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
import slicer, mmap, qt

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

    def setDefaultParameters(self, parameterNode):
        """
        Initialize parameter node with default settings.
        """
        # if not parameterNode.GetParameter("Threshold"):
        #     parameterNode.SetParameter("Threshold", "100.0")
        # if not parameterNode.GetParameter("Invert"):
        #     parameterNode.SetParameter("Invert", "false")

    def processImgSync(self):

        if self._flag_img_sync:

            # img = qt.QPixmap.grabWidget(slicer.util.mainWindow()).toImage()
            # img = QImageToCvMat(img)
            # input_bytes = img.tobytes()

            # SHARED_MEMORY_SIZE = len(input_bytes)
            # TAG_NAME = "xr_MEM_MAP_SYNC_VIEW"

            # map = mmap.mmap(0, SHARED_MEMORY_SIZE, TAG_NAME, mmap.ACCESS_WRITE)
            # map.write(input_bytes)

            # qt.QTimer.singleShot(60, self.processViewSync)
