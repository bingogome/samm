"""
MIT License
Copyright (c) 2022 Yihao Liu
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

import vtk, math, slicer, json, qt
import ctypes

import numpy as np
import clr 
import System
from System import Array, Int32
from System.Runtime.InteropServices import GCHandle, GCHandleType

_MAP_NP_NET = {
    np.dtype(np.float32): System.Single,
    np.dtype(np.float64): System.Double,
    np.dtype(np.int8)   : System.SByte,
    np.dtype(np.int16)  : System.Int16,
    np.dtype(np.int32)  : System.Int32,
    np.dtype(np.int64)  : System.Int64,
    np.dtype(np.uint8)  : System.Byte,
    np.dtype(np.uint16) : System.UInt16,
    np.dtype(np.uint32) : System.UInt32,
    np.dtype(np.uint64) : System.UInt64,
    np.dtype(np.bool)   : System.Boolean,
}
_MAP_NET_NP = {
    'Single' : np.dtype(np.float32),
    'Double' : np.dtype(np.float64),
    'SByte'  : np.dtype(np.int8),
    'Int16'  : np.dtype(np.int16), 
    'Int32'  : np.dtype(np.int32),
    'Int64'  : np.dtype(np.int64),
    'Byte'   : np.dtype(np.uint8),
    'UInt16' : np.dtype(np.uint16),
    'UInt32' : np.dtype(np.uint32),
    'UInt64' : np.dtype(np.uint64),
    'Boolean': np.dtype(np.bool),
}

def setTranslation(p, T):
    T.SetElement(0,3,p[0])
    T.SetElement(1,3,p[1])
    T.SetElement(2,3,p[2])

def setRotation(rotm, T):
    T.SetElement(0,0,rotm[0][0])
    T.SetElement(0,1,rotm[0][1])
    T.SetElement(0,2,rotm[0][2])
    T.SetElement(1,0,rotm[1][0])
    T.SetElement(1,1,rotm[1][1])
    T.SetElement(1,2,rotm[1][2])
    T.SetElement(2,0,rotm[2][0])
    T.SetElement(2,1,rotm[2][1])
    T.SetElement(2,2,rotm[2][2])

def setTransform(rotm, p, T):
    setRotation(rotm, T)
    setTranslation(p, T)

def QImageToCvMat(incomingImage):
    '''  Converts a QImage into an opencv MAT format  '''

    incomingImage = incomingImage.convertToFormat(qt.QImage.Format_RGB888)

    width = incomingImage.width()
    height = incomingImage.height()

    ptr = incomingImage.constBits()
    arr = np.array(ptr).reshape(height, width, 3)  #  Copies the data
    return arr

def asNetArray(npArray):
    """
    https://gist.github.com/robbmcleod/73ca42da5984e6d0e5b6ad28bc4a504e
    Converts a NumPy array to a .NET array. See `_MAP_NP_NET` for 
    the mapping of CLR types to Numpy ``dtype``.

    Parameters
    ----------
    npArray: numpy.ndarray
        The array to be converted

    Returns
    -------
    System.Array

    Warning
    -------
    ``complex64`` and ``complex128`` arrays are converted to ``float32``
    and ``float64`` arrays respectively with shape ``[m,n,...] -> [m,n,...,2]``

    """
    dims = npArray.shape
    dtype = npArray.dtype

    # For complex arrays, we must make a view of the array as its corresponding 
    # float type as if it's (real, imag)
    if dtype == np.complex64:
        dtype = np.dtype(np.float32)
        dims += (2,)
        npArray = npArray.view(dtype).reshape(dims)
    elif dtype == np.complex128:
        dtype = np.dtype(np.float64)
        dims += (2,)
        npArray = npArray.view(dtype).reshape(dims)

    if not npArray.flags.c_contiguous or not npArray.flags.aligned:
        npArray = np.ascontiguousarray(npArray)
    assert npArray.flags.c_contiguous

    try:
        netArray = Array.CreateInstance(_MAP_NP_NET[dtype], *dims)
    except KeyError:
        raise NotImplementedError(f'asNetArray does not yet support dtype {dtype}')

    try: # Memmove 
        destHandle = GCHandle.Alloc(netArray, GCHandleType.Pinned)
        sourcePtr = npArray.__array_interface__['data'][0]
        destPtr = destHandle.AddrOfPinnedObject().ToInt64()
        ctypes.memmove(destPtr, sourcePtr, npArray.nbytes)
    finally:
        if destHandle.IsAllocated: 
            destHandle.Free()
    return netArray