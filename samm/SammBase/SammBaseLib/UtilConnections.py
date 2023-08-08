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

import zmq
import numpy as np
from SammBaseLib.UtilMsgFactory import *

class UtilConnections():

    def __init__(self, ip, portControl):
        self.ip = ip
        self.portControl = portControl
        self.context = zmq.Context()

    def pushRequest(self, requireType, MSG):
        commandByte = np.array([requireType.value], dtype='int32').tobytes()
        msgByte = SammMsgSolverMapper[requireType](MSG).getEncodedData()
        sock = self.context.socket(zmq.REQ)
        # if no receiption, try extending the wait time. The first setup time takes longer
        sock.setsockopt(zmq.RCVTIMEO, 10000) 
        sock.connect("tcp://%s:%s" % (self.ip, self.portControl))
        sock.send_multipart([commandByte, msgByte])
        feedback = sock.recv()
        sock.close()
        return feedback

    def clear(self):
        pass
