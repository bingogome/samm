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

from datetime import datetime
import pickle

class latency_logger():

    def __init__(self, workspace):

        # Latency logging
        # log latency?
        self.workspace              = workspace
        self.flag_loglat            = False
        if self.flag_loglat:
            now = datetime.now()
            self.logctrmax          = 300
            self.timearr_RCV_INF    = [now for idx in range(self.logctrmax)]
            self.timearr_CPL_INF    = [now for idx in range(self.logctrmax)]
            self.timearr_EMB        = [now, now]
            self.ctr_RCV_INF        = 0
            self.ctr_CPL_INF        = 0
    
    def event_start_computeembedding(self):
        if self.flag_loglat:
            self.timearr_EMB[0] = datetime.now()

    def event_complete_computeembedding(self):
        if self.flag_loglat:
            self.timearr_EMB[1] = datetime.now()
            file_name = self.workspace + "timearr_EMB.pkl"
            with open(file_name, 'wb') as file:
                pickle.dump(self.timearr_EMB, file)
                print("[SAMM INFO] Time for embedding computing is saved.")

    def event_receive_inferencerequest(self):
        if self.flag_loglat:
            self.timearr_RCV_INF[self.ctr_RCV_INF] = datetime.now()
            self.ctr_RCV_INF = self.ctr_RCV_INF + 1

    def event_complete_inference(self):
        if self.flag_loglat:
            self.timearr_CPL_INF[self.ctr_CPL_INF] = datetime.now()
            self.ctr_CPL_INF = self.ctr_CPL_INF + 1
            if self.ctr_RCV_INF >= self.logctrmax - 1 or self.ctr_CPL_INF >= self.logctrmax - 1:
                file_name = self.workspace + "/timearr_RCV_INF.pkl"
                with open(file_name, 'wb') as file:
                    pickle.dump(self.timearr_RCV_INF, file)
                file_name = self.workspace + "/timearr_CPL_INF.pkl"
                with open(file_name, 'wb') as file:
                    pickle.dump(self.timearr_CPL_INF, file)
                print("[SAMM INFO] Time for inference is saved.")
                