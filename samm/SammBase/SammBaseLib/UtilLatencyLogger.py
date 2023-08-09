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

class LatencyLogger:

    def __init__(self):
        # Latency logging
        # log latency?
        self.flag_loglat            = True
        if self.flag_loglat:
            now                     = datetime.now()
            self.logctrmax          = 300
            self.timearr_SND_INF    = [now for idx in range(self.logctrmax)]
            self.timearr_RCV_MSK    = [now for idx in range(self.logctrmax)]
            self.timearr_APL_MSK    = [now for idx in range(self.logctrmax)]
            self.ctr_SND_INF        = 0
            self.ctr_RCV_MSK        = 0
            self.ctr_APL_MSK        = 0

    def event_receive_mask(self):
        if self.flag_loglat:
            self.timearr_RCV_MSK[self.ctr_RCV_MSK] = datetime.now()
            self.ctr_RCV_MSK = self.ctr_RCV_MSK + 1
            if self.ctr_RCV_MSK >= self.logctrmax - 1:
                self.processSaveLatencyLog()
                self.flag_loglat = False

    def event_apply_mask(self):
        if self.flag_loglat:
            self.timearr_APL_MSK[self.ctr_APL_MSK] = datetime.now()
            self.ctr_APL_MSK = self.ctr_APL_MSK + 1
            if self.ctr_APL_MSK >= self.logctrmax - 1:
                self.processSaveLatencyLog()
                self.flag_loglat = False

    def event_send_inferencerequest(self):
        if self.flag_loglat:
            self.timearr_SND_INF[self.ctr_SND_INF] = datetime.now()
            self.ctr_SND_INF = self.ctr_SND_INF + 1
            if self.ctr_SND_INF >= self.logctrmax - 1:
                self.processSaveLatencyLog()
                self.flag_loglat = False

    def processSaveLatencyLog(self):

        file_name = self.workspace + "/timearr_SND_INF.pkl"
        with open(file_name, 'wb') as file:
            pickle.dump(self.timearr_SND_INF, file)

        file_name = self.workspace + "/timearr_RCV_MSK.pkl"
        with open(file_name, 'wb') as file:
            pickle.dump(self.timearr_RCV_MSK, file)

        file_name = self.workspace + "/timearr_APL_MSK.pkl"
        with open(file_name, 'wb') as file:
            pickle.dump(self.timearr_APL_MSK, file)

        print("[SAMM INFO] Time for inference is saved.")