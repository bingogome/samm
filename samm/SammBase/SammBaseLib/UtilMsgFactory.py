import zmq
from enum import Enum
import numpy as np

class SammMsgType(Enum):
    SET_IMAGE_SIZE = 0
    SET_NTH_IMAGE = 1
    CALCULATE_EMBEDDINGS = 2
    INFERENCE = 3
    MODEL_SELECTION = 4

class SammMsgTypeCommandTemplate:
    def __init__(self, msg):
        self.msg = msg

    def getEncodedData(self):
        return b''
    
    @staticmethod
    def getDecodedData(msg):
        return {}

'''
x : int 32
y
z
'''

class SammMsgType_SET_IMAGE_SIZE(SammMsgTypeCommandTemplate):
    def getEncodedData(self):
        x = self.msg["r"]
        y = self.msg["g"]
        z = self.msg["y"]
        msg = np.array([x], dtype='int32').tobytes() + \
              np.array([y], dtype='int32').tobytes() + \
              np.array([z], dtype='int32').tobytes()
        
        return msg
    
    @staticmethod
    def getDecodedData(msgbyte):
        msgDecode = {
            "r" : np.frombuffer(msgbyte[0:4], dtype="int32").reshape([1])[0],
            "g" : np.frombuffer(msgbyte[4:8], dtype="int32").reshape([1])[0],
            "y" : np.frombuffer(msgbyte[8:12], dtype="int32").reshape([1])[0]
        }

        return msgDecode
    

'''
n : int 32
N : 3 int 32
image : uint8
'''

class SammMsgType_SET_NTH_IMAGE(SammMsgTypeCommandTemplate):
    def getEncodedData(self):
        n = self.msg["n"]
        N = self.msg["N"]
        image = self.msg["image"].astype(np.uint8)
        msg = np.array([n], dtype='int32').tobytes() + \
              np.array([N["R"]], dtype='int32').tobytes() + \
              np.array([N["G"]], dtype='int32').tobytes() + \
              np.array([N["Y"]], dtype='int32').tobytes() + \
              np.array([image.shape[0]], dtype='int32').tobytes() + \
              np.array([image.shape[1]], dtype='int32').tobytes() + \
              image.tobytes()
        
        return msg
    
    @staticmethod
    def getDecodedData(msgbyte):
        n = np.frombuffer(msgbyte[0:4], dtype="int32").reshape([1])[0]
        N = {"R" : [], "G" : [], "Y" : []}
        N["R"] = np.frombuffer(msgbyte[4:8], dtype="int32").reshape([1])[0]
        N["G"] = np.frombuffer(msgbyte[8:12], dtype="int32").reshape([1])[0]
        N["Y"] = np.frombuffer(msgbyte[12:16], dtype="int32").reshape([1])[0]
        imageH = np.frombuffer(msgbyte[16:20], dtype="int32").reshape([1])[0]
        imageW = np.frombuffer(msgbyte[20:24], dtype="int32").reshape([1])[0]
        image = np.frombuffer(msgbyte[24:], dtype=np.uint8).reshape(imageH, imageW)

        msgDecode = {
            "n" : n,
            "N" : N,
            "image" : image
        }

        return msgDecode

'''
'''

class SammMsgType_CALCULATE_EMBEDDINGS(SammMsgTypeCommandTemplate):
    def getEncodedData(self):
        saveToLocal = self.msg["saveToLocal"]
        loadLocal = self.msg["loadLocal"]
        msg = b''
        msg += np.array([saveToLocal], dtype='int32').tobytes()
        msg += np.array([loadLocal], dtype='int32').tobytes()
        return msg

    @staticmethod
    def getDecodedData(msgbyte):
        msg = {}
        msg["saveToLocal"] = np.frombuffer(msgbyte[0:4], dtype="int32").reshape([1])[0]
        msg["loadLocal"] = np.frombuffer(msgbyte[4:8], dtype="int32").reshape([1])[0]
        return msg

'''
n : int
positivePrompts, int 32, n * 2
negativePrompts, int 32
'''

class SammMsgType_INFERENCE(SammMsgTypeCommandTemplate):
    def getEncodedData(self):
        
        n = self.msg["n"]
        view = SammViewMapper[self.msg["view"]]
        positivePoints = self.msg["positivePrompts"]
        negativePoints = self.msg["negativePrompts"]
        
        msg = b''
        msg += np.array([n], dtype='int32').tobytes()
        msg += np.array([view], dtype='int32').tobytes()

        if positivePoints is not None and positivePoints.shape[0] > 0:
            msg += np.array([positivePoints.shape[0]], dtype='int32').tobytes()
            msg += positivePoints.astype("int32").tobytes()
        else:
            msg += np.array([0], dtype='int32').tobytes()
        
        if negativePoints is not None and negativePoints.shape[0] > 0:
            msg += np.array([negativePoints.shape[0]], dtype='int32').tobytes()
            msg += negativePoints.astype("int32").tobytes()
        else:
            msg += np.array([0], dtype='int32').tobytes()

        return msg
    
    @staticmethod
    def getDecodedData(msgbyte):
        msg = {}
        pt = 0

        msg["n"] = np.frombuffer(msgbyte[0:4], dtype="int32").reshape([1])[0]
        pt += 4

        msg["view"] = SammViewMapper["DICT"][np.frombuffer(msgbyte[pt:pt+4], dtype="int32").reshape([1])[0]]
        pt += 4

        positivePromptNum = np.frombuffer(msgbyte[pt:pt+4], dtype="int32").reshape([1])[0]
        pt += 4

        if positivePromptNum > 0:
            positivePrompt = np.frombuffer(msgbyte[pt:pt+4*2*positivePromptNum], dtype="int32").reshape([positivePromptNum, 2])
            pt += 4*2*positivePromptNum
            msg["positivePrompts"] = positivePrompt
        else:
            msg["positivePrompts"] = None

        negativePromptNum = np.frombuffer(msgbyte[pt:pt+4], dtype="int32").reshape([1])[0]
        pt += 4

        if negativePromptNum > 0:
            negativePrompt = np.frombuffer(msgbyte[pt:pt+4*2*negativePromptNum], dtype="int32").reshape([positivePromptNum, 2])
            pt += 4 * 2 * negativePromptNum
            msg["negativePrompts"] = negativePrompt
        else:
            msg["negativePrompts"] = None

        return msg

class SammMsgType_MODEL_SELECTION(SammMsgTypeCommandTemplate):
    def getEncodedData(self):
        model = SammModelMapper[self.msg["model"]]
        msg = np.array([model], dtype='int32').tobytes()
        
        return msg
    
    @staticmethod
    def getDecodedData(msgbyte):
        msgDecode = {
            "model" : SammModelMapper["DICT"][np.frombuffer(msgbyte[0:4], dtype="int32").reshape([1])[0]]
        }

        return msgDecode

SammMsgSolverMapper = {
    SammMsgType.SET_IMAGE_SIZE : SammMsgType_SET_IMAGE_SIZE,
    SammMsgType.SET_NTH_IMAGE : SammMsgType_SET_NTH_IMAGE,
    SammMsgType.CALCULATE_EMBEDDINGS : SammMsgType_CALCULATE_EMBEDDINGS,
    SammMsgType.INFERENCE : SammMsgType_INFERENCE,
    SammMsgType.MODEL_SELECTION : SammMsgType_MODEL_SELECTION
}

SammViewMapper = {
    "R" : 0,
    "G" : 1,
    "Y" : 2,
    "DICT" : "RGY"
}

SammModelMapper = {
    "vit_b" : 0,
    "vit_l" : 1,
    "vit_h" : 2,
    "DICT" : ["vit_b", "vit_l", "vit_h"]
}