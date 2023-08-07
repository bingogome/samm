from utl_sam_msg import *
import numpy as np
from tqdm import tqdm
import sys,os, cv2, matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "0"#
from segment_anything import sam_model_registry, SamPredictor

def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance

@singleton
class SammParameterNode:
    def __init__(self):
        ## properties
        self.mainVolume = []
        self.N = {"R": 0, "G": 0, "Y": 0}
        self.imageSize = []

        ## features
        self.features = {"R": [], "G": [], "Y": []}
        
        ## pred
        self.samPredictor = None
        self.initNetWork()
    
    def initNetWork(self):
        workspace = os.path.dirname(os.path.abspath(__file__))
        workspace = os.path.join(workspace, 'samm-workspace')
        if not os.path.exists(workspace):
            os.makedirs(workspace)
        self.workspace = workspace
        self.sam_checkpoint = self.workspace + "/sam_vit_h_4b8939.pth" 
        if not os.path.isfile(self.sam_checkpoint):
            raise Exception("[SAMM ERROR] SAM model file is not in " + self.sam_checkpoint)
        model_type = "vit_h"
        sam = sam_model_registry[model_type](checkpoint=self.sam_checkpoint)
        sam.to(device="cuda")
        self.samPredictor = SamPredictor(sam)

def sammProcessingCallBack_SET_IMAGE_SIZE(msg):
    dataNode = SammParameterNode()
    dataNode.mainVolume = np.zeros([msg["r"], msg["g"], msg["y"]], dtype = np.uint8)
    dataNode.N = {"R": msg["r"], "G": msg["g"], "Y": msg["y"]}
    dataNode.imageSize = [msg["r"], msg["g"], msg["y"]]
    return np.array([1],dtype=np.uint8).tobytes(), None

def sammProcessingCallBack_SET_NTH_IMAGE(msg):
    dataNode = SammParameterNode()
    dataNode.mainVolume[msg["n"],:,:] = msg["image"][:,:]
    return np.array([1],dtype = np.uint8).tobytes(), None 
    
def CalculateEmbeddings():
    dataNode = SammParameterNode()
    dataNode.features = {
        "R": [None for i in range(dataNode.N["R"])], 
        "G": [None for i in range(dataNode.N["G"])], 
        "Y": [None for i in range(dataNode.N["Y"])]
    }
    print("[SAMM INFO] Red View Progress:")
    for i in tqdm(range(dataNode.N["R"])):
        dataNode.samPredictor.set_image(cv2.cvtColor(dataNode.mainVolume[i,:,:],cv2.COLOR_GRAY2RGB))
        dataNode.features["R"][i] = dataNode.samPredictor.features.to('cpu')
    print("[SAMM INFO] Green View Progress:")
    for i in tqdm(range(dataNode.N["G"])):
        dataNode.samPredictor.set_image(cv2.cvtColor(dataNode.mainVolume[:,i,:],cv2.COLOR_GRAY2RGB))
        dataNode.features["G"][i] = dataNode.samPredictor.features.to('cpu')
    print("[SAMM INFO] Yellow View Progress:")
    for i in tqdm(range(dataNode.N["Y"])):
        dataNode.samPredictor.set_image(cv2.cvtColor(dataNode.mainVolume[:,:,i],cv2.COLOR_GRAY2RGB))
        dataNode.features["Y"][i] = dataNode.samPredictor.features.to('cpu')
    print("[SAMM INFO] Embeddings Cached.")

def sammProcessingCallBack_INFERENCE(msg):
    dataNode = SammParameterNode()
    positivePoints = msg["positivePrompts"]
    negativePoints = msg["negativePrompts"]

    points = []
    labels = []
    if positivePoints is not None:
        for i in range(positivePoints.shape[0]):
            points.append([positivePoints[i,1], positivePoints[i,0]])
            labels.append(1)

    if negativePoints is not None:
        for i in range(negativePoints.shape[0]):
            points.append([negativePoints[i,1], negativePoints[i,0]])
            labels.append(0)

    seg = None
    if  len(points) > 0:
        dataNode.samPredictor.features = dataNode.features[msg["n"]].to("cuda")
        seg, _, _ = dataNode.samPredictor.predict(
            point_coords = np.array(points),
            point_labels = np.array(labels),
            multimask_output = False,)
        
    else:
        seg = np.zeros([dataNode.imageSize[1], dataNode.imageSize[2]],dtype=np.uint8)

    return seg[:].astype(np.uint8).tobytes(), None

def sammProcessingCallBack_CALCULATE_EMBEDING(msg):
    print("[SAMM INFO] Received Embeddings Request.")
    return np.array([1],dtype=np.uint8).tobytes(), CalculateEmbeddings 

callBackList = {
    SammMsgType.SET_IMAGE_SIZE : sammProcessingCallBack_SET_IMAGE_SIZE,
    SammMsgType.SET_NTH_IMAGE : sammProcessingCallBack_SET_NTH_IMAGE,
    SammMsgType.INFERENCE : sammProcessingCallBack_INFERENCE,
    SammMsgType.CALCULATE_EMBEDING : sammProcessingCallBack_CALCULATE_EMBEDING
}

def sammProcessingCallBack(cmd, msg):
    cmdType = SammMsgType(np.frombuffer(cmd,dtype="int32").reshape([1])[0])
    solverType = SammMsgSolverMapper[cmdType]
    msgDeserializaed = solverType.getDecodedData(msg)
    msgBack, lateUpdate = callBackList[cmdType](msgDeserializaed)
    return msgBack, lateUpdate
