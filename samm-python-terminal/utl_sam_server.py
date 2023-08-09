from utl_sam_msg import *
import numpy as np
from tqdm import tqdm
import sys,os, cv2, matplotlib.pyplot as plt
import torch, functools, pickle
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"#

from segment_anything import sam_model_registry as sam_model_registry_sam
from segment_anything import SamPredictor as SamPredictor_sam
from mobile_sam import sam_model_registry as sam_model_registry_mobile
from mobile_sam import SamPredictor as SamPredictor_mobile

from utl_latencylogger import latency_logger

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
        self.samPredictor = {"R": None, "G": None, "Y": None}
        self.initNetwork()
    
    def initNetwork(self, model = "vit_b"):

        # Load the segmentation model
        if torch.cuda.is_available():
            self.device = "cuda"
            print("[SAMM INFO] CUDA detected. Waiting for Model ...")
        if torch.backends.mps.is_available():
            self.device = "mps"
            print("[SAMM INFO] MPS detected. Waiting for Model ...")

        workspace = os.path.dirname(os.path.abspath(__file__))
        workspace = os.path.join(workspace, 'samm-workspace')
        if not os.path.exists(workspace):
            os.makedirs(workspace)
        self.workspace = workspace

        if model.startswith('vit_'):
            self.initNetworkSam(model)
        if model.startswith('mobile_'):
            self.initNetworkMobile(model)
        if model.startswith('medsam_'):
            self.initNetworkMedSam(model)

    def initNetworkSam(self, model):
        dictpath = {
            "vit_h" : "sam_vit_h_4b8939.pth",
            "vit_l" : "sam_vit_l_0b3195.pth",
            "vit_b" : "sam_vit_b_01ec64.pth"
        }
        self.sam_checkpoint = self.workspace + "/" +  dictpath[model]
        if not os.path.isfile(self.sam_checkpoint):
            raise Exception("[SAMM ERROR] SAM model file is not in " + self.sam_checkpoint)
        model_type = model
        sam = sam_model_registry_sam[model_type](checkpoint=self.sam_checkpoint)
        sam.to(device=self.device)

        self.samPredictor["R"] = SamPredictor_sam(sam)
        self.samPredictor["G"] = SamPredictor_sam(sam)
        self.samPredictor["Y"] = SamPredictor_sam(sam)
        print(f'[SAMM INFO] Model initialzed to: "{model}"')
    
    def initNetworkMobile(self, model):
        dictpath = {
            "mobile_vit_t" : "mobile_sam.pt"
        }
        self.sam_checkpoint = self.workspace + "/" +  dictpath[model]
        if not os.path.isfile(self.sam_checkpoint):
            raise Exception("[SAMM ERROR] SAM model file is not in " + self.sam_checkpoint)
        model_type = model[7:]
        sam = sam_model_registry_mobile[model_type](checkpoint=self.sam_checkpoint)
        sam.to(device=self.device)

        self.samPredictor["R"] = SamPredictor_mobile(sam)
        self.samPredictor["G"] = SamPredictor_mobile(sam)
        self.samPredictor["Y"] = SamPredictor_mobile(sam)
        print(f'[SAMM INFO] Model initialzed to: "{model}"')

    def initNetworkMedSam(self, model):
        dictpath = {
            "medsam_vit_b" : "medsam_vit_b.pth"
        }
        self.sam_checkpoint = self.workspace + "/" +  dictpath[model]
        if not os.path.isfile(self.sam_checkpoint):
            raise Exception("[SAMM ERROR] SAM model file is not in " + self.sam_checkpoint)
        model_type = model[7:]
        sam = sam_model_registry_sam[model_type](checkpoint=self.sam_checkpoint)
        sam.to(device=self.device)

        self.samPredictor["R"] = SamPredictor_sam(sam)
        self.samPredictor["G"] = SamPredictor_sam(sam)
        self.samPredictor["Y"] = SamPredictor_sam(sam)
        print(f'[SAMM INFO] Model initialzed to: "{model}"')

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
    
def CalculateEmbeddings(msg):
    dataNode = SammParameterNode()
    laglog = latency_logger(dataNode.workspace)

    dataNode.features = {
        "R": [None for i in range(dataNode.N["R"])], 
        "G": [None for i in range(dataNode.N["G"])], 
        "Y": [None for i in range(dataNode.N["Y"])]
    }

    laglog.event_start_computeembedding()
    
    output_folder = os.path.join(dataNode.workspace, 'emb')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    pkl_file = os.path.join(output_folder, "emb" + ".pkl")

    if msg["loadLocal"]:
        dataNode.samPredictor["R"].input_size = (dataNode.imageSize[1], dataNode.imageSize[2])
        dataNode.samPredictor["R"].original_size = (dataNode.imageSize[1], dataNode.imageSize[2])
        dataNode.samPredictor["G"].input_size = (dataNode.imageSize[0], dataNode.imageSize[2])
        dataNode.samPredictor["G"].original_size = (dataNode.imageSize[0], dataNode.imageSize[2])
        dataNode.samPredictor["Y"].input_size = (dataNode.imageSize[0], dataNode.imageSize[1])
        dataNode.samPredictor["Y"].original_size = (dataNode.imageSize[0], dataNode.imageSize[1])
        with open(pkl_file, 'rb') as f:
            dataNode.features = pickle.load(f)
        print("[SAMM INFO] Red View Progress:")
        dataNode.samPredictor["R"].is_image_set = True
        for i in tqdm(range(dataNode.N["R"])):
            dataNode.features["R"][i] = dataNode.features["R"][i].to('cpu')
        print("[SAMM INFO] Green View Progress:")
        dataNode.samPredictor["G"].is_image_set = True
        for i in tqdm(range(dataNode.N["G"])):
            dataNode.features["G"][i] = dataNode.features["G"][i].to('cpu')
        print("[SAMM INFO] Yellow View Progress:")
        dataNode.samPredictor["Y"].is_image_set = True
        for i in tqdm(range(dataNode.N["Y"])):
            dataNode.features["Y"][i] = dataNode.features["Y"][i].to('cpu')
        print("[SAMM INFO] Loaded Local Embeddings.")
    else:
        print("[SAMM INFO] Red View Progress:")
        for i in tqdm(range(dataNode.N["R"])):
            dataNode.samPredictor["R"].set_image(cv2.cvtColor(dataNode.mainVolume[i,:,:],cv2.COLOR_GRAY2RGB))
            if i == 127:
                testImage(dataNode, 127, np.array([[50,50]]), "R")
            dataNode.features["R"][i] = dataNode.samPredictor["R"].features.to('cpu')
        print("[SAMM INFO] Green View Progress:")
        for i in tqdm(range(dataNode.N["G"])):
            dataNode.samPredictor["G"].set_image(cv2.cvtColor(dataNode.mainVolume[:,i,:],cv2.COLOR_GRAY2RGB))
            dataNode.features["G"][i] = dataNode.samPredictor["G"].features.to('cpu')
        print("[SAMM INFO] Yellow View Progress:")
        for i in tqdm(range(dataNode.N["Y"])):
            dataNode.samPredictor["Y"].set_image(cv2.cvtColor(dataNode.mainVolume[:,:,i],cv2.COLOR_GRAY2RGB))
            dataNode.features["Y"][i] = dataNode.samPredictor["Y"].features.to('cpu')

        if msg["saveToLocal"]:
            with open(pkl_file, 'wb') as f:
                pickle.dump(dataNode.features, f)
                print(f'[SAMM INFO] Predictor successfully saved to "{f}"')

    print("[SAMM INFO] Embeddings Cached.")
    laglog.event_complete_computeembedding()

def testImage(dataNode, n, points, view):
    from PIL import Image
    img = dataNode.mainVolume[n,:,:]
    for i in range(5):
        for j in range(5):
            img[points[0][0]+i][points[0][1]+j] = 0
            img[points[0][0]+i][points[0][1]-j] = 0
            img[points[0][0]-i][points[0][1]-j] = 0
            img[points[0][0]-i][points[0][1]+j] = 0
    img = Image.fromarray(img)
    img.save("testimg.png")

    seg, _, _ = dataNode.samPredictor[view].predict(
            point_coords = points,
            point_labels = [1],
            multimask_output = True)
    img = Image.fromarray(seg[0])
    img.save("testseg.png")
    
def helperPredict(dataNode, msg, points, labels, bbox2d):

    dataNode.samPredictor[msg["view"]].features = dataNode.features[msg["view"]][msg["n"]].to("cuda")
    if isinstance(bbox2d, (np.ndarray, np.generic)):
        seg, _, _ = dataNode.samPredictor[msg["view"]].predict(
            point_coords = points,
            point_labels = labels,
            box = bbox2d,
            multimask_output = True)
    else:
        seg, _, _ = dataNode.samPredictor[msg["view"]].predict(
            point_coords = points,
            point_labels = labels,
            multimask_output = True)
    seg = seg[0]
    return seg

def sammProcessingCallBack_INFERENCE(msg):
    dataNode = SammParameterNode()
    laglog = latency_logger(dataNode.workspace)

    laglog.event_receive_inferencerequest()
    positivePoints = msg["positivePrompts"]
    negativePoints = msg["negativePrompts"]
    bbox2d = msg["bbox2D"]

    points = []
    labels = []
    if positivePoints is not None:
        for i in range(positivePoints.shape[0]):
            points.append([positivePoints[i,0], positivePoints[i,1]])
            labels.append(1)

    if negativePoints is not None:
        for i in range(negativePoints.shape[0]):
            points.append([negativePoints[i,0], negativePoints[i,1]])
            labels.append(0)

    seg = None
    if len(points) > 0 and (bbox2d[0]!=-404):

        points = np.array(points)
        point_labels = np.array(labels)
        bbox2d = np.array(bbox2d)
        bbox2d[bbox2d<1] = 1
        seg = helperPredict(dataNode, msg, points, point_labels, bbox2d)
    
    elif len(points) == 0 and (bbox2d[0]!=-404):
        
        points = None
        point_labels = None
        bbox2d = np.array(bbox2d)
        bbox2d[bbox2d<1] = 1
        seg = helperPredict(dataNode, msg, points, point_labels, bbox2d)

    elif len(points) > 0 and (bbox2d[0]==-404):
        
        points = np.array(points)
        point_labels = np.array(labels)
        bbox2d = None
        seg = helperPredict(dataNode, msg, points, point_labels, bbox2d)
        
    else:
        if msg["view"] == "R":
            seg = np.zeros([dataNode.imageSize[1], dataNode.imageSize[2]],dtype=np.uint8)
        if msg["view"] == "G":
            seg = np.zeros([dataNode.imageSize[0], dataNode.imageSize[2]],dtype=np.uint8)
        if msg["view"] == "Y":
            seg = np.zeros([dataNode.imageSize[0], dataNode.imageSize[1]],dtype=np.uint8)

    laglog.event_complete_inference()

    return seg[:].astype(np.uint8).tobytes(), None

def sammProcessingCallBack_CALCULATE_EMBEDDINGS(msg):
    print("[SAMM INFO] Received Embeddings Request.")
    return np.array([1],dtype=np.uint8).tobytes(), functools.partial(CalculateEmbeddings, msg)

def SwitchModel(msg):
    dataNode = SammParameterNode()
    dataNode.initNetwork(msg["model"])

def sammProcessingCallBack_MODEL_SELECTION(msg):
    print(f'[SAMM INFO] Model switched to: "{msg["model"]}"')
    return np.array([1],dtype = np.uint8).tobytes(), functools.partial(SwitchModel, msg) 

callBackList = {
    SammMsgType.SET_IMAGE_SIZE : sammProcessingCallBack_SET_IMAGE_SIZE,
    SammMsgType.SET_NTH_IMAGE : sammProcessingCallBack_SET_NTH_IMAGE,
    SammMsgType.INFERENCE : sammProcessingCallBack_INFERENCE,
    SammMsgType.CALCULATE_EMBEDDINGS : sammProcessingCallBack_CALCULATE_EMBEDDINGS,
    SammMsgType.MODEL_SELECTION : sammProcessingCallBack_MODEL_SELECTION
}

def sammProcessingCallBack(cmd, msg):
    cmdType = SammMsgType(np.frombuffer(cmd,dtype="int32").reshape([1])[0])
    solverType = SammMsgSolverMapper[cmdType]
    msgDeserialized = solverType.getDecodedData(msg)
    msgBack, lateUpdate = callBackList[cmdType](msgDeserialized)
    return msgBack, lateUpdate
