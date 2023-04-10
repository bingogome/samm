from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import yaml, cv2, os, pickle, zmq, json

class sam_server():

    def __init__(self):

        # create a workspace
        workspace = os.path.dirname(os.path.abspath(__file__))
        workspace = os.path.join(workspace, 'samm-workspace')
        if not os.path.exists(workspace):
            os.makedirs(workspace)
        self.workspace = workspace

        # check if model exists
        self.sam_checkpoint = self.workspace + "/sam_vit_h_4b8939.pth" 
        if not os.path.isfile(self.sam_checkpoint):
            raise Exception("SAM model file is not in " + self.sam_checkpoint)
        
        # Load the segmentation model
        self.model_type = "vit_h"
        self.device = "cuda"
        sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        sam.to(device=self.device)
        self.predictor = SamPredictor(sam)

        # temp files, just for initialization, will be overwritten
        self.imgsize_path = self.workspace + "/imgsize"
        if not os.path.isfile(self.imgsize_path):
            f = open(self.workspace + "/imgsize", "w+")
            f.write("IMAGE_WIDTH: " + str(240) + "\n" + "IMAGE_HEIGHT: " + str(352) + "\n" )
            f.close()
        if not os.path.isfile(self.workspace + "/imgsize_input_size"):
            f = open(self.workspace + "/imgsize_input_size", "w")
            f.write("INPUT_WIDTH: " + str(698) + "\n" \
                + "INPUT_HEIGHT: " + str(1024) + "\n" )
            f.close()
        if not os.path.isfile(self.workspace + "/imgsize_original_size"):
            f = open(self.workspace + "/imgsize_original_size", "w")
            f.write("ORIGINAL_WIDTH: " + str(240) + "\n" \
                + "ORIGINAL_HEIGHT: " + str(352) + "\n" )
            f.close()
        if not os.path.isfile(self.workspace + "/config.yaml"):
            with open(self.workspace + "/config.yaml", 'w') as fp:
                pass

        # initialize some parameters for testing (assumes the embeddings are saved)
        self.predictor.is_image_set = True
        with open(self.workspace+"/imgsize_input_size", 'r') as file:
            yaml_file = yaml.safe_load(file)
        self.predictor.input_size = \
            (int(yaml_file["INPUT_WIDTH"]), int(yaml_file["INPUT_HEIGHT"]))
        with open(self.workspace+"/imgsize_original_size", 'r') as file:
            yaml_file = yaml.safe_load(file)
        self.predictor.original_size = \
            (int(yaml_file["ORIGINAL_WIDTH"]), int(yaml_file["ORIGINAL_HEIGHT"]))
        
        masks = np.full(self.predictor.original_size, False)
        memmap = np.memmap(self.workspace + '/mask.memmap', dtype='bool', mode='w+', shape=masks.shape)
        memmap[:] = masks[:]
        memmap.flush()

        # create a folder to store slices
        self.slices_folder_path = os.path.join(self.workspace, 'slices')
        if not os.path.exists(self.slices_folder_path):
            os.makedirs(self.slices_folder_path)

    def computeEmbedding(self):

        # create a folder to store segmented data
        output_folder = os.path.join(self.workspace, 'segmented_images')
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # read size of images
        with open(self.imgsize_path, 'r') as file:
            yaml_file = yaml.safe_load(file)
        image_width = int(yaml_file["IMAGE_WIDTH"])
        image_height = int(yaml_file["IMAGE_HEIGHT"])

        # Loop through all files in the folder
        for filename in os.listdir(self.slices_folder_path):

            data = np.memmap(os.path.join(self.slices_folder_path, filename), \
                dtype='float64', mode='r+') 
            data = data.reshape((image_width,image_height,1))# reshape
            data = 255 * data / data.max()
            data = cv2.cvtColor(data.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            self.predictor.set_image(data)
            
            # store all pred_data according to the filename in a folder
            pkl_file = os.path.join(output_folder, "segmented_" + str(filename) + ".pkl")
            with open(pkl_file, 'wb') as f:
                pickle.dump(self.predictor.features, f)
                print(f'Predictor successfully saved to "{f}"')

        f = open(self.workspace + "/imgsize_input_size", "w")
        f.write("INPUT_WIDTH: " + str(self.predictor.input_size[0]) + "\n" \
            + "INPUT_HEIGHT: " + str(self.predictor.input_size[1]) + "\n" )
        f.close()
        f = open(self.workspace + "/imgsize_original_size", "w")
        f.write("ORIGINAL_WIDTH: " + str(self.predictor.original_size[0]) + "\n" \
            + "ORIGINAL_HEIGHT: " + str(self.predictor.original_size[1]) + "\n" )
        f.close()

        print("All images have been processed.")

    def load_feature(self, feature_path: str):
        self.feature_path = feature_path
        with open(self.feature_path, 'rb') as f:
            features = pickle.load(f)
        
        self.predictor.features = features
    
    def predict(self, input_point:np.ndarray, input_label:np.ndarray):
        self.input_point = input_point
        self.input_label = input_label
        self.masks, self.scores, self.logits = \
            self.predictor.predict( \
                point_coords=input_point, \
                point_labels=input_label, \
                multimask_output=True )

    def infer_image(self, input_point, input_label, image_name):
        # input_point = np.array([[200, 100]])
        # input_label = np.array([1])
        if len(input_label) != 0:
            self.load_feature(self.workspace + "/segmented_images/segmented_" + image_name + ".pkl")
            self.predict(input_point,input_label)
        else:
            self.masks = np.full(self.predictor.original_size, False)
        # self.imageshow(self.workspace + "/slices/" + image_name)
        memmap = np.memmap(self.workspace + '/mask.memmap', dtype='bool', mode='w+', shape=self.masks[0].shape)
        memmap[:] = self.masks[0][:]
        memmap.flush()

    def imageshow(self, image_path):
        with open(self.imgsize_path, 'r') as file:
            yaml_file = yaml.safe_load(file)
        image_width = int(yaml_file["IMAGE_WIDTH"])
        image_height = int(yaml_file["IMAGE_HEIGHT"])
        data = np.fromfile(image_path,dtype=np.float64)
        data = data.reshape((image_width,image_height,1))# reshape
        image = 255 * data / data.max()
        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        for i, (mask, score) in enumerate(zip(self.masks, self.scores)):
            plt.figure(figsize=(10,10))
            plt.imshow(image)
            self.show_mask(mask, plt.gca())
            self.show_points(self.input_point,self.input_label, plt.gca())
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            plt.axis('off')
            plt.show() 

    def show_mask(self, mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
    
    def show_points(self, coords, labels, ax, marker_size=375):
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
        
    def show_box(self, box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  

def main():

    print("Initializing SAM server  ... ")
    srv = sam_server()
    print("SAM server initialized ... ")
    context = zmq.Context()
    zmqsocket = context.socket(zmq.PULL)
    zmqsocket.bind("tcp://*:5555")
    zmqsocket.setsockopt(zmq.RCVTIMEO, 30)
    srv.sock_rcv = zmqsocket

    print("Starting To Wait for Messages ... ")

    while True:
        try:
            msg = json.loads(srv.sock_rcv.recv_json())
            if msg["command"] == "COMPUTE_EMBEDDING":
                srv.computeEmbedding()
            if msg["command"] == "INFER_IMAGE":
                srv.infer_image( \
                    np.array(msg["parameters"]["point"]), \
                    np.array(msg["parameters"]["label"]), \
                    msg["parameters"]["name"])
        except:
            continue
        
if __name__=="__main__":
    main()