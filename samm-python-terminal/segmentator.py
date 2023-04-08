import pickle, yaml, os, cv2
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import matplotlib.pyplot as plt

class segmentator():
    def __init__(self,sam_checkpoint_path:str) -> None:

        # self.sam_checkpoint = "/home/yl/software/segment-anything/notebooks/sam_vit_h_4b8939.pth" 
        self.sam_checkpoint_path = sam_checkpoint_path
        model_type = "vit_h"

        sam = sam_model_registry[model_type](checkpoint=self.sam_checkpoint_path)
        sam.to(device="cuda")
        self.predictor = SamPredictor(sam)
        self.predictor.is_image_set = True
        self.predictor.original_size = (240, 352)
        self.predictor.input_size = (698, 1024)

    def load_feature(self, folder_path: str):
        self.folder_path = folder_path
        with open(self.folder_path, 'rb') as f:
            features = pickle.load(f)
        
        self.predictor.features = features
    
    def predict(self, input_point:np.ndarray, input_label:np.ndarray):
        self.input_point = input_point
        self.input_label = input_label
        self.masks, self.scores, self.logits = self.predictor.predict(
                                                point_coords=input_point,
                                                point_labels=input_label,
                                                multimask_output=True)
        
    def imageshow(self,config_path,image_path):
        with open(config_path, 'r') as file:
            prime_service = yaml.safe_load(file)
        image_width = int(prime_service["IMAGE_WIDTH"])
        image_height = int(prime_service["IMAGE_HEIGHT"])
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
        print(type(mask))
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
    seg_obj_ = segmentator("/home/yl/software/segment-anything/notebooks/sam_vit_h_4b8939.pth")
    folder_path = "/home/yl/software/mmaptest/segmented_images/segmented_slc50.pkl"
    config_path= "/home/yl/software/mmaptest/config.yaml"
    image_path = "/home/yl/software/mmaptest/slices/slc50"
    seg_obj_.load_feature(folder_path)
    input_point = np.array([[200, 100]])
    input_label = np.array([1])
    seg_obj_.predict(input_point,input_label)
    seg_obj_.imageshow(config_path, image_path)

if __name__=="__main__":
    main()