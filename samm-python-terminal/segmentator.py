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
        self.predictor.original_size = (480, 576)
        self.predictor.input_size = (853, 1024)

    def load_feature(self, folder_path: str):
        folder_path = "/home/yl/software/mmaptest/slices/segmented_images/segmented_slc98.pkl"
        with open(folder_path, 'rb') as f:
            features = pickle.load(f)
        
        self.predictor.features = features
    
    def predict(self, input_point:np.ndarray, input_label:np.ndarray):
        self.masks, self.scores, self.logits = self.predictor.predict(
                                                point_coords=input_point,
                                                point_labels=input_label,
                                                multimask_output=True)
    def imageshow(self,config_path,folder_path):
        with open(config_path, 'r') as file:
            prime_service = yaml.safe_load(file)

        image_width = int(prime_service["IMAGE_WIDTH"])
        image_height = int(prime_service["IMAGE_HEIGHT"])
        

def main():
    seg_obj_ = segmentator("sam_vit_h_4b8939.pth")




if __name__=="__main__":
    main()