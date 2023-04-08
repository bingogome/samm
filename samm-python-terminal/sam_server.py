from PIL import Image
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
import json, yaml, cv2, os, pickle

class sam_server():

    def __init__(self):
        # code from Segment_anything
        self.folder_path = "/home/yl/software/mmaptest/slices"#
        self.sam_checkpoint = "/home/yl/software/segment-anything/notebooks/sam_vit_h_4b8939.pth" #
        self.model_type = "vit_h"
        self.device = "cuda"

        # Load the segmentation model
        sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        sam.to(device=self.device)
        self.predictor = SamPredictor(sam)

    def computeEmbedding(self):

        # create a folder to store segmented data
        output_folder = os.path.join(self.folder_path, 'segmented_images')
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # read size of images
        with open('/home/yl/software/mmaptest/config.yaml', 'r') as file:
            prime_service = yaml.safe_load(file)
        image_width = int(prime_service["IMAGE_WIDTH"])
        image_height = int(prime_service["IMAGE_HEIGHT"])

        # Loop through all files in the folder
        for filename in os.listdir(self.folder_path):

            data = np.fromfile(os.path.join(self.folder_path, filename),dtype=np.uint8)
            data = data.reshape((image_width,image_height,3))# reshape
            data = np.flip(data, axis = 0)
            image = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)

            self.predictor.set_image(image)
            print(self.predictor.input_size)
            print(self.predictor.original_size)
            
            # store all pred_data according to the filename in a folder
            pkl_file = os.path.join(output_folder, "segmented_" + str(filename) + ".pkl")
            with open(pkl_file, 'wb') as f:
                pickle.dump(self.predictor.features, f)
                print(f'Predictor successfully saved to "{f}"')

        print("All images have been processed.")

def main():
    srv = sam_server()
    while True:
        

if __name__=="__main__":
    main()