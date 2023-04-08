import os
from PIL import Image
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
import json

#code from Segment_anything
folder_path = '/path/to/folder' #
sam_checkpoint = "sam_vit_h_4b8939.pth" #
model_type = "vit_h"
device = "cuda"
# Load the segmentation model
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# create a folder to store segmented data
output_folder = os.path.join(folder_path, 'segmented_images')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through all files in the folder
for filename in os.listdir(folder_path):
        data = np.fromfile(os.path.join(folder_path, filename),dtype=np.uint8)
        data = data.reshape((480,576,3))# reshape
        data = np.flip(data, axis = 0)
        pred_data = predictor.set_image(data)
        # store all pred_data according to the filename in a folder
        json_data = json.dumps(pred_data.__dict__)
        json_file = os.path.join(output_folder, "segmented_" + str(filename) + ".json")
        with open(json_file, 'w') as f:
            f.write(json_data)



print("All images have been processed.")