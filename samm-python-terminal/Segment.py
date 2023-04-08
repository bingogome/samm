from PIL import Image
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
import json, yaml, cv2, os, pickle

#code from Segment_anything
folder_path = "/home/yl/software/mmaptest/slices"#
sam_checkpoint = "/home/yl/software/segment-anything/notebooks/sam_vit_h_4b8939.pth" #
model_type = "vit_h"
device = "cuda"

# create a folder to store segmented data
output_folder = os.path.join(folder_path + "/..", 'segmented_images')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# read size of images
with open('/home/yl/software/mmaptest/config.yaml', 'r') as file:
    prime_service = yaml.safe_load(file)
image_width = int(prime_service["IMAGE_WIDTH"])
image_height = int(prime_service["IMAGE_HEIGHT"])

# Load the segmentation model
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    
    data = np.fromfile(os.path.join(folder_path, filename),dtype=np.float64)
    data = data.reshape((image_width,image_height,1))# reshape
    image = 255 * data / data.max()
    image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    predictor.set_image(image)
    print(predictor.input_size)
    print(predictor.original_size)
    
    # store all pred_data according to the filename in a folder
    pkl_file = os.path.join(output_folder, "segmented_" + str(filename) + ".pkl")
    with open(pkl_file, 'wb') as f:
        pickle.dump(predictor.features, f)
        print(f'Predictor successfully saved to "{f}"')

print("All images have been processed.")