import pickle, yaml, os, cv2
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import matplotlib.pyplot as plt

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))   

sam_checkpoint = "/home/yl/software/segment-anything/notebooks/sam_vit_h_4b8939.pth" #
model_type = "vit_h"
device = "cuda"

folder_path = "/home/yl/software/mmaptest/slices/segmented_images/segmented_slc96.pkl"
with open(folder_path, 'rb') as f:
    features = pickle.load(f)

# Load the segmentation model
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)
input_point = np.array([[360, 200]])
input_label = np.array([1])

predictor.is_image_set = True
predictor.original_size = (480, 576)
predictor.input_size = (853, 1024)

predictor.features = features
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

with open('/home/yl/software/mmaptest/config.yaml', 'r') as file:
    prime_service = yaml.safe_load(file)
image_width = int(prime_service["IMAGE_WIDTH"])
image_height = int(prime_service["IMAGE_HEIGHT"])
data = np.fromfile("/home/yl/software/mmaptest/slices/slc96",dtype=np.uint8)
data = data.reshape((image_width,image_height,3))# reshape
data = np.flip(data, axis = 0)
image = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)

print(1)
for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_mask(mask, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.show()  

folder_path = "/home/yl/software/mmaptest/slices/segmented_images/segmented_slc4.pkl"
with open(folder_path, 'rb') as f:
    features = pickle.load(f)

predictor.features = features
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

data = np.fromfile("/home/yl/software/mmaptest/slices/slc4",dtype=np.uint8)
data = data.reshape((image_width,image_height,3))# reshape
data = np.flip(data, axis = 0)
image = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)

for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_mask(mask, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.show()  

print(2)

folder_path = "/home/yl/software/mmaptest/slices/segmented_images/segmented_slc98.pkl"
with open(folder_path, 'rb') as f:
    features = pickle.load(f)

predictor.features = features
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

data = np.fromfile("/home/yl/software/mmaptest/slices/slc98",dtype=np.uint8)
data = data.reshape((image_width,image_height,3))# reshape
data = np.flip(data, axis = 0)
image = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)

for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_mask(mask, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.show()  

print(3)
