# pred.py on a single selected picture

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import yaml
import os
import argparse
import random
import matplotlib.pyplot as plt
from skimage import segmentation
import matplotlib.cm as cm
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

from scripts.dataset import SegmentationDataset
from scripts.model import CustomUNet, UNet
from scripts.metric import compute_miou

from utils.select_model import select_model

parser = argparse.ArgumentParser(description='SegNet Single Pred.')
parser.add_argument('--pic', type=str, required=False)
parser.add_argument('--config_file', type=str, default="config/single_pred.yaml",required=False)
parser.add_argument('--seed', type=int, default=42, required=False,
                    help='set the seed to reproduce result')
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load the configuration
with open(args.config_file, "r") as reader:
    config = yaml.load(reader, Loader=yaml.FullLoader)

pic_path=args.pic
# if not os.path.exists(pic_path):
    # raise ValueError(f"Pic {pic_path} does not exist")

(id, model_name) = select_model()

# The requested model must exist before making predictions
if not os.path.exists(os.path.join(config["output_dir"]["model"],model_name)):
    raise ValueError(f"Model {model_name} does not exist")

model_path = os.path.join(config["output_dir"]["model"],model_name,"checkpoints")

# Data transformations
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
])

# Hyperparameters from the configuration
num_classes = config["model"]["num_classes"]

os.makedirs(os.path.join(config["output_dir"]["model"],model_name,"val"),exist_ok=True)

# Initialize the U-Net model
in_channels = 3  # RGB 
out_channels = num_classes 
model = UNet(in_channels, out_channels)

# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# predictions
model.eval()
if device==torch.device("cpu"):
    model.load_state_dict(torch.load(os.path.join(model_path,"best.pth"),map_location=torch.device('cpu')))
else:
    model.load_state_dict(torch.load(os.path.join(model_path,"best.pth")))

names = os.listdir("/dssg/home/acct-zhaochaoxian/zhaochaoxian-user1/InstanceSegment/pathology_maskrcnn/result")
# L = [454,1046,1563,1751,4327,6221,6473,6732]
# for l in L:
for name in tqdm(names):
# image_path="/dssg/home/acct-zhaochaoxian/zhaochaoxian-user1/data/dataset/val/F21-5777F_47616_44288_47872_44544.png"
    image_path = os.path.join("/dssg/home/acct-zhaochaoxian/zhaochaoxian-user1/InstanceSegment/pathology_maskrcnn/result",name,"origin_{}.png".format(name))
    original_image = cv2.imread(image_path)#Image.open(image_path).convert("RGB")
    image = Image.fromarray(original_image)
    image = transform(image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1).squeeze(0)

    cnt=0
    view_result_path=os.path.join(config["output_dir"]["model"],model_name,"val")
    os.makedirs(view_result_path,exist_ok=True)
    colors = [(0, 255, 0), (0, 0, 255), (0, 255, 0)] #cm.hsv(np.linspace(0, 1, num_classes)) * 255 # exclude background
    deep_colors=[]

    # Define weights for blending
    alpha = 0.5
    beta = 0.5
    gamma = 0

    depth_factor = 1.0
    # convert original color to a deeper one
    for rgb_color in colors:
        # Convert the original RGB color to HSV
        hsv_color=cv2.cvtColor(np.uint8([[rgb_color[:3]]]), cv2.COLOR_RGB2HSV)[0][0]
        deeper_value = int(hsv_color[2] * depth_factor)
        adjusted_hsv = np.array([hsv_color[0], hsv_color[1], deeper_value], dtype=np.uint8)
        # Convert the adjusted HSV color back to RGB
        deeper_color = cv2.cvtColor(np.uint8([[adjusted_hsv]]), cv2.COLOR_HSV2RGB)[0][0]
        deep_colors.append(deeper_color)

    # Create a half-transparent version of each color
    colors_with_alpha = np.zeros((num_classes, 4))
    colors_with_alpha[:, :3] = colors
    colors_with_alpha[:, 3] = 0.05
    image_transparency=0.6 # closer to 0, the more apparent mask is

    # tensor2pil = transforms.ToPILImage()
    # original_image=tensor2pil(original_image.squeeze())
    # Convert the mask to a numpy array
    segmentation_mask_np = np.array(prediction)

    # Apply the mask to the original image
    segmented_image = np.array(original_image)
    image_outline = np.array(original_image)

    for class_idx in range(1, num_classes):  # Start from 1 to exclude the background class
        binary_mask = segmentation_mask_np == class_idx
        mask_rgb = colors_with_alpha[class_idx] # * 255
        segmented_image[binary_mask] = (segmented_image[binary_mask] * image_transparency + mask_rgb[:3] * (1-image_transparency)).astype(np.uint8)
        image_outline = segmentation.mark_boundaries(image_outline, binary_mask,color=colors[class_idx][:3],mode='thick')
    image_outline_new = np.array(original_image)
    for i in range(1024):
        for j in range(1024):
            if 255 in image_outline[i][j]:
                image_outline_new[i][j] = 0
    image_outline = image_outline + image_outline_new
