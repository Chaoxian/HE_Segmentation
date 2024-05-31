import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from sklearn.utils.class_weight import compute_class_weight
import yaml
import os
import argparse
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from PIL import Image
from tqdm import tqdm

from scripts.dataset import SegmentationDataset
from scripts.model import CustomUNet, UNet
from scripts.metric import compute_miou

from utils.transform import transform1, transform_val
from utils.eda import PixelWiseCnt

parser = argparse.ArgumentParser(description='Train semantic segmentation network.')
parser.add_argument('--config_file', type=str, required=True)
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

print(config["name"])
# Hyperparameters from the configuration
num_classes = config["model"]["num_classes"]
learning_rate = config["train"]["learning_rate"]
num_epochs = config["train"]["num_epochs"]
data_dir = config["data"]["data_dir"]
train_ratio = config["data"]["train_ratio"]

model_path = os.path.join(config["output_dir"]["model"],config["name"],"checkpoints")
os.makedirs(config["output_dir"]["model"],exist_ok=True)
os.makedirs(os.path.join(config["output_dir"]["model"],config["name"]),exist_ok=True)
os.makedirs(os.path.join(config["output_dir"]["model"],config["name"],"checkpoints"),exist_ok=True)
os.makedirs(config["output_dir"]["log"],exist_ok=True)

## Load dataset
dataset = SegmentationDataset(data_dir=data_dir, transform=None)
train_size = int(train_ratio * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_dataset.dataset.transform = transform_val
val_dataset.dataset.transform = transform_val
train_loader = DataLoader(train_dataset, batch_size=config["train"]["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["val"]["batch_size"], shuffle=False)

# Initialize the U-Net model
in_channels = 3  # RGB 
out_channels = num_classes 
model = UNet(in_channels, out_channels)

# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define loss function and optimizer
cnt,sum=PixelWiseCnt(os.path.join(config["data"]["data_dir"],"sem_masks"))
class_weights=torch.tensor(np.array([(sum/cnt[i]) for i in range(num_classes)])).to(device).float()
criterion = nn.CrossEntropyLoss(weight=class_weights,reduction='mean')

optimizer = getattr(optim, config["optimizer"]["type"])(
    model.parameters(),
    **config["optimizer"]["args"])

lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
    **config["lr_scheduler"]["args"])

with open(os.path.join(config["output_dir"]["model"],config["name"], 'config.yaml'), "w") as writer:
    yaml.dump(config, writer, default_flow_style=False)

best_miou = 0.0
early_stop_counter = 0

for epoch in range(1,num_epochs+1):
    # train
    model.train()
    train_loss=0
    train_miou=0
    for batch_idx, (images, masks, image_name) in enumerate(tqdm(train_loader)):
        images, masks = images.to(device), masks.to(device)
        masks=masks.to(torch.long)
        # Forward pass
        outputs = model(images)
        # results=torch.argmax(outputs,dim=1)   # executed in func: compute_miou
        loss=criterion(outputs,masks)  
        miou=compute_miou(outputs=outputs,targets=masks,num_classes=num_classes)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_miou += miou

    train_loss /= ((batch_idx+1) * config["train"]["batch_size"])
    train_miou /= (batch_idx+1 * config["train"]["batch_size"])

    # val
    model.eval()
    val_loss=0
    val_miou=0
    
    for batch_idx, (images, masks, image_name) in enumerate(tqdm(val_loader)):
        images, masks = images.to(device), masks.to(device)
        masks=masks.to(torch.long)
        # Forward pass
        outputs = model(images)

        loss=criterion(outputs,masks)
        miou=compute_miou(outputs=outputs,targets=masks,num_classes=num_classes)

        val_loss += loss.item()
        val_miou += miou

    val_loss /= ((batch_idx+1) * config["val"]["batch_size"])
    val_miou /= ((batch_idx+1) * config["val"]["batch_size"])

    with open(os.path.join(config["output_dir"]["model"],config["name"],"train_log.txt"),"a") as f:
        f.write(f"epoch: {epoch} | Train loss: {train_loss:.6f} ; miou: {train_miou:.6f} | Val   loss: {val_loss:.6f} ; miou: {val_miou:.6f}\n")
    print(f"epoch: {epoch} | Train loss: {train_loss:.6f} ; miou: {train_miou:.6f} | Val   loss: {val_loss:.6f} ; miou: {val_miou:.6f}")
    
    lr_scheduler.step(val_loss)

    if epoch % config["train"]["save_interval"] == 0:
        torch.save(model.state_dict(),os.path.join(model_path,"epoch_{}.pth".format(epoch)))
    if val_miou > best_miou:
        early_stop_counter = 0
        best_miou = val_miou
        torch.save(model.state_dict(),os.path.join(model_path,"best.pth"))
    else:
        early_stop_counter += 1
    if early_stop_counter >= config["train"]["early_stop"]:
        print(f"epoch: {epoch} | Early stop raised.")
        break

"""
# visualize validation results
view_result_path=os.path.join(config["output_dir"]["model"],config["name"],"val")
os.makedirs(view_result_path,exist_ok=True)
results={}
# Create a colormap with transparency
transparency=0.4
colors = cm.hsv(np.linspace(0, 1, num_classes))
colors_with_alpha = np.zeros((num_classes, 4))
colors_with_alpha[:, :4] = colors
colors_with_alpha[:, 3] = transparency   # transparency

tensor2pil = transforms.ToPILImage()

for batch_idx, (images, masks, image_name) in enumerate(tqdm(val_loader)):
    images, masks = images.to(device), masks.to(device)
    masks=masks.to(torch.long)
    # Forward pass
    outputs = model(images)
    pred = torch.argmax(outputs, dim=1).squeeze(0)
    miou=compute_miou(outputs=outputs,targets=masks,num_classes=num_classes)

    results[batch_idx]=(image_name,miou,pred)

cnt=0
sorted_results = dict(sorted(results.items(), key=lambda x: x[1]))

for (idx,result) in sorted_results.items():
    cnt+=1
    image, true_mask, img_name = val_dataset[idx]
    image_name=result[0]
    miou=result[1]
    pred_mask=result[2]

    original_image=tensor2pil(image)
    segmented_image = np.array(original_image)
    # pred_mask=tensor2pil(pred_mask)
    # Convert the mask to a numpy array
    segmentation_mask_np = np.array(pred_mask)

    for class_idx in range(1, num_classes):  # Start from 1 to exclude the background class
        binary_mask = segmentation_mask_np == class_idx
        mask_rgb = colors_with_alpha[class_idx] * 255

    segmented_image[binary_mask] = (segmented_image[binary_mask] * 0.5 + mask_rgb[:3] * 0.5).astype(np.uint8)

    segmented_image_pil = Image.fromarray(segmented_image)
    segmented_image_pil.save(os.path.join(view_result_path,f"{img_name[:-4]}.png"))

    if cnt>=config["val"]["view_cnt"]:
        break
"""
