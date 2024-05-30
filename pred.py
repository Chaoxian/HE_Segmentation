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
import openslide
import slide
import torch.nn.functional as F


from scripts.dataset import SegmentationDataset_2cls
from scripts.model import CustomUNet, UNet
from scripts.metric import compute_miou


def split_names_into_files(names, num_files):
    # 计算每个文件中名字的数量
    names_per_file = len(names) // num_files
    
    # 确保目标文件夹存在，如果不存在，则创建它
    if not os.path.exists('split_names'):
        os.makedirs('split_names')
    
    # 将名字列表分成指定数量的子列表
    split_names = [names[i:i+names_per_file] for i in range(0, len(names), names_per_file)]
    
    # 保存每个子列表到不同的文件中
    for i, name_list in enumerate(split_names):
        with open(f'split_names/names_{i+1}.txt', 'w') as file:
            for name in name_list:
                file.write(name + '\n')
                
    if len(names) % num_files != 0:
        with open(f'split_names/names_{num_files}.txt', 'a') as file:
            for name in names[num_files * names_per_file:]:
                file.write(name + '\n')

def read_names_from_files(num_files):
    # 初始化一个空列表来存储所有的名字
    all_names = []

    # 读取每个文件中的名字并添加到列表中
    filename = f'split_names/names_{num_files}.txt'
    with open(filename, 'r') as file:
        names = file.read().splitlines()
        all_names.extend(names)

    return all_names

parser = argparse.ArgumentParser(description='SegNet Pred.')
parser.add_argument('--config_file', type=str, default="config/pred.yaml",required=False)
parser.add_argument('--model', type=str, default="", required=False,
                    help='the name of the model')
parser.add_argument('--seed', type=int, default=42, required=False,
                    help='set the seed to reproduce result')
parser.add_argument("--id", type=int)
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

if args.model=="":
    model_name=config["name"]
else:
    model_name=args.model

# The requested model must exist before making predictions
if not os.path.exists(os.path.join(config["output_dir"]["model"],model_name)):
    raise ValueError(f"Model {model_name} does not exist")

model_path = os.path.join(config["output_dir"]["model"],model_name,"checkpoints")

# Data transformations
transform = transforms.Compose([
    # transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Hyperparameters from the configuration
num_classes = config["model"]["num_classes"]
data_dir = config["data"]["data_dir"]
train_ratio = config["data"]["train_ratio"]
all_files = read_names_from_files(args.id)
# os.makedirs(os.path.join(config["output_dir"]["model"],model_name,"val"),exist_ok=True)
for tmp_i, file in enumerate(tqdm(all_files)):
    out_path = os.path.join(config["output_dir"]["model"],model_name,"pred2",file)
    os.makedirs(out_path,exist_ok=True)
    ## Load dataset
    val_dataset = SegmentationDataset_2cls(data_dir=data_dir, transform=transform,train=file)
    # train_size = int(train_ratio * len(dataset))
    # val_size = len(dataset) - train_size
    # train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    # train_loader = DataLoader(train_dataset, batch_size=config["train"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["val"]["batch_size"], shuffle=False)

    # Initialize the U-Net model
    in_channels = 3  # RGB 
    out_channels = num_classes 
    model = UNet(in_channels, out_channels)

    # Set device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)

    # predictions
    model.eval()
    # if device==torch.device("cpu"):
    model.load_state_dict(torch.load(os.path.join(model_path,"best.pth"),map_location=torch.device('cpu')))
    # else:
    #     model.load_state_dict(torch.load(os.path.join(model_path,"best.pth")))
    model.to(device)
    # results={}
    sum_pixel = 0
    slide_img = openslide.OpenSlide("../HE_20240220/"+file+".ndpi")
    slide_image = np.array(slide.read_slide_at_mag(slide_img, 5).convert('RGB'))
    image_transparency = 0.6
    mask_rgb = [0,255,0]
    slide_img.close()
    preds = []
    img_names = []
    with torch.no_grad():
        for batch_idx, (images, _, image_name) in enumerate(tqdm(val_loader)):
            images = images.to(device)
            # import ipdb; ipdb.set_trace()
            # masks=masks.to(torch.long)
            # Forward pass
            outputs = model(images)
            prediction = torch.argmax(outputs, dim=1).cpu().float()
            sum_pixel+=prediction.sum()
            prediction = F.interpolate(prediction.unsqueeze(1), size=(128, 128), mode='bilinear').squeeze(1).numpy().astype(np.uint8)
            # import ipdb; ipdb.set_trace()
            prediction[prediction>1] = 1
            preds.append(prediction)
            img_names.extend(image_name)
            # if batch_idx==100:
            #     break
           # import ipdb; ipdb.set_trace()
            # print("1",prediction.shape)
            # miou=compute_miou(outputs=outputs,targets=masks,num_classes=num_classes)
            
            # print(image_name)
        preds = np.concatenate(preds,axis=0)
        funX = lambda x: int(int(x.split("_")[3].split("-")[0])/8)
        funY = lambda x: int(int(x.split("_")[3].split("-")[1])/8)
        X,Y = list(map(funX,img_names)),list(map(funY,img_names))
            
            # results[batch_idx]=(image_name,prediction,)
            # image_save = np.zeros_like(prediction)
            # image_save[prediction==1]=255
            # Image.fromarray(image_save.astype(np.uint8),mode="L").save(os.path.join(out_path,image_name[0]))
        # position = [np.ix_(np.arange(y,y+128),np.arange(x,x+128)) for x,y in zip(X,Y)]
        masks = np.zeros_like(slide_image)[:,:,0]
        for i in range(len(X)):
            masks[Y[i]:Y[i]+128,X[i]:X[i]+128] = preds[i]
        Image.fromarray(masks).save(f"test_{file}.png")
        binary_mask = masks==1
        slide_image[binary_mask] = (slide_image[binary_mask] * image_transparency + np.array(mask_rgb[:3]) * (1-image_transparency)).astype(np.uint8)
        percentage = sum_pixel/(1024.0*1024.0*(preds.shape[0]))
        print(percentage)
        Image.fromarray(slide_image).save(file+f"_{str(percentage)[6:]}.png")
        # Image.fromarray(slide_image).save(os.path.join(out_path,file+f"_{str(percentage)[6:]}.png"))
# sorted_results = dict(sorted(results.items(), key=lambda x: x[1]))
# # print(sorted_results)
# cnt=0
# view_result_path=os.path.join(config["output_dir"]["model"],config["name"],"val")
# os.makedirs(view_result_path,exist_ok=True)
# colors = cm.hsv(np.linspace(0, 1, num_classes)) * 255 # exclude background
# deep_colors=[]

# # Define weights for blending
# alpha = 0.5
# beta = 0.5
# gamma = 0

# depth_factor = 1.0
# # convert original color to a deeper one
# for rgb_color in colors:
#     # Convert the original RGB color to HSV
#     hsv_color=cv2.cvtColor(np.uint8([[rgb_color[:3]]]), cv2.COLOR_RGB2HSV)[0][0]
#     deeper_value = int(hsv_color[2] * depth_factor)
#     adjusted_hsv = np.array([hsv_color[0], hsv_color[1], deeper_value], dtype=np.uint8)
#     # Convert the adjusted HSV color back to RGB
#     deeper_color = cv2.cvtColor(np.uint8([[adjusted_hsv]]), cv2.COLOR_HSV2RGB)[0][0]
#     deep_colors.append(deeper_color)

# # Create a half-transparent version of each color
# colors_with_alpha = np.zeros((num_classes, 4))
# colors_with_alpha[:, :4] = colors
# colors_with_alpha[:, 3] = 0.05
# image_transparency=0.6 # closer to 0, the more apparent mask is

# tensor2pil = transforms.ToPILImage()

# for (idx,result) in sorted_results.items():
#     cnt+=1
#     image, true_mask, img_name = val_dataset[idx]
#     image_name=result[0]
#     miou=result[1]
#     pred_mask=result[2]

#     original_image=tensor2pil(image)
#     # pred_mask=tensor2pil(pred_mask)

#     # Convert the mask to a numpy array
#     segmentation_mask_np = np.array(pred_mask.cpu())
    
#     # Apply the mask to the original image
#     segmented_image = np.array(original_image)
#     image_outline = np.array(original_image)

#     for class_idx in range(1, num_classes):  # Start from 1 to exclude the background class
#         binary_mask = segmentation_mask_np == class_idx
#         # import ipdb; ipdb.set_trace()
#         class_true_mask = np.where(true_mask != class_idx , 0, class_idx)
#         # print(class_idx,np.count_nonzero(binary_mask))
#         mask_rgb = colors_with_alpha[class_idx][[2,1,0,3]] # * 255
#         segmented_image[binary_mask] = (segmented_image[binary_mask] * image_transparency + mask_rgb[:3] * (1-image_transparency)).astype(np.uint8)
#         image_outline = segmentation.mark_boundaries(image_outline, class_true_mask,
#                                                      outline_color=colors[class_idx][:3][[1,0,2]]/255,mode='thick')
    
#     combined_image = segmented_image.copy()

#     for class_idx in range(1, num_classes):  # Start from 1 to exclude the background class
#         class_true_mask = np.where(true_mask != class_idx , 0, class_idx)
#         combined_image = segmentation.mark_boundaries(combined_image, class_true_mask
#                                                       ,outline_color=colors[class_idx][:3][[1,0,2]]/255,mode='thick')

#     # print(colors_with_alpha*255)
#     # print(deep_colors)
#     #original_hsv = cv2.cvtColor(np.uint8([[original_color]]), cv2.COLOR_RGB2HSV)[0][0]

#     # combined_image=
#     # segmented_image_pil = Image.fromarray(segmented_image)
#     # segmented_image_pil.save(os.path.join(view_result_path,f"{img_name[:-4]}.png"))

#     # GT_Pred_pic=cv2.addWeighted(image_outline.astype(segmented_image.dtype), alpha, segmented_image, beta, gamma)
#     # Create a figure and an array of axes (subplots)
#     fig, axes = plt.subplots(2, 2, figsize=(10, 5))  # 1 row, 2 columns

#     axes[0][0].imshow(original_image, interpolation='bicubic')
#     axes[0][0].set_title('Ori')
#     axes[0][0].axis('off') 

#     axes[0][1].imshow(image_outline, interpolation='bicubic')
#     axes[0][1].set_title('GT')
#     axes[0][1].axis('off') 
#     print(np.unique(image_outline))

#     axes[1][0].imshow(segmented_image, interpolation='bicubic')
#     axes[1][0].set_title('Pred')
#     axes[1][0].axis('off') 
#     print(np.unique(segmented_image))

#     axes[1][1].imshow(combined_image, interpolation='bicubic')
#     axes[1][1].set_title('GT+Pred')
#     axes[1][1].axis('off') 
#     print(np.unique(combined_image))

#     plt.suptitle(f"{miou:.4f}", fontsize=16)
#     plt.tight_layout()  # Adjust spacing between subplots
#     plt.savefig(os.path.join(view_result_path,f"{img_name[:-4]}.png"))  # Show the entire plot
    
"""
print(colors)
print("")
print(np.rint(colors_with_alpha*255))

def show(arr):
    return arr[:5,:5],type(arr[0][0])

# Random colors for each class
colors = np.random.randint(0, 255, (num_classes, 3), dtype=np.uint8)  

for (idx,result) in sorted_results.items():
    cnt+=1
    image, true_mask, _ = val_dataset[idx]
    image_name=result[0]
    miou=result[1]
    pred_mask=result[2]

    image = image.numpy().astype(int)
    print(show(image))
    image = np.uint8(np.rollaxis(image, 0, 3))
    print(show(image))
    true_mask_np = true_mask.numpy().astype(int)
    print(show(true_mask_np))
    pred_mask_np = pred_mask.numpy().astype(int)
    print(show(pred_mask_np))

    # Convert the predicted mask and ground truth mask to 3-channel (RGB) images with different colors for each class
    predicted_mask_rgb = colors[pred_mask_np]
    print(show(predicted_mask_rgb))
    ground_truth_mask_rgb = colors[true_mask_np]
    print(show(ground_truth_mask_rgb))

    
    # Create a blank alpha (transparency) channel with the same shape as the original image
    alpha_channel = np.ones_like(image) * 255  # Fully opaque alpha channel (255 indicates full opacity)
    image_transparency=0.1

    # Combine the original image and the predicted mask with transparency
    combined_image_with_predicted = cv2.addWeighted(image, 1, predicted_mask_rgb, image_transparency, 0)
    #combined_image_with_predicted = np.concatenate((combined_image_with_predicted, alpha_channel), axis=2)

    # Combine the original image and the ground truth mask with transparency
    combined_image_with_ground_truth = cv2.addWeighted(image, 1, ground_truth_mask_rgb, image_transparency, 0)
    # print("1",combined_image_with_ground_truth.shape)
    #combined_image_with_ground_truth = np.concatenate((combined_image_with_ground_truth, alpha_channel), axis=2)
    # print("2",combined_image_with_ground_truth.shape)


    # print(combined_image_with_predicted.shape) #,type(image[0,0,0]))
    # print(combined_image_with_ground_truth.shape) #,type(predicted_mask_rgb[0,0,0]))

    # Stack the images horizontally with some space in between
    space_between_images = 10  # Adjust this value to control the space between images
    combined_image = np.hstack((image, np.zeros((image.shape[0], space_between_images, 3), dtype=np.uint8),
                                combined_image_with_predicted, np.zeros((predicted_mask_rgb.shape[0], space_between_images, 3), dtype=np.uint8),
                                combined_image_with_ground_truth))

    cv2.imwrite(os.path.join(view_result_path,f"{idx}_{image_name}_{miou:.4f}.png"),combined_image)
    if cnt>config["val"]["view_cnt"]:
        break




for (idx,result) in sorted_results.items():
    cnt+=1
    image, true_mask, _ = val_dataset[idx]
    image_name=result[0]
    miou=result[1]
    pred_mask=result[2]

    image = tensor2pil(image)
    true_mask = tensor2pil(true_mask)
    true_mask_np = true_mask.numpy().astype(int)
    pred_mask_np = pred_mask.numpy().astype(int)
    
    colored_true_mask_np = cmap(true_mask)
    colored_pred_mask_np = cmap(pred_mask_np)

    pred_mask = tensor2pil(pred_mask.to(torch.uint8))
    # print(true_mask,"\n")
    # print(pred_mask)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))

    GroundTruth_image = image.copy()
    WrongPred_image = image.copy()

    colored_true_mask_pil = Image.fromarray((colored_true_mask_np[:, :, :3] * 255).astype('uint8'))
    colored_true_mask_pil=colored_true_mask_pil.convert("RGBA")
    colored_pred_mask_pil = Image.fromarray((colored_pred_mask_np[:, :, :3] * 255).astype('uint8'))
    colored_pred_mask_pil=colored_pred_mask_pil.convert("RGBA")

    print(colored_true_mask_pil.size)
    print(colored_pred_mask_pil.size)
    # Show each image on a separate subplot
    axes[0].imshow(image)
    axes[0].set_title('Original')

    GroundTruth_image.paste(image, colored_true_mask_pil)
    axes[1].imshow(GroundTruth_image)
    axes[1].set_title('GroundTruth')

    WrongPred_image.paste(image, colored_pred_mask_pil)
    axes[2].imshow(WrongPred_image)
    axes[2].set_title('WrongPred')

    # Hide axis ticks and labels
    for ax in axes:
        ax.axis('off')
    # Add a title to the entire figure
    fig.suptitle(f'{idx}_{image_name}_{miou}', fontsize=8)
    # Save the figure as a PNG file
    plt.savefig(f'{idx}_{image_name}_{miou}.jpg', bbox_inches='tight', dpi=300)
"""
