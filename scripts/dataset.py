import os
from PIL import Image
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

class SegmentationDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_list = os.listdir(os.path.join(data_dir, "images"))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        image_path = os.path.join(self.data_dir, "images", image_name)
        mask_path = os.path.join(self.data_dir, "sem_masks", image_name.replace("img_", "sem_"))

        image = Image.open(image_path).convert("RGB")
        mask=cv2.imread(mask_path,0)
        mask=torch.from_numpy(mask)

        if self.transform:
            image = self.transform(image)

        return image, mask, image_name

class InstDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_list = os.listdir(os.path.join(data_dir, "images"))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        image_path = os.path.join(self.data_dir, "images", image_name)
        mask_path = os.path.join(self.data_dir, "sem_masks", image_name.replace("img_", "sem_"))
        inst_path = os.path.join(self.data_dir, "inst_masks", image_name.replace("img_", "inst_"))
        
        image = Image.open(image_path).convert("RGB")
        mask=cv2.imread(mask_path,0)
        mask=torch.from_numpy(mask)
        inst=cv2.imread(inst_path,0)
        inst=torch.from_numpy(inst)
        if self.transform:
            image = self.transform(image)

        return image, mask, inst,image_name

class SegmentationDataset_2cls(Dataset):
    def __init__(self, data_dir, transform=None,train="train"):
        self.data_dir = data_dir
        self.transform = transform
        self.image_list = os.listdir(os.path.join(data_dir,train))#,"images")) #modified by jidong at 2024/4/11
        self.train = train

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        # modified by jidong at 2024/4/10 to accomodate to the he_dataset
        image_name = self.image_list[idx]
        image_path = os.path.join(self.data_dir, self.train, image_name)#modified by jidong at 2024/4/11
        # mask_path = os.path.join(self.data_dir, self.train, "sem_masks", image_name.replace("img_", "sem_"))
        # print(mask_path)
        image = Image.open(image_path).convert("RGB")
        # mask=cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # mask=torch.from_numpy(mask)
        # mask[mask>0]=1
        mask = torch.zeros(1) #for test /jidong at 2024/4/21
        # import ipdb; ipdb.set_trace()
        if self.transform:
            image = self.transform(image)

        return image, mask, image_name


if __name__ == "__main__":
    data_dir = "/dssg/home/acct-zhaochaoxian/zhaochaoxian-user1/InstanceSegment/pathology_maskrcnn/final_pannuke_dataset_2"
    dataset = SegmentationDataset_2cls(data_dir)
    for image, mask, image_name in tqdm(dataset):
        # print(image_name)
        pass
    dataset = SegmentationDataset_2cls(data_dir,None,"val")
    for image, mask, image_name in tqdm(dataset):
        # print(image_name)
        pass
    # image, mask, inst, image_name = dataset[0]
    # # inst = inst.to(torch.uint8)
    # mask = mask.numpy()
    # mask *= (255//5)
    # cv2.imwrite("mask.png", mask)
    # inst = inst.numpy()
    # inst *= 255
    # cv2.imwrite("inst.png", inst)