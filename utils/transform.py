import torch
from torchvision import transforms

class GammaCorrection(object):
    def __init__(self, gamma_range=(0.5, 2.0)):
        self.gamma_range = gamma_range

    def __call__(self, img):
        gamma = torch.rand(1) * (self.gamma_range[1] - self.gamma_range[0]) + self.gamma_range[0]
        return img ** gamma

# Define the data augmentation transforms for training
transform1 = transforms.Compose([
    transforms.RandomVerticalFlip(),      # Randomly flip the image vertically
    transforms.RandomHorizontalFlip(),    # Randomly flip the image horizontally
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Randomly change color
    transforms.RandomApply([transforms.Grayscale(num_output_channels=3)], p=0.2),    # Randomly convert to grayscale (optional)
    GammaCorrection(gamma_range=(0.5, 2.0)),
    transforms.ToTensor(),    # Convert the image and mask to PyTorch tensors
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
])


