import torch
import numpy as np

def compute_miou(outputs, targets, num_classes):
    """
    Compute mean Intersection over Union (mIoU) for semantic segmentation.

    Args:
        outputs (torch.Tensor): Predicted output from the model (shape: [batch_size, num_classes, H, W]).
        targets (torch.Tensor): Ground truth segmentation masks (shape: [batch_size, H, W]).
        num_classes (int): Number of classes in the segmentation task.

    Returns:
        float: Mean Intersection over Union (mIoU) value.
    """
    with torch.no_grad():
        preds = torch.argmax(outputs, dim=1)  # Get predicted class labels (shape: [batch_size, H, W])
        miou_sum = 0.0
        for i in range(num_classes):
            intersection = torch.sum((preds == i) & (targets == i))
            union = torch.sum((preds == i) | (targets == i))
            iou = intersection.float() / (union.float() + 1e-8)  # Add a small epsilon to avoid division by zero
            # print(i,iou.item(),np.sum(preds.numpy()==i),np.sum(targets.numpy()==i))
            miou_sum += iou.item()

        miou = miou_sum / num_classes
        return miou


