import numpy as np
import torch

def iou(pred, target, n_classes = 21):
    """
    Calculate the Intersection over Union (IoU) for predictions.

    Args:
        pred (tensor): Predicted output from the model.
        target (tensor): Ground truth labels.
        n_classes (int, optional): Number of classes. Default is 21.

    Returns:
        float: Mean IoU across all classes.
    """
    
    if pred.dim() == 4:  # If shape is (B, C, H, W), convert to (B, H, W)
        pred = torch.argmax(pred, dim=1)

    target = target.long()  # Make sure target is long

    ious = []
    for cls in range(n_classes):
        # True positives
        intersection = torch.sum((pred == cls) & (target == cls))

        # Denominator
        union = torch.sum((pred == cls) | (target == cls))

        if union > 0:
            ious.append(intersection.float() / union.float())

    return torch.stack(ious).mean() if ious else torch.tensor(0.0)


def pixel_acc(pred, target):
    """
    Calculate pixel-wise accuracy between predictions and targets.

    Args:
        pred (tensor): Predicted output from the model.
        target (tensor): Ground truth labels.

    Returns:
        float: Pixel-wise accuracy.

    Example: 
    pred: [0, 5, 2, 3]
    target: [0, 3, 4, 3]

    float = .75
    """
    if pred.dim() == 4:  # If shape is (B, C, H, W), convert to (B, H, W)
        pred = torch.argmax(pred, dim=1)
        
    # This assumes the type is int
    pred[pred == 255] = 0
    correct = torch.sum(pred == target)
    total = target.numel()
    
    return correct / total if total > 0 else torch.tensor(0.0)


import os
import numpy as np
import matplotlib.pyplot as plt

def plot_losses(train_loss, val_loss, early_stop=None, best_model=None):
    """
    Plots training and validation loss over epochs.

    Args:
        train_loss (list): List of training loss values per epoch.
        val_loss (list): List of validation loss values per epoch.
        early_stop (int, optional): The epoch where early stopping occurred.
    """
    
    epochs = np.arange(1, len(train_loss) + 1)

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_loss, 'r', label="Training Loss")
    plt.plot(epochs, val_loss, 'g', label="Validation Loss")
    
    if early_stop is not None:
        plt.scatter(epochs[early_stop], val_loss[early_stop], marker='x', c='g', s=200, label='Early Stop Epoch')
    if best_model is not None:
        plt.scatter(epochs[best_model], val_loss[best_model], marker='o', c='r', s=400, label='Best Model Epoch')

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('Loss Plot', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Cross Entropy Loss', fontsize=14)
    plt.legend(loc="upper right", fontsize=14)
    plt.grid(True)
        
    plt.savefig('loss_plot.png')

    plt.show()

