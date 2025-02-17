from basic_fcn import FCN
from resnet_fcn import ResNet
import time
from torch.utils.data import DataLoader
import torch
import gc
import voc
import torchvision.transforms as standard_transforms
import util
import numpy as np
import multiprocessing
from util import iou, pixel_acc, plot_losses
import copy
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

num_workers = multiprocessing.cpu_count()

class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.normal_(m.bias.data) #xavier not applicable for biases



def getClassWeights(train_dataset):
    """
    Computes class weights for a segmentation dataset.
    Each sample in train_dataset is assumed to be a tuple (image, mask)
    where mask is a 2D tensor of size (H, W) with integer class labels.
    """
    all_labels = []
    # Iterate over the training dataset (not the loader)
    for image, label in train_dataset:
        # Flatten the label image into 1D array and append it
        all_labels.append(label.numpy().flatten())

    all_labels = np.concatenate(all_labels)
    n_class = 21  
    class_counts = np.bincount(all_labels, minlength=n_class)
    class_weights = 1.0 / (torch.tensor(class_counts, dtype=torch.float) + 1e-6)
    return class_weights

best_model_path = "best_model/basic/best_model.pt"
# normalize using imagenet averages
mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 

input_transform = standard_transforms.Compose([
    standard_transforms.ToTensor(),
    standard_transforms.Normalize(*mean_std)
])
augmentations = [('crop', .3), ('horizontal_flip', .3)]
# augmentations = None

target_transform = MaskToTensor()

train_dataset = voc.VOC('train', transform=input_transform, target_transform=target_transform, augmentations=augmentations)
val_dataset = voc.VOC('val', transform=input_transform, target_transform=target_transform, augmentations=None)

# test_dataset = voc.VOC('test', transform=input_transform, target_transform=target_transform)

val_dataset,test_dataset = torch.utils.data.random_split(val_dataset,[0.5,0.5])

train_loader = DataLoader(dataset=train_dataset, batch_size= 16, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(dataset=val_dataset, batch_size= 16, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(dataset=test_dataset, batch_size= 16, shuffle=False, num_workers=num_workers)

epochs = 30

n_class = 21

learning_rate = 0.001

early_stop = True
early_stop_epoch = 5
fcn_model = FCN(n_class=n_class)
fcn_model.apply(init_weights)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

optimizer = torch.optim.Adam(fcn_model.parameters(), lr=learning_rate)

scheduler = CosineAnnealingLR(
            optimizer, 
            T_max = epochs,
            eta_min=1e-6 
        )



do_weighted_loss=True
if do_weighted_loss:
    loss_weights = getClassWeights(train_dataset)
    loss_weights = loss_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=loss_weights)
else:
    criterion = nn.CrossEntropyLoss()

criterion = nn.CrossEntropyLoss()
fcn_model = fcn_model.to(device)


# TODO
def train():
    """
    Train a deep learning model using mini-batches.

    - Perform forward propagation in each epoch.
    - Compute loss and conduct backpropagation.
    - Update model weights.
    - Evaluate model on validation set for mIoU score.
    - Save model state if mIoU score improves.
    - Implement early stopping if necessary.

    Returns:
        None.
    """

    best_iou_score = 0.0
    best_val_loss = float('inf')
    best_val_epoch = 0
    scheduler_step=True
    patience = 0

    training_loss = []
    val_loss = []
    for epoch in range(epochs):
        fcn_model.train()
        ts = time.time()
        print(f"epoch {epoch}:")
        epoch_losses = []
        for iter, (inputs, labels) in enumerate(train_loader):
            # TODO  reset optimizer gradients

            # both inputs and labels have to reside in the same device as the model's
            inputs =  inputs.to(device)# TODO transfer the input to the same device as the model's
            labels =  labels.to(device) # TODO transfer the labels to the same device as the model's

            optimizer.zero_grad()

            outputs =  fcn_model(inputs) # TODO  Compute outputs. we will not need to transfer the output, it will be automatically in the same device as the model's!

            loss =  criterion(outputs, labels.long())
            epoch_losses.append(loss.item())
            
            loss.backward()
            optimizer.step()


            if iter % 20 == 0:
                print("    iteration {}, loss: {}".format(iter, loss.item()))
        
        training_loss.append(np.mean(epoch_losses))
        

        print("Finished epoch {}, time elapsed {} seconds".format(epoch, int(time.time() - ts)))

        current_iou_score, current_val_loss = val(epoch)
        val_loss.append(current_val_loss)
        if current_val_loss<best_val_loss:
            best_val_loss=current_val_loss
            best_val_epoch=epoch
        
        print("\n")
        if scheduler_step:

          scheduler.step()

        if early_stop or epoch == 0:
            if current_iou_score > best_iou_score:
                best_iou_score = current_iou_score
                patience = 0
                best_model = copy.deepcopy(fcn_model)
                best_model_epoch = epoch
            else:
                patience += 1
                if patience >= early_stop_epoch:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

    torch.save(best_model.state_dict(), best_model_path)
    plot_losses(training_loss, val_loss, early_stop=best_val_epoch if best_val_epoch and patience >= early_stop_epoch else None, best_model = best_model_epoch if best_model_epoch > 0 else None)
    return best_model


def val(epoch=-1):
    """
    Validate the deep learning model on a validation dataset.

    - Set model to evaluation mode.
    - Disable gradient calculations.
    - Iterate over validation data loader:
        - Perform forward pass to get outputs.
        - Compute loss and accumulate it.
        - Calculate and accumulate mean Intersection over Union (IoU) scores and pixel accuracy.
    - Print average loss, IoU, and pixel accuracy for the epoch.
    - Switch model back to training mode.

    Args:
        epoch (int): The current epoch number.

    Returns:
        tuple: Mean IoU score and mean loss for this validation epoch.
    """
    fcn_model.eval() # Put in eval mode (disables batchnorm/dropout) !
    
    losses = []
    mean_iou_scores = []
    accuracy = []

    with torch.no_grad(): # we don't need to calculate the gradient in the validation/testing

        for iter, (inputs, labels) in enumerate(val_loader):

            inputs =  inputs.to(device)
            labels =  labels.to(device)
            
            outputs = fcn_model(inputs)
            
            loss = criterion(outputs, labels.long())
            losses.append(loss.cpu().item())

            iou_score = iou(outputs, labels)
            mean_iou_scores.append(iou_score.cpu().item())

            acc = pixel_acc(outputs, labels)
            accuracy.append(acc.cpu().item())

    if epoch != -1:
        print(f"Loss at epoch {epoch}: {np.mean(losses)}")
        print(f"IoU at epoch {epoch}: {np.mean(mean_iou_scores)}")
        print(f"Pixel Accuracy at epoch {epoch}: {np.mean(accuracy)}")
    else:
        print(f"Initial Loss: {np.mean(losses)}")
        print(f"Initial IoU {np.mean(mean_iou_scores)}")
        print(f"Initial Pixel Accuracy {np.mean(accuracy)}")

    fcn_model.train() #TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!

    return np.mean(mean_iou_scores), np.mean(losses)


def modelTest(fcn_model):
    """
    Test the deep learning model using a test dataset.

    - Load the model with the best weights.
    - Set the model to evaluation mode.
    - Iterate over the test data loader:
        - Perform forward pass and compute loss.
        - Accumulate loss, IoU scores, and pixel accuracy.
    - Print average loss, IoU, and pixel accuracy for the test data.
    - Switch model back to training mode.

    Returns:
        None. Outputs average test metrics to the console.
    """
    fcn_model.eval() # Put in eval mode (disables batchnorm/dropout) !
    
    losses = []
    mean_iou_scores = []
    accuracy = []

    with torch.no_grad(): # we don't need to calculate the gradient in the validation/testing

        for iter, (inputs, labels) in enumerate(test_loader):

            inputs =  inputs.to(device)
            labels =  labels.to(device)
            
            outputs = fcn_model(inputs)
            
            loss = criterion(outputs, labels.long())
            losses.append(loss.cpu().item())

            iou_score = iou(outputs, labels)
            mean_iou_scores.append(iou_score.cpu().item())

            acc = pixel_acc(outputs, labels)
            accuracy.append(acc.cpu().item())


    print(f"Test Loss: {np.mean(losses)}")
    print(f"Test IoU: {np.mean(mean_iou_scores)}")
    print(f"Test Pixel acc: {np.mean(accuracy)}")

    fcn_model.train() #TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!

    return np.mean(mean_iou_scores)


def exportModel(inputs):    
    """
    Export the output of the model for given inputs.

    - Set the model to evaluation mode.
    - Load the model with the best saved weights.
    - Perform a forward pass with the model to get output.
    - Switch model back to training mode.

    Args:
        inputs: Input data to the model.

    Returns:
        Output from the model for the given inputs.
    """
    
    # fcn_model = FCN(n_class=21)
    fcn_model = FCN(n_class=21)
    fcn_model.to(device)
    fcn_model.load_state_dict(torch.load(best_model_path, weights_only=True))
    
    fcn_model.eval() # Put in eval mode (disables batchnorm/dropout) !
    
    inputs = inputs.to(device)
    
    output_image = fcn_model(inputs)
    
    fcn_model.train()  #TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!
    
    return output_image

if __name__ == "__main__":
    print("Accuracy before training: ")
    val()  # show the accuracy before training
    print("\n Starting training.")
    model = train()
    modelTest(model)
    
    

    # housekeeping
    gc.collect()
    torch.cuda.empty_cache()
