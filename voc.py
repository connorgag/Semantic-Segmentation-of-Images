import os
from PIL import Image
from torch.utils import data
from torchvision.transforms.v2 import RandomHorizontalFlip, RandomVerticalFlip, RandomResizedCrop
import random
import torchvision.transforms as standard_transforms

num_classes = 21
ignore_label = 255
root = './data'

'''
color map
0=background, 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle # 6=bus, 7=car, 8=cat, 9=chair, 10=cow, 11=diningtable,
12=dog, 13=horse, 14=motorbike, 15=person # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
'''


#Feel free to convert this palette to a map
palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128,
           128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128,
           64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128]  #3 values- R,G,B for every class. First 3 values for class 0, next 3 for
#class 1 and so on......


def make_dataset(mode):
    """
    Creates a list of tuples (image_path, mask_path) for a given dataset mode.

    Asserts that the mode is one of 'train', 'val', or 'test'.
    Based on the mode, it reads corresponding image and mask paths from the VOC dataset.
    For each image, it pairs its path with the corresponding mask path.

    TODO: Make similar for val and test set.
    data_list for train is with name train.txt,
    data_list for validation is with name trainval.txt,
    data_list for test is with name val.txt.

    Args:
        mode (str): The mode of the dataset, either 'train', 'val', or 'test'.

    Returns:
        list of tuples: Each tuple contains paths (image_path, mask_path).
    """
    assert mode in ['train', 'val', 'test']
    items = []
    if mode == 'train':
        img_path = os.path.join(root, 'VOCdevkit', 'VOC2012', 'JPEGImages')
        mask_path = os.path.join(root, 'VOCdevkit', 'VOC2012', 'SegmentationClass')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'VOCdevkit', 'VOC2012', 'ImageSets', 'Segmentation', 'train.txt')).readlines()]
        for it in data_list:
            item = (os.path.join(img_path, it + '.jpg'), os.path.join(mask_path, it + '.png'))
            items.append(item)
    elif mode == 'val':
        img_path = os.path.join(root, 'VOCdevkit', 'VOC2012', 'JPEGImages')
        mask_path = os.path.join(root, 'VOCdevkit', 'VOC2012', 'SegmentationClass')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'VOCdevkit', 'VOC2012', 'ImageSets', 'Segmentation', 'val.txt')).readlines()]
        for it in data_list:
            item = (os.path.join(img_path, it + '.jpg'), os.path.join(mask_path, it + '.png'))
            items.append(item)
    else:
        img_path = os.path.join(root, 'VOCdevkit', 'VOC2012', 'JPEGImages')
        mask_path = os.path.join(root, 'VOCdevkit', 'VOC2012', 'SegmentationClass')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'VOCdevkit', 'VOC2012', 'ImageSets', 'Segmentation', 'val.txt')).readlines()]
        for it in data_list:
            item = (os.path.join(img_path, it + '.jpg'), os.path.join(mask_path, it + '.png'))
            items.append(item)
        
    return items


def apply_random_transformation(image, mask, transformations):
    for input_tranformation, percent in transformations:
        if random.random() <= percent:
            if input_tranformation == 'crop':
                crop_size = (224, 224)
                scale = (0.9, 1.0)  # Fraction of original image area to be used
                ratio = (3/4, 4/3)   # Aspect ratio range
                
                # Get random crop parameters
                top, left, height, width = standard_transforms.RandomResizedCrop.get_params(image, scale=scale, ratio=ratio)
                
                if mask.dim() == 2:
                    mask = mask.unsqueeze(0) 

                image = standard_transforms.functional.resized_crop(image, top, left, height, width, size=(224, 224), interpolation= standard_transforms.functional.InterpolationMode.NEAREST)
                mask = standard_transforms.functional.resized_crop(mask, top, left, height, width, size=(224, 224), interpolation= standard_transforms.functional.InterpolationMode.NEAREST)

                # image = standard_transforms.functional.crop(image, top, left, height, width)
                # mask = standard_transforms.functional.crop(mask, top, left, height, width)
                
                mask = mask.squeeze()
                
            elif input_tranformation == 'horizontal_flip':
                image = RandomHorizontalFlip(p=1.0)(image)
                mask = RandomHorizontalFlip(p=1.0)(mask)
                
            elif input_tranformation == 'vertical_flip':
                image = RandomVerticalFlip(p=1.0)(image)
                mask = RandomVerticalFlip(p=1.0)(mask)

            else:
                print("Transformation not found")
            
    return image, mask

    
class VOC(data.Dataset):
    """
    A custom dataset class for VOC dataset.
    Maintain the structure of this so that it is easily compatible with Pytorch's dataloader.

    - Resizes images and masks to a specified width and height.
    - Implements methods to get dataset items and dataset length.

    - TIP: You may add an additional argument for common transformation for both the image and mask
           to help with data augmentation in part 4.c

    Args:
        mode (str): Mode of the dataset ('train', 'val', etc.).
        transform (callable, optional): Transform to be applied to the images.
        target_transform (callable, optional): Transform to be applied to the masks.
    """

    def __init__(self, mode, transform=None, target_transform=None, augmentations=None):
        self.imgs = make_dataset(mode)
        print(f"Processing data for {mode} data. Found {len(self.imgs)} images.")

        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform
        self.width = 224
        self.height = 224
        self.augmentations = augmentations

        for index in range(len(self.imgs)):
            img_path, mask_path = self.imgs[index]
            img = Image.open(img_path).convert('RGB').resize((self.width, self.height))
            mask = Image.open(mask_path).resize((self.width, self.height))
            
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                mask = self.target_transform(mask)
    
            # Apply any augmentations:
            if self.augmentations is not None:
                img, mask = apply_random_transformation(img, mask, self.augmentations)
                                          
            mask[mask==ignore_label]=0
            
            self.imgs[index] = [img, mask]
            
    
    def __getitem__(self, index):

        return self.imgs[index][0], self.imgs[index][1]

    def __len__(self):
        return len(self.imgs)