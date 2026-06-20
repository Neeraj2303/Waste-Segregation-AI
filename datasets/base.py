import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet normalization statistics
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def get_transforms(img_size=224, is_train=True):
    """
    Get Albumentations transforms for train/val.
    Includes rotations, flips, blur, brightness/contrast, weather, and occlusion simulations.
    """
    if is_train:
        return A.Compose([
            A.Resize(img_size, img_size),
            # Flips & Rotations
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0)),
            
            # Blurs
            A.OneOf([
                A.Blur(blur_limit=3, p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
            ], p=0.3),
            
            # Brightness & Contrast
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
            
            # Weather / Lighting simulation (e.g. Shadows, Fog)
            A.OneOf([
                A.RandomShadow(p=1.0),
                # Note: A.RandomFog is deprecated or might have installation issues in some envs, 
                # so shadow + brightness fluctuations are safer, but we can add CLAHE / solarize.
                A.CLAHE(clip_limit=2.0, p=1.0),
            ], p=0.2),
            
            # Occlusion simulation (CoarseDropout / Cutout)
            A.CoarseDropout(max_holes=8, max_height=16, max_width=16, 
                            min_holes=1, min_height=8, min_width=8, 
                            fill_value=0, mask_fill_value=0, p=0.3),
            
            # Normalize and convert to PyTorch Tensor
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])

class BaseWasteDataset(Dataset):
    """
    Base class for waste classification and segmentation datasets.
    """
    def __init__(self, transforms=None):
        self.transforms = transforms

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
        
    def _apply_transforms(self, image, mask=None):
        """
        Apply Albumentations transforms to image and mask.
        """
        if mask is None:
            mask = np.zeros(image.shape[:2], dtype=np.int64)
            
        if self.transforms:
            # Albumentations expects mask to be 2D or 3D
            transformed = self.transforms(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
            
        # If mask is a tensor, ensure it is LongTensor and shape is [H, W]
        if isinstance(mask, torch.Tensor):
            mask = mask.long()
        else:
            mask = torch.tensor(mask, dtype=torch.long)
            
        return image, mask
