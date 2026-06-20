import os
import glob
import cv2
import numpy as np
import torch
from datasets.base import BaseWasteDataset

class ZeroWasteDataset(BaseWasteDataset):
    """
    Dataset loader for ZeroWaste.
    Structure:
      root_dir/
        images/
          train/
          val/
          test/
        masks/
          train/
          val/
          test/
    """
    def __init__(self, root_dir, split="train", class_names=None, transforms=None):
        super().__init__(transforms=transforms)
        self.root_dir = root_dir
        self.split = split
        if class_names is None:
            self.class_names = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
        else:
            self.class_names = class_names
            
        self.image_paths = []
        self.mask_paths = []
        
        img_dir = os.path.join(root_dir, split, "data")
        mask_dir = os.path.join(root_dir, split, "sem_seg")
        
        if os.path.exists(img_dir) and os.path.exists(mask_dir):
            patterns = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
            img_files = []
            for pattern in patterns:
                img_files.extend(glob.glob(os.path.join(img_dir, pattern)))
                
            img_files = list(set(os.path.normpath(p) for p in img_files))
            for img_path in img_files:
                base_name = os.path.basename(img_path)
                # Assume mask has same base name or similar (often png)
                name_without_ext = os.path.splitext(base_name)[0]
                
                # Check for mask with png or same ext
                mask_path_png = os.path.join(mask_dir, f"{name_without_ext}.png")
                mask_path_jpg = os.path.join(mask_dir, f"{name_without_ext}.jpg")
                
                if os.path.exists(mask_path_png):
                    self.image_paths.append(img_path)
                    self.mask_paths.append(mask_path_png)
                elif os.path.exists(mask_path_jpg):
                    self.image_paths.append(img_path)
                    self.mask_paths.append(mask_path_jpg)
        else:
            print(f"Warning: ZeroWaste directories for split '{split}' not found under '{root_dir}'. Dataset will be empty.")
            
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load mask (1-channel semantic labels)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # ZeroWaste classes: 0: background, 1: cardboard, 2: metal, 3: rigid_plastic, 4: soft_plastic
        # We map them to our target classes:
        # Target classes: ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
        # So:
        # ZeroWaste 1 -> cardboard (target_idx 0, mask_val 1)
        # ZeroWaste 2 -> metal (target_idx 2, mask_val 3)
        # ZeroWaste 3 -> plastic (target_idx 4, mask_val 5)
        # ZeroWaste 4 -> plastic (target_idx 4, mask_val 5)
        # Others/Background -> background (0)
        
        h, w = mask.shape
        target_mask = np.zeros((h, w), dtype=np.int64)
        
        target_mask[mask == 1] = 1 # cardboard
        target_mask[mask == 2] = 3 # metal
        target_mask[mask == 3] = 5 # plastic
        target_mask[mask == 4] = 5 # plastic
        
        # Classification label: find the dominant class present in mask (excluding background)
        unique_vals, counts = np.unique(target_mask, return_counts=True)
        # Exclude background (0)
        valid_indices = unique_vals != 0
        unique_vals = unique_vals[valid_indices]
        counts = counts[valid_indices]
        
        if len(unique_vals) > 0:
            dominant_mask_val = unique_vals[np.argmax(counts)]
            label = dominant_mask_val - 1 # map back to target_idx (0..5)
        else:
            label = 5 # Default is trash if background only
            
        img_t, mask_t = self._apply_transforms(img, target_mask)
        
        return img_t, mask_t, torch.tensor(label, dtype=torch.long)
