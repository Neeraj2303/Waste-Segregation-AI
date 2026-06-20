import os
import glob
import cv2
import numpy as np
import torch
from datasets.base import BaseWasteDataset

class TrashNetDataset(BaseWasteDataset):
    """
    Dataset loader for TrashNet.
    Structure:
    trashnet_path/
      cardboard/
      glass/
      metal/
      paper/
      plastic/
      trash/
    """
    def __init__(self, root_dir, class_names=None, transforms=None):
        super().__init__(transforms=transforms)
        self.root_dir = root_dir
        if class_names is None:
            self.class_names = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
        else:
            self.class_names = class_names
            
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.image_paths = []
        self.labels = []
        
        if os.path.exists(root_dir):
            for class_name in self.class_names:
                class_path = os.path.join(root_dir, class_name)
                if os.path.isdir(class_path):
                    # Find all images (jpg, jpeg, png)
                    patterns = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
                    class_images = []
                    for pattern in patterns:
                        class_images.extend(glob.glob(os.path.join(class_path, pattern)))
                    
                    class_images = list(set(os.path.normpath(p) for p in class_images))
                    for img_path in class_images:
                        self.image_paths.append(img_path)
                        self.labels.append(self.class_to_idx[class_name])
        else:
            print(f"Warning: TrashNet path '{root_dir}' not found. Dataset will be empty.")
            
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # TrashNet classification has no semantic masks, return empty mask
        h, w, _ = img.shape
        mask = np.zeros((h, w), dtype=np.int64)
        
        img_t, mask_t = self._apply_transforms(img, mask)
        
        return img_t, mask_t, torch.tensor(label, dtype=torch.long)
