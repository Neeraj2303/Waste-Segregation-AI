import os
import json
import cv2
import numpy as np
import torch
from datasets.base import BaseWasteDataset

class TACODataset(BaseWasteDataset):
    """
    Dataset loader for TACO.
    Expects coco-style annotations JSON file and an images directory.
    """
    def __init__(self, root_dir, ann_file="annotations.json", class_names=None, transforms=None):
        super().__init__(transforms=transforms)
        self.root_dir = root_dir
        if class_names is None:
            self.class_names = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
        else:
            self.class_names = class_names
            
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.image_paths = []
        self.annotations = []
        self.image_ids = []
        self.coco_data = None
        
        ann_path = os.path.join(root_dir, ann_file)
        if os.path.exists(ann_path):
            with open(ann_path, 'r') as f:
                self.coco_data = json.load(f)
                
            # Create category mapping to target classes
            cat_map = self._create_category_mapping(self.coco_data['categories'])
            
            # Map image_id to annotations
            img_to_anns = {}
            for ann in self.coco_data['annotations']:
                img_id = ann['image_id']
                if img_id not in img_to_anns:
                    img_to_anns[img_id] = []
                img_to_anns[img_id].append(ann)
                
            # Load images
            for img_info in self.coco_data['images']:
                img_id = img_info['id']
                file_name = img_info['file_name']
                # Join path
                img_path = os.path.join(root_dir, file_name)
                
                # Check if image has annotations and file exists
                if img_id in img_to_anns and os.path.exists(img_path):
                    self.image_paths.append(img_path)
                    self.image_ids.append(img_id)
                    self.annotations.append(img_to_anns[img_id])
                    
            self.cat_map = cat_map
        else:
            print(f"Warning: TACO annotations file '{ann_path}' not found. Dataset will be empty.")
            
    def _create_category_mapping(self, coco_categories):
        """
        Map 60 TACO categories to our 6 target categories:
        0: cardboard, 1: glass, 2: metal, 3: paper, 4: plastic, 5: trash
        """
        cat_map = {}
        for cat in coco_categories:
            name = cat['name'].lower()
            cat_id = cat['id']
            
            if any(k in name for k in ['cardboard', 'carton', 'corrugated']):
                target_idx = 0
            elif any(k in name for k in ['glass', 'jar', 'bottle glass']):
                target_idx = 1
            elif any(k in name for k in ['can', 'metal', 'foil', 'aluminium', 'tin', 'steel']):
                target_idx = 2
            elif any(k in name for k in ['paper', 'newspaper', 'magazine', 'receipt', 'book']):
                target_idx = 3
            elif any(k in name for k in ['plastic', 'bottle', 'cup', 'tub', 'bag', 'film', 'wrapper', 'polystyrene', 'styrofoam', 'pet', 'pe']):
                target_idx = 4
            else:
                target_idx = 5 # default is trash
                
            cat_map[cat_id] = target_idx
            
        return cat_map
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        anns = self.annotations[idx]
        
        # Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        
        # Create segmentation mask: 0 is background, target_idx + 1 is the class mask
        mask = np.zeros((h, w), dtype=np.int32)
        
        # Track classes present in this image to determine classification label
        active_classes = []
        
        for ann in anns:
            cat_id = ann['category_id']
            if cat_id not in self.cat_map:
                continue
                
            target_idx = self.cat_map[cat_id]
            mask_val = target_idx + 1 # 1 to 6
            active_classes.append(target_idx)
            
            # Rasterize polygons
            if 'segmentation' in ann:
                segs = ann['segmentation']
                # COCO polygons are lists of floats
                if isinstance(segs, list):
                    for poly in segs:
                        if len(poly) >= 6: # Need at least 3 points (x,y)
                            poly_pts = np.array(poly, dtype=np.int32).reshape((-1, 2))
                            cv2.fillPoly(mask, [poly_pts], mask_val)
                elif isinstance(segs, dict) and 'counts' in segs:
                    # RLE encoding - skip or decode if needed. For TACO it is mostly polygons.
                    pass
                    
        # Classification label for multi-task classification
        # If there are active classes, pick the dominant one (mode) or first one.
        if active_classes:
            # We can use the most frequent class, or if multiple, the one with largest mask area
            label = max(set(active_classes), key=active_classes.count)
        else:
            label = 5 # Fallback to trash
            
        img_t, mask_t = self._apply_transforms(img, mask)
        
        return img_t, mask_t, torch.tensor(label, dtype=torch.long)
