import numpy as np
import cv2
import torch
from datasets.base import BaseWasteDataset

class SyntheticWasteDataset(BaseWasteDataset):
    """
    Generates synthetic waste classification and segmentation data.
    Draws random shapes (circles, rectangles, polygons) to simulate waste objects
    on textured backgrounds.
    """
    def __init__(self, num_samples=100, img_size=224, num_classes=6, transforms=None):
        super().__init__(transforms=transforms)
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_classes = num_classes
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        # Create background (can be solid, gradient, or random noise)
        np.random.seed(idx + 9999) # Keep deterministic per index
        
        # Background: noise or gradient
        bg_type = np.random.choice(['noise', 'gradient', 'solid'])
        if bg_type == 'solid':
            bg_color = np.random.randint(180, 240, size=3, dtype=np.uint8)
            img = np.ones((self.img_size, self.img_size, 3), dtype=np.uint8) * bg_color
        elif bg_type == 'gradient':
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            color1 = np.random.randint(180, 240, size=3)
            color2 = np.random.randint(140, 200, size=3)
            for y in range(self.img_size):
                ratio = y / self.img_size
                img[y, :, :] = (1 - ratio) * color1 + ratio * color2
        else: # noise
            img = np.random.randint(150, 230, size=(self.img_size, self.img_size, 3), dtype=np.uint8)
            # smooth the noise slightly
            img = cv2.GaussianBlur(img, (5, 5), 0)

        mask = np.zeros((self.img_size, self.img_size), dtype=np.int32)
        
        # Decide active label for this image (1 to num_classes-1)
        # 0: background (not a waste class in our model), or we can treat classification classes as 0 to 5.
        # Let's say classification label is 0..5
        label = np.random.randint(0, self.num_classes)
        
        # Add a shape corresponding to that label (so mask values match the label+1)
        # Wait, classification label: 0 to 5.
        # Segmentation mask: 0 (background), and 1 to 6 (cardboard, glass, metal, paper, plastic, trash)
        mask_val = label + 1
        
        # Draw 1 or 2 shapes
        num_shapes = np.random.randint(1, 3)
        for _ in range(num_shapes):
            shape_type = np.random.choice(['circle', 'rectangle', 'polygon'])
            color = np.random.randint(20, 150, size=3).tolist()
            
            center_x = np.random.randint(int(self.img_size * 0.25), int(self.img_size * 0.75))
            center_y = np.random.randint(int(self.img_size * 0.25), int(self.img_size * 0.75))
            size = np.random.randint(int(self.img_size * 0.15), int(self.img_size * 0.35))
            
            if shape_type == 'circle':
                cv2.circle(img, (center_x, center_y), size, color, -1)
                cv2.circle(mask, (center_x, center_y), size, mask_val, -1)
            elif shape_type == 'rectangle':
                x1 = center_x - size
                y1 = center_y - size
                x2 = center_x + size
                y2 = center_y + size
                cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
                cv2.rectangle(mask, (x1, y1), (x2, y2), mask_val, -1)
            else: # polygon
                pts = np.array([
                    [center_x - np.random.randint(0, size), center_y - np.random.randint(0, size)],
                    [center_x + np.random.randint(0, size), center_y - np.random.randint(0, size)],
                    [center_x + np.random.randint(0, size), center_y + np.random.randint(0, size)],
                    [center_x - np.random.randint(0, size), center_y + np.random.randint(0, size)],
                ], dtype=np.int32)
                cv2.fillPoly(img, [pts], color)
                cv2.fillPoly(mask, [pts], mask_val)

        # Apply standard augmentations
        img_t, mask_t = self._apply_transforms(img, mask)
        
        # Return format: image, mask, label
        return img_t, mask_t, torch.tensor(label, dtype=torch.long)
