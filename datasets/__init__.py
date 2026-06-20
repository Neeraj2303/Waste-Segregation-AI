import os
import torch
from torch.utils.data import DataLoader, random_split
from datasets.base import get_transforms
from datasets.synthetic import SyntheticWasteDataset
from datasets.trashnet import TrashNetDataset
from datasets.taco import TACODataset
from datasets.zerowaste import ZeroWasteDataset

def get_datasets(config, stage="stage1"):
    """
    Load datasets and return DataLoaders for train, val, and test.
    Handles stage-specific data loading and automatic synthetic fallbacks.
    """
    img_size = config['dataset']['img_size']
    batch_size = config['dataset']['batch_size']
    num_workers = config['dataset']['num_workers']
    num_classes = config['dataset']['num_classes']
    class_names = config['dataset']['class_names']
    
    use_synthetic = config['dataset']['synthetic']
    
    train_transform = get_transforms(img_size=img_size, is_train=True)
    val_transform = get_transforms(img_size=img_size, is_train=False)
    
    train_dataset = None
    val_dataset = None
    test_dataset = None
    
    if stage == "stage1":
        # TrashNet
        path = config['dataset']['trashnet_path']
        if use_synthetic or not os.path.exists(path):
            print("INFO: Using Synthetic dataset for Stage 1 (TrashNet classification)")
            train_dataset = SyntheticWasteDataset(num_samples=200, img_size=img_size, num_classes=num_classes, transforms=train_transform)
            val_dataset = SyntheticWasteDataset(num_samples=50, img_size=img_size, num_classes=num_classes, transforms=val_transform)
            test_dataset = SyntheticWasteDataset(num_samples=50, img_size=img_size, num_classes=num_classes, transforms=val_transform)
        else:
            full_dataset = TrashNetDataset(root_dir=path, class_names=class_names, transforms=None)
            if len(full_dataset) == 0:
                print("WARNING: TrashNet dataset is empty! Falling back to synthetic.")
                train_dataset = SyntheticWasteDataset(num_samples=200, img_size=img_size, num_classes=num_classes, transforms=train_transform)
                val_dataset = SyntheticWasteDataset(num_samples=50, img_size=img_size, num_classes=num_classes, transforms=val_transform)
                test_dataset = SyntheticWasteDataset(num_samples=50, img_size=img_size, num_classes=num_classes, transforms=val_transform)
            else:
                # Split 80/10/10
                generator = torch.Generator().manual_seed(config['seed'])
                n = len(full_dataset)
                n_train = int(0.8 * n)
                n_val = int(0.1 * n)
                n_test = n - n_train - n_val
                
                train_subset, val_subset, test_subset = random_split(full_dataset, [n_train, n_val, n_test], generator=generator)
                
                # Apply appropriate transforms to subsets (by wrapping them)
                train_dataset = SubsetWrapper(train_subset, train_transform)
                val_dataset = SubsetWrapper(val_subset, val_transform)
                test_dataset = SubsetWrapper(test_subset, val_transform)
                
    elif stage == "stage2":
        # TACO
        path = config['dataset']['taco_path']
        if use_synthetic or not os.path.exists(path):
            print("INFO: Using Synthetic dataset for Stage 2 (TACO object classification + segmentation)")
            train_dataset = SyntheticWasteDataset(num_samples=150, img_size=img_size, num_classes=num_classes, transforms=train_transform)
            val_dataset = SyntheticWasteDataset(num_samples=40, img_size=img_size, num_classes=num_classes, transforms=val_transform)
            test_dataset = SyntheticWasteDataset(num_samples=40, img_size=img_size, num_classes=num_classes, transforms=val_transform)
        else:
            full_dataset = TACODataset(root_dir=path, class_names=class_names, transforms=None)
            if len(full_dataset) == 0:
                print("WARNING: TACO dataset is empty! Falling back to synthetic.")
                train_dataset = SyntheticWasteDataset(num_samples=150, img_size=img_size, num_classes=num_classes, transforms=train_transform)
                val_dataset = SyntheticWasteDataset(num_samples=40, img_size=img_size, num_classes=num_classes, transforms=val_transform)
                test_dataset = SyntheticWasteDataset(num_samples=40, img_size=img_size, num_classes=num_classes, transforms=val_transform)
            else:
                # Split 80/10/10
                generator = torch.Generator().manual_seed(config['seed'])
                n = len(full_dataset)
                n_train = int(0.8 * n)
                n_val = int(0.1 * n)
                n_test = n - n_train - n_val
                
                train_subset, val_subset, test_subset = random_split(full_dataset, [n_train, n_val, n_test], generator=generator)
                
                train_dataset = SubsetWrapper(train_subset, train_transform)
                val_dataset = SubsetWrapper(val_subset, val_transform)
                test_dataset = SubsetWrapper(test_subset, val_transform)
                
    elif stage == "stage3":
        # ZeroWaste
        path = config['dataset']['zerowaste_path']
        if use_synthetic or not os.path.exists(path):
            print("INFO: Using Synthetic dataset for Stage 3 (ZeroWaste segmentation fine-tuning)")
            train_dataset = SyntheticWasteDataset(num_samples=150, img_size=img_size, num_classes=num_classes, transforms=train_transform)
            val_dataset = SyntheticWasteDataset(num_samples=40, img_size=img_size, num_classes=num_classes, transforms=val_transform)
            test_dataset = SyntheticWasteDataset(num_samples=40, img_size=img_size, num_classes=num_classes, transforms=val_transform)
        else:
            train_raw = ZeroWasteDataset(root_dir=path, split="train", class_names=class_names, transforms=None)
            val_raw = ZeroWasteDataset(root_dir=path, split="val", class_names=class_names, transforms=None)
            test_raw = ZeroWasteDataset(root_dir=path, split="test", class_names=class_names, transforms=None)
            
            if len(train_raw) == 0:
                print("WARNING: ZeroWaste dataset is empty! Falling back to synthetic.")
                train_dataset = SyntheticWasteDataset(num_samples=150, img_size=img_size, num_classes=num_classes, transforms=train_transform)
                val_dataset = SyntheticWasteDataset(num_samples=40, img_size=img_size, num_classes=num_classes, transforms=val_transform)
                test_dataset = SyntheticWasteDataset(num_samples=40, img_size=img_size, num_classes=num_classes, transforms=val_transform)
            else:
                train_dataset = SubsetWrapper(train_raw, train_transform)
                val_dataset = SubsetWrapper(val_raw, val_transform)
                test_dataset = SubsetWrapper(test_raw, val_transform)
                
    else:
        raise ValueError(f"Unknown training stage: {stage}")
        
    # Create DataLoaders
    # Pin memory for faster GPU transfer if cuda is active
    pin_memory = (config['device'] == 'cuda')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    
    return train_loader, val_loader, test_loader

class SubsetWrapper(torch.utils.data.Dataset):
    """
    Wraps a subset of a dataset and applies custom transformations to the items
    at load-time (since PyTorch random_split subset doesn't allow changing transforms).
    """
    def __init__(self, subset, transforms):
        self.subset = subset
        self.transforms = transforms
        
    def __len__(self):
        return len(self.subset)
        
    def __getitem__(self, idx):
        # We need to temporarily disable transforms on the base dataset if any
        # and apply our own transforms here.
        # But we designed TrashNet, TACO, ZeroWaste to support None transforms.
        # Let's read raw image and mask by calling base getitem, but we want
        # to ensure it returns raw np array first.
        # Let's inspect the underlying dataset type.
        if hasattr(self.subset, 'dataset'):
            dataset = self.subset.dataset
            actual_idx = self.subset.indices[idx]
        else:
            dataset = self.subset
            actual_idx = idx
        
        # Save original transform
        orig_transform = dataset.transforms
        dataset.transforms = None
        
        try:
            # Get raw items
            img_t, mask_t, label = dataset[actual_idx]
            # Since transforms is None, img_t is raw numpy array, mask_t is raw numpy mask
            # Wait, our dataset class returns img_t and mask_t after applying self._apply_transforms
            # If self.transforms is None in _apply_transforms, it will return:
            # A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD) and ToTensorV2 is NOT applied.
            # Wait, let's verify what _apply_transforms does if self.transforms is None:
            # It just converts mask to tensor. It doesn't normalize or turn image to tensor.
            # So if we set dataset.transforms = None, it will return raw numpy array for image.
            # That is perfect!
            raw_img = img_t
            raw_mask = mask_t
            
            # If they are already tensors (e.g. if dataset is something else), convert back to numpy
            if isinstance(raw_img, torch.Tensor):
                raw_img = raw_img.permute(1, 2, 0).numpy()
            if isinstance(raw_mask, torch.Tensor):
                raw_mask = raw_mask.numpy()
                
            # Apply our subset transform
            transformed = self.transforms(image=raw_img, mask=raw_mask)
            img_tensor = transformed['image']
            mask_tensor = transformed['mask']
            
            if not isinstance(mask_tensor, torch.Tensor):
                mask_tensor = torch.tensor(mask_tensor, dtype=torch.long)
            else:
                mask_tensor = mask_tensor.long()
                
            return img_tensor, mask_tensor, label
        finally:
            # Restore original transform
            dataset.transforms = orig_transform
