import torch
import numpy as np
import scipy.stats
import cv2

def compute_xai_metrics(model, image_tensor, heatmap, target_class, grid_size=8, device="cuda"):
    """
    Computes Faithfulness, Insertion, and Deletion scores for an explanation heatmap.
    Uses patch-level masking to speed up computation.
    
    Args:
        model: PyTorch model
        image_tensor: [3, H, W] tensor
        heatmap: [H, W] numpy array normalized to [0, 1]
        target_class: Target class index
        grid_size: Size of grid to split image (e.g., 8 means 8x8 = 64 patches)
    Returns:
        dict with keys: 'faithfulness', 'insertion_auc', 'deletion_auc'
    """
    model.eval()
    H, W = image_tensor.shape[1:]
    
    # 1. Prepare base images
    img_np = image_tensor.cpu().numpy() # [3, H, W]
    mean_color = img_np.mean(axis=(1, 2), keepdims=True) # [3, 1, 1] background filler
    
    # Resize heatmap to match grid size to get patch-level attributions
    heatmap_grid = cv2.resize(heatmap, (grid_size, grid_size), interpolation=cv2.INTER_AREA)
    flat_attributions = heatmap_grid.flatten()
    
    # Get indices of patches sorted by importance (descending)
    sorted_patch_indices = np.argsort(flat_attributions)[::-1]
    
    # Helper to generate mask for patch coordinates
    def get_patch_mask(patch_idx):
        row = patch_idx // grid_size
        col = patch_idx % grid_size
        h_step = H // grid_size
        w_step = W // grid_size
        
        mask = np.zeros((H, W), dtype=bool)
        mask[row*h_step:(row+1)*h_step, col*w_step:(col+1)*w_step] = True
        return mask

    # Get baseline prediction probability (original image)
    with torch.no_grad():
        out = model(image_tensor.unsqueeze(0).to(device))
        probs = torch.softmax(out['cls_logits'], dim=1)[0]
        p_orig = probs[target_class].item()
        l_orig = out['cls_logits'][0, target_class].item()

    # --- 1. FAITHFULNESS (Correlation) ---
    # Mask each patch one-by-one and record changes in logit
    logit_drops = []
    patch_values = []
    
    for idx in range(grid_size * grid_size):
        mask = get_patch_mask(idx)
        # Create image with this single patch masked
        masked_img = img_np.copy()
        masked_img[:, mask] = mean_color
        
        with torch.no_grad():
            out_masked = model(torch.tensor(masked_img).unsqueeze(0).to(device))
            l_masked = out_masked['cls_logits'][0, target_class].item()
            
        logit_drop = l_orig - l_masked
        logit_drops.append(logit_drop)
        patch_values.append(flat_attributions[idx])
        
    # Compute Pearson Correlation
    # Pearson correlation measures if higher attribution values correlate with larger drops in logit
    correlation, _ = scipy.stats.pearsonr(patch_values, logit_drops)
    if np.isnan(correlation):
        correlation = 0.0
        
    # --- 2. INSERTION SCORE (AUC) ---
    # Start with blank image (mean-filled) and insert patches in descending order of importance
    insertion_img = np.repeat(mean_color, H*W, axis=1).reshape(3, H, W).copy()
    insertion_probs = []
    
    # Step 0: Blank image probability
    with torch.no_grad():
        out_ins = model(torch.tensor(insertion_img).unsqueeze(0).to(device))
        p_ins = torch.softmax(out_ins['cls_logits'], dim=1)[0, target_class].item()
        insertion_probs.append(p_ins)
        
    for patch_idx in sorted_patch_indices:
        mask = get_patch_mask(patch_idx)
        # Insert patch content from original image
        insertion_img[:, mask] = img_np[:, mask]
        
        with torch.no_grad():
            out_ins = model(torch.tensor(insertion_img).unsqueeze(0).to(device))
            p_ins = torch.softmax(out_ins['cls_logits'], dim=1)[0, target_class].item()
            insertion_probs.append(p_ins)
            
    # Calculate AUC
    insertion_auc = np.mean(insertion_probs)
    
    # --- 3. DELETION SCORE (AUC) ---
    # Start with original image and delete patches in descending order of importance
    deletion_img = img_np.copy()
    deletion_probs = []
    
    # Step 0: Original image probability
    deletion_probs.append(p_orig)
    
    for patch_idx in sorted_patch_indices:
        mask = get_patch_mask(patch_idx)
        # Delete patch content (replace with mean color)
        deletion_img[:, mask] = mean_color
        
        with torch.no_grad():
            out_del = model(torch.tensor(deletion_img).unsqueeze(0).to(device))
            p_del = torch.softmax(out_del['cls_logits'], dim=1)[0, target_class].item()
            deletion_probs.append(p_del)
            
    # Calculate AUC
    deletion_auc = np.mean(deletion_probs)
    
    return {
        'faithfulness': correlation,
        'insertion_auc': insertion_auc,
        'deletion_auc': deletion_auc
    }
