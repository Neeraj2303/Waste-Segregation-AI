import torch
import torch.nn.functional as F
import numpy as np
import cv2

class AttentionRollout:
    """
    Attention Rollout for Vision Transformer (ViT) branches.
    Computes cumulative attention flow from the CLS token to the input patches.
    """
    def __init__(self, model, discard_ratio=0.9, head_fusion="mean"):
        """
        Args:
            model: AdaptiveExplainableHybridModel (with ViT branch having attention history)
            discard_ratio: Ratio of lower attention weights to discard (helps denoise)
            head_fusion: How to fuse attention maps across heads ('mean', 'max', 'min')
        """
        self.model = model
        self.discard_ratio = discard_ratio
        self.head_fusion = head_fusion

    def generate_heatmap(self, input_tensor, device="cuda"):
        """
        Generate Attention Rollout map.
        """
        self.model.eval()
        input_tensor = input_tensor.to(device)
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
            
        # Run forward pass (which populates the attention weights in vit_branch)
        with torch.no_grad():
            _ = self.model(input_tensor)
            
        # Get attention weights from all blocks: list of [B, num_heads, N, N]
        # For a single image, B = 1
        attns = self.model.get_vit_attention_maps()
        
        if len(attns) == 0:
            print("WARNING: AttentionRollout received empty attention history.")
            # Return dummy map
            h, w = input_tensor.shape[2:]
            return np.zeros((h, w), dtype=np.float32)
            
        N = attns[0].shape[2] # number of tokens (e.g. 257)
        # Initialize identity matrix
        result = torch.eye(N).to(device)
        
        for attn in attns:
            # attn shape: [1, num_heads, N, N]
            attn = attn[0] # [num_heads, N, N]
            
            # Fuse heads
            if self.head_fusion == "mean":
                attn_fused = attn.mean(dim=0)
            elif self.head_fusion == "max":
                attn_fused, _ = attn.max(dim=0)
            elif self.head_fusion == "min":
                attn_fused, _ = attn.min(dim=0)
            else:
                raise ValueError(f"Unknown head fusion: {self.head_fusion}")
                
            # Discard low attention weights to denoise
            if self.discard_ratio > 0:
                flat = attn_fused.flatten()
                _, indices = torch.sort(flat)
                threshold_idx = int(self.discard_ratio * len(flat))
                threshold = flat[indices[threshold_idx]]
                attn_fused[attn_fused < threshold] = 0
                
            # Add identity matrix for residual connection, re-normalize
            I = torch.eye(N).to(device)
            attn_fused = 0.5 * attn_fused + 0.5 * I
            # Re-normalize rows
            row_sums = attn_fused.sum(dim=-1, keepdim=True)
            attn_fused = attn_fused / row_sums
            
            # Rollout multiplication
            result = torch.matmul(attn_fused, result)
            
        # Extract class-to-patch attention (row 0 represents CLS token, cols 1: are patch tokens)
        cls_to_patches = result[0, 1:] # [N-1]
        
        # Reshape to 2D grid
        grid_size = int(len(cls_to_patches) ** 0.5)
        if grid_size * grid_size != len(cls_to_patches):
            # If patch count is not a perfect square, handle padding/cropping
            # Usually it is (e.g. 16*16 = 256 patches)
            # Find closest square
            grid_size = int(np.floor(len(cls_to_patches) ** 0.5))
            cls_to_patches = cls_to_patches[:grid_size*grid_size]
            
        heatmap = cls_to_patches.reshape(grid_size, grid_size)
        
        # Normalize heatmap between 0 and 1
        h_min = heatmap.min()
        h_max = heatmap.max()
        if h_max > h_min:
            heatmap = (heatmap - h_min) / (h_max - h_min)
        else:
            heatmap = heatmap - h_min
            
        # Convert to numpy and upsample to original image size
        heatmap_np = heatmap.cpu().numpy()
        h_in, w_in = input_tensor.shape[2:]
        
        # Resize to input shape
        heatmap_resized = cv2.resize(heatmap_np, (w_in, h_in), interpolation=cv2.INTER_CUBIC)
        
        return heatmap_resized
