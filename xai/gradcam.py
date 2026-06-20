import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping) for CNN branch.
    Hooks into activations and gradients of a target layer to compute heatmaps.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.handlers = []
        self._register_hooks()
        
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
            
        self.handlers.append(self.target_layer.register_forward_hook(forward_hook))
        # Support older PyTorch versions and newer ones (register_full_backward_hook)
        if hasattr(self.target_layer, 'register_full_backward_hook'):
            self.handlers.append(self.target_layer.register_full_backward_hook(backward_hook))
        else:
            self.handlers.append(self.target_layer.register_backward_hook(backward_hook))
            
    def generate_heatmap(self, input_tensor, class_idx=None, device="cuda"):
        """
        Generate a Grad-CAM heatmap for a given input and target class.
        """
        self.model.eval()
        input_tensor = input_tensor.to(device)
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
            
        # Run forward pass
        outputs = self.model(input_tensor)
        cls_logits = outputs['cls_logits']
        
        if class_idx is None:
            class_idx = torch.argmax(cls_logits, dim=1).item()
            
        # Run backward pass for target class
        self.model.zero_grad()
        score = cls_logits[0, class_idx]
        score.backward()
        
        # Calculate weights from gradients (average pooling over spatial dimensions)
        # self.gradients: [B, C_feat, H_feat, W_feat]
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True) # [B, C_feat, 1, 1]
        
        # Weighted sum of activations
        # self.activations: [B, C_feat, H_feat, W_feat]
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True) # [B, 1, H_feat, W_feat]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize between 0 and 1
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = cam - cam_min
            
        # Upsample to input shape [H, W]
        h_in, w_in = input_tensor.shape[2:]
        cam = F.interpolate(cam, size=(h_in, w_in), mode='bilinear', align_corners=False)
        
        return cam.squeeze().cpu().numpy()

    def remove_hooks(self):
        for handler in self.handlers:
            handler.remove()
        self.handlers.clear()
        
    def __del__(self):
        self.remove_hooks()
