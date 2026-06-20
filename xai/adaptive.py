import torch
import numpy as np
from xai.gradcam import GradCAM
from xai.rollout import AttentionRollout
from xai.metrics import compute_xai_metrics

class AdaptiveXAISelector:
    """
    Adaptive XAI Selector.
    Dynamically routes explanation queries to either Grad-CAM (CNN branch)
    or Attention Rollout (ViT branch) based on either:
      1. Learnable Gated weights (gating weights in the fusion block).
      2. Quantitative evaluation (maximizing Faithfulness).
    """
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
        # Target layer for Grad-CAM is the final deformable block in stage 4 of the CNN
        self.gradcam_layer = model.cnn_branch.layer4[-1].deform_conv
        
        self.gradcam = GradCAM(model, self.gradcam_layer)
        self.rollout = AttentionRollout(model)
        
    def get_explanation_by_gating(self, image_tensor, class_idx=None, device="cuda"):
        """
        Selects explainer dynamically based on the learnable gating weights.
        """
        # Get gating weights from model: [alpha_vit, alpha_cnn]
        with torch.no_grad():
            gating_weights = self.model.get_gating_weights()
            
        alpha_vit = gating_weights[0].item()
        alpha_cnn = gating_weights[1].item()
        
        if alpha_cnn > alpha_vit:
            method_used = "gradcam"
            heatmap = self.gradcam.generate_heatmap(image_tensor, class_idx, device)
        else:
            method_used = "rollout"
            heatmap = self.rollout.generate_heatmap(image_tensor, device)
            
        return heatmap, method_used, {"alpha_vit": alpha_vit, "alpha_cnn": alpha_cnn}

    def select_best_explainer_quantitatively(self, val_loader, num_samples=5, device="cuda"):
        """
        Evaluate both explainers quantitatively on a set of validation samples
        and return the name of the one that achieves higher average Faithfulness.
        """
        print("Evaluating XAI methods quantitatively to select the best explanation method...")
        
        self.model.eval()
        gradcam_faithfulness = []
        rollout_faithfulness = []
        
        count = 0
        for images, masks, labels in val_loader:
            if count >= num_samples:
                break
                
            for idx in range(len(images)):
                if count >= num_samples:
                    break
                    
                image = images[idx]
                label = labels[idx].item()
                
                # Grad-CAM explanation and metrics
                try:
                    gc_map = self.gradcam.generate_heatmap(image, label, device)
                    gc_metrics = compute_xai_metrics(self.model, image, gc_map, label, grid_size=8, device=device)
                    gradcam_faithfulness.append(gc_metrics['faithfulness'])
                except Exception as e:
                    print(f"Error computing Grad-CAM metrics: {e}")
                    
                # Rollout explanation and metrics
                try:
                    ro_map = self.rollout.generate_heatmap(image, device)
                    ro_metrics = compute_xai_metrics(self.model, image, ro_map, label, grid_size=8, device=device)
                    rollout_faithfulness.append(ro_metrics['faithfulness'])
                except Exception as e:
                    print(f"Error computing Rollout metrics: {e}")
                    
                count += 1
                
        mean_gc_f = np.mean(gradcam_faithfulness) if gradcam_faithfulness else 0.0
        mean_ro_f = np.mean(rollout_faithfulness) if rollout_faithfulness else 0.0
        
        print(f"  Grad-CAM Mean Faithfulness: {mean_gc_f:.4f}")
        print(f"  Rollout Mean Faithfulness:  {mean_ro_f:.4f}")
        
        best_method = "gradcam" if mean_gc_f >= mean_ro_f else "rollout"
        print(f"Automatically selected optimal XAI method: {best_method.upper()}")
        
        return best_method

    def get_explanation_by_method(self, image_tensor, method="gradcam", class_idx=None, device="cuda"):
        """
        Generate explanation using a specific method name.
        """
        if method == "gradcam":
            return self.gradcam.generate_heatmap(image_tensor, class_idx, device)
        else:
            return self.rollout.generate_heatmap(image_tensor, device)

    def remove_hooks(self):
        self.gradcam.remove_hooks()
