import os
import argparse
import yaml
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2
import albumentations as A

from models.hybrid_model import AdaptiveExplainableHybridModel
from xai.adaptive import AdaptiveXAISelector
from datasets.synthetic import SyntheticWasteDataset
from datasets.base import get_transforms

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def parse_args():
    parser = argparse.ArgumentParser(description="Single Image Inference & XAI Visualization")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config file")
    parser.add_argument("--image", type=str, default=None, help="Path to input image")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model weights checkpoint")
    parser.add_argument("--test-synthetic", action="store_true", help="Test inference using a generated synthetic image")
    parser.add_argument("--method", type=str, default="adaptive", choices=["adaptive", "gradcam", "rollout"], help="XAI method")
    return parser.parse_args()

def preprocess_image(img_path, img_size=224):
    """
    Load and preprocess image using standard validation settings.
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    transformed = transform(image=img)
    tensor = transformed['image']
    return img, tensor

def main():
    args = parse_args()
    config = load_config(args.config)
    
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Running inference on: {device}")
    
    # 1. Load Model
    model = AdaptiveExplainableHybridModel(config)
    
    # Checkpoint loading
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        # Fallback checkpoints: check Stage 3, then Stage 2, then Stage 1 best checkpoints
        checkpoint_dir = config['training']['checkpoint_dir']
        stage3_path = os.path.join(checkpoint_dir, "stage3_zerowaste_stage3_best.pt")
        stage2_path = os.path.join(checkpoint_dir, "stage2_taco_stage2_best.pt")
        stage1_path = os.path.join(checkpoint_dir, "stage1_trashnet_stage1_best.pt")
        
        for path in [stage3_path, stage2_path, stage1_path]:
            if os.path.exists(path):
                ckpt_path = path
                break
            
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Loading checkpoint weights from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("WARNING: No checkpoints found under config['training']['checkpoint_dir']. Initializing model with random/ImageNet weights.")
        
    model = model.to(device)
    model.eval()
    
    # 2. Get Input Image
    raw_img = None
    img_tensor = None
    gt_mask = None
    
    if args.test_synthetic or args.image is None:
        print("Generating synthetic image for test inference...")
        # Instantiating a synthetic dataset to get one sample
        val_transform = get_transforms(img_size=config['dataset']['img_size'], is_train=False)
        syn_ds = SyntheticWasteDataset(num_samples=1, img_size=config['dataset']['img_size'], num_classes=config['dataset']['num_classes'], transforms=val_transform)
        img_tensor, gt_mask, label = syn_ds[0]
        
        # Reconstruct raw image from normalized tensor
        raw_img = img_tensor.cpu().numpy().transpose(1, 2, 0)
        raw_img = raw_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        raw_img = np.clip(raw_img * 255.0, 0, 255).astype(np.uint8)
        gt_mask = gt_mask.numpy()
        print(f"  Synthetic Image Label: {config['dataset']['class_names'][label.item()]}")
    else:
        if not os.path.exists(args.image):
            raise FileNotFoundError(f"Requested input image '{args.image}' not found.")
        raw_img, img_tensor = preprocess_image(args.image, img_size=config['dataset']['img_size'])
        
    # Resize raw_img to match the model input dimensions (224x224) for overlays
    raw_img = cv2.resize(raw_img, (config['dataset']['img_size'], config['dataset']['img_size']))
        
    # 3. Model Forward Pass
    img_tensor_batch = img_tensor.unsqueeze(0).to(device)
    
    # To run backward pass in GradCAM we need grad enabled
    with torch.set_grad_enabled(True):
        outputs = model(img_tensor_batch)
        
    cls_logits = outputs['cls_logits'].detach()
    seg_logits = outputs['seg_logits'].detach()
    
    probs = torch.softmax(cls_logits, dim=1)[0]
    pred_idx = torch.argmax(probs).item()
    pred_prob = probs[pred_idx].item()
    pred_class = config['dataset']['class_names'][pred_idx]
    
    print(f"\nPrediction Summary:")
    print(f"  Classification: {pred_class} (Confidence: {pred_prob:.4f})")
    
    for i, class_name in enumerate(config['dataset']['class_names']):
        print(f"    - {class_name:10s}: {probs[i].item():.4f}")
        
    pred_mask = torch.argmax(seg_logits, dim=1)[0].cpu().numpy()
    
    # 4. Generate Explainability Maps
    xai_selector = AdaptiveXAISelector(model, config)
    
    # Generate Gated Explanation
    gated_heatmap, method_used, gating_info = xai_selector.get_explanation_by_gating(
        img_tensor, pred_idx, device
    )
    print(f"\nAdaptive Gating Mechanism Allocation:")
    print(f"  Transformer Branch (alpha_vit): {gating_info['alpha_vit']:.4f}")
    print(f"  Deformable CNN Branch (alpha_cnn): {gating_info['alpha_cnn']:.4f}")
    print(f"  Selected Explanation Method:    {method_used.upper()}")
    
    # Generate explicit maps for comparison
    gradcam_map = xai_selector.get_explanation_by_method(img_tensor, method="gradcam", class_idx=pred_idx, device=device)
    rollout_map = xai_selector.get_explanation_by_method(img_tensor, method="rollout", device=device)
    
    # Clean hooks
    xai_selector.remove_hooks()
    
    # 5. Visualizing and Saving
    os.makedirs(config['xai']['visualization_dir'], exist_ok=True)
    
    # Plot layout: 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), dpi=300)
    
    # Subplot 0,0: Input image with segmentation overlay
    axes[0, 0].imshow(raw_img)
    axes[0, 0].set_title("Input Image", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Subplot 0,1: Segmentation prediction overlay
    # Create colored overlay
    # Colors: black, red, green, blue, orange, cyan, magenta
    colors = ['black', 'red', 'green', 'blue', 'orange', 'cyan', 'magenta']
    cmap = plt.cm.colors.ListedColormap(colors)
    bounds = list(range(8))
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
    
    axes[0, 1].imshow(raw_img)
    # Overlay the mask with transparency
    axes[0, 1].imshow(pred_mask, cmap=cmap, norm=norm, alpha=0.5)
    axes[0, 1].set_title(f"Segmentation Mask Overlay ({pred_class})", fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Subplot 1,0: Grad-CAM (CNN local/irregular attention)
    axes[1, 0].imshow(raw_img)
    # Show heatmap overlay
    gc_heatmap_rgb = cv2.applyColorMap(np.uint8(255 * gradcam_map), cv2.COLORMAP_JET)
    gc_heatmap_rgb = cv2.cvtColor(gc_heatmap_rgb, cv2.COLOR_BGR2RGB)
    axes[1, 0].imshow(gc_heatmap_rgb, alpha=0.5)
    title_gc = "Grad-CAM (CNN Branch)"
    if method_used == "gradcam":
        title_gc += " [ACTIVE]"
    axes[1, 0].set_title(title_gc, fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Subplot 1,1: Attention Rollout (ViT global attention)
    axes[1, 1].imshow(raw_img)
    ro_heatmap_rgb = cv2.applyColorMap(np.uint8(255 * rollout_map), cv2.COLORMAP_JET)
    ro_heatmap_rgb = cv2.cvtColor(ro_heatmap_rgb, cv2.COLOR_BGR2RGB)
    axes[1, 1].imshow(ro_heatmap_rgb, alpha=0.5)
    title_ro = "Attention Rollout (ViT Branch)"
    if method_used == "rollout":
        title_ro += " [ACTIVE]"
    axes[1, 1].set_title(title_ro, fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Add info labels on the figure
    fig.suptitle(
        f"Waste Classification: {pred_class.upper()} ({pred_prob:.2%})\n"
        f"Gating: ViT={gating_info['alpha_vit']:.2f} | CNN={gating_info['alpha_cnn']:.2f}",
        fontsize=14, fontweight='bold', y=0.98
    )
    
    plt.tight_layout()
    save_fig_path = os.path.join(config['xai']['visualization_dir'], "inference_xai_visualization.png")
    plt.savefig(save_fig_path, bbox_inches='tight')
    plt.close()
    
    print(f"\nInference completed successfully. Plot saved to: {save_fig_path}")

if __name__ == "__main__":
    main()
