import os
import io
import base64
import yaml
import cv2
import numpy as np
import torch
from flask import Flask, request, jsonify, render_template

import albumentations as A
from albumentations.pytorch import ToTensorV2

from models.hybrid_model import AdaptiveExplainableHybridModel
from xai.adaptive import AdaptiveXAISelector
from datasets.synthetic import SyntheticWasteDataset
from datasets.base import get_transforms

app = Flask(__name__)

# Load base configuration
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

config = load_config('configs/default_config.yaml')
device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')

# Load the hybrid model
print(f"Flask Server initializing model on: {device}")
model = AdaptiveExplainableHybridModel(config)

# Load checkpoint weights dynamically
checkpoint_dir = config['training']['checkpoint_dir']
STAGE_PATHS = {
    'stage1': os.path.join(checkpoint_dir, "stage1_trashnet_stage1_best.pt"),
    'stage2': os.path.join(checkpoint_dir, "stage2_taco_stage2_best.pt"),
    'stage3': os.path.join(checkpoint_dir, "stage3_zerowaste_stage3_best.pt")
}

current_stage = None

def load_stage_weights(stage_name):
    global current_stage
    if current_stage == stage_name:
        return True
        
    path = STAGE_PATHS.get(stage_name)
    if not path or not os.path.exists(path):
        print(f"WARNING: Checkpoint for {stage_name} not found at {path}")
        return False
        
    print(f"Loading checkpoint weights for {stage_name} from: {path}")
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    current_stage = stage_name
    return True

# Initial weights loading (fall back to stage3, stage2, or stage1)
loaded = False
for stage in ['stage3', 'stage2', 'stage1']:
    if load_stage_weights(stage):
        loaded = True
        break

if not loaded:
    print("WARNING: No checkpoints found under 'checkpoints/'. Model initialized with random/ImageNet weights.")

# Preprocess image
def preprocess_image(img_bytes, img_size=224):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    transformed = transform(image=img_rgb)
    img_tensor = transformed['image']
    return img_rgb, img_tensor

# Convert RGB image to base64 string
def to_base64(img_rgb):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.png', img_bgr)
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{b64_str}"

# Render heatmap overlay
def get_heatmap_overlay(img_rgb, heatmap_mask):
    heatmap_rgb = cv2.applyColorMap(np.uint8(255 * heatmap_mask), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_rgb, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_rgb, 0.5, heatmap_rgb, 0.5, 0)
    return overlay

# Render segmentation overlay
def get_segmentation_overlay(img_rgb, pred_mask):
    # Color mapping for classes: 0: bg, 1: cardboard, 2: glass, 3: metal, 4: paper, 5: plastic, 6: trash
    # Let's map segmentation labels (0..6) to specific visual colors
    colors = [
        [0, 0, 0],         # 0: Background (black)
        [255, 50, 50],     # 1: Cardboard (red)
        [50, 255, 50],     # 2: Glass (green)
        [50, 50, 255],     # 3: Metal (blue)
        [255, 165, 0],     # 4: Paper (orange)
        [0, 255, 255],     # 5: Plastic (cyan)
        [255, 0, 255],     # 6: Trash (magenta)
    ]
    
    mask_colored = np.zeros_like(img_rgb)
    for cls in range(7):
        mask_colored[pred_mask == cls] = colors[cls]
        
    return mask_colored

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        stage = request.form.get('stage', 'stage3')
        load_stage_weights(stage)
        
        is_synthetic = request.form.get('synthetic', 'false').lower() == 'true'
        raw_img = None
        img_tensor = None
        gt_mask_val = None
        
        if is_synthetic:
            # Generate synthetic sample
            val_transform = get_transforms(img_size=config['dataset']['img_size'], is_train=False)
            syn_ds = SyntheticWasteDataset(num_samples=1, img_size=config['dataset']['img_size'], num_classes=config['dataset']['num_classes'], transforms=val_transform)
            img_tensor, gt_mask, label = syn_ds[0]
            
            # Convert normalized tensor back to original RGB image
            raw_img = img_tensor.cpu().numpy().transpose(1, 2, 0)
            raw_img = raw_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            raw_img = np.clip(raw_img * 255.0, 0, 255).astype(np.uint8)
        else:
            if 'image' not in request.files:
                return jsonify({'error': 'No image file uploaded'}), 400
                
            file = request.files['image']
            img_bytes = file.read()
            raw_img, img_tensor = preprocess_image(img_bytes, img_size=config['dataset']['img_size'])
            
        # Resize raw_img to match the model input dimensions (224x224) for overlays
        raw_img = cv2.resize(raw_img, (config['dataset']['img_size'], config['dataset']['img_size']))
            
        # Model forward pass
        img_tensor_batch = img_tensor.unsqueeze(0).to(device)
        
        with torch.set_grad_enabled(True):
            outputs = model(img_tensor_batch)
            
        cls_logits = outputs['cls_logits'].detach()
        seg_logits = outputs['seg_logits'].detach()
        
        # Classification stats
        probs = torch.softmax(cls_logits, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        pred_prob = probs[pred_idx].item()
        pred_class = config['dataset']['class_names'][pred_idx]
        
        detailed_probs = {}
        for idx, class_name in enumerate(config['dataset']['class_names']):
            detailed_probs[class_name] = float(probs[idx].item())
            
        # Segmentation stats
        pred_mask = torch.argmax(seg_logits, dim=1)[0].cpu().numpy()
        seg_overlay = get_segmentation_overlay(raw_img, pred_mask)
        
        # XAI computation
        xai_selector = AdaptiveXAISelector(model, config)
        
        # Heatmap calculations
        gated_map, method_used, gating_info = xai_selector.get_explanation_by_gating(img_tensor, pred_idx, device)
        gradcam_map = xai_selector.get_explanation_by_method(img_tensor, method="gradcam", class_idx=pred_idx, device=device)
        rollout_map = xai_selector.get_explanation_by_method(img_tensor, method="rollout", device=device)
        
        xai_selector.remove_hooks()
        
        # Overlay heatmaps onto original image
        gated_overlay = get_heatmap_overlay(raw_img, gated_map)
        gradcam_overlay = get_heatmap_overlay(raw_img, gradcam_map)
        rollout_overlay = get_heatmap_overlay(raw_img, rollout_map)
        
        # Convert all to base64 images
        res_data = {
            'class': pred_class,
            'confidence': float(pred_prob),
            'detailed_probabilities': detailed_probs,
            'gating': {
                'alpha_vit': float(gating_info['alpha_vit']),
                'alpha_cnn': float(gating_info['alpha_cnn']),
                'method_used': method_used.upper()
            },
            'images': {
                'input': to_base64(raw_img),
                'segmentation': to_base64(seg_overlay),
                'gated_xai': to_base64(gated_overlay),
                'gradcam': to_base64(gradcam_overlay),
                'rollout': to_base64(rollout_overlay)
            }
        }
        return jsonify(res_data)
        
    except Exception as e:
        print(f"Error during predict API: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create static & templates folders if they don't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=False)
