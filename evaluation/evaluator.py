import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from evaluation.metrics import compute_classification_metrics, compute_segmentation_metrics

def evaluate_model(model, loader, device, num_classes=6):
    """
    Run evaluation loop over a dataloader.
    Accumulates logits and targets, and returns computed metrics.
    """
    model.eval()
    all_cls_logits = []
    all_cls_targets = []
    all_seg_logits = []
    all_seg_targets = []
    
    # Track images for visualization
    sample_images = []
    sample_masks = []
    sample_preds = []
    
    with torch.no_grad():
        for i, (images, masks, labels) in enumerate(loader):
            images = images.to(device)
            outputs = model(images)
            
            if 'cls_logits' in outputs:
                all_cls_logits.append(outputs['cls_logits'].cpu())
                all_cls_targets.append(labels)
                
            if 'seg_logits' in outputs:
                all_seg_logits.append(outputs['seg_logits'].cpu())
                all_seg_targets.append(masks)
                
            # Save a few sample images for overlays (first batch)
            if i == 0:
                n_save = min(len(images), 5)
                # Denormalize image for visualization
                # mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)
                imgs_np = images[:n_save].cpu().numpy().transpose(0, 2, 3, 1)
                imgs_np = imgs_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                imgs_np = np.clip(imgs_np, 0, 1)
                
                sample_images.extend(imgs_np)
                sample_masks.extend(masks[:n_save].numpy())
                
                if 'seg_logits' in outputs:
                    seg_preds = torch.argmax(outputs['seg_logits'], dim=1)[:n_save].cpu().numpy()
                    sample_preds.extend(seg_preds)
                    
    results = {}
    
    if all_cls_logits:
        all_cls_logits = torch.cat(all_cls_logits, dim=0).numpy()
        all_cls_targets = torch.cat(all_cls_targets, dim=0).numpy()
        cls_metrics = compute_classification_metrics(all_cls_logits, all_cls_targets, num_classes)
        results['cls'] = cls_metrics
        results['cls_logits'] = all_cls_logits
        results['cls_targets'] = all_cls_targets
        
    if all_seg_logits:
        all_seg_logits = torch.cat(all_seg_logits, dim=0).numpy()
        all_seg_targets = torch.cat(all_seg_targets, dim=0).numpy()
        seg_metrics = compute_segmentation_metrics(all_seg_logits, all_seg_targets, num_classes + 1)
        results['seg'] = seg_metrics
        results['seg_logits'] = all_seg_logits
        results['seg_targets'] = all_seg_targets
        
    results['samples'] = {
        'images': sample_images,
        'masks': sample_masks,
        'preds': sample_preds
    }
    
    return results


def plot_confusion_matrix(conf_mat, class_names, save_path):
    """
    Plot and save a publication-quality confusion matrix.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, 6), dpi=300)
    
    # Normalize confusion matrix
    conf_mat_norm = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
    conf_mat_norm = np.nan_to_num(conf_mat_norm)
    
    # Draw heatmap
    sns.heatmap(
        conf_mat_norm, 
        annot=conf_mat, 
        fmt="d", 
        cmap="Blues", 
        xticklabels=class_names, 
        yticklabels=class_names, 
        cbar=True,
        annot_kws={"size": 11, "weight": "bold"}
    )
    
    plt.title("Normalized Confusion Matrix", fontsize=14, weight='bold', pad=15)
    plt.xlabel("Predicted Class", fontsize=12, labelpad=10)
    plt.ylabel("True Class", fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_roc_curves(logits, targets, class_names, save_path):
    """
    Plot and save publication-quality multiclass ROC curves.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, 6), dpi=300)
    
    # Compute probabilities via softmax
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    num_classes = len(class_names)
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, num_classes))
    
    for i in range(num_classes):
        # One-vs-rest binary targets
        binary_targets = (targets == i).astype(int)
        
        # Check if class exists in target
        if np.sum(binary_targets) > 0:
            fpr[i], tpr[i], _ = roc_curve(binary_targets, probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(
                fpr[i], tpr[i], color=colors[i], lw=2,
                label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})'
            )
            
    # Compute micro-average ROC curve
    # Flatten targets and probs
    binary_targets_flat = []
    probs_flat = []
    for i in range(num_classes):
        binary_targets_flat.append((targets == i).astype(int))
        probs_flat.append(probs[:, i])
        
    binary_targets_flat = np.array(binary_targets_flat).ravel()
    probs_flat = np.array(probs_flat).ravel()
    
    fpr_micro, tpr_micro, _ = roc_curve(binary_targets_flat, probs_flat)
    roc_auc_micro = auc(fpr_micro, tpr_micro)
    
    plt.plot(
        fpr_micro, tpr_micro, color='deeppink', linestyle=':', lw=3,
        label=f'micro-average (AUC = {roc_auc_micro:.2f})'
    )
    
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, labelpad=8)
    plt.ylabel('True Positive Rate', fontsize=12, labelpad=8)
    plt.title('Receiver Operating Characteristic (ROC)', fontsize=14, weight='bold', pad=15)
    plt.legend(loc="lower right", fontsize=9, frameon=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_segmentation_predictions(images, masks, preds, class_names, save_path):
    """
    Plot and save prediction overlays comparing Ground Truth and Predicted Masks.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    n_samples = len(images)
    if n_samples == 0:
        return
        
    fig, axes = plt.subplots(n_samples, 3, figsize=(10, 3 * n_samples), dpi=300)
    if n_samples == 1:
        axes = np.expand_dims(axes, 0)
        
    # Setup custom discrete color palette for segmentation masks
    # 0 is background (transparent/black), 1..6 are the classes
    # We define a list of 7 colors
    colors = ['black', 'red', 'green', 'blue', 'orange', 'cyan', 'magenta']
    cmap = plt.cm.colors.ListedColormap(colors)
    bounds = list(range(8))
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    for i in range(n_samples):
        # Column 1: Original Image
        axes[i, 0].imshow(images[i])
        axes[i, 0].set_title("Input Image", fontsize=10)
        axes[i, 0].axis('off')
        
        # Column 2: Ground Truth Mask
        im_mask = axes[i, 1].imshow(masks[i], cmap=cmap, norm=norm)
        axes[i, 1].set_title("Ground Truth Mask", fontsize=10)
        axes[i, 1].axis('off')
        
        # Column 3: Predicted Mask
        if i < len(preds):
            axes[i, 2].imshow(preds[i], cmap=cmap, norm=norm)
            axes[i, 2].set_title("Predicted Mask", fontsize=10)
        else:
            axes[i, 2].text(0.5, 0.5, "N/A", ha='center', va='center')
            axes[i, 2].set_title("Predicted Mask", fontsize=10)
        axes[i, 2].axis('off')
        
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
