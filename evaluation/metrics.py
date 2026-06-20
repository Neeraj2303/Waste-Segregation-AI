import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix

def compute_classification_metrics(logits, targets, num_classes=6):
    """
    Calculate classification metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix.
    
    Args:
        logits: numpy array or tensor of shape [N, num_classes]
        targets: numpy array or tensor of shape [N]
    """
    if isinstance(logits, torch.Tensor):
        logits = logits.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
        
    preds = np.argmax(logits, axis=1)
    
    # Probabilities for ROC-AUC
    # Apply softmax to logits
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    acc = accuracy_score(targets, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(targets, preds, average='macro', zero_division=0)
    
    # Multi-class ROC-AUC (one-vs-rest)
    try:
        # Check if targets contains all classes, if not, ROC-AUC can fail in sklearn unless handled
        if len(np.unique(targets)) > 1:
            roc_auc = roc_auc_score(targets, probs, multi_class='ovr', average='macro')
        else:
            roc_auc = 0.0
    except Exception as e:
        print(f"WARNING: ROC-AUC computation failed: {e}. Defaulting to 0.0")
        roc_auc = 0.0
        
    conf_mat = confusion_matrix(targets, preds, labels=list(range(num_classes)))
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': conf_mat
    }


def compute_segmentation_metrics(logits, targets, num_classes=7):
    """
    Calculate semantic segmentation metrics: IoU, mIoU, Dice Score, and Pixel Accuracy.
    
    Args:
        logits: numpy array or tensor of shape [N, num_classes, H, W]
        targets: numpy array or tensor of shape [N, H, W]
    """
    if isinstance(logits, torch.Tensor):
        logits = logits.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
        
    preds = np.argmax(logits, axis=1) # [N, H, W]
    
    # Calculate pixel accuracy
    pixel_acc = np.mean(preds == targets)
    
    ious = []
    dice_scores = []
    
    # Compute metrics for each class (including background 0)
    for cls in range(num_classes):
        pred_cls = (preds == cls)
        target_cls = (targets == cls)
        
        intersection = np.sum(pred_cls & target_cls)
        union = np.sum(pred_cls | target_cls)
        
        if union == 0:
            # Class not present in either ground truth or prediction, skip it from average
            iou = np.nan
            dice = np.nan
        else:
            iou = intersection / union
            dice = (2.0 * intersection) / (np.sum(pred_cls) + np.sum(target_cls) + 1e-8)
            
        ious.append(iou)
        dice_scores.append(dice)
        
    # Calculate means, ignoring NaNs (classes not present in test subset)
    mean_iou = np.nanmean(ious)
    mean_dice = np.nanmean(dice_scores)
    
    return {
        'pixel_accuracy': pixel_acc,
        'class_ious': ious,
        'miou': mean_iou,
        'class_dice': dice_scores,
        'mean_dice': mean_dice
    }
