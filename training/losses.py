import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, C] - classification logits
            targets: [B] - class indices
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss) # p_t
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """
    Multi-class Dice Loss for semantic segmentation.
    Computes class-wise Dice Coefficient and returns 1 - Dice.
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits: [B, C_seg, H, W] - segmentation logits
            targets: [B, H, W] - pixel labels (0 to C_seg - 1)
        """
        B, C, H, W = logits.shape
        probs = F.softmax(logits, dim=1)
        
        # One-hot encode targets
        # targets shape: [B, H, W] -> [B, C, H, W]
        targets_one_hot = F.one_hot(targets, num_classes=C).permute(0, 3, 1, 2).float()
        
        # Flatten tensors for calculation
        probs_flat = probs.reshape(B, C, -1)
        targets_flat = targets_one_hot.reshape(B, C, -1)
        
        # Compute intersection and union
        intersection = torch.sum(probs_flat * targets_flat, dim=2)
        union = torch.sum(probs_flat, dim=2) + torch.sum(targets_flat, dim=2)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Average across classes and batch
        dice_loss = 1.0 - dice.mean()
        return dice_loss


class JointLoss(nn.Module):
    """
    Combined Loss:
    Total Loss = cls_weight * Cls_Loss + seg_weight * Seg_Loss
    Where Cls_Loss is CrossEntropy or Focal Loss,
    and Seg_Loss is dice_weight * Dice_Loss + ce_weight * Seg_CE_Loss.
    Supports task masking (skipping loss calculation if weight is 0.0).
    """
    def __init__(self, config):
        super().__init__()
        self.cls_weight = config['loss']['cls_weight']
        self.seg_weight = config['loss']['seg_weight']
        
        cls_type = config['loss']['cls_loss_type']
        if cls_type == 'focal':
            self.cls_loss_fn = FocalLoss(alpha=config['loss']['focal_alpha'], gamma=config['loss']['focal_gamma'])
        else:
            self.cls_loss_fn = nn.CrossEntropyLoss()
            
        self.seg_ce_loss_fn = nn.CrossEntropyLoss()
        self.seg_dice_loss_fn = DiceLoss()
        
        self.dice_w = config['loss']['dice_weight']
        self.ce_w = config['loss']['ce_weight']
        
    def forward(self, outputs, cls_targets, seg_targets):
        """
        Args:
            outputs: dict containing 'cls_logits' and 'seg_logits'
            cls_targets: [B] tensor of classification labels
            seg_targets: [B, H, W] tensor of segmentation masks
        Returns:
            dict containing:
              - 'loss': total scalar loss
              - 'cls_loss': classification loss scalar
              - 'seg_loss': segmentation loss scalar
        """
        total_loss = 0.0
        cls_loss_val = 0.0
        seg_loss_val = 0.0
        
        # 1. Classification Loss
        if self.cls_weight > 0.0 and 'cls_logits' in outputs:
            cls_logits = outputs['cls_logits']
            cls_loss_val = self.cls_loss_fn(cls_logits, cls_targets)
            total_loss += self.cls_weight * cls_loss_val
            
        # 2. Segmentation Loss
        if self.seg_weight > 0.0 and 'seg_logits' in outputs:
            seg_logits = outputs['seg_logits']
            # Compute Cross Entropy on pixels
            seg_ce = self.seg_ce_loss_fn(seg_logits, seg_targets)
            # Compute Dice Loss
            seg_dice = self.seg_dice_loss_fn(seg_logits, seg_targets)
            
            seg_loss_val = self.ce_w * seg_ce + self.dice_w * seg_dice
            total_loss += self.seg_weight * seg_loss_val
            
        return {
            'loss': total_loss,
            'cls_loss': cls_loss_val if isinstance(cls_loss_val, float) else cls_loss_val.item(),
            'seg_loss': seg_loss_val if isinstance(seg_loss_val, float) else seg_loss_val.item()
        }
