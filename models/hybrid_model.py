import torch
import torch.nn as nn
from models.backbones import EVA02Backbone, DeformableResNet18
from models.fusion import CrossAttentionFusion
from models.heads import ClassificationHead, SegmentationHead

class AdaptiveExplainableHybridModel(nn.Module):
    """
    Adaptive Explainable Hybrid Transformer-CNN Model.
    Combines:
      - Branch A: EVA-02 Vision Transformer (Global)
      - Branch B: Deformable CNN (Local / Irregular)
      - Fusion: Cross-Attention Fusion with Gated Routing
      - Heads: Classification and U-Net Segmentation
    """
    def __init__(self, config):
        super().__init__()
        num_classes = config['dataset']['num_classes']
        vit_name = config['model']['vit_backbone']
        embed_dim = config['model']['embed_dim']
        num_heads = config['model']['fusion']['num_heads']
        dropout = config['model']['fusion']['dropout']
        
        # Branch A: EVA-02 ViT
        self.vit_branch = EVA02Backbone(model_name=vit_name, pretrained=True)
        vit_dim = self.vit_branch.embed_dim
        
        # Branch B: Deformable CNN (ResNet18)
        self.cnn_branch = DeformableResNet18(in_channels=3)
        cnn_dim = self.cnn_branch.channels[-1] # channels of stage 4 (512)
        
        # Fusion Module
        self.fusion = CrossAttentionFusion(
            vit_dim=vit_dim,
            cnn_dim=cnn_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Classification Head
        self.class_head = ClassificationHead(
            embed_dim=embed_dim,
            num_classes=num_classes,
            dropout=0.2
        )
        
        # Segmentation Head
        self.seg_head = SegmentationHead(
            embed_dim=embed_dim,
            cnn_channels=self.cnn_branch.channels,
            num_classes=num_classes
        )

    def get_gating_weights(self):
        """
        Get the normalized branch contributions [alpha_vit, alpha_cnn].
        """
        return self.fusion.get_gating_weights()

    def get_vit_attention_maps(self):
        """
        Retrieve attention matrices recorded by the EVA-02 backbone blocks.
        """
        return self.vit_branch.get_attention_history()

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: Input image tensor of shape [B, 3, H, W]
        Returns:
            dict containing:
              - 'cls_logits': classification logits [B, num_classes]
              - 'seg_logits': segmentation logits [B, num_seg_classes, H, W]
        """
        # 1. Feature Extraction
        # Branch A: ViT global features
        # F_vit shape: [B, N, vit_dim] where N = 1 + patch_count (e.g. 257)
        F_vit = self.vit_branch(x)
        
        # Branch B: CNN local features
        # skip_features contains: [c1, c2, c3, c4]
        # c4 shape: [B, 512, H/32, W/32] -> e.g. [B, 512, 7, 7]
        skip_features = self.cnn_branch(x)
        F_cnn = skip_features[-1]
        
        # 2. Cross-Attention Fusion
        # F_vit_fused shape: [B, N, embed_dim]
        # F_cnn_fused shape: [B, H_c * W_c, embed_dim]
        F_vit_fused, F_cnn_fused = self.fusion(F_vit, F_cnn)
        
        # 3. Heads
        # Classification prediction
        cls_logits = self.class_head(F_vit_fused)
        
        # Segmentation prediction
        # Spatial size of the bottleneck feature map (e.g., 7x7)
        spatial_shape = F_cnn.shape[2:] # (H_c, W_c)
        seg_logits = self.seg_head(F_cnn_fused, skip_features, spatial_shape)
        
        return {
            'cls_logits': cls_logits,
            'seg_logits': seg_logits
        }
