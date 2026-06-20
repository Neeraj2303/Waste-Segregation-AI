import torch
import torch.nn as nn
import timm
try:
    from transformers import SegformerConfig, SegformerForSemanticSegmentation
except ImportError:
    SegformerForSemanticSegmentation = None

class BaselineModel(nn.Module):
    """
    Unified Baseline Model wrapper.
    Attaches a standard classification head and a semantic segmentation decoder
    to standard backbones, enabling multi-task comparison.
    """
    def __init__(self, model_name="resnet50", num_classes=6, img_size=224, pretrained=True):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.num_seg_classes = num_classes + 1
        
        # Load backbone and determine feature dimensions
        # Check for Segformer
        if "segformer" in model_name:
            self.is_segformer = True
            # Build Segformer (if transformers is available)
            if SegformerForSemanticSegmentation is not None:
                # Use a small Segformer for efficiency
                config = SegformerConfig(
                    num_labels=self.num_seg_classes,
                    image_size=img_size,
                    num_encoder_blocks=4,
                    depths=[2, 2, 2, 2],
                    hidden_sizes=[32, 64, 160, 256],
                    num_attention_heads=[1, 2, 5, 8],
                )
                self.backbone = SegformerForSemanticSegmentation(config)
                self.embed_dim = 256 # final hidden_size dimension
            else:
                # Fallback to standard CNN if library is missing
                self.is_segformer = False
                self.backbone = timm.create_model("resnet34", pretrained=pretrained, features_only=True)
                self.feature_channels = self.backbone.feature_info.channels()
                self.embed_dim = self.feature_channels[-1]
        else:
            self.is_segformer = False
            # Load standard TIMM backbones
            if model_name == "resnet50":
                timm_name = "resnet50"
            elif model_name == "efficientnet_b3":
                timm_name = "efficientnet_b3"
            elif model_name == "vit_b16":
                timm_name = "vit_base_patch16_224"
            elif model_name == "swin":
                timm_name = "swin_base_patch4_window7_224"
            elif model_name == "convnext":
                timm_name = "convnext_tiny"
            else:
                timm_name = "resnet34"
                
            # Use features_only for segmentation
            if "vit" in timm_name:
                # ViT models don't support features_only directly in TIMM sometimes,
                # so we load them normally and use standard representations.
                self.backbone = timm.create_model(timm_name, pretrained=pretrained)
                self.embed_dim = self.backbone.num_features
                self.backbone.reset_classifier(0)
            else:
                self.backbone = timm.create_model(timm_name, pretrained=pretrained, features_only=True)
                self.feature_channels = self.backbone.feature_info.channels()
                self.embed_dim = self.feature_channels[-1]

        # Multi-task Heads
        # Classification Head
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1) if not ("vit" in model_name or self.is_segformer) else nn.Identity(),
            nn.Flatten(),
            nn.Linear(self.embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
        # Segmentation Head (Simple FPN/Bilinear Decoder)
        if not self.is_segformer:
            if "vit" in model_name:
                # ViT single-scale features
                self.seg_head = nn.Sequential(
                    nn.Conv2d(self.embed_dim, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.Conv2d(128, self.num_seg_classes, kernel_size=1)
                )
            else:
                # CNN multi-scale features
                # Simple FPN-like upsamplers
                self.seg_conv = nn.Conv2d(self.embed_dim, 128, kernel_size=3, padding=1)
                self.seg_head = nn.Sequential(
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.Conv2d(128, self.num_seg_classes, kernel_size=1)
                )

    def forward(self, x):
        if self.is_segformer:
            outputs = self.backbone(x)
            # outputs.logits is [B, num_seg_classes, H/4, W/4]
            # Upsample to match H, W
            seg_logits = nn.functional.interpolate(outputs.logits, size=x.shape[2:], mode="bilinear", align_corners=True)
            
            # Classification logits derived from mean pool of final logits
            cls_feats = seg_logits.mean(dim=[2, 3]) # [B, num_seg_classes]
            cls_logits = nn.functional.adaptive_avg_pool1d(cls_feats.unsqueeze(1), self.num_classes).squeeze(1)
            
            return {
                'cls_logits': cls_logits,
                'seg_logits': seg_logits
            }
            
        if "vit" in self.model_name:
            # ViT forward
            features = self.backbone.forward_features(x) # [B, N, C]
            # Extract CLS token
            cls_token = features[:, 0] # [B, C]
            cls_logits = self.cls_head(cls_token)
            
            # Reconstruct patch grid for segmentation
            patch_tokens = features[:, 1:] # [B, N-1, C]
            B, N_minus_1, C = patch_tokens.shape
            grid_size = int(N_minus_1 ** 0.5)
            # Reshape to 2D
            feats_2d = patch_tokens.transpose(1, 2).reshape(B, C, grid_size, grid_size)
            
            # Upsample to image size
            feats_2d = nn.functional.interpolate(feats_2d, size=x.shape[2:], mode="bilinear", align_corners=True)
            seg_logits = self.seg_head(feats_2d)
            
        else:
            # CNN multi-scale features
            features = self.backbone(x) # list of features
            final_feat = features[-1] # bottleneck feature
            
            # Classification
            # Get GAP of bottleneck feature
            cls_logits = self.cls_head(final_feat)
            
            # Segmentation
            seg_feat = self.seg_conv(final_feat)
            seg_feat = nn.functional.interpolate(seg_feat, size=x.shape[2:], mode="bilinear", align_corners=True)
            seg_logits = self.seg_head(seg_feat)
            
        return {
            'cls_logits': cls_logits,
            'seg_logits': seg_logits
        }
