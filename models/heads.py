import torch
import torch.nn as nn

class ClassificationHead(nn.Module):
    """
    MLP Classification Head.
    Takes the fused ViT class token representation and projects it to class logits.
    """
    def __init__(self, embed_dim=256, num_classes=6, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )
        
    def forward(self, F_vit_fused):
        # Extract class token (index 0) from the fused sequence
        # F_vit_fused is [B, N, embed_dim]
        cls_token = F_vit_fused[:, 0] # [B, embed_dim]
        return self.net(cls_token)


class DecoderBlock(nn.Module):
    """
    U-Net style Decoder Block with bilinear upsampling and skip connection fusion.
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x, skip=None):
        # Upsample spatial dimension by 2
        x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out


class SegmentationHead(nn.Module):
    """
    Transformer/CNN Decoder Segmentation Head.
    Combines the fused bottleneck representation with multi-scale skip connections
    from the CNN backbone, producing pixel-level class predictions.
    """
    def __init__(self, embed_dim=256, cnn_channels=[64, 128, 256, 512], num_classes=6):
        super().__init__()
        # Target classes + 1 (background is 0, waste classes are 1..num_classes)
        self.num_seg_classes = num_classes + 1 
        
        # Decoder blocks
        # DB1: Input (embed_dim, 7x7) + Skip c3 (256, 14x14) -> Out (256, 14x14)
        self.db1 = DecoderBlock(in_channels=embed_dim, skip_channels=cnn_channels[2], out_channels=256)
        
        # DB2: Input (256, 14x14) + Skip c2 (128, 28x28) -> Out (128, 28x28)
        self.db2 = DecoderBlock(in_channels=256, skip_channels=cnn_channels[1], out_channels=128)
        
        # DB3: Input (128, 28x28) + Skip c1 (64, 56x56) -> Out (64, 56x56)
        self.db3 = DecoderBlock(in_channels=128, skip_channels=cnn_channels[0], out_channels=64)
        
        # Final block upsampling from 56x56 to 224x224
        self.final_upsample = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, self.num_seg_classes, kernel_size=1)
        )
        
    def forward(self, F_cnn_fused, skip_features, spatial_shape):
        """
        Args:
            F_cnn_fused: [B, H_c * W_c, embed_dim] - bottleneck fused features
            skip_features: List of tensors [c1, c2, c3, c4] from CNN backbone
            spatial_shape: (H_c, W_c) for reconstructing F_cnn_fused spatial grid
        """
        B = F_cnn_fused.shape[0]
        H_c, W_c = spatial_shape
        C_fuse = F_cnn_fused.shape[2]
        
        # Reconstruct spatial feature map
        # F_cnn_fused is [B, H_c*W_c, embed_dim] -> [B, embed_dim, H_c, W_c]
        x = F_cnn_fused.permute(0, 2, 1).reshape(B, C_fuse, H_c, W_c)
        
        c1, c2, c3, _ = skip_features # Skip connections: c1 (56x56), c2 (28x28), c3 (14x14)
        
        # DB1: 7x7 -> 14x14
        x = self.db1(x, c3)
        
        # DB2: 14x14 -> 28x28
        x = self.db2(x, c2)
        
        # DB3: 28x28 -> 56x56
        x = self.db3(x, c1)
        
        # Final prediction map: 56x56 -> 224x224
        x = nn.functional.interpolate(x, scale_factor=4, mode="bilinear", align_corners=True)
        logits = self.final_upsample(x)
        
        return logits
