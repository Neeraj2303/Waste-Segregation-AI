import torch
import torch.nn as nn

class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention Fusion Module.
    Aligns and fuses global features (ViT) and local features (Deformable CNN)
    using bidirectional cross-attention with a learnable gating parameter.
    """
    def __init__(self, vit_dim, cnn_dim, embed_dim=256, num_heads=4, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Feature Alignment Layers
        self.align_vit = nn.Linear(vit_dim, embed_dim)
        self.align_cnn = nn.Linear(cnn_dim, embed_dim)
        
        # Bidirectional Multi-head Cross-Attention
        # ViT attends to CNN features
        self.cross_attn_vit = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        # CNN attends to ViT features
        self.cross_attn_cnn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        # Norm and MLP blocks
        self.norm_vit = nn.LayerNorm(embed_dim)
        self.norm_cnn = nn.LayerNorm(embed_dim)
        
        self.mlp_vit = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout)
        )
        self.mlp_cnn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout)
        )
        
        self.norm_mlp_vit = nn.LayerNorm(embed_dim)
        self.norm_mlp_cnn = nn.LayerNorm(embed_dim)
        
        # Learnable gating weights for the two branches
        # Initialized such that Softmax(gating_weights) yields [0.5, 0.5]
        self.gating_weights = nn.Parameter(torch.tensor([0.0, 0.0], dtype=torch.float32))
        
    def get_gating_weights(self):
        """
        Get the normalized gate contributions: [alpha_vit, alpha_cnn]
        """
        return torch.softmax(self.gating_weights, dim=0)
        
    def forward(self, F_vit, F_cnn):
        """
        Args:
            F_vit: [B, N, C_vit] - sequence of patch + class tokens
            F_cnn: [B, C_cnn, H_c, W_c] - CNN spatial feature map
        Returns:
            F_vit_fused: [B, N, embed_dim]
            F_cnn_fused: [B, H_c * W_c, embed_dim]
        """
        B, C_cnn, H_c, W_c = F_cnn.shape
        
        # 1. Feature Alignment
        # Project ViT features
        F_vit_aligned = self.align_vit(F_vit) # [B, N, embed_dim]
        
        # Project CNN features after reshaping
        F_cnn_flat = F_cnn.permute(0, 2, 3, 1).reshape(B, H_c * W_c, C_cnn) # [B, H_c*W_c, C_cnn]
        F_cnn_aligned = self.align_cnn(F_cnn_flat) # [B, H_c*W_c, embed_dim]
        
        # 2. Get Softmax Gated weights
        alpha = self.get_gating_weights()
        alpha_vit, alpha_cnn = alpha[0], alpha[1]
        
        # 3. Bidirectional Cross-Attention
        # ViT queries CNN
        attn_vit_out, _ = self.cross_attn_vit(
            query=F_vit_aligned,
            key=F_cnn_aligned,
            value=F_cnn_aligned
        )
        F_vit_res = F_vit_aligned + alpha_vit * attn_vit_out
        F_vit_res = self.norm_vit(F_vit_res)
        F_vit_fused = F_vit_res + self.mlp_vit(F_vit_res)
        F_vit_fused = self.norm_mlp_vit(F_vit_fused)
        
        # CNN queries ViT
        attn_cnn_out, _ = self.cross_attn_cnn(
            query=F_cnn_aligned,
            key=F_vit_aligned,
            value=F_vit_aligned
        )
        F_cnn_res = F_cnn_aligned + alpha_cnn * attn_cnn_out
        F_cnn_res = self.norm_cnn(F_cnn_res)
        F_cnn_fused = F_cnn_res + self.mlp_cnn(F_cnn_res)
        F_cnn_fused = self.norm_mlp_cnn(F_cnn_fused)
        
        return F_vit_fused, F_cnn_fused
