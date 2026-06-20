import torch
import torch.nn as nn
import torchvision
import timm

class EVA02Backbone(nn.Module):
    """
    Wrapper for TIMM EVA-02 Vision Transformer.
    Extracts global features and records attention matrices for Rollout.
    """
    def __init__(self, model_name="eva02_tiny_patch14_224", pretrained=True):
        super().__init__()
        self.hooks = [] # Initialize early to safeguard __del__
        self.attention_weights = []
        
        # Load backbone
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.embed_dim = self.model.num_features
        
        # Clear classification head (we will build custom classification heads)
        self.model.reset_classifier(0)
        
        # Register hooks
        self._register_attention_hooks()

    def _register_attention_hooks(self):
        """
        Register forward hooks on self-attention blocks to record attention weights.
        """
        # Find all attention modules in the model blocks
        # For standard timm ViT models: self.model.blocks[i].attn
        if hasattr(self.model, 'blocks'):
            for block in self.model.blocks:
                if hasattr(block, 'attn'):
                    attn_module = block.attn
                    hook = attn_module.register_forward_hook(self._make_attn_hook())
                    self.hooks.append(hook)
        else:
            print("WARNING: Could not find blocks in ViT model. Attention hooks not registered.")

    def _make_attn_hook(self):
        """
        Create a hook that intercepts and records attention maps.
        """
        def hook(module, input, output):
            # Recalculate or intercept attention maps inside the hook
            # Input to the hook is x of shape [B, N, C]
            x = input[0]
            B, N, C = x.shape
            
            # Replicate the forward attention matrix calculation
            qkv = module.qkv(x) # [B, N, 3 * C]
            qkv = qkv.reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0) # [B, num_heads, N, head_dim]
            
            # Calculate attention map
            scale = module.scale if hasattr(module, 'scale') else (C // module.num_heads) ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)
            
            self.attention_weights.append(attn.detach())
            
        return hook

    def clear_attention_history(self):
        self.attention_weights.clear()

    def get_attention_history(self):
        return self.attention_weights

    def forward(self, x):
        self.clear_attention_history()
        
        # Pass through TIMM model to extract features
        # For ViT: returns [B, N, C]
        # In EVA-02, N is patch_count + 1 (class token)
        features = self.model.forward_features(x)
        return features

    def __del__(self):
        # Remove hooks when object is destroyed
        for hook in self.hooks:
            hook.remove()


class DeformableBasicBlock(nn.Module):
    """
    ResNet-style BasicBlock where the first 3x3 conv is replaced with a Deformable Conv2d.
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        
        # Offsets generator for Deformable Conv
        # Offsets output shape: [B, 2 * kh * kw, H_out, W_out]
        # For a 3x3 kernel: 2 * 3 * 3 = 18 channels of offsets
        self.offset_conv = nn.Conv2d(in_channels, 2 * 3 * 3, kernel_size=3, padding=1, stride=stride, bias=True)
        
        # Deformable Convolution
        self.deform_conv = torchvision.ops.DeformConv2d(
            in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False
        )
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        
        # Compute offsets
        offsets = self.offset_conv(x)
        
        # Deformable Conv
        out = self.deform_conv(x, offsets)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Second Conv
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
            
        out += residual
        out = self.relu(out)
        return out


class DeformableResNet18(nn.Module):
    """
    A lightweight Deformable ResNet-18 model.
    Produces multi-scale features for U-Net style skip connections.
    """
    def __init__(self, in_channels=3):
        super().__init__()
        self.in_channels = 64
        
        # Initial stem
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Stages
        self.layer1 = self._make_layer(64, 2, stride=1)   # H/4, W/4
        self.layer2 = self._make_layer(128, 2, stride=2)  # H/8, W/8
        self.layer3 = self._make_layer(256, 2, stride=2)  # H/16, W/16
        self.layer4 = self._make_layer(512, 2, stride=2)  # H/32, W/32
        
        self.channels = [64, 128, 256, 512]
        
    def _make_layer(self, out_channels, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * DeformableBasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * DeformableBasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * DeformableBasicBlock.expansion),
            )
            
        blocks = []
        blocks.append(DeformableBasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * DeformableBasicBlock.expansion
        for _ in range(1, num_blocks):
            blocks.append(DeformableBasicBlock(self.in_channels, out_channels))
            
        return nn.Sequential(*blocks)
        
    def forward(self, x):
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x_pool = self.maxpool(x)
        
        # Feature extraction stages
        c1 = self.layer1(x_pool) # [B, 64, H/4, W/4]
        c2 = self.layer2(c1)     # [B, 128, H/8, W/8]
        c3 = self.layer3(c2)     # [B, 256, H/16, W/16]
        c4 = self.layer4(c3)     # [B, 512, H/32, W/32]
        
        # Return multi-scale features
        return [c1, c2, c3, c4]
