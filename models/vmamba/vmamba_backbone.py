"""
VMamba Backbone Adapter for HoverNet
This module adapts VMamba as a backbone for the HoverNet framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from typing import List, Tuple, Optional

# 动态添加VMamba路径
vmamba_path = os.path.join(os.path.dirname(__file__), '../../../VMamba')
if vmamba_path not in sys.path:
    sys.path.append(vmamba_path)

try:
    from vmamba import Backbone_VSSM
    VMAMBA_AVAILABLE = True
except ImportError:
    print("Warning: VMamba not found, using dummy implementation")
    VMAMBA_AVAILABLE = False
    
    class Backbone_VSSM(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.dummy = True
            
        def forward(self, x):
            # Dummy implementation for testing
            B, C, H, W = x.shape
            return [
                torch.randn(B, 96, H//4, W//4),    # Stage 0: 1/4 scale
                torch.randn(B, 192, H//8, W//8),   # Stage 1: 1/8 scale  
                torch.randn(B, 384, H//16, W//16), # Stage 2: 1/16 scale
                torch.randn(B, 768, H//32, W//32), # Stage 3: 1/32 scale
            ]


class FeatureAdapter(nn.Module):
    """特征适配层，将VMamba特征适配到HoverNet期望的维度"""
    
    def __init__(self, in_channels: int, out_channels: int, use_norm: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=not use_norm)
        self.norm = nn.BatchNorm2d(out_channels) if use_norm else nn.Identity()
        self.act = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class VMambaBackbone(nn.Module):
    """
    VMamba Backbone for HoverNet
    将VMamba作为特征提取器，输出多尺度特征图供HoverNet解码器使用
    """
    
    def __init__(self, 
                 input_ch: int = 3,
                 patch_size: int = 4,
                 embed_dims: List[int] = [96, 192, 384, 768],
                 depths: List[int] = [2, 2, 9, 2],
                 ssm_d_state: int = 16,
                 ssm_ratio: float = 2.0,
                 mlp_ratio: float = 4.0,
                 drop_path_rate: float = 0.1,
                 use_checkpoint: bool = False,
                 pretrained: Optional[str] = None):
        """
        Args:
            input_ch: 输入通道数，默认3 (RGB)
            patch_size: patch大小，默认4
            embed_dims: 各阶段的embedding维度
            depths: 各阶段的层数
            ssm_d_state: SSM状态维度
            ssm_ratio: SSM比例
            mlp_ratio: MLP扩展比例
            drop_path_rate: DropPath率
            use_checkpoint: 是否使用gradient checkpointing节省显存
            pretrained: 预训练权重路径
        """
        super().__init__()
        
        self.embed_dims = embed_dims
        self.use_checkpoint = use_checkpoint
        
        # VMamba backbone
        if VMAMBA_AVAILABLE:
            self.vmamba = Backbone_VSSM(
                patch_size=patch_size,
                in_chans=input_ch,
                dims=embed_dims[0],  # 基础维度，会自动扩展
                depths=depths,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank="auto",
                ssm_act_layer="silu",
                ssm_conv=3,
                ssm_conv_bias=True,
                ssm_drop_rate=0.0,
                ssm_init="v0",
                forward_type="v0",
                mlp_ratio=mlp_ratio,
                mlp_act_layer="gelu",
                mlp_drop_rate=0.0,
                drop_path_rate=drop_path_rate,
                patch_norm=True,
                norm_layer="ln",
                out_indices=(0, 1, 2, 3),  # 输出所有4个阶段的特征
                pretrained=pretrained,
            )
        else:
            self.vmamba = Backbone_VSSM()
            
        # 特征适配层 - 将VMamba特征适配到HoverNet期望的维度
        # HoverNet期望的特征维度: [64, 256, 512, 1024, 2048]
        self.adapt_layers = nn.ModuleList([
            FeatureAdapter(embed_dims[0], 64),    # Stage 0: 96 -> 64
            FeatureAdapter(embed_dims[1], 256),   # Stage 1: 192 -> 256  
            FeatureAdapter(embed_dims[2], 512),   # Stage 2: 384 -> 512
            FeatureAdapter(embed_dims[3], 1024),  # Stage 3: 768 -> 1024
        ])
        
        # 额外的卷积层产生2048维特征 (对应HoverNet的d3)
        self.final_conv = nn.Sequential(
            nn.Conv2d(1024, 2048, 3, padding=1, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            x: 输入图像 [B, C, H, W]
            
        Returns:
            Tuple of feature maps at different scales:
            - d0: [B, 64, H/4, W/4]     (1/4 scale)
            - d1: [B, 256, H/8, W/8]    (1/8 scale)
            - d2: [B, 512, H/16, W/16]  (1/16 scale) 
            - d3: [B, 1024, H/32, W/32] (1/32 scale)
            - d4: [B, 2048, H/32, W/32] (1/32 scale, final features)
        """
        # VMamba前向传播
        if self.use_checkpoint and self.training:
            import torch.utils.checkpoint as checkpoint
            features = checkpoint.checkpoint(self.vmamba, x)
        else:
            features = self.vmamba(x)
            
        # 适配特征维度
        adapted_features = []
        for i, (feat, adapter) in enumerate(zip(features, self.adapt_layers)):
            adapted_feat = adapter(feat)
            adapted_features.append(adapted_feat)
            
        # 生成最终的2048维特征
        d4 = self.final_conv(adapted_features[-1])
        
        return tuple(adapted_features + [d4])
        
    def get_feature_dims(self) -> List[int]:
        """返回各阶段的特征维度"""
        return [64, 256, 512, 1024, 2048]
        
    def freeze_backbone(self, freeze: bool = True):
        """冻结backbone参数"""
        for param in self.vmamba.parameters():
            param.requires_grad = not freeze
            
    def unfreeze_backbone(self):
        """解冻backbone参数"""
        self.freeze_backbone(False)
