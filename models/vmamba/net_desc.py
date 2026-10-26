import math
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..hovernet.net_utils import (DenseBlock, Net, TFSamepaddingLayer, UpSample2x)
from ..hovernet.utils import crop_op, crop_to_shape
from .vmamba_backbone import VMambaBackbone

class VMambaHoVerNet(Net):
    """VMamba-based HoVerNet for nuclei segmentation and classification"""
    
    def __init__(self, input_ch=3, nr_types=None, freeze=False, mode='original'):
        super().__init__()
        self.mode = mode
        self.freeze = freeze
        self.nr_types = nr_types
        self.output_ch = 3 if nr_types is None else 4

        # VMamba backbone for feature extraction
        self.backbone = VMambaBackbone(
            input_ch=input_ch,
            patch_size=4,
            embed_dims=[96, 192, 384, 768],
            depths=[2, 2, 9, 2],
            ssm_d_state=16,
            ssm_ratio=2.0,
            mlp_ratio=4.0,
            drop_path_rate=0.1,
            use_checkpoint=False,  # 可根据显存情况调整
        )
        
        # 如果需要冻结backbone
        if freeze:
            self.backbone.freeze_backbone()

        # 解码器分支
        def create_decoder_branch(out_ch=2, ksize=3):
            """创建解码器分支"""
            module_list = [
                ("conva", nn.Conv2d(2048, 256, ksize, stride=1, padding=ksize//2, bias=False)),
                ("dense", DenseBlock(256, [1, ksize], [128, 32], 8, split=4)),
                ("convf", nn.Conv2d(512, 512, 1, stride=1, padding=0, bias=False)),
            ]
            u3 = nn.Sequential(OrderedDict(module_list))

            module_list = [
                ("conva", nn.Conv2d(512, 128, ksize, stride=1, padding=ksize//2, bias=False)),
                ("dense", DenseBlock(128, [1, ksize], [128, 32], 4, split=4)),
                ("convf", nn.Conv2d(256, 256, 1, stride=1, padding=0, bias=False)),
            ]
            u2 = nn.Sequential(OrderedDict(module_list))

            module_list = [
                ("conva", nn.Conv2d(256, 64, ksize, stride=1, padding=ksize//2, bias=False)),
            ]
            u1 = nn.Sequential(OrderedDict(module_list))

            module_list = [
                ("bn", nn.BatchNorm2d(64, eps=1e-5)),
                ("relu", nn.ReLU(inplace=True)),
                ("conv", nn.Conv2d(64, out_ch, 1, stride=1, padding=0, bias=True)),
            ]
            u0 = nn.Sequential(OrderedDict(module_list))

            decoder = nn.Sequential(
                OrderedDict([("u3", u3), ("u2", u2), ("u1", u1), ("u0", u0)])
            )
            return decoder

        # 创建解码器分支
        ksize = 3
        if nr_types is None:
            self.decoder = nn.ModuleDict(
                OrderedDict([
                    ("np", create_decoder_branch(ksize=ksize, out_ch=2)),
                    ("hv", create_decoder_branch(ksize=ksize, out_ch=2)),
                ])
            )
        else:
            self.decoder = nn.ModuleDict(
                OrderedDict([
                    ("tp", create_decoder_branch(ksize=ksize, out_ch=nr_types)),
                    ("np", create_decoder_branch(ksize=ksize, out_ch=2)),
                    ("hv", create_decoder_branch(ksize=ksize, out_ch=2)),
                ])
            )

        self.upsample2x = UpSample2x()
        
        # 权重初始化
        self._init_decoder_weights()

    def _init_decoder_weights(self):
        """初始化解码器权重"""
        for m in self.decoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, imgs):
        """前向传播"""
        # 输入预处理
        if imgs.max() > 1.0:
            imgs = imgs / 255.0
        imgs = torch.clamp(imgs, 0.0, 1.0)

        # 通过backbone提取特征
        features = self.backbone(imgs)  # [d0, d1, d2, d3, d4]
        
        # 使用最高层特征进行解码
        d4 = features[-1]  # [B, 2048, H/32, W/32]
        
        # 解码器前向传播
        out_dict = OrderedDict()
        for branch_name, branch_desc in self.decoder.items():
            # 第一层解码
            u3 = branch_desc.u3(d4)
            u3 = self.upsample2x(u3)
            
            # 第二层解码 
            u2 = branch_desc.u2(u3)
            u2 = self.upsample2x(u2)
            
            # 第三层解码
            u1 = branch_desc.u1(u2)
            u1 = self.upsample2x(u1)
            
            # 最终输出层
            u0 = branch_desc.u0(u1)
            
            # 调整到目标输出尺寸
            target_size = 80 if self.mode == "original" else 164
            if u0.size(2) != target_size or u0.size(3) != target_size:
                u0 = F.interpolate(
                    u0, size=(target_size, target_size), 
                    mode='bilinear', align_corners=False
                )

            out_dict[branch_name] = u0

        return out_dict
    
    def freeze_encoder(self):
        """冻结编码器"""
        self.backbone.freeze_backbone()
        
    def unfreeze_encoder(self):
        """解冻编码器"""
        self.backbone.unfreeze_backbone()
        
    def get_trainable_parameters(self):
        """获取可训练参数"""
        return [p for p in self.parameters() if p.requires_grad]

def create_model(mode=None, **kwargs):
    if mode not in ['original', 'fast']:
        assert "Unknown Model Mode %s" % mode
    return VMambaHoVerNet(mode=mode, **kwargs)