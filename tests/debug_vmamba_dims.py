#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import torch.nn as nn
from VMamba.vmamba import Backbone_VSSM

def test_vmamba_dimensions():
    """测试VMamba的实际输出维度"""
    print("=" * 50)
    print("VMamba维度测试")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 初始化VMamba
    vmamba = Backbone_VSSM(
        patch_size=4,
        in_chans=3,
        depths=[2, 2, 9, 2],
        dims=96,  # 这会自动扩展为 [96, 192, 384, 768]
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",
        ssm_act_layer="silu",
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0,
        ssm_init="v0",
        forward_type="v0",  # 使用v0避免backend问题
        mlp_ratio=4.0,
        mlp_act_layer="gelu",
        mlp_drop_rate=0.0,
        drop_path_rate=0.1,
        patch_norm=True,
        norm_layer="ln",
        out_indices=(0, 1, 2, 3),
    ).to(device)
    
    print(f"VMamba dims配置: {vmamba.dims}")
    
    # 测试输入
    test_input = torch.randn(1, 3, 270, 270).to(device)
    print(f"输入形状: {test_input.shape}")
    
    try:
        with torch.no_grad():
            features = vmamba(test_input)
            
        print(f"\nVMamba输出特征数量: {len(features)}")
        for i, feat in enumerate(features):
            print(f"  特征{i}: {feat.shape}")
            
    except Exception as e:
        print(f"❌ VMamba前向传播失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_vmamba_dimensions()
