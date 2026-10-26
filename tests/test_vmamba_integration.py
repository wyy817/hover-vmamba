#!/usr/bin/env python3

"""
VMamba-HoverNet集成测试脚本
检查模型是否能正确初始化和前向传播
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
from models.vmamba import create_model

def test_vmamba_hovernet():
    print("=" * 60)
    print("VMamba-HoverNet集成测试")
    print("=" * 60)
    
    # 检查GPU可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    try:
        # 1. 测试模型初始化
        print("\n1. 测试模型初始化...")
        model = create_model(
            input_ch=3,
            nr_types=None,  # 先不使用类型分类
            freeze=False,
            mode='original'
        )
        model = model.to(device)
        print("✓ 模型初始化成功")
        
        # 2. 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   总参数量: {total_params:,}")
        print(f"   可训练参数: {trainable_params:,}")
        
        # 3. 测试前向传播
        print("\n2. 测试前向传播...")
        batch_size = 2
        input_size = 270  # original mode
        
        # 创建测试输入
        test_input = torch.randint(0, 255, (batch_size, 3, input_size, input_size), 
                                  dtype=torch.float32, device=device)
        print(f"   输入形状: {test_input.shape}")
        
        # 前向传播
        model.eval()
        with torch.no_grad():
            outputs = model(test_input)
        
        print("✓ 前向传播成功")
        
        # 4. 检查输出形状
        print("\n3. 检查输出...")
        expected_output_size = 80  # original mode
        for branch_name, output in outputs.items():
            expected_channels = 2  # np和hv分支都是2通道
            print(f"   {branch_name}: {output.shape}")
            
            # 验证输出形状
            assert output.shape[0] == batch_size, f"批次大小错误: {output.shape[0]} != {batch_size}"
            assert output.shape[1] == expected_channels, f"通道数错误: {output.shape[1]} != {expected_channels}"
            assert output.shape[2] == expected_output_size, f"高度错误: {output.shape[2]} != {expected_output_size}"
            assert output.shape[3] == expected_output_size, f"宽度错误: {output.shape[3]} != {expected_output_size}"
        
        print("✓ 输出形状正确")
        
        # 5. 测试梯度计算
        print("\n4. 测试梯度计算...")
        model.train()
        test_input.requires_grad_(True)
        
        outputs = model(test_input)
        
        # 计算简单损失
        loss = 0
        for output in outputs.values():
            loss += output.mean()
        
        loss.backward()
        print("✓ 梯度计算成功")
        
        # 6. 检查梯度
        grad_norm = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm += param.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        print(f"   梯度范数: {grad_norm:.6f}")
        
        if grad_norm > 0:
            print("✓ 梯度传播正常")
        else:
            print("⚠ 警告: 梯度为零")
        
        # 7. 内存使用情况
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1e9
            memory_cached = torch.cuda.memory_reserved() / 1e9
            print(f"\n5. GPU内存使用:")
            print(f"   已分配: {memory_used:.2f} GB")
            print(f"   已缓存: {memory_cached:.2f} GB")
        
        print("\n" + "=" * 60)
        print("✓ 所有测试通过！VMamba-HoverNet集成正常工作")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_vmamba_hovernet()
    sys.exit(0 if success else 1)
