#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import torch.nn as nn

def test_final_integration():
    """最终集成测试"""
    print("=" * 60)
    print("VMamba-HoverNet最终集成测试")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name}")
        print(f"GPU内存: {gpu_memory:.1f} GB")
    
    try:
        # 导入模型
        from models.vmamba.net_desc import create_model
        
        print("\n1. 测试模型创建...")
        model = create_model(
            input_ch=3,
            nr_types=None,  # 不使用类型分类
            freeze=False,
            mode='original'
        ).to(device)
        
        # 计算参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✓ 模型创建成功")
        print(f"  总参数量: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        
        print("\n2. 测试前向传播...")
        # 测试不同的batch size
        for batch_size in [1, 2, 4]:
            test_input = torch.randn(batch_size, 3, 270, 270).to(device)
            print(f"  批次大小 {batch_size}, 输入形状: {test_input.shape}")
            
            with torch.no_grad():
                outputs = model(test_input)
            
            print(f"  ✓ 输出分支数: {len(outputs)}")
            for branch_name, output in outputs.items():
                print(f"    {branch_name}: {output.shape}")
        
        print("\n3. 测试梯度计算...")
        model.train()
        test_input = torch.randn(2, 3, 270, 270).to(device)
        test_input.requires_grad_(True)
        
        outputs = model(test_input)
        
        # 简单的损失计算
        loss = 0
        for branch_name, output in outputs.items():
            target = torch.randn_like(output)
            loss += nn.MSELoss()(output, target)
        
        loss.backward()
        print(f"✓ 梯度计算成功, 总损失: {loss.item():.6f}")
        
        # 检查梯度
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
        
        if grad_norms:
            print(f"  梯度范数: 最小={min(grad_norms):.6f}, 最大={max(grad_norms):.6f}")
        
        print("\n4. 内存使用情况...")
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
            memory_cached = torch.cuda.memory_reserved(0) / 1024**3
            print(f"  GPU内存使用: {memory_allocated:.2f} GB")
            print(f"  GPU内存缓存: {memory_cached:.2f} GB")
        
        print("\n" + "=" * 60)
        print("🎉 所有测试通过！VMamba-HoverNet集成成功！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_final_integration()
