"""
VMamba-HoVerNet训练脚本
展示如何使用VMamba作为backbone训练HoVerNet
"""

import os
import sys
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)


def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='VMamba-HoVerNet Training')
    
    parser.add_argument('--gpu', type=str, default='0', 
                       help='Comma separated list of GPU(s) to use.')
    parser.add_argument('--mode', type=str, default='original',
                       choices=['original', 'fast'],
                       help='Model mode: original or fast')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=75,
                       help='Total number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Initial learning rate')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--save_dir', type=str, 
                       default='./checkpoints_vmamba',
                       help='Directory to save checkpoints')
    
    return parser.parse_args()


def setup_environment(gpu_list):
    """设置训练环境"""
    # 设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    
    # 设置随机种子
    torch.manual_seed(10)
    np.random.seed(10)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(10)
        print(f"Using GPU: {gpu_list}")
        print(f"Available GPUs: {torch.cuda.device_count()}")
    else:
        print("Using CPU")


def create_model(args):
    """创建VMamba-HoVerNet模型"""
    from models.vmamba.net_desc import create_model
    
    model = create_model(
        input_ch=3,
        nr_types=5,  # CoNSeP数据集的核类型数量
        freeze=False,
        mode=args.mode
    )
    
    # 如果有多个GPU，使用DataParallel
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    return model


def create_optimizer(model, args):
    """创建优化器"""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    
    return optimizer, scheduler


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """加载检查点"""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint {checkpoint_path} not found!")
        return 0
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 加载模型权重
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # 加载优化器和调度器状态
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint['epoch']
    print(f"Loaded checkpoint from epoch {epoch}")
    
    return epoch + 1


def save_checkpoint(model, optimizer, scheduler, epoch, save_dir):
    """保存检查点"""
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    
    checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch:03d}.pth')
    torch.save(checkpoint, checkpoint_path)
    
    # 保存最新的检查点
    latest_path = os.path.join(save_dir, 'latest_checkpoint.pth')
    torch.save(checkpoint, latest_path)
    
    print(f"Saved checkpoint: {checkpoint_path}")


def print_model_info(model):
    """打印模型信息"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n" + "="*50)
    print("模型信息:")
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    print(f"模型大小: {total_params * 4 / (1024**2):.1f} MB")
    print("="*50 + "\n")


def main():
    """主训练函数"""
    args = get_args()
    
    # 设置环境
    setup_environment(args.gpu)
    
    # 创建模型
    print("创建VMamba-HoVerNet模型...")
    model = create_model(args)
    print_model_info(model)
    
    # 创建优化器
    optimizer, scheduler = create_optimizer(model, args)
    
    # 加载检查点(如果有)
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, scheduler, args.resume)
    
    print(f"开始训练，从epoch {start_epoch} 到 {args.epochs}")
    print(f"模型模式: {args.mode}")
    print(f"批次大小: {args.batch_size}")
    print(f"初始学习率: {args.lr}")
    print(f"保存目录: {args.save_dir}")
    
    # 注意：这里只是展示框架，实际训练需要配置数据加载器和训练循环
    print("\n⚠️  注意: 这只是训练脚本框架")
    print("实际训练需要:")
    print("1. 配置数据加载器")
    print("2. 实现训练循环")
    print("3. 配置损失函数") 
    print("4. 添加验证逻辑")
    
    print("\n建议使用HoverNet原有的训练框架:")
    print("python run_train.py --config_path=config_vmamba.py")


if __name__ == "__main__":
    main()
