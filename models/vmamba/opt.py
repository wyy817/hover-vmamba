import torch.optim as optim

from run_utils.callbacks.base import (
    AccumulateRawOutput,
    PeriodicSaver,
    ProcessAccumulatedRawOutput,
    ScalarMovingAverage,
    ScheduleLr,
    TrackLr,
    VisualizeOutput,
    TriggerEngine,
)
from run_utils.callbacks.logging import LoggingEpochOutput, LoggingGradient
from run_utils.engine import Events

from ..hovernet.targets import gen_targets, prep_sample
from .net_desc import create_model
from ..hovernet.run_desc import proc_valid_step_output, train_step, valid_step, viz_step_output

def get_config(nr_type, mode):
    """
    VMamba-HoVerNet训练配置
    采用三阶段训练策略：
    1. 预热阶段：较小学习率，训练解码器
    2. 主训练阶段：正常学习率，端到端训练
    3. 微调阶段：小学习率，精细调整
    """
    return {
        "phase_list": [
            # Phase 1: 预热阶段 - 较小学习率训练解码器
            {
                "run_info": {
                    "net": {
                        "desc": lambda: create_model(
                            input_ch=3, 
                            nr_types=None,  # 先不训练类型预测
                            freeze=True,  # 冻结backbone
                            mode=mode
                        ),
                        "optimizer": [
                            optim.AdamW,  # 使用AdamW优化器
                            {
                                "lr": 1.0e-4,
                                "betas": (0.9, 0.999),
                                "eps": 1e-8,
                                "weight_decay": 1e-4,
                            },
                        ],
                        "lr_scheduler": lambda x: optim.lr_scheduler.CosineAnnealingLR(
                            x, T_max=15, eta_min=1e-6
                        ),
                        "extra_info": {
                            "loss": {
                                "np": {"bce": 1, "dice": 1},
                                "hv": {"mse": 1, "msge": 1},
                            },
                        },
                        "pretrained": None,
                    },
                },
                "target_info": {"gen": (gen_targets, {}), "viz": (prep_sample, {})},
                "batch_size": {"train": 6, "valid": 8},  # 适中的batch size
                "nr_epochs": 15,  # 预热阶段
            },
            
            # Phase 2: 主训练阶段 - 端到端训练
            {
                "run_info": {
                    "net": {
                        "desc": lambda: create_model(
                            input_ch=3, 
                            nr_types=nr_type,  # 启用类型预测
                            freeze=False,  # 解冻backbone
                            mode=mode
                        ),
                        "optimizer": [
                            optim.AdamW,
                            {
                                "lr": 2.0e-5,  # 较小的学习率用于端到端训练
                                "betas": (0.9, 0.999),
                                "eps": 1e-8,
                                "weight_decay": 1e-4,
                            },
                        ],
                        "lr_scheduler": lambda x: optim.lr_scheduler.CosineAnnealingLR(
                            x, T_max=35, eta_min=1e-7
                        ),
                        "extra_info": {
                            "loss": {
                                "np": {"bce": 1, "dice": 1},
                                "hv": {"mse": 1, "msge": 1},
                                "tp": {"bce": 1, "dice": 1},  # 类型预测损失
                            },
                        },
                        "pretrained": -1,  # 从上一阶段加载
                    },
                },
                "target_info": {"gen": (gen_targets, {}), "viz": (prep_sample, {})},
                "batch_size": {"train": 4, "valid": 6},  # 减小batch size
                "nr_epochs": 35,
            },
            
            # Phase 3: 微调阶段 - 精细调整
            {
                "run_info": {
                    "net": {
                        "desc": lambda: create_model(
                            input_ch=3, 
                            nr_types=nr_type,
                            freeze=False,
                            mode=mode
                        ),
                        "optimizer": [
                            optim.AdamW,
                            {
                                "lr": 5.0e-6,  # 很小的学习率用于微调
                                "betas": (0.9, 0.999),
                                "eps": 1e-8,
                                "weight_decay": 5e-5,  # 轻微正则化
                            },
                        ],
                        "lr_scheduler": lambda x: optim.lr_scheduler.StepLR(
                            x, step_size=10, gamma=0.5
                        ),
                        "extra_info": {
                            "loss": {
                                "np": {"bce": 1, "dice": 1},
                                "hv": {"mse": 1, "msge": 1},
                                "tp": {"bce": 1, "dice": 1},
                            },
                        },
                        "pretrained": -1,  # 从上一阶段加载
                    },
                },
                "target_info": {"gen": (gen_targets, {}), "viz": (prep_sample, {})},
                "batch_size": {"train": 4, "valid": 6},
                "nr_epochs": 25,
            },
        ],
        "run_engine": {
            "train": {
                "dataset": "",
                "nr_procs": 8,
                "run_step": train_step,
                "reset_per_run": False,
                "callbacks": {
                    Events.STEP_COMPLETED: [
                        ScalarMovingAverage(),
                    ],
                    Events.EPOCH_COMPLETED: [
                        TrackLr(),
                        PeriodicSaver(),
                        VisualizeOutput(viz_step_output),
                        LoggingEpochOutput(),
                        TriggerEngine("valid"),
                        ScheduleLr(),
                    ],
                },
            },
            "valid": {
                "dataset": "",
                "nr_procs": 4,
                "run_step": valid_step,
                "reset_per_run": True,
                "callbacks": {
                    Events.STEP_COMPLETED: [AccumulateRawOutput(),],
                    Events.EPOCH_COMPLETED: [
                        ProcessAccumulatedRawOutput(
                            lambda a: proc_valid_step_output(a, nr_types=nr_type)
                        ),
                        LoggingEpochOutput(),
                    ],
                },
            },
        },
    }