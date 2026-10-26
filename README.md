# HoVer-VMamba: Replacing the HoVer-Net Encoder with a Vision State-Space Model

Code from a summer research project (XJTLU Summer Undergraduate Research Fellowship, SURF-2025-0347, May-Aug 2025, supervisor Dr Jingxin Liu): the Preact-ResNet50 encoder of [HoVer-Net](https://github.com/vqdang/hover_net) is replaced with a [VMamba](https://github.com/MzeroMiko/VMamba) backbone (2-D Selective Scan; linear complexity, global receptive field) while the three-branch decoder (NP: nuclear pixel, HV: horizontal/vertical distance maps, NC: nuclear classification) is kept, with channel dimensions and feature resolutions adapted. 把 HoVer-Net 的 CNN 编码器换成 VMamba，保留三分支解码器。

## Results (CoNSeP)

Quantitative comparison from the project poster (baseline HoVer-Net -> HoVer-VMamba):

| Metric | HoVer-Net | HoVer-VMamba |
|---|---|---|
| Dice | 0.833 | 0.850 |
| Type Dice | 0.636 | 0.720 |

Training curves for HoVer-VMamba were smoother and converged faster in the poster; those logs are not included here. Single dataset, single reported run: treat as a preliminary result.

## This repository

This repository contains only the **files added for the VMamba integration**. It is meant to be overlaid on a checkout of upstream HoVer-Net (MIT licence, see `LICENSE.hover_net`) plus a checkout of VMamba, which are not redistributed here.

```
models/vmamba/
  vmamba_backbone.py   VMamba backbone adapter producing HoVer-Net-compatible multi-scale features
  net_desc.py          VMamba encoder + HoVer-Net decoder
  opt.py               training schedule (warm-up / main / fine-tune stages)
  targets.py, run_desc.py
config_vmamba.py       config (model_name = "vmamba")
train_vmamba.py        stand-alone training script
tests/                 shape / integration checks
```

## How to use (not re-verified)

1. Clone upstream `vqdang/hover_net`, copy this repository's files over it, and clone `MzeroMiko/VMamba` next to it (the backbone adapter looks for it at `../../../VMamba` relative to `models/vmamba/`); install the VMamba / mamba-ssm dependencies and CUDA kernels.
2. Prepare CoNSeP patches with upstream `extract_patches.py` and set the paths in `config_vmamba.py`.
3. `python tests/test_vmamba_integration.py` (shape checks), then `python train_vmamba.py --gpu 0 --mode original --batch_size 4`.

I did not re-run these steps when preparing this repository.

## Contributions
Team project (5 members: Yuyao Wang, Weiqi Yan, Jinyang Zheng, Minghao Zhang, Pengqian Gong). I was team lead: coordinated the team, chose the baseline, adapted the model, ran the backbone-replacement experiments and gave technical guidance to teammates. The code was developed by the team; individual authorship of each file is not itemised here.

## Tech stack
Python, PyTorch, VMamba / Mamba SSM, HoVer-Net, CoNSeP.

## Not included
Datasets, pretrained and trained checkpoints, training logs, and the upstream HoVer-Net / VMamba source.
