# PyTorch port

A single-GPU-friendly PyTorch port of WarpNet, for anyone who wants to
use or extend it without TF 1.1.0/CUDA 8/multi-GPU. See the
[top-level README](../README.md) for what WarpNet is, and `model.py`'s
module docstring for exactly how each term of the warp operator is
computed, including the two approximations the paper makes to keep the
"grad" (F') branch cheap — dropping BatchNorm from it entirely, and
approximating ReLU's derivative as a bare sign mask.

## Status

- [x] Model (`model.py`) — both K=2 and K=3 warp operators ported;
      self-test passes (forward + backward on random input).
- [x] Data pipeline (`dataset.py`) — CIFAR-10/100 via torchvision, with
      the original's exact preprocessing (pad+crop, flip, per-image
      whitening).
- [x] Training loop / CLI (`train.py`) — smoke-tested end-to-end
      (download, train, validate, checkpoint).

Not yet done: multi-GPU (the original's whole point was splitting F1/F2/
F3/F' across GPUs — this port deliberately runs everything on one
device instead) and a from-scratch full training run to confirm parity
with the paper's reported validation errors.

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

```python
from model import WarpNet

# WarpNet1-73-4, K=2 (the CIFAR-10 code's default config)
model = WarpNet(num_residual_blocks=6, warp_factor=2, k=4, num_classes=10)

# WarpNet1-73-4, K=3 (the CIFAR-100 code's default config)
model = WarpNet(num_residual_blocks=4, warp_factor=3, k=4, num_classes=100)

logits = model(images)  # images: (N, 3, 32, 32) float tensor
```

```bash
python train.py --dataset cifar10                                       # K=2
python train.py --dataset cifar100 --warp_factor 3 --num_residual_blocks 4  # K=3
```

`survival_rate` (default `1.0`) mirrors the original code's stochastic-
width mechanism, which was dead weight in practice (its drop-rate
constant was hardcoded to 0, so nothing was ever actually dropped when
the paper's results were produced, and the paper doesn't mention it) —
kept only for completeness.

See `train.py`'s module docstring for one deliberate fix versus the
original code: the learning-rate schedule now applies all three decay
steps the paper describes (factor 0.1 at ~epochs 60/120/160), where the
original code defined but never used the third one and defaulted to a
0.2 factor. Pass `--lr_decay_factor 0.2` to reproduce the original's
exact (likely unintended) behavior.
