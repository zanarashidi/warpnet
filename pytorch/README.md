# PyTorch implementation

A PyTorch implementation of WarpNet. See the [top-level README](../README.md)
for what WarpNet is, and `model.py`'s module docstring for exactly how
each term of the warp operator is computed, including the two
approximations the paper makes to keep the "grad" (F') branch cheap —
dropping BatchNorm from it entirely, and approximating ReLU's
derivative as a bare sign mask.

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

```python
from model import WarpNet

# WarpNet1-73-4, K=2 (the CIFAR-10 config)
model = WarpNet(num_residual_blocks=6, warp_factor=2, k=4, num_classes=10)

# WarpNet1-73-4, K=3 (the CIFAR-100 config)
model = WarpNet(num_residual_blocks=4, warp_factor=3, k=4, num_classes=100)

logits = model(images)  # images: (N, 3, 32, 32) float tensor
```

```bash
python train.py --dataset cifar10                                          # K=2
python train.py --dataset cifar100 --warp_factor 3 --num_residual_blocks 4  # K=3
```

`survival_rate` (default `1.0`) mirrors the original code's stochastic-
width mechanism, which never actually dropped anything in the runs that
produced the paper's results (its drop-rate constant was hardcoded to
0, and the paper doesn't mention stochastic depth) — kept only for
completeness.

`train.py`'s learning-rate schedule applies all three decay steps the
paper describes (factor 0.1 at ~epochs 60/120/160). Pass
`--lr_decay_factor 0.2` to instead reproduce the original code's
behavior, which only ever applied two of its three declared decay
steps at a 0.2 factor.

## Multi-GPU

Each `WarpBlock` computes F1, F2, (F3) and the warp term(s) as
independent functions of the same input `x` — the whole point of the
paper's "decoupling" — so they can run on separate GPUs, exactly as
the paper's experiments do (Tables 2-3). Pass `devices` to `WarpNet`
(or `--devices` to `train.py`) as a comma-separated/list of device
strings:

```python
# K=2: 3 devices -- [F1, F2, F2']
model = WarpNet(num_residual_blocks=6, warp_factor=2, k=4, num_classes=10,
                 devices=["cuda:0", "cuda:1", "cuda:2"])

# K=3: 4 devices -- [F1, F2, F3, {F2', F3'}]
model = WarpNet(num_residual_blocks=4, warp_factor=3, k=4, num_classes=100,
                 devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
```

```bash
python train.py --dataset cifar10 --devices cuda:0,cuda:1,cuda:2
python train.py --dataset cifar100 --warp_factor 3 --num_residual_blocks 4 \
    --devices cuda:0,cuda:1,cuda:2,cuda:3
```

The stem, final BatchNorm/ReLU and classifier head always stay on the
first listed device; only the branches inside each `WarpBlock` are
split, matching the paper (which never distributes anything else).
F2's and F3's warp terms reuse those blocks' own conv weights (as in
the original TF code's `reuse_variables()`), so running a warp term on
a different device than its source block involves copying that
block's weights across — the actual communication cost of sharing
weights between GPUs, and the reason the paper reports the speed-up
net of that cost rather than assuming it away.

Omit `devices`/`--devices` to run the whole network on one device
(CPU, MPS, or a single GPU).
