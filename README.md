# WarpNet

Code accompanying:

> Ricky Fok, Aijun An, Zana Rashidi, Xiaogang Wang. **"Decoupling the
> Layers in Residual Networks."** ICLR Workshop, 2018.
> https://openreview.net/pdf?id=SyMvJrdaW

WarpNet approximates a wide ResNet by Taylor-expanding a chain of `K`
consecutive residual units to first order, yielding a "warp operator"
that computes all `K` units' outputs in parallel from a single shared
input instead of sequentially. For `K=2` with sub-blocks F1, F2 (each
BN-Conv-BN-ReLU-Conv-BN):

    x_out = x + F1(x) + F2(x) + F2'(x)·x

where `F2'(x)` is F2's Jacobian evaluated at `x`, applied to `x` itself
(the paper's cheaper "WarpNet1" variant, used for all reported CIFAR
results). Because F1, F2 and F2' only depend on the block's input, they
can be computed on separate GPUs — that's the "decoupling," and the
source of WarpNet's speed-up over a plain wide ResNet at comparable
accuracy. See `pytorch/model.py`'s module docstring for the full
breakdown, including the approximations the paper makes in computing
F2' cheaply (dropping BatchNorm from it entirely, approximating ReLU's
derivative as a sign mask).

## Layout

- **`pytorch/`** — a modern, single-GPU-friendly PyTorch port. This is
  the recommended way to use or extend the model today. See
  `pytorch/README.md`.
- **`archive/tensorflow/`** — the original TensorFlow 1.1.0 implementation
  used to produce the paper's results (`cifar10/` = the K=2 config,
  `cifar100/` = the K=3 config). Kept for reproducibility, not
  recommended for new work — see `archive/tensorflow/README.md` for how
  to run it (requires CUDA 8, TF 1.1.0, and 3-4 GPUs).

## Citation

If you use this code, please cite the paper (see the OpenReview link
above for citation details).
