"""
PyTorch port of WarpNet.

    Fok, An, Rashidi, Wang. "Decoupling the Layers in Residual Networks."
    ICLR 2018. https://openreview.net/pdf?id=SyMvJrdaW

Ported from the original TensorFlow 1.x implementation
(`tensorflow/cifar10/parallel_Jumping_k.py`, `tensorflow/cifar100/parallel_Jumping_k.py`,
forked from https://github.com/wenxinxu/resnet_in_tensorflow) for single-GPU use.

## What WarpNet actually is

WarpNet approximates a wide ResNet (WRN) by treating each residual unit's
output as a small perturbation of its input and Taylor-expanding a chain
of K consecutive residual units to first order (paper Eq. 3-5). For
K=2, denoting the two residual sub-blocks F1, F2 (each BN-Conv-BN-ReLU-
Conv-BN) with independent weights W1, W2:

    x3 = x1 + F1(x1, W1) + F2(x1, W2) + F2'(x1, W2) F1(x1, W1)      (exact 1st-order term, Eq. 5)

F2' denotes the Jacobian of F2 w.r.t. its input, evaluated at x1. The
point of this "warp operator" is that F1, F2 and F2' can all be computed
in parallel (each depends only on x1, not on each other's output),
whereas the original ResNet's F2(x1 + F1(x1)) is a sequential dependency
— that's the "decoupling" in the paper's title, and what makes WarpNet
useful for model-parallel training across GPUs (paper puts F1, F2, F2'
on separate GPUs; see `block_jump` in the original TF code).

The paper's cheaper **WarpNet1** variant (used for essentially all
reported CIFAR results) additionally replaces F2'(x1)F1(x1) with
F2'(x1)x1 — i.e. the Jacobian-vector product is taken against the block's
own input x1 instead of against F1's output — for a further speed-up
at similar accuracy. That is what this port implements (matching the
original code, which always calls the "grad" branch with `input` as
the vector, never with F1's output).

Two more approximations, both explicit in the paper's "Experiments"
section (they found BN's backward pass was the compute bottleneck) and
both reproduced here for fidelity:
  - BatchNorm is dropped entirely from F2' (its gradient is not computed
    at all, effectively treated as identity).
  - The convolution term in F2' is computed as a plain convolution with
    a spatially-flipped kernel (the true adjoint/transpose of a
    conv only when in_channels == out_channels, which always holds in
    this architecture, so it's not an approximation here — just a
    non-obvious identity worth calling out).
  - The ReLU term in F2' is approximated as a bare 0/1 sign mask of the
    running Jacobian-vector product itself (not evaluated at the original
    pre-activation, and not multiplied against anything — used as-is).
    This is a genuine, code-level approximation to d(ReLU)/dx not spelled
    out in the paper's equations, presumably part of "removing BN" from
    the derivative pass, but replicated here since it's what the reported
    results were produced with.

K=3 (`warp_factor=3`) adds a third sub-block F3 and its own warp term
F3'(x1)x1, matching `block_jump3` in the CIFAR-100 code. Following the
paper ("we ... drop the term F3'F2'F1 ... due to the limited GPUs we have
in the experiments"), the second-order cross term between F2 and F3 is
omitted.

Stochastic width (`survival_rate`) mirrors `F_block_stoch_width` in the
original code, but is dead weight in practice: the code's drop-rate
constant was hardcoded to 0, so `survival_rate` was always 1.0 (nothing
was ever dropped) when the paper's results were produced, and the paper
itself never mentions stochastic depth/width. Kept only for completeness;
defaults to 1.0.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

BN_EPSILON = 0.001


def _init_conv_weight(out_channels: int, in_channels: int, kernel_size: int) -> nn.Parameter:
    weight = torch.empty(out_channels, in_channels, kernel_size, kernel_size)
    nn.init.kaiming_normal_(weight, mode="fan_in", nonlinearity="relu")
    return nn.Parameter(weight)


class ConvBNReLU(nn.Module):
    """conv_bn_relu_layer: conv -> BN -> ReLU. Used only for the stem."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.weight = _init_conv_weight(out_channels, in_channels, 3)
        self.stride = stride
        self.bn = nn.BatchNorm2d(out_channels, eps=BN_EPSILON)

    def forward(self, x):
        x = F.conv2d(x, self.weight, stride=self.stride, padding=1)
        x = self.bn(x)
        return F.relu(x)


class FBlock(nn.Module):
    """
    F_block: BN(x) -> conv1 -> BN -> ReLU -> conv2 -> BN.

    Also implements `warp_forward`, computing F'(x1) applied to a vector
    `v` (in WarpNet1, v == x1 itself) using this same block's weights —
    see module docstring for exactly which approximations that involves.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.bn_in = nn.BatchNorm2d(in_channels, eps=BN_EPSILON)
        self.conv1_weight = _init_conv_weight(out_channels, in_channels, 3)
        self.bn_mid = nn.BatchNorm2d(out_channels, eps=BN_EPSILON)
        self.conv2_weight = _init_conv_weight(out_channels, out_channels, 3)
        self.bn_out = nn.BatchNorm2d(out_channels, eps=BN_EPSILON)

    def forward(self, x):
        h = self.bn_in(x)
        h = F.conv2d(h, self.conv1_weight, padding=1)
        h = self.bn_mid(h)
        h = F.relu(h)
        h = F.conv2d(h, self.conv2_weight, padding=1)
        return self.bn_out(h)

    def warp_forward(self, v):
        """F'(x1) applied to `v`, reusing this block's own conv weights."""
        flipped1 = self.conv1_weight.flip(2, 3)
        h = F.conv2d(v, flipped1, padding=1)

        # Not `h * mask` -- the original code uses the mask itself as the
        # "post-ReLU-derivative" value. See module docstring.
        relu_mask = (h > 0).float()

        flipped2 = self.conv2_weight.flip(2, 3)
        return F.conv2d(relu_mask, flipped2, padding=1)


class WarpBlock(nn.Module):
    """
    One warp operator, compressing `warp_factor` (K) consecutive residual
    units into a single block that computes all its terms in parallel:

      K=2: out = x + F1(x) + F2(x) + F2'(x)x   (`block_jump` in the original)
      K=3: out = x + F1(x) + F2(x) + F3(x) + F2'(x)x + F3'(x)x
                                       (`block_jump3`, dropping F3'F2'F1)

    All F blocks require in_channels == out_channels == `channels`, true
    everywhere this is used in the network.
    """

    def __init__(self, channels: int, warp_factor: int = 2, survival_rate: float = 1.0):
        super().__init__()
        if warp_factor not in (2, 3):
            raise ValueError("warp_factor must be 2 or 3 (matches the original code)")
        self.warp_factor = warp_factor
        self.survival_rate = survival_rate

        self.F1 = FBlock(channels, channels)
        self.F2 = FBlock(channels, channels)
        if warp_factor == 3:
            self.F3 = FBlock(channels, channels)

    def _maybe_run(self, fn, x):
        if not self.training or self.survival_rate >= 1.0:
            return fn(x)
        if torch.rand(()).item() < self.survival_rate:
            return fn(x)
        return torch.zeros_like(x)

    def forward(self, x):
        out = x + self._maybe_run(self.F1, x)
        out = out + self._maybe_run(self.F2, x)
        out = out + self._maybe_run(self.F2.warp_forward, x)
        if self.warp_factor == 3:
            out = out + self._maybe_run(self.F3, x)
            out = out + self._maybe_run(self.F3.warp_forward, x)
        return out


class WarpNet(nn.Module):
    """
    Three-stage wide-ResNet-style network built from WarpBlocks (Table 1
    of the paper). Layer count n = 6 * warp_factor * num_residual_blocks + 1,
    where num_residual_blocks is Nwarp (warp operators per stage) and
    warp_factor is K -- e.g. num_residual_blocks=6, warp_factor=2, k=4 is
    "WarpNet1-73-4" (the config the CIFAR-10 code defaults to); the
    CIFAR-100 code defaults to num_residual_blocks=4, warp_factor=3, k=4,
    which is also a "-73-4" network (6*3*4+1=73).

    :param num_residual_blocks: warp operators per stage (`Nwarp`)
    :param warp_factor: residual units compressed per warp operator (`K`, 2 or 3)
    :param k: width multiplier (`kw`)
    :param num_classes: 10 for CIFAR-10, 100 for CIFAR-100
    :param survival_rate: see module docstring; kept for completeness, default 1.0
    """

    def __init__(
        self,
        num_residual_blocks: int = 6,
        warp_factor: int = 2,
        k: int = 4,
        num_classes: int = 10,
        survival_rate: float = 1.0,
    ):
        super().__init__()
        self.stem = ConvBNReLU(3, 16, stride=1)

        stage1_channels = 16 * k
        stage2_channels = 32 * k
        stage3_channels = 64 * k

        def make_stage(channels):
            return nn.ModuleList(
                [WarpBlock(channels, warp_factor, survival_rate) for _ in range(num_residual_blocks)]
            )

        self.stage1 = make_stage(stage1_channels)
        self.stage2 = make_stage(stage2_channels)
        self.stage3 = make_stage(stage3_channels)

        self.final_bn = nn.BatchNorm2d(stage3_channels, eps=BN_EPSILON)
        self.fc = nn.Linear(stage3_channels, num_classes)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

        self.k = k

    def forward(self, x):
        out = self.stem(x)
        # conv0's 16 output channels are duplicated k times along the
        # channel axis to seed the first stage at width 16k.
        out = out.repeat(1, self.k, 1, 1)

        for block in self.stage1:
            out = block(out)

        out = F.avg_pool2d(out, kernel_size=2, stride=2)
        out = out.repeat(1, 2, 1, 1)

        for block in self.stage2:
            out = block(out)

        out = F.avg_pool2d(out, kernel_size=2, stride=2)
        out = out.repeat(1, 2, 1, 1)

        for block in self.stage3:
            out = block(out)
            assert out.shape[-2:] == (8, 8), f"unexpected spatial size {out.shape[-2:]}"

        out = self.final_bn(out)
        out = F.relu(out)
        out = out.mean(dim=[2, 3])
        return self.fc(out)


def num_params(model: nn.Module) -> int:
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total number of parameters: {total}")
    return total


if __name__ == "__main__":
    import time

    for warp_factor, num_residual_blocks, label in [(2, 6, "WarpNet1-73-4 (K=2, Nwarp=6)"), (3, 4, "WarpNet1-73-4 (K=3, Nwarp=4)")]:
        print(f"\n=== {label} ===")
        model = WarpNet(num_residual_blocks=num_residual_blocks, warp_factor=warp_factor, k=4, num_classes=10)
        num_params(model)

        x = torch.randn(128, 3, 32, 32)

        model.eval()
        with torch.no_grad():
            model(x)
            start = time.time()
            model(x)
            print("Forward prop time!", time.time() - start)

        model.train()
        x.requires_grad_(True)
        model(x).sum().backward()
        start = time.time()
        model(x).sum().backward()
        print("Full fwd+bwd time!", time.time() - start)
