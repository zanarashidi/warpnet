# TensorFlow implementation (original)

This is the original code used to produce the results in "Decoupling
the Layers in Residual Networks" (see the [top-level README](../../README.md)).
Forked from [wenxinxu/resnet_in_tensorflow](https://github.com/wenxinxu/resnet_in_tensorflow).
Kept as-is for reproducibility; for a runnable modern port, see
[`../../pytorch`](../../pytorch).

## Requirements

- TensorFlow 1.1.0 (GPU build), CUDA 8.0
- `pip install -r requirements.txt`
- `cifar10/` (warp factor K=2) needs 3 GPUs; `cifar100/` (K=3) needs 4 —
  each residual sub-block and warp term in `block_jump`/`block_jump3` is
  pinned to its own `/gpu:N` in `parallel_Jumping_k.py`.

## Directories

- `cifar10/` — the K=2 warp operator (`block_jump`: F1, F2, F2'(x)x),
  `num_residual_blocks=6`, `k=4` → "WarpNet1-73-4, K=2" in the paper's
  Table 2/5.
- `cifar100/` — the K=3 warp operator (`block_jump3`: F1, F2, F3,
  F2'(x)x, F3'(x)x), `num_residual_blocks=4`, `k=4` → "WarpNet1-73-4,
  K=3" in Table 3. (`block_jump4` in this file is leftover, unused,
  broken dead code — `inference()` never calls it.)

## Running

```bash
cd cifar10   # or cifar100
python cifar10_train.py
```

CIFAR-10/100 data is downloaded automatically on first run
(`maybe_download_and_extract()` in `cifar10_input.py`). Hyperparameters
are TensorFlow flags in `hyper_parameters.py`, overridable on the
command line, e.g. `--train_steps=1000`.

Note: the paper's Section 4.1 states the learning rate drops by a
factor of 0.1 at epochs 60/120/160 (steps ~23460/46920/62560 at batch
size 128). This code's `hyper_parameters.py` defines all three decay
steps but its training loop (`cifar10_train.py`) only actually checks
the first two, and its default decay factor is 0.2 — the third decay
step is defined but never fires. The PyTorch port
(`../../pytorch/train.py`) applies all three decays at a default factor
of 0.1 to match the paper's stated schedule, with a flag to reproduce
this exact original behavior instead.

`run.sh` is a SLURM batch script for a 4-GPU node; adjust for your
cluster/scheduler as needed.
