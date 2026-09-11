"""
Single-GPU training script for the PyTorch WarpNet port.

Mirrors the original TF training loop (`cifar10_train.py`/`cifar100_train.py`):
training runs for a fixed number of steps (not epochs), does full
validation over the whole 10k-image validation/test set every
`report_freq` steps, decays the learning rate at three fixed steps
(~epochs 60/120/160, per the paper's stated schedule), and checkpoints
every 10000 steps plus at the end of training.

One deliberate deviation from the original code, to match what the
paper says was actually done rather than an apparent bug in the
original flags: the original defined `decay_step2` but its training
loop only ever checked `decay_step0`/`decay_step1`, and its default
decay factor was 0.2, while the paper's Section 4.1 states the LR drops
by a factor of 0.1 at all three of epochs 60/120/160. This script
applies all three decay steps with a default factor of 0.1; pass
`--lr_decay_factor 0.2` to reproduce the original code's two-decay
behavior instead (decay_step2 will still fire, unlike the original).

Weight decay (L2) is applied only to conv/fc weights, not BatchNorm
affine parameters, matching the original's per-variable regularizer
(which was never attached to BN's beta/gamma).

Usage:
    python train.py --dataset cifar10                        # WarpNet1-73-4, K=2
    python train.py --dataset cifar100 --warp_factor 3 --num_residual_blocks 4  # WarpNet1-73-4, K=3
"""

import argparse
import csv
import itertools
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import build_datasets
from model import WarpNet, num_params


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--version", default="warpnet_run")
    p.add_argument("--train_dir", default=None, help="defaults to logs_<version>/")

    p.add_argument("--train_steps", type=int, default=80000)
    p.add_argument("--train_batch_size", type=int, default=128)
    p.add_argument("--validation_batch_size", type=int, default=250)
    p.add_argument("--report_freq", type=int, default=391)

    p.add_argument("--init_lr", type=float, default=0.1)
    p.add_argument(
        "--lr_decay_factor", type=float, default=0.1,
        help="the paper states 0.1 (dropping at epochs 60/120/160); the original code's flag "
             "default was 0.2 and it only ever applied decay_step0/1, never decay_step2 -- pass "
             "--lr_decay_factor 0.2 to reproduce that exact (likely unintended) behavior instead",
    )
    p.add_argument("--decay_step0", type=int, default=23460, help="~epoch 60")
    p.add_argument("--decay_step1", type=int, default=46920, help="~epoch 120")
    p.add_argument("--decay_step2", type=int, default=62560, help="~epoch 160")

    p.add_argument("--num_residual_blocks", type=int, default=6, help="Nwarp: warp operators per stage")
    p.add_argument("--warp_factor", type=int, choices=[2, 3], default=2, help="K: residual units per warp operator")
    p.add_argument("--k", type=int, default=4, help="kw: width multiplier")
    p.add_argument("--weight_decay", type=float, default=0.0005)
    p.add_argument("--padding_size", type=int, default=2)

    p.add_argument("--ckpt_path", default=None, help="checkpoint to resume from")
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def top1_error(logits, labels):
    predictions = logits.argmax(dim=1)
    correct = (predictions == labels).sum().item()
    return 1.0 - correct / labels.size(0)


def build_optimizer(model, init_lr, weight_decay):
    """conv/fc weights get weight decay; BatchNorm affine params don't."""
    decay, no_decay = [], []
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            no_decay.extend([module.weight, module.bias])
    no_decay_ids = {id(p) for p in no_decay}
    decay = [p for p in model.parameters() if id(p) not in no_decay_ids]

    return torch.optim.SGD(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=init_lr,
        momentum=0.9,
    )


@torch.no_grad()
def full_validation(model, test_loader, device, criterion):
    model.eval()
    losses, errors = [], []
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        losses.append(criterion(logits, labels).item())
        errors.append(top1_error(logits, labels))
    model.train()
    return sum(losses) / len(losses), sum(errors) / len(errors)


def main():
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    train_dir = args.train_dir or f"logs_{args.version}/"
    os.makedirs(train_dir, exist_ok=True)

    train_set, test_set, num_classes = build_datasets(args.dataset, args.data_dir, args.padding_size)
    train_loader = DataLoader(
        train_set, batch_size=args.train_batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True, persistent_workers=args.num_workers > 0,
    )
    test_loader = DataLoader(
        test_set, batch_size=args.validation_batch_size, shuffle=False,
        num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
    )
    train_iter = itertools.cycle(train_loader)

    model = WarpNet(
        num_residual_blocks=args.num_residual_blocks, warp_factor=args.warp_factor,
        k=args.k, num_classes=num_classes,
    ).to(device)
    num_params(model)

    start_step = 0
    if args.ckpt_path:
        ckpt = torch.load(args.ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        start_step = ckpt["step"] + 1
        print(f"Restored from checkpoint at step {ckpt['step']}")

    optimizer = build_optimizer(model, args.init_lr, args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    lr = args.init_lr

    csv_path = os.path.join(train_dir, f"{args.version}_error.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["step", "train_error", "validation_error"])

    print("Start training...")
    print("----------------------------")
    model.train()

    for step in range(start_step, args.train_steps):
        images, labels = next(train_iter)
        images, labels = images.to(device), labels.to(device)

        if step % args.report_freq == 0:
            val_loss, val_error = full_validation(model, test_loader, device, criterion)
        else:
            val_error = None

        start_time = time.time()

        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        train_error = top1_error(logits.detach(), labels)

        duration = time.time() - start_time

        if step % args.report_freq == 0:
            examples_per_sec = args.train_batch_size / duration
            print(
                f"step {step}, loss = {loss.item():.4f} "
                f"({examples_per_sec:.1f} examples/sec; {duration:.3f} sec/batch)"
            )
            print(f"Train top1 error = {train_error:.4f}")
            print(f"Validation top1 error = {val_error:.4f}")
            print(f"Validation loss = {val_loss:.4f}")
            print("----------------------------")
            csv_writer.writerow([step, train_error, val_error])
            csv_file.flush()

        if step in (args.decay_step0, args.decay_step1, args.decay_step2):
            lr *= args.lr_decay_factor
            print(f"Learning rate decayed to {lr}")

        if step % 10000 == 0 or (step + 1) == args.train_steps:
            ckpt_path = os.path.join(train_dir, f"model_step{step}.pt")
            torch.save({"model": model.state_dict(), "step": step}, ckpt_path)

    csv_file.close()


if __name__ == "__main__":
    main()
