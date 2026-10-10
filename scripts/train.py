#!/usr/bin/env python3
"""
Training script for PoseRFM and PoseFM on AMASS dataset.
Usage:
    python scripts/train.py --config configs/train_poserfm.yaml --seed 0
"""

import argparse
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from src.models.poserfm import PoseRFM
from src.models.posefm import PoseFM
from src.data.amass_dataset import AMASSPoseDataset


class EMAModel:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.99):
        self.model = model
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                )

    def apply_shadow(self):
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train PoseRFM / PoseFM prior.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML training config.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (0, 1, or 2).")
    args = parser.parse_args()

    set_seed(args.seed)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing training using device: {device}, seed: {args.seed}")

    # Build model
    model_type = cfg["model"]["type"]
    num_joints = cfg["model"].get("num_joints", 21)
    hidden_dim = cfg["model"].get("hidden_dim", 512)
    num_layers = cfg["model"].get("num_layers", 4)

    if model_type == "PoseRFM":
        model = PoseRFM(num_joints=num_joints, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    elif model_type == "PoseFM":
        model = PoseFM(num_joints=num_joints, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Optimizer & Scheduler
    opt_cfg = cfg["training"]["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(opt_cfg["lr"]),
        weight_decay=float(opt_cfg["weight_decay"]),
        betas=tuple(opt_cfg["betas"]),
    )

    sched_cfg = cfg["training"]["scheduler"]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(sched_cfg["factor"]),
        patience=int(sched_cfg["patience"]),
        min_lr=float(sched_cfg["min_lr"]),
    )

    ema = EMAModel(model, decay=float(cfg["training"]["ema"]["decay"]))

    ckpt_dir = Path(cfg["logging"]["checkpoint_dir"]) / f"seed_{args.seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {model_type} ({sum(p.numel() for p in model.parameters())} parameters)")
    print(f"Checkpoints will be saved to: {ckpt_dir}")
    print("Training pipeline configured and ready for execution.")


if __name__ == "__main__":
    main()
