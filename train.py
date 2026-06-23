"""
Training script for the proposed Residual + Attention U-Net++ model,
following the experimental setup in Section 4.3 of the paper:

  - PyTorch, Adam optimizer, lr=1e-4, weight_decay=1e-5
  - ReduceLROnPlateau: factor=0.5, patience=10 (on validation loss)
  - Max 100 epochs, early stopping if val Dice doesn't improve for 15 epochs
  - Batch size 8
  - Five-fold cross-validation with patient-level splits
  - Global seed = 42 for reproducibility (torch, numpy, random)

Usage
-----
    python train.py --config configs/default.yaml --fold 0

This script expects you to plug in your own LIDC-IDRI loading logic to
populate `images`, `masks`, and `patient_ids` (see `data/lidc_dataset.py`
for the preprocessing primitives this repo provides). A `--dry-run` mode
with synthetic data is included so the full training loop can be sanity
checked without the real dataset.
"""

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from models import build_model
from losses import HybridDiceFocalLoss
from utils.metrics import compute_all_metrics, aggregate_metrics
from data.lidc_dataset import LungNoduleSliceDataset


def set_global_seed(seed: int = 42):
    """Fix the seed for Python's random, NumPy, and PyTorch (Sec 4.3)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    opt_cfg = cfg["optimizer"]
    if opt_cfg["name"].lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=opt_cfg["lr"], weight_decay=opt_cfg["weight_decay"])
    elif opt_cfg["name"].lower() == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=opt_cfg["lr"], weight_decay=opt_cfg["weight_decay"])
    raise ValueError(f"Unknown optimizer: {opt_cfg['name']}")


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: dict):
    sch_cfg = cfg["scheduler"]
    if sch_cfg["name"] == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=sch_cfg["factor"], patience=sch_cfg["patience"]
        )
    raise ValueError(f"Unknown scheduler: {sch_cfg['name']}")


class EarlyStopping:
    """Stops training if the monitored metric doesn't improve for `patience` epochs."""

    def __init__(self, patience: int = 15, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.best = None
        self.counter = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        if self.best is None:
            self.best = value
            return False

        improved = (value > self.best) if self.mode == "max" else (value < self.best)
        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, total_dice_loss, total_focal_loss = 0.0, 0.0, 0.0
    n_batches = 0

    with torch.set_grad_enabled(train):
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)

            if train:
                optimizer.zero_grad()

            logits = model(images)
            if isinstance(logits, dict):
                logits = logits["out"]

            loss_dict = criterion(logits, masks)
            loss = loss_dict["loss"]

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_dice_loss += loss_dict["dice_loss"].item()
            total_focal_loss += loss_dict["focal_loss"].item()
            n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "dice_loss": total_dice_loss / max(n_batches, 1),
        "focal_loss": total_focal_loss / max(n_batches, 1),
    }


@torch.no_grad()
def evaluate_metrics(model, loader, device, threshold: float = 0.5, spacing: float = 1.0):
    model.eval()
    all_metrics = []
    for images, masks in loader:
        images = images.to(device)
        logits = model(images)
        if isinstance(logits, dict):
            logits = logits["out"]
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float().cpu().numpy()
        gts = masks.cpu().numpy()

        for b in range(preds.shape[0]):
            m = compute_all_metrics(preds[b, 0], gts[b, 0], spacing=spacing)
            all_metrics.append(m)

    return aggregate_metrics(all_metrics)


def train_one_fold(
    cfg: dict,
    train_dataset: LungNoduleSliceDataset,
    val_dataset: LungNoduleSliceDataset,
    fold_idx: int,
    device: torch.device,
):
    train_cfg = cfg["training"]
    ckpt_cfg = cfg["checkpoint"]

    num_workers = train_cfg.get("num_workers", 4)
    train_loader = DataLoader(
        train_dataset, batch_size=train_cfg["batch_size"], shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(), drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=train_cfg["batch_size"], shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )

    model = build_model(cfg).to(device)
    criterion = HybridDiceFocalLoss(
        lambda_dice=cfg["loss"]["lambda_dice"],
        lambda_focal=cfg["loss"]["lambda_focal"],
        focal_alpha=cfg["loss"]["focal_alpha"],
        focal_gamma=cfg["loss"]["focal_gamma"],
        dice_eps=cfg["loss"]["dice_eps"],
    )
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    early_stopper = EarlyStopping(patience=train_cfg["early_stopping_patience"], mode="max")

    out_dir = Path(ckpt_cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val_dice = -1.0
    best_ckpt_path = out_dir / f"fold{fold_idx}_best.pt"

    for epoch in range(1, train_cfg["max_epochs"] + 1):
        t0 = time.time()

        train_stats = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_stats = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        val_metrics = evaluate_metrics(model, val_loader, device, spacing=cfg["data"]["voxel_spacing_mm"])

        scheduler.step(val_stats["loss"])
        elapsed = time.time() - t0

        print(
            f"[fold {fold_idx}] epoch {epoch:03d} | "
            f"train_loss={train_stats['loss']:.4f} val_loss={val_stats['loss']:.4f} "
            f"val_dice={val_metrics['dice']:.4f} val_iou={val_metrics['iou']:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e} time={elapsed:.1f}s"
        )

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "config": cfg,
                },
                best_ckpt_path,
            )

        if early_stopper.step(val_metrics["dice"]):
            print(f"[fold {fold_idx}] Early stopping triggered at epoch {epoch} "
                  f"(no val_dice improvement for {train_cfg['early_stopping_patience']} epochs).")
            break

    return best_ckpt_path, best_val_dice


def build_dry_run_data(n_train: int = 12, n_val: int = 4, size: int = 64):
    """Synthetic data generator for sanity-checking the training loop
    without the real LIDC-IDRI dataset."""
    rng = np.random.RandomState(0)

    def make_split(n):
        images, masks = [], []
        for _ in range(n):
            img = rng.rand(size, size).astype(np.float32)
            mask = np.zeros((size, size), dtype=np.float32)
            cy, cx = rng.randint(size // 4, 3 * size // 4, size=2)
            r = rng.randint(5, 15)
            yy, xx = np.ogrid[:size, :size]
            mask[(yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2] = 1.0
            images.append(img)
            masks.append(mask)
        return images, masks

    train_imgs, train_masks = make_split(n_train)
    val_imgs, val_masks = make_split(n_val)
    return train_imgs, train_masks, val_imgs, val_masks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="Run with synthetic data to sanity-check the pipeline.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg["training"]["seed"])

    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.dry_run:
        print("Running in --dry-run mode with synthetic data.")
        # Shrink epochs for a fast sanity check.
        cfg["training"]["max_epochs"] = 2
        train_imgs, train_masks, val_imgs, val_masks = build_dry_run_data()
        train_ds = LungNoduleSliceDataset(train_imgs, train_masks, augment=True)
        val_ds = LungNoduleSliceDataset(val_imgs, val_masks, augment=False)
    else:
        raise NotImplementedError(
            "Plug in your LIDC-IDRI loading here: build `train_ds`/`val_ds` "
            "(LungNoduleSliceDataset instances) using the preprocessing "
            "primitives in data/lidc_dataset.py, with patient-level folds "
            "from `patient_level_kfold_splits`. Use --dry-run to test the "
            "training loop with synthetic data."
        )

    best_ckpt, best_dice = train_one_fold(cfg, train_ds, val_ds, args.fold, device)
    print(f"Best checkpoint for fold {args.fold}: {best_ckpt} (val_dice={best_dice:.4f})")


if __name__ == "__main__":
    main()
