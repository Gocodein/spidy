#!/usr/bin/env python
"""
DETECTOR AI — EfficientNetV2-S Species Classifier Training
=============================================================
Train a fine-grained species classifier on cropped animal images.

Expected directory layout:
    training/tiger_dataset/crops/
    ├── bengal_tiger/       (positive class)
    │   ├── img_0001.jpg
    │   └── ...
    └── not_tiger/          (negative class — other animals / background)
        ├── img_0001.jpg
        └── ...

Usage:
    python training/train_classifier.py --data training/tiger_dataset/crops
    python training/train_classifier.py --data crops/ --epochs 30 --batch 8

Hardware: Optimized for NVIDIA RTX 4050 (6 GB VRAM) with AMP.
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image

# Resolve project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detector_ai.config import ClassifierConfig, MODELS_DIR

logger = logging.getLogger("detector_ai.train_classifier")


# ──────────────────────────────────────────────── Dataset ──
class SpeciesCropDataset(Dataset):
    """ImageFolder-style dataset for species crops.

    Expects one sub-folder per class inside *root_dir*.  The folder
    names become the class labels, sorted alphabetically (so
    ``bengal_tiger`` = 0, ``not_tiger`` = 1 with the default layout).

    Parameters
    ----------
    root_dir : Path
        Root directory containing one sub-folder per class.
    transform : transforms.Compose | None
        Torchvision transform pipeline.
    """

    def __init__(
        self,
        root_dir: Path,
        transform: transforms.Compose | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform

        # Discover classes from sub-folder names
        self.classes = sorted(
            [d.name for d in self.root_dir.iterdir() if d.is_dir()]
        )
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples: list[tuple[Path, int]] = []
        for cls_name in self.classes:
            cls_dir = self.root_dir / cls_name
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
                for img_path in cls_dir.glob(ext):
                    self.samples.append((img_path, self.class_to_idx[cls_name]))

        if not self.samples:
            raise FileNotFoundError(
                f"No images found under {self.root_dir}. "
                "Expected sub-folders with images (e.g. bengal_tiger/, not_tiger/)."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class TransformSubset(Dataset):
    """Apply a transform to a ``random_split`` Subset.

    Defined at module level (not inside ``main``) so that Windows
    multi-process DataLoader workers can pickle it.
    """

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img_path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ──────────────────────────────────────────── Transforms ──
def build_train_transform(img_size: int) -> transforms.Compose:
    """Augmentation pipeline including IR-simulation and motion-blur proxy."""
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandomGrayscale(p=0.15),        # simulates IR / night vision
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),  # motion blur proxy
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.15)),
    ])


def build_val_transform(img_size: int) -> transforms.Compose:
    """Deterministic validation transform (resize + center-crop)."""
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ──────────────────────────────────────────── Training ──
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float]:
    """Train for one epoch; return (avg_loss, accuracy)."""

    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float]:
    """Evaluate on validation set; return (avg_loss, accuracy)."""

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


# ──────────────────────────────────────────── Main ──
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    cc = ClassifierConfig()
    parser = argparse.ArgumentParser(
        description="Train EfficientNetV2-S species classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data", required=True,
        help="Root directory of species crops (sub-folders = classes).",
    )
    parser.add_argument("--output-dir", default=str(MODELS_DIR))
    parser.add_argument("--model-name", default=cc.model_name)
    parser.add_argument("--img-size", type=int, default=cc.img_size)
    parser.add_argument("--batch", type=int, default=cc.batch_size)
    parser.add_argument("--epochs", type=int, default=cc.epochs)
    parser.add_argument("--lr", type=float, default=cc.lr)
    parser.add_argument("--weight-decay", type=float, default=cc.weight_decay)
    parser.add_argument("--device", default=cc.device)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Fraction of data to use for validation.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(name)-32s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args(argv)

    data_root = Path(args.data)
    if not data_root.is_dir():
        logger.error("Data directory not found: %s", data_root)
        sys.exit(1)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"

    logger.info("=" * 60)
    logger.info("  DETECTOR AI — Species Classifier Training")
    logger.info("=" * 60)
    logger.info("  Data       : %s", data_root)
    logger.info("  Model      : %s", args.model_name)
    logger.info("  Image size : %d", args.img_size)
    logger.info("  Batch      : %d", args.batch)
    logger.info("  Epochs     : %d", args.epochs)
    logger.info("  LR         : %g", args.lr)
    logger.info("  Device     : %s", device)
    logger.info("  AMP        : %s", use_amp)
    logger.info("=" * 60)

    # ── Dataset ──
    full_ds = SpeciesCropDataset(data_root, transform=None)  # transform applied per split
    num_classes = len(full_ds.classes)
    logger.info("Classes (%d): %s", num_classes, full_ds.classes)
    logger.info("Total samples: %d", len(full_ds))

    val_size = int(len(full_ds) * args.val_split)
    train_size = len(full_ds) - val_size
    train_ds_raw, val_ds_raw = random_split(
        full_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    # Wrap subsets with appropriate transforms
    train_transform = build_train_transform(args.img_size)
    val_transform = build_val_transform(args.img_size)

    train_ds = TransformSubset(train_ds_raw, train_transform)
    val_ds = TransformSubset(val_ds_raw, val_transform)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    logger.info("Train: %d samples  |  Val: %d samples", len(train_ds), len(val_ds))

    # ── Model ──
    model = timm.create_model(
        args.model_name,
        pretrained=True,
        num_classes=num_classes,
    )
    model = model.to(device)
    logger.info("Model params: %.2f M", sum(p.numel() for p in model.parameters()) / 1e6)

    # ── Optimiser / scheduler / criterion ──
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler(enabled=use_amp)

    # ── Training loop ──
    best_val_acc = 0.0
    best_state = None
    patience_counter = 0
    patience_limit = 15

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp,
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device, use_amp)
        scheduler.step()

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        logger.info(
            "Epoch %3d/%d  │  train loss=%.4f  acc=%.2f%%  │  "
            "val loss=%.4f  acc=%.2f%%  │  lr=%.2e  │  %.1fs",
            epoch, args.epochs,
            train_loss, train_acc * 100,
            val_loss, val_acc * 100,
            lr_now, elapsed,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            logger.info("  ↑ New best val accuracy: %.2f%%", best_val_acc * 100)
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                logger.info("  Early stopping triggered (patience=%d).", patience_limit)
                break

    # ── Save best model ──
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "species_classifier_best.pth"

    if best_state is not None:
        checkpoint = {
            "model_name": args.model_name,
            "num_classes": num_classes,
            "classes": full_ds.classes,
            "img_size": args.img_size,
            "state_dict": best_state,
            "best_val_acc": best_val_acc,
        }
        torch.save(checkpoint, save_path)
        logger.info("Best model saved to: %s", save_path)
    else:
        logger.warning("No best state to save — training may have failed.")

    # ── Summary ──
    logger.info("")
    logger.info("=" * 60)
    logger.info("  Training complete!")
    logger.info("  Best val accuracy : %.2f%%", best_val_acc * 100)
    logger.info("  Saved weights     : %s", save_path)
    logger.info("")
    logger.info("  Next steps:")
    logger.info("    1. Update config.py: DEFAULT_CLASSIFIER_WEIGHTS = '%s'", save_path.name)
    logger.info("    2. Run pipeline:     python run_detector.py --species-model %s", save_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
