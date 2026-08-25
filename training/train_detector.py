#!/usr/bin/env python
"""
DETECTOR AI — YOLOv8 Detector Fine-Tuning
============================================
Fine-tune YOLOv8n (or any variant) on the Bengal Tiger dataset
prepared by ``setup_tiger_dataset.py``.

Usage:
    python training/train_detector.py --data tiger_dataset/tiger_data.yaml
    python training/train_detector.py --data tiger_data.yaml --epochs 50 --batch 16
    python training/train_detector.py --data tiger_data.yaml --model yolov8s.pt

Hardware: Optimized for NVIDIA RTX 4050 (6 GB VRAM).
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

# Resolve project root so imports work even when running from training/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detector_ai.config import TrainingConfig, MODELS_DIR, PROJECT_ROOT

logger = logging.getLogger("detector_ai.train_detector")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments, layering on top of TrainingConfig defaults."""

    tc = TrainingConfig()
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8 for Bengal Tiger detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data", required=True, type=str,
        help="Path to the YOLO data.yaml (e.g. tiger_dataset/tiger_data.yaml).",
    )
    parser.add_argument(
        "--model", default="yolov8n.pt",
        help="Pre-trained YOLO checkpoint to start from.",
    )
    parser.add_argument(
        "--output-dir", default=str(MODELS_DIR),
        help="Directory to copy the best weights into.",
    )
    parser.add_argument(
        "--weights-name", default=None,
        help="Filename for the best weights file (e.g. multispecies_best.pt). Defaults to <name>_best.pt.",
    )
    parser.add_argument("--epochs", type=int, default=tc.epochs)
    parser.add_argument("--batch", type=int, default=tc.batch_size)
    parser.add_argument("--imgsz", type=int, default=tc.img_size)
    parser.add_argument("--patience", type=int, default=tc.patience)
    parser.add_argument("--device", default=tc.device)
    parser.add_argument("--workers", type=int, default=tc.workers)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed-precision.")
    parser.add_argument("--project", default=str(PROJECT_ROOT / "runs" / "detect"))
    parser.add_argument("--name", default="multispecies_finetune")
    parser.add_argument("--erasing", type=float, default=0.4, help="Random erasing / cutout probability for occlusion simulation.")
    parser.add_argument("--degrees", type=float, default=10.0, help="Random rotation degrees.")
    parser.add_argument("--scale", type=float, default=0.5, help="Image scale gain (+/- gain).")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Load a YOLO model, fine-tune on the tiger dataset, and save best weights."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(name)-30s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args(argv)
    tc = TrainingConfig()

    data_path = Path(args.data)
    if not data_path.exists():
        logger.error("data.yaml not found: %s", data_path)
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("  DETECTOR AI — YOLOv8 Fine-Tuning")
    logger.info("=" * 60)
    logger.info("  Data YAML  : %s", data_path)
    logger.info("  Base model : %s", args.model)
    logger.info("  Epochs     : %d", args.epochs)
    logger.info("  Batch size : %d", args.batch)
    logger.info("  Image size : %d", args.imgsz)
    logger.info("  Device     : %s", args.device)
    logger.info("  AMP        : %s", not args.no_amp)
    logger.info("=" * 60)

    # ── Load model ──
    model = YOLO(args.model)

    # ── Train ──
    results = model.train(
        data=str(data_path.resolve()),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        patience=args.patience,
        optimizer=tc.optimizer,
        lr0=tc.lr0,
        lrf=tc.lrf,
        warmup_epochs=tc.warmup_epochs,
        augment=tc.augment,
        mosaic=tc.mosaic,
        mixup=tc.mixup,
        hsv_h=tc.hsv_h,
        hsv_s=tc.hsv_s,
        hsv_v=tc.hsv_v,
        flipud=tc.flipud,
        fliplr=tc.fliplr,
        degrees=args.degrees,
        scale=args.scale,
        erasing=args.erasing,
        device=args.device,
        amp=not args.no_amp,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=True,
        verbose=True,
    )

    # ── Copy best weights to models/ ──
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dir = Path(args.project) / args.name
    best_pt = run_dir / "weights" / "best.pt"
    weights_name = args.weights_name or ("multispecies_best.pt" if "multispecies" in str(data_path).lower() else f"{args.name}_best.pt")
    dest_pt = out_dir / weights_name

    if best_pt.exists():
        shutil.copy2(best_pt, dest_pt)
        logger.info("Best weights copied to: %s", dest_pt)
    else:
        logger.warning("best.pt not found at %s — check training output.", best_pt)

    # ── Summary ──
    logger.info("")
    logger.info("=" * 60)
    logger.info("  Training complete!")
    logger.info("  Run dir  : %s", run_dir)
    logger.info("  Best wts : %s", dest_pt if best_pt.exists() else "(not found)")
    logger.info("")
    logger.info("  Next steps:")
    logger.info("    1. Update config.py: DEFAULT_YOLO_WEIGHTS = '%s'", dest_pt.name)
    logger.info("    2. Run validation: yolo detect val model=%s data=%s", dest_pt, data_path)
    logger.info("    3. Launch pipeline: python run_detector.py -m %s", dest_pt)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
