#!/usr/bin/env python
"""
DETECTOR AI — CLI Entry Point
================================
Launch the real-time detection pipeline from the command line.

Usage examples:
    python run_detector.py                         # webcam, default settings
    python run_detector.py -s wildlife_clip.mp4    # video file
    python run_detector.py -s rtsp://cam:8554/live # RTSP stream
    python run_detector.py -m models/tiger_best.pt --conf 0.4
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from detector_ai.config import (
    PipelineConfig,
    DEFAULT_YOLO_WEIGHTS,
    DEFAULT_DB_PATH,
    DETECTION_CONF_THRESHOLD,
    DISTURBANCE_DISTANCE_PX,
)
from detector_ai.pipeline import DetectorAIPipeline

BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   ██████╗ ███████╗████████╗███████╗ ██████╗████████╗         ║
║   ██╔══██╗██╔════╝╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝        ║
║   ██║  ██║█████╗     ██║   █████╗  ██║        ██║            ║
║   ██║  ██║██╔══╝     ██║   ██╔══╝  ██║        ██║            ║
║   ██████╔╝███████╗   ██║   ███████╗╚██████╗   ██║            ║
║   ╚═════╝ ╚══════╝   ╚═╝   ╚══════╝ ╚═════╝   ╚═╝            ║
║                      🐯  DETECTOR AI  🐯                     ║
║   Endangered Species Detection & Tracking System              ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        prog="run_detector",
        description="DETECTOR AI — Real-time endangered species detection & tracking.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:  python run_detector.py -s wildlife.mp4 -m models/tiger_best.pt",
    )

    parser.add_argument(
        "-s", "--source",
        default="0",
        help="Video source: webcam index (0), file path, or RTSP URL. (default: 0)",
    )
    parser.add_argument(
        "-m", "--model",
        default=DEFAULT_YOLO_WEIGHTS,
        help=f"Path to YOLO weights. (default: {DEFAULT_YOLO_WEIGHTS})",
    )
    parser.add_argument(
        "--species-model",
        default=None,
        help="Path to species classifier weights. (default: None — skip classification)",
    )
    parser.add_argument(
        "--db",
        default=str(DEFAULT_DB_PATH),
        help=f"SQLite database file for event logging. (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--show", "--no-show",
        dest="show",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Display the live annotated window. (default: --show)",
    )
    parser.add_argument(
        "--save-video",
        default=None,
        help="If provided, save annotated output to this path (e.g. output.mp4).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DETECTION_CONF_THRESHOLD,
        help=f"Detection confidence threshold. (default: {DETECTION_CONF_THRESHOLD})",
    )
    parser.add_argument(
        "--disturbance-dist",
        type=int,
        default=DISTURBANCE_DISTANCE_PX,
        help=f"Disturbance proximity in pixels. (default: {DISTURBANCE_DISTANCE_PX})",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=0,
        help="Process every N-th frame only (0 = process all). (default: 0)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Main entry point — parse args, build config, run pipeline."""

    args = parse_args(argv)

    # ── Logging ──
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s │ %(name)-28s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Banner ──
    print(BANNER)

    # ── Build pipeline config ──
    source = int(args.source) if args.source.isdigit() else args.source

    config = PipelineConfig(
        video_source=source,
        yolo_weights=args.model,
        classifier_weights=args.species_model,
        db_path=Path(args.db),
        show_display=args.show,
        save_video=args.save_video is not None,
        save_video_path=args.save_video,
        detection_conf=args.conf,
        disturbance_distance=args.disturbance_dist,
        frame_skip=args.frame_skip,
    )

    logger = logging.getLogger("detector_ai.cli")
    logger.info("Source      : %s", config.video_source)
    logger.info("YOLO model  : %s", config.yolo_weights)
    logger.info("Species mdl : %s", config.classifier_weights or "(none)")
    logger.info("Database    : %s", config.db_path)
    logger.info("Confidence  : %.2f", config.detection_conf)
    logger.info("Disturbance : %d px", config.disturbance_distance)

    # ── Run ──
    pipeline = DetectorAIPipeline(config)
    try:
        pipeline.process_video()
    except Exception:
        logger.exception("Pipeline failed.")
        sys.exit(1)
    finally:
        pipeline.cleanup()

    logger.info("Session complete.")


if __name__ == "__main__":
    main()
