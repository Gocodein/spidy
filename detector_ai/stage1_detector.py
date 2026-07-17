"""
DETECTOR AI — Stage 1: Generic Object Detector
Wraps YOLOv8 for animal / human / vehicle detection with optional
ByteTrack-based multi-object tracking.

Classes detected via COCO pre-trained weights are mapped to categories
('animal', 'human', 'vehicle') using the class-ID sets defined in
``detector_ai.config``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from ultralytics import YOLO

from detector_ai.config import (
    COCO_ANIMAL_CLASS_IDS,
    COCO_HUMAN_CLASS_ID,
    COCO_VEHICLE_CLASS_IDS,
    DETECTION_CONF_THRESHOLD,
    TRACKER_TYPE,
    TRACK_PERSIST,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Detection dataclass
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """A single bounding-box detection from Stage 1.

    Attributes:
        bbox: ``(x1, y1, x2, y2)`` pixel coordinates of the bounding box.
        center: ``(cx, cy)`` center point of the bounding box.
        width: Width of the bounding box in pixels.
        height: Height of the bounding box in pixels.
        class_id: Integer class ID from the model's output.
        class_name: Human-readable class label (e.g. ``'cat'``).
        category: High-level category – one of ``'animal'``,
            ``'human'``, or ``'vehicle'``.
        confidence: Detection confidence score in ``[0, 1]``.
        track_id: Unique track identifier assigned by the tracker, or
            ``None`` when tracking is not active.
    """

    bbox: tuple  # (x1, y1, x2, y2)
    center: tuple  # (cx, cy)
    width: float
    height: float
    class_id: int
    class_name: str
    category: str  # 'animal', 'human', 'vehicle'
    confidence: float
    track_id: Optional[int] = None


# ---------------------------------------------------------------------------
# Helper: map a COCO class ID to a high-level category
# ---------------------------------------------------------------------------

def _categorize(class_id: int) -> Optional[str]:
    """Return ``'animal'``, ``'human'``, ``'vehicle'``, or ``None``."""
    if class_id in COCO_ANIMAL_CLASS_IDS:
        return "animal"
    if class_id == COCO_HUMAN_CLASS_ID:
        return "human"
    if class_id in COCO_VEHICLE_CLASS_IDS:
        return "vehicle"
    return None


# ---------------------------------------------------------------------------
# AnimalDetector
# ---------------------------------------------------------------------------

class AnimalDetector:
    """YOLOv8-based detector that returns structured :class:`Detection` objects.

    The detector supports two modes:

    * **Tracking mode** (``detect``) — uses ByteTrack/BotSort via
      ``model.track()`` so that each detection carries a persistent
      ``track_id``.
    * **Single-frame mode** (``detect_no_track``) — plain ``model()``
      inference with no tracking overhead.

    Parameters:
        weights:
            Path to a YOLOv8 ``.pt`` weights file (local or Ultralytics
            hub name such as ``'yolov8n.pt'``).
        conf_threshold:
            Minimum confidence score to keep a detection.
        tracker:
            Tracker config filename recognised by Ultralytics (e.g.
            ``'bytetrack.yaml'`` or ``'botsort.yaml'``).
    """

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        conf_threshold: float = DETECTION_CONF_THRESHOLD,
        tracker: str = TRACKER_TYPE,
    ) -> None:
        self.conf_threshold = conf_threshold
        self.tracker = tracker

        logger.info("Loading YOLO model from '%s' ...", weights)
        self.model = YOLO(weights)
        self._class_names: dict = self.model.names  # {int: str}
        logger.info(
            "YOLO model loaded — %d classes available.", len(self._class_names)
        )

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run detection **with** ByteTrack tracking on *frame*.

        Parameters:
            frame: BGR image as a NumPy array (H × W × 3, ``uint8``).

        Returns:
            List of :class:`Detection` instances for the frame.  Only
            detections whose COCO class maps to a known category
            (animal / human / vehicle) and whose confidence ≥
            ``conf_threshold`` are included.
        """
        results = self.model.track(
            frame,
            persist=TRACK_PERSIST,
            tracker=self.tracker,
            conf=self.conf_threshold,
            verbose=False,
        )
        return self._parse_results(results, with_tracking=True)

    def detect_no_track(self, frame: np.ndarray) -> List[Detection]:
        """Run detection **without** tracking (single-image inference).

        Parameters:
            frame: BGR image as a NumPy array (H × W × 3, ``uint8``).

        Returns:
            List of :class:`Detection` instances.
        """
        results = self.model(
            frame,
            conf=self.conf_threshold,
            verbose=False,
        )
        return self._parse_results(results, with_tracking=False)

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _parse_results(
        self,
        results,
        *,
        with_tracking: bool,
    ) -> List[Detection]:
        """Convert raw Ultralytics results into a list of :class:`Detection`.

        Parameters:
            results: Output of ``model()`` or ``model.track()``.
            with_tracking: Whether to extract ``track_id`` from results.

        Returns:
            Filtered and structured detection list.
        """
        detections: List[Detection] = []

        if not results or results[0].boxes is None:
            return detections

        boxes = results[0].boxes

        # Extract tensors --------------------------------------------------
        xyxy = boxes.xyxy.cpu().numpy()        # (N, 4)
        confs = boxes.conf.cpu().numpy()       # (N,)
        class_ids = boxes.cls.cpu().numpy().astype(int)  # (N,)

        # Track IDs (may be None when tracker hasn't locked on yet)
        track_ids = None
        if with_tracking and boxes.id is not None:
            track_ids = boxes.id.cpu().numpy().astype(int)

        for idx in range(len(xyxy)):
            cid = int(class_ids[idx])
            conf = float(confs[idx])

            # Filter by confidence (redundant if YOLO honours conf= but
            # acts as a safety net).
            if conf < self.conf_threshold:
                continue

            category = _categorize(cid)
            if category is None:
                continue  # skip classes we don't care about

            x1, y1, x2, y2 = xyxy[idx]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w = float(x2 - x1)
            h = float(y2 - y1)

            tid: Optional[int] = None
            if track_ids is not None and idx < len(track_ids):
                tid = int(track_ids[idx])

            class_name = self._class_names.get(cid, f"class_{cid}")

            detections.append(
                Detection(
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    center=(float(cx), float(cy)),
                    width=w,
                    height=h,
                    class_id=cid,
                    class_name=class_name,
                    category=category,
                    confidence=conf,
                    track_id=tid,
                )
            )

        return detections
