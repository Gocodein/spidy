"""
DETECTOR AI — Pipeline Orchestrator
=====================================
Coordinates all 6 stages of the detection pipeline:
  1. Detection  →  2. Classification  →  3. Tracking
  4. Behavior   →  5. Disturbance     →  6. Logging

Each frame flows through every stage; results are drawn as an
annotated overlay with a live HUD before display / recording.

Author: DETECTOR AI Team
"""

from __future__ import annotations

import time
import logging
from typing import Dict, List, Optional, Set

import cv2
import numpy as np

from detector_ai.config import (
    PipelineConfig,
    COLOR_ANIMAL_SAFE,
    COLOR_ANIMAL_DISTURBED,
    COLOR_HUMAN,
    COLOR_VEHICLE,
    COLOR_ALERT_TEXT,
    COLOR_INFO_TEXT,
    COLOR_TRACK_LINE,
    FONT,
    FONT_SCALE,
    FONT_THICKNESS,
    BBOX_THICKNESS,
    BEHAVIOR_HISTORY_WINDOW,
)
from detector_ai.stage1_detector import AnimalDetector, Detection
from detector_ai.stage2_classifier import SpeciesClassifier
from detector_ai.stage3_tracker import MultiObjectTracker, TrackedObject
from detector_ai.stage4_behavior import BehaviorEstimator
from detector_ai.stage5_disturbance import DisturbanceAnalyzer, DisturbanceEvent
from detector_ai.stage6_logging import WildlifeDB, AlertManager

logger = logging.getLogger("detector_ai.pipeline")

# ── Trajectory visualisation ────────────────────────────────────
_TRAJECTORY_MAX_PTS = 60  # max points kept per track for drawing


class DetectorAIPipeline:
    """End-to-end real-time wildlife detection & monitoring pipeline.

    Parameters
    ----------
    config : PipelineConfig
        Runtime configuration (video source, model paths, thresholds …).
    """

    # ------------------------------------------------------------------ init
    def __init__(self, config: PipelineConfig) -> None:
        self.cfg = config

        # Stage 1 — Detector
        self.detector = AnimalDetector(
            weights=self.cfg.yolo_weights,
            conf_threshold=self.cfg.detection_conf,
            tracker=self.cfg.tracker,
        )

        # Stage 2 — Species classifier (optional — skip if no weights)
        self.classifier: Optional[SpeciesClassifier] = None
        if self.cfg.classifier_weights:
            self.classifier = SpeciesClassifier(
                weights_path=self.cfg.classifier_weights,
                conf_threshold=self.cfg.species_conf,
            )

        # Stage 3 — Multi-object tracker
        self.tracker = MultiObjectTracker(
            max_history=self.cfg.behavior_window,
        )

        # Stage 4 — Behavior estimator
        self.behavior = BehaviorEstimator(
            window=self.cfg.behavior_window,
        )

        # Stage 5 — Disturbance analyser
        self.disturbance = DisturbanceAnalyzer(
            distance_threshold=self.cfg.disturbance_distance,
        )

        # Stage 6 — Logging & alerts
        self.db: Optional[WildlifeDB] = None
        self._session_id: Optional[int] = None
        if self.cfg.log_to_db:
            self.db = WildlifeDB(db_path=self.cfg.db_path)
        self.alert_mgr = AlertManager()

        # Per-track trajectory history for drawing: {track_id: [(cx, cy), …]}
        self._trajectories: Dict[int, list] = {}

        # FPS bookkeeping
        self._fps: float = 0.0
        self._frame_times: list = []

        # Video I/O handles (set in process_video)
        self._cap: Optional[cv2.VideoCapture] = None
        self._writer: Optional[cv2.VideoWriter] = None

    # --------------------------------------------------------------- video loop
    def process_video(self) -> None:
        """Open the configured video source, run the pipeline, and display/save results."""

        source = self.cfg.video_source
        # Allow string-ified integers for webcam indices
        if isinstance(source, str) and source.isdigit():
            source = int(source)

        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            logger.error("Cannot open video source: %s", source)
            raise IOError(f"Cannot open video source: {source}")

        # Retrieve source properties for the writer
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        src_fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0

        if self.cfg.save_video and self.cfg.save_video_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(
                self.cfg.save_video_path, fourcc, src_fps, (width, height)
            )

        # Start DB session
        if self.db is not None:
            self._session_id = self.db.start_session(str(source))

        frame_num = 0
        logger.info("Pipeline started — source=%s  resolution=%dx%d", source, width, height)

        try:
            while True:
                ret, frame = self._cap.read()
                if not ret:
                    break

                frame_num += 1

                # Optional frame-skipping
                if self.cfg.frame_skip > 0 and frame_num % (self.cfg.frame_skip + 1) != 0:
                    continue

                t0 = time.perf_counter()

                # ── Run pipeline stages ──
                results = self.process_frame(frame, frame_num)

                # ── Visualise ──
                vis = self._draw_annotations(frame.copy(), results)
                vis = self._draw_hud(vis, results)

                # ── FPS tracking ──
                elapsed = time.perf_counter() - t0
                self._frame_times.append(elapsed)
                if len(self._frame_times) > 30:
                    self._frame_times.pop(0)
                self._fps = len(self._frame_times) / max(sum(self._frame_times), 1e-9)

                # ── Display / record ──
                if self.cfg.show_display:
                    cv2.imshow("DETECTOR AI", vis)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):  # ESC or 'q' to quit
                        break

                if self._writer is not None:
                    self._writer.write(vis)

        except KeyboardInterrupt:
            logger.info("Interrupted — shutting down.")
        finally:
            self.cleanup()

    # ------------------------------------------------------------- single frame
    def process_frame(self, frame: np.ndarray, frame_num: int) -> dict:
        """Run a single frame through all 6 pipeline stages.

        Parameters
        ----------
        frame : np.ndarray
            BGR image from OpenCV.
        frame_num : int
            1-based frame counter.

        Returns
        -------
        dict
            Keys: ``detections``, ``tracked``, ``behaviors``, ``disturbance_events``,
            ``animal_tracks``, ``human_tracks``, ``vehicle_tracks``,
            ``disturbed_ids`` (set of track IDs flagged as disturbed).
        """

        # ── Stage 1: Detection ──────────────────────────────────
        detections: List[Detection] = self.detector.detect(frame)

        # ── Stage 3: Tracking (runs before classifier to provide crops) ──
        tracked: List[TrackedObject] = self.tracker.update(detections)

        # Partition by category (accessing through .detection attribute)
        animal_tracks = [t for t in tracked if t.detection.category == "animal"]
        human_tracks = [t for t in tracked if t.detection.category == "human"]
        vehicle_tracks = [t for t in tracked if t.detection.category == "vehicle"]

        # ── Stage 2: Species classification (animal crops only) ──
        if self.classifier is not None and self.classifier.is_available:
            for track in animal_tracks:
                x1, y1, x2, y2 = track.detection.bbox
                crop = frame[
                    max(0, int(y1)):min(frame.shape[0], int(y2)),
                    max(0, int(x1)):min(frame.shape[1], int(x2)),
                ]
                if crop.size == 0:
                    continue
                result = self.classifier.classify(crop)
                if result is not None:
                    species, conf = result
                    track.species_override = species
                    track.species_confidence = conf

        # ── Stage 4: Behavior estimation ────────────────────────
        behaviors: Dict[int, str] = {}
        for track in animal_tracks:
            cx, cy = track.detection.center
            self.behavior.update(track.track_id, (cx, cy))
            behaviors[track.track_id] = self.behavior.classify(track.track_id)

            # Maintain trajectory buffer for drawing
            traj = self._trajectories.setdefault(track.track_id, [])
            traj.append((int(cx), int(cy)))
            if len(traj) > _TRAJECTORY_MAX_PTS:
                traj.pop(0)

        # ── Stage 5: Disturbance analysis ───────────────────────
        # Convert TrackedObjects to dicts expected by DisturbanceAnalyzer
        animal_dicts = []
        for track in animal_tracks:
            species_name = track.species_override or track.detection.class_name
            animal_dicts.append({
                "track_id": track.track_id,
                "center": track.detection.center,
                "species": species_name,
                "behavior": behaviors.get(track.track_id, "Observing"),
            })

        human_dicts = []
        for track in human_tracks:
            human_dicts.append({
                "track_id": track.track_id,
                "center": track.detection.center,
            })

        disturbance_events: List[DisturbanceEvent] = self.disturbance.analyze(
            animal_dicts, human_dicts,
        )
        disturbed_ids: Set[int] = {
            evt.animal_track_id for evt in disturbance_events
        }

        # ── Stage 6: Logging ────────────────────────────────────
        if self.db is not None and self._session_id is not None:
            for track in animal_tracks:
                species_name = track.species_override or track.detection.class_name
                cx, cy = track.detection.center
                is_disturbed = track.track_id in disturbed_ids
                self.db.log_detection(
                    session_id=self._session_id,
                    frame_num=frame_num,
                    track_id=track.track_id,
                    class_name=track.detection.class_name,
                    species=species_name,
                    confidence=track.detection.confidence,
                    behavior=behaviors.get(track.track_id, "Observing"),
                    x=cx,
                    y=cy,
                    disturbed=is_disturbed,
                )
            for evt in disturbance_events:
                self.db.log_disturbance(
                    session_id=self._session_id,
                    event=evt,
                )

        # Alerts
        for evt in disturbance_events:
            self.alert_mgr.send_alert(evt)

        return {
            "detections": detections,
            "tracked": tracked,
            "animal_tracks": animal_tracks,
            "human_tracks": human_tracks,
            "vehicle_tracks": vehicle_tracks,
            "behaviors": behaviors,
            "disturbance_events": disturbance_events,
            "disturbed_ids": disturbed_ids,
        }

    # ─────────────────────────────────────────── annotation drawing
    def _draw_annotations(self, frame: np.ndarray, results: dict) -> np.ndarray:
        """Draw bounding boxes, labels, trajectories, and disturbance warnings.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (will be mutated in-place and returned).
        results : dict
            Output of :meth:`process_frame`.

        Returns
        -------
        np.ndarray
            The annotated frame.
        """

        disturbed_ids: Set[int] = results["disturbed_ids"]
        behaviors: Dict[int, str] = results["behaviors"]

        # ── Animal tracks ──
        for track in results["animal_tracks"]:
            is_disturbed = track.track_id in disturbed_ids
            color = COLOR_ANIMAL_DISTURBED if is_disturbed else COLOR_ANIMAL_SAFE

            x1, y1, x2, y2 = map(int, track.detection.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, BBOX_THICKNESS)

            # Build label text
            species = track.species_override or track.detection.class_name
            beh = behaviors.get(track.track_id, "")
            label = f"#{track.track_id} {species}"
            if beh:
                label += f" | {beh}"
            if is_disturbed:
                label += " [DISTURBED]"

            # Label background
            (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                frame, label, (x1 + 2, y1 - 4),
                FONT, FONT_SCALE, (0, 0, 0), FONT_THICKNESS, cv2.LINE_AA,
            )

            # Trajectory polyline
            pts = self._trajectories.get(track.track_id, [])
            if len(pts) > 1:
                for i in range(1, len(pts)):
                    alpha = i / len(pts)  # fade old points
                    line_color = tuple(int(c * alpha) for c in COLOR_TRACK_LINE)
                    cv2.line(frame, pts[i - 1], pts[i], line_color, 2, cv2.LINE_AA)

        # ── Human tracks ──
        for track in results["human_tracks"]:
            x1, y1, x2, y2 = map(int, track.detection.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_HUMAN, BBOX_THICKNESS)
            label = f"#{track.track_id} human"
            (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), COLOR_HUMAN, -1)
            cv2.putText(
                frame, label, (x1 + 2, y1 - 4),
                FONT, FONT_SCALE, (0, 0, 0), FONT_THICKNESS, cv2.LINE_AA,
            )

        # ── Vehicle tracks ──
        for track in results["vehicle_tracks"]:
            x1, y1, x2, y2 = map(int, track.detection.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_VEHICLE, BBOX_THICKNESS)
            label = f"#{track.track_id} vehicle"
            (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), COLOR_VEHICLE, -1)
            cv2.putText(
                frame, label, (x1 + 2, y1 - 4),
                FONT, FONT_SCALE, (0, 0, 0), FONT_THICKNESS, cv2.LINE_AA,
            )

        return frame

    # ──────────────────────────────────────────────── HUD overlay
    def _draw_hud(self, frame: np.ndarray, results: dict) -> np.ndarray:
        """Draw the heads-up display: FPS, track counts, disturbance banner.

        Parameters
        ----------
        frame : np.ndarray
            Annotated frame.
        results : dict
            Output of :meth:`process_frame`.

        Returns
        -------
        np.ndarray
            Frame with HUD overlay.
        """

        h, w = frame.shape[:2]

        # ── Semi-transparent HUD background (top-left) ──
        hud_lines = [
            f"DETECTOR AI v1.0",
            f"FPS: {self._fps:.1f}",
            f"Animals: {len(results['animal_tracks'])}",
            f"Humans:  {len(results['human_tracks'])}",
            f"Vehicles:{len(results['vehicle_tracks'])}",
        ]

        line_h = 22
        hud_h = line_h * len(hud_lines) + 16
        hud_w = 220
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (hud_w, hud_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        for i, text in enumerate(hud_lines):
            color = (0, 200, 255) if i == 0 else COLOR_INFO_TEXT  # title in gold
            cv2.putText(
                frame, text, (8, 20 + i * line_h),
                FONT, 0.50, color, 1, cv2.LINE_AA,
            )

        # ── Disturbance warning banner ──
        if results["disturbance_events"]:
            severity_order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
            severities = [e.severity for e in results["disturbance_events"]]
            worst = max(severities, key=lambda s: (
                severity_order.index(s) if s in severity_order else -1
            ))
            banner = f"  !! DISTURBANCE DETECTED - {worst} !!  "

            overlay2 = frame.copy()
            cv2.rectangle(overlay2, (0, 0), (w, 40), (0, 0, 180), -1)
            cv2.addWeighted(overlay2, 0.70, frame, 0.30, 0, frame)

            (tw, th), _ = cv2.getTextSize(banner, FONT, 0.7, 2)
            tx = (w - tw) // 2
            cv2.putText(
                frame, banner, (tx, 28),
                FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
            )

        return frame

    # ──────────────────────────────────────────────── cleanup
    def cleanup(self) -> None:
        """Release all held resources (video capture, writer, database)."""

        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self.cfg.show_display:
            cv2.destroyAllWindows()
        if self.db is not None:
            if self._session_id is not None:
                try:
                    self.db.end_session(self._session_id)
                except Exception:
                    logger.warning("Failed to end session cleanly.", exc_info=True)
            self.db.close()

        logger.info("Pipeline resources released.")
