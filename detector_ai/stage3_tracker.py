"""
DETECTOR AI — Stage 3: Multi-Object Tracking Wrapper
Maintains per-track state (trajectory history, age, species override)
on top of the track IDs already assigned by YOLO's ByteTrack / BotSort
in Stage 1.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

from detector_ai.config import BEHAVIOR_HISTORY_WINDOW
from detector_ai.stage1_detector import Detection

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TrackedObject dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrackedObject:
    """Persistent state for a single tracked entity.

    Attributes:
        detection:
            The most recent :class:`Detection` from Stage 1.
        track_id:
            Unique integer assigned by the upstream tracker.
        trajectory:
            FIFO history of ``(cx, cy)`` centre-point positions (most
            recent entry is last).
        age:
            Total number of frames in which this track has been observed.
        frames_since_seen:
            Number of consecutive frames since the last observation.
            Reset to ``0`` on every :meth:`MultiObjectTracker.update`
            that includes this track.
        species_override:
            Fine-grained species label injected by the Stage 2
            classifier (e.g. ``'bengal_tiger'``).  ``None`` until the
            classifier runs.
        species_confidence:
            Confidence of the species classification.
    """

    detection: Detection
    track_id: int
    trajectory: Deque[tuple] = field(default_factory=deque)
    age: int = 0
    frames_since_seen: int = 0
    species_override: Optional[str] = None
    species_confidence: float = 0.0


# ---------------------------------------------------------------------------
# MultiObjectTracker
# ---------------------------------------------------------------------------

class MultiObjectTracker:
    """Lightweight tracking wrapper that enriches YOLO detections with
    trajectory history, age counters, and species overrides.

    Track IDs are **not** computed here — they come from the ByteTrack /
    BotSort tracker embedded in YOLO (see
    :meth:`AnimalDetector.detect`).  This class simply maintains
    persistent state keyed by those IDs.

    Parameters:
        max_history:
            Maximum number of ``(cx, cy)`` positions to retain per
            track for trajectory / behaviour analysis.
    """

    def __init__(self, max_history: int = BEHAVIOR_HISTORY_WINDOW) -> None:
        self.max_history = max_history
        self._tracks: Dict[int, TrackedObject] = {}

    # -----------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------

    @property
    def active_tracks(self) -> Dict[int, TrackedObject]:
        """Return the full dictionary of currently-tracked objects."""
        return self._tracks

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def update(self, detections: List[Detection]) -> List[TrackedObject]:
        """Ingest a new batch of detections and return updated tracks.

        For every detection that carries a ``track_id``:

        * If the track already exists its state is updated in-place.
        * If the track is new a :class:`TrackedObject` is created.
        * All active tracks have ``frames_since_seen`` incremented
          **before** matched tracks are reset to ``0``.

        Detections without a ``track_id`` (``None``) are silently
        skipped because no persistent state can be maintained for them.

        Parameters:
            detections: List of :class:`Detection` from the current
                frame (produced by Stage 1).

        Returns:
            List of :class:`TrackedObject` that were observed in this
            frame (i.e. ``frames_since_seen == 0``).
        """
        # Age all existing tracks by one frame --------------------------------
        for tracked in self._tracks.values():
            tracked.frames_since_seen += 1

        seen_this_frame: List[TrackedObject] = []

        for det in detections:
            tid = det.track_id
            if tid is None:
                continue  # cannot track without an ID

            if tid in self._tracks:
                # --- Update existing track -----------------------------------
                tracked = self._tracks[tid]
                tracked.detection = det
                tracked.age += 1
                tracked.frames_since_seen = 0
                tracked.trajectory.append(det.center)
                # Trim trajectory to max_history
                while len(tracked.trajectory) > self.max_history:
                    tracked.trajectory.popleft()
            else:
                # --- Create new track ----------------------------------------
                traj: Deque[tuple] = deque(maxlen=self.max_history)
                traj.append(det.center)
                tracked = TrackedObject(
                    detection=det,
                    track_id=tid,
                    trajectory=traj,
                    age=1,
                    frames_since_seen=0,
                )
                self._tracks[tid] = tracked
                logger.debug("New track id=%d  class=%s", tid, det.class_name)

            seen_this_frame.append(tracked)

        return seen_this_frame

    def get_tracks_by_category(self, category: str) -> List[TrackedObject]:
        """Return active tracks whose detection category matches *category*.

        Parameters:
            category: One of ``'animal'``, ``'human'``, ``'vehicle'``.

        Returns:
            Filtered list (references, not copies).
        """
        return [
            t
            for t in self._tracks.values()
            if t.detection.category == category
        ]

    def set_species(
        self,
        track_id: int,
        species: str,
        confidence: float,
    ) -> None:
        """Inject a species classification into a track.

        Called by the pipeline after Stage 2 classifies an animal crop.

        Parameters:
            track_id: The track to update.
            species: Species label (e.g. ``'bengal_tiger'``).
            confidence: Classification confidence.
        """
        if track_id in self._tracks:
            tracked = self._tracks[track_id]
            # Only overwrite if the new classification is more confident
            if confidence > tracked.species_confidence:
                tracked.species_override = species
                tracked.species_confidence = confidence

    def cleanup_lost_tracks(self, max_lost_frames: int = 30) -> int:
        """Remove tracks that have not been observed for too long.

        Parameters:
            max_lost_frames: Number of consecutive unobserved frames
                after which a track is purged.

        Returns:
            Number of tracks removed.
        """
        lost_ids = [
            tid
            for tid, t in self._tracks.items()
            if t.frames_since_seen > max_lost_frames
        ]
        for tid in lost_ids:
            del self._tracks[tid]

        if lost_ids:
            logger.debug(
                "Cleaned up %d lost track(s): %s", len(lost_ids), lost_ids
            )

        return len(lost_ids)

    def reset(self) -> None:
        """Clear all tracked state (e.g. when switching video sources)."""
        self._tracks.clear()
        logger.info("Tracker state reset — all tracks cleared.")
