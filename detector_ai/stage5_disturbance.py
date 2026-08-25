"""
DETECTOR AI — Stage 5: Disturbance Analysis
Detects human–wildlife disturbance by correlating animal proximity to
humans/vehicles with behavioural shifts (calm → alert/fleeing).

Severity levels (ascending):
    LOW       — animal within threshold but calm
    MEDIUM    — animal within threshold and alert
    HIGH      — animal within critical range OR shifted to fleeing
    CRITICAL  — animal within critical range AND behaviour shifted
"""

from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from detector_ai.config import (
    BEHAVIOR_SHIFT_WINDOW,
    DISTURBANCE_CRITICAL_PX,
    DISTURBANCE_DISTANCE_PX,
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    SEVERITY_LOW,
    SEVERITY_MEDIUM,
    SPECIES_DISTURBANCE_THRESHOLDS,
)

# ---------------------------------------------------------------------------
# Calm / alert behaviour sets (used for shift detection)
# ---------------------------------------------------------------------------
_CALM_STATES = frozenset({"Resting", "Walking/Grazing", "Observing", "Stalking"})
_ALERT_STATES = frozenset({"Alert/Pacing", "Fleeing/Running"})


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DisturbanceEvent:
    """Record of a single disturbance incident."""

    timestamp: float
    animal_track_id: int
    animal_species: str
    animal_behavior_before: str
    animal_behavior_after: str
    human_track_id: Optional[int]
    distance_px: float
    severity: str          # LOW | MEDIUM | HIGH | CRITICAL
    description: str


@dataclass
class _TrackRecord:
    """Internal ring-buffer of recent behaviours for one animal track."""

    behaviors: deque = field(default_factory=lambda: deque(maxlen=30))
    last_species: str = "unknown"


# ---------------------------------------------------------------------------
# Main analyser
# ---------------------------------------------------------------------------

class DisturbanceAnalyzer:
    """Analyse animal–human proximity and flag disturbance events.

    Parameters
    ----------
    distance_threshold : int
        Pixel distance at which proximity counts as a potential
        disturbance (default from config).
    critical_distance : int
        Pixel distance for critical alerts (default from config).
    shift_window : int
        Number of recent behaviour records to check for a
        calm → alert shift (default from config).
    """

    def __init__(
        self,
        distance_threshold: int = DISTURBANCE_DISTANCE_PX,
        critical_distance: int = DISTURBANCE_CRITICAL_PX,
        shift_window: int = BEHAVIOR_SHIFT_WINDOW,
    ) -> None:
        self.distance_threshold = distance_threshold
        self.critical_distance = critical_distance
        self.shift_window = shift_window

        # Per-animal-track behaviour history
        self._records: Dict[int, _TrackRecord] = defaultdict(_TrackRecord)

    # ------------------------------------------------------------------
    # Public helpers — call each frame to feed behaviour history
    # ------------------------------------------------------------------

    def record_behavior(
        self, track_id: int, behavior: str, species: str = "unknown"
    ) -> None:
        """Store a behaviour observation for later shift detection.

        Should be called once per animal track per frame *before*
        calling :meth:`analyze`.
        """
        rec = self._records[track_id]
        rec.behaviors.append(behavior)
        rec.last_species = species

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        animal_tracks: List[Dict],
        human_tracks: List[Dict],
        behavior_estimator=None,
    ) -> List[DisturbanceEvent]:
        """Run disturbance analysis for the current frame.

        Parameters
        ----------
        animal_tracks : list[dict]
            Each dict must contain at minimum:
            ``{"track_id": int, "center": (x, y), "species": str,
              "behavior": str}``.
        human_tracks : list[dict]
            Each dict must contain at minimum:
            ``{"track_id": int, "center": (x, y)}``.
        behavior_estimator : BehaviorEstimator, optional
            If provided, the analyser will pull the latest behaviour
            from it instead of the dicts (handy when behaviours are
            classified externally).

        Returns
        -------
        list[DisturbanceEvent]
        """
        if not animal_tracks or not human_tracks:
            return []

        events: List[DisturbanceEvent] = []
        now = time.time()

        for animal in animal_tracks:
            a_id: int = animal["track_id"]
            a_center: Tuple[float, float] = animal["center"]
            a_species: str = animal.get("species", "unknown")

            # Current behaviour — prefer estimator if available
            if behavior_estimator is not None:
                current_behavior = behavior_estimator.classify(a_id)
            else:
                current_behavior = animal.get("behavior", "Observing")

            # Feed history
            self.record_behavior(a_id, current_behavior, a_species)

            # Species-specific thresholds
            sp_thresh = SPECIES_DISTURBANCE_THRESHOLDS.get(
                a_species, SPECIES_DISTURBANCE_THRESHOLDS["default"]
            )
            sp_distance = sp_thresh["distance"]
            sp_critical = sp_thresh["critical"]

            # Check against every human / vehicle track
            closest_dist = float("inf")
            closest_human_id: Optional[int] = None

            for human in human_tracks:
                h_id: int = human["track_id"]
                h_center: Tuple[float, float] = human["center"]
                dist = _euclidean(a_center, h_center)

                if dist < closest_dist:
                    closest_dist = dist
                    closest_human_id = h_id

            # Skip if the closest human is beyond the outer threshold
            if closest_dist > sp_distance:
                continue

            # Detect behaviour shift
            shifted, prev_behavior = self._detect_shift(a_id)

            # Compute severity
            severity = self._compute_severity_species(
                closest_dist, shifted, sp_distance, sp_critical
            )

            # Build description
            desc = self._build_description(
                a_species,
                closest_dist,
                severity,
                shifted,
                prev_behavior,
                current_behavior,
            )

            events.append(
                DisturbanceEvent(
                    timestamp=now,
                    animal_track_id=a_id,
                    animal_species=a_species,
                    animal_behavior_before=prev_behavior,
                    animal_behavior_after=current_behavior,
                    human_track_id=closest_human_id,
                    distance_px=round(closest_dist, 1),
                    severity=severity,
                    description=desc,
                )
            )

        return events

    # ------------------------------------------------------------------
    # Severity computation
    # ------------------------------------------------------------------

    def compute_severity(self, distance: float, behavior_shifted: bool) -> str:
        """Determine disturbance severity.

        Parameters
        ----------
        distance : float
            Pixel distance between animal and closest human.
        behavior_shifted : bool
            Whether a calm → alert/fleeing shift was detected
            within the shift window.

        Returns
        -------
        str
            One of ``SEVERITY_LOW``, ``SEVERITY_MEDIUM``,
            ``SEVERITY_HIGH``, ``SEVERITY_CRITICAL``.
        """
        within_critical = distance < self.critical_distance

        if within_critical and behavior_shifted:
            return SEVERITY_CRITICAL
        if within_critical or behavior_shifted:
            return SEVERITY_HIGH
        # Within outer threshold — check current behaviour
        # (shift not detected, but animal may still be alert right now)
        # Caller can refine with current_behavior; we use shift as proxy
        if distance < self.distance_threshold:
            return SEVERITY_MEDIUM

        return SEVERITY_LOW

    def _compute_severity_species(
        self, distance: float, behavior_shifted: bool,
        sp_distance: float, sp_critical: float,
    ) -> str:
        within_critical = distance < sp_critical
        if within_critical and behavior_shifted:
            return SEVERITY_CRITICAL
        if within_critical or behavior_shifted:
            return SEVERITY_HIGH
        if distance < sp_distance:
            return SEVERITY_MEDIUM
        return SEVERITY_LOW

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_shift(self, track_id: int) -> Tuple[bool, str]:
        """Check if *track_id* shifted from calm → alert recently.

        Returns
        -------
        tuple[bool, str]
            ``(shifted, previous_calm_behavior)``.
        """
        rec = self._records.get(track_id)
        if rec is None or len(rec.behaviors) < 2:
            return False, "Observing"

        window = list(rec.behaviors)[-self.shift_window:]
        if len(window) < 2:
            return False, window[0]

        # Look for the last calm state followed by an alert state
        last_calm: Optional[str] = None
        for b in window:
            if b in _CALM_STATES:
                last_calm = b
            elif b in _ALERT_STATES and last_calm is not None:
                return True, last_calm

        # No shift detected — return the earliest behaviour in the window
        return False, window[0]

    @staticmethod
    def _build_description(
        species: str,
        distance: float,
        severity: str,
        shifted: bool,
        prev_behavior: str,
        current_behavior: str,
    ) -> str:
        """Build a human-readable event description."""
        parts = [
            f"{severity} disturbance: {species} at {distance:.0f}px from human."
        ]
        if shifted:
            parts.append(
                f"Behavior shifted from '{prev_behavior}' to '{current_behavior}'."
            )
        else:
            parts.append(f"Current behavior: {current_behavior}.")
        return " ".join(parts)

    # ------------------------------------------------------------------
    # House-keeping
    # ------------------------------------------------------------------

    def remove_track(self, track_id: int) -> None:
        """Drop internal state for a lost animal track."""
        self._records.pop(track_id, None)

    def __repr__(self) -> str:
        return (
            f"DisturbanceAnalyzer(threshold={self.distance_threshold}px, "
            f"critical={self.critical_distance}px, "
            f"tracks={len(self._records)})"
        )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance between two 2-D points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])
