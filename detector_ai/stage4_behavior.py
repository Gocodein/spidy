"""
DETECTOR AI — Stage 4: Behavior Estimation
Classifies animal behavior from trajectory analysis over a sliding window
of tracked positions.

Behavior states (priority order):
    1. Resting       — very low speed, low direction variance
    2. Fleeing/Running — high speed
    3. Stalking       — moderate speed, very low direction variance
    4. Alert/Pacing   — low speed, high direction variance
    5. Walking/Grazing — moderate speed, moderate direction variance
    6. Observing       — default / insufficient data
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Dict, Optional, Tuple

from detector_ai.config import (
    ACCEL_STARTLE,
    BEHAVIOR_HISTORY_WINDOW,
    DIR_VAR_ALERT,
    DIR_VAR_STALKING,
    SPEED_RESTING,
    SPEED_RUNNING,
    SPEED_WALKING,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DWELL_RADIUS_PX = 5          # pixel radius for dwell-time accumulation
_MIN_FRAMES_FOR_CLASSIFY = 3  # need at least this many frames to classify


class BehaviorEstimator:
    """Estimate animal behavior from per-frame centre positions.

    Parameters
    ----------
    window : int
        Number of most-recent positions to keep per track.
    speed_thresholds : dict
        ``{"resting": float, "walking": float, "running": float}``
    dir_var_thresholds : dict
        ``{"alert": float, "stalking": float}``
    """

    def __init__(
        self,
        window: int = BEHAVIOR_HISTORY_WINDOW,
        speed_thresholds: Optional[Dict[str, float]] = None,
        dir_var_thresholds: Optional[Dict[str, float]] = None,
    ) -> None:
        self.window = window

        # Speed thresholds ---------------------------------------------------
        st = speed_thresholds or {}
        self.speed_resting: float = st.get("resting", SPEED_RESTING)
        self.speed_walking: float = st.get("walking", SPEED_WALKING)
        self.speed_running: float = st.get("running", SPEED_RUNNING)

        # Direction-variance thresholds --------------------------------------
        dv = dir_var_thresholds or {}
        self.dir_var_alert: float = dv.get("alert", DIR_VAR_ALERT)
        self.dir_var_stalking: float = dv.get("stalking", DIR_VAR_STALKING)

        # Per-track position history: track_id → deque of (x, y)
        self._history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.window)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update(self, track_id: int, center: Tuple[float, float]) -> None:
        """Append a centre position ``(x, y)`` for *track_id*.

        Parameters
        ----------
        track_id : int
            Unique tracker ID.
        center : tuple[float, float]
            ``(x, y)`` pixel coordinates of the bounding-box centre.
        """
        self._history[track_id].append((float(center[0]), float(center[1])))

    def classify(self, track_id: int) -> str:
        """Return a behaviour-state string for *track_id*.

        Returns one of:
        ``"Resting"``, ``"Fleeing/Running"``, ``"Stalking"``,
        ``"Alert/Pacing"``, ``"Walking/Grazing"``, ``"Observing"``.
        """
        feats = self.get_features(track_id)
        if feats is None:
            return "Observing"

        avg_speed: float = feats["avg_speed"]
        dir_var: float = feats["direction_variance"]

        # Priority 1 — Resting
        if avg_speed < self.speed_resting and dir_var < self.dir_var_alert:
            return "Resting"

        # Priority 2 — Fleeing / Running
        if avg_speed > self.speed_running:
            return "Fleeing/Running"

        # Priority 3 — Stalking (moderate speed, very consistent heading)
        if (
            self.speed_resting * 3 <= avg_speed <= self.speed_walking * 1.25
            and dir_var < self.dir_var_stalking
        ):
            return "Stalking"

        # Priority 4 — Alert / Pacing (slow but erratic)
        if avg_speed < self.speed_walking and dir_var > self.dir_var_alert:
            return "Alert/Pacing"

        # Priority 5 — Walking / Grazing
        if avg_speed <= self.speed_running:
            return "Walking/Grazing"

        # Fallback
        return "Observing"

    def get_features(self, track_id: int) -> Optional[Dict[str, float]]:
        """Return raw trajectory features for *track_id*, or ``None``
        if fewer than ``_MIN_FRAMES_FOR_CLASSIFY`` positions have been
        recorded.

        Returns
        -------
        dict or None
            Keys: ``avg_speed``, ``acceleration``, ``direction_variance``,
            ``direction_consistency``, ``dwell_time``, ``max_speed``.
        """
        pts = self._history.get(track_id)
        if pts is None or len(pts) < _MIN_FRAMES_FOR_CLASSIFY:
            return None

        positions = list(pts)
        n = len(positions)

        # ---- Per-frame speeds (Euclidean distance between consecutive) ----
        speeds: list[float] = []
        for i in range(1, n):
            dx = positions[i][0] - positions[i - 1][0]
            dy = positions[i][1] - positions[i - 1][1]
            speeds.append(math.hypot(dx, dy))

        avg_speed = sum(speeds) / len(speeds) if speeds else 0.0
        max_speed = max(speeds) if speeds else 0.0

        # ---- Acceleration (rate of speed change) ----
        accels: list[float] = []
        for i in range(1, len(speeds)):
            accels.append(speeds[i] - speeds[i - 1])
        acceleration = (
            sum(abs(a) for a in accels) / len(accels) if accels else 0.0
        )

        # ---- Direction angles & variance ----
        angles: list[float] = []
        for i in range(1, n):
            dx = positions[i][0] - positions[i - 1][0]
            dy = positions[i][1] - positions[i - 1][1]
            if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                angles.append(math.atan2(dy, dx))

        if len(angles) >= 2:
            direction_variance = _circular_variance(angles)
        else:
            direction_variance = 0.0

        direction_consistency = 1.0 - min(direction_variance / math.pi, 1.0)

        # ---- Dwell time (frames where centre barely moved) ----
        dwell_time = 0
        if n >= 2:
            anchor = positions[-1]
            for i in range(n - 2, -1, -1):
                dx = positions[i][0] - anchor[0]
                dy = positions[i][1] - anchor[1]
                if math.hypot(dx, dy) <= _DWELL_RADIUS_PX:
                    dwell_time += 1
                else:
                    break

        return {
            "avg_speed": avg_speed,
            "max_speed": max_speed,
            "acceleration": acceleration,
            "direction_variance": direction_variance,
            "direction_consistency": direction_consistency,
            "dwell_time": float(dwell_time),
        }

    # ------------------------------------------------------------------
    # House-keeping
    # ------------------------------------------------------------------
    def remove_track(self, track_id: int) -> None:
        """Drop history for a lost track to free memory."""
        self._history.pop(track_id, None)

    def active_tracks(self) -> list[int]:
        """Return list of track IDs that have history."""
        return list(self._history.keys())

    def __repr__(self) -> str:
        return (
            f"BehaviorEstimator(window={self.window}, "
            f"tracks={len(self._history)})"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _circular_variance(angles: list[float]) -> float:
    """Compute circular variance of a list of angles (radians).

    Returns a value in ``[0, π]``: 0 → all angles identical,
    π → maximally dispersed.
    """
    if not angles:
        return 0.0

    n = len(angles)
    sum_sin = sum(math.sin(a) for a in angles)
    sum_cos = sum(math.cos(a) for a in angles)
    r = math.hypot(sum_sin / n, sum_cos / n)  # mean resultant length [0,1]
    # Circular variance = 1 - R.  Scale to [0, π] for comparability.
    return (1.0 - r) * math.pi
