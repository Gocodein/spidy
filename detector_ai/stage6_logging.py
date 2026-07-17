"""
DETECTOR AI — Stage 6: Logging & Alerting
SQLite-backed logging for detections, disturbance events, and sessions.
Includes batch-buffered writes, CSV export, and a colour-coded console
alert manager.

Tables:
    sessions           — one row per pipeline run
    detections         — one row per tracked detection per frame
    disturbance_events — one row per flagged disturbance incident
"""

from __future__ import annotations

import csv
import datetime
import sqlite3
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from detector_ai.config import (
    DEFAULT_DB_PATH,
    ENABLE_CONSOLE_ALERTS,
    LOG_BATCH_SIZE,
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    SEVERITY_LOW,
    SEVERITY_MEDIUM,
)
from detector_ai.stage5_disturbance import DisturbanceEvent

# ---------------------------------------------------------------------------
# ANSI colour helpers (Windows ≥ 10 supports ANSI in newer terminals)
# ---------------------------------------------------------------------------
_RESET = "\033[0m"
_BOLD = "\033[1m"
_SEVERITY_COLOURS = {
    SEVERITY_LOW: "\033[32m",       # green
    SEVERITY_MEDIUM: "\033[33m",    # yellow
    SEVERITY_HIGH: "\033[91m",      # bright red
    SEVERITY_CRITICAL: "\033[41m",  # red background
}

# ---------------------------------------------------------------------------
# Schema SQL
# ---------------------------------------------------------------------------
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    video_source    TEXT    NOT NULL,
    start_time      TEXT    NOT NULL,
    end_time        TEXT,
    total_frames    INTEGER DEFAULT 0,
    total_detections INTEGER DEFAULT 0,
    total_disturbances INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS detections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      INTEGER NOT NULL,
    frame_num       INTEGER NOT NULL,
    timestamp       TEXT    NOT NULL,
    track_id        INTEGER,
    class_name      TEXT,
    species         TEXT,
    confidence      REAL,
    behavior        TEXT,
    x               REAL,
    y               REAL,
    disturbance_flag INTEGER DEFAULT 0,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS disturbance_events (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id          INTEGER NOT NULL,
    timestamp           TEXT    NOT NULL,
    animal_track_id     INTEGER,
    animal_species      TEXT,
    behavior_before     TEXT,
    behavior_after      TEXT,
    human_track_id      INTEGER,
    distance_px         REAL,
    severity            TEXT,
    description         TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_det_session
    ON detections(session_id);
CREATE INDEX IF NOT EXISTS idx_det_track
    ON detections(session_id, track_id);
CREATE INDEX IF NOT EXISTS idx_dist_session
    ON disturbance_events(session_id);
"""


# ===================================================================
# WildlifeDB
# ===================================================================

class WildlifeDB:
    """SQLite database for wildlife detection logging.

    Parameters
    ----------
    db_path : str | Path
        Path to the ``.db`` file.  Created if it doesn't exist.
    batch_size : int
        Number of detection rows buffered before a batch INSERT.
    """

    def __init__(
        self,
        db_path: str | Path = DEFAULT_DB_PATH,
        batch_size: int = LOG_BATCH_SIZE,
    ) -> None:
        self.db_path = Path(db_path)
        self.batch_size = batch_size
        self._buffer: List[tuple] = []

        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def start_session(self, video_source: str) -> int:
        """Create a new session and return its ``session_id``.

        Parameters
        ----------
        video_source : str
            Camera index or video file path.
        """
        cur = self._conn.execute(
            "INSERT INTO sessions (video_source, start_time) VALUES (?, ?)",
            (str(video_source), _now_iso()),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def end_session(self, session_id: int) -> None:
        """Finalise session: flush buffer, write end-time, update stats.

        Parameters
        ----------
        session_id : int
            The session to close.
        """
        self.flush()

        # Aggregate stats
        row = self._conn.execute(
            "SELECT COUNT(*) AS cnt FROM detections WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        total_det = row["cnt"] if row else 0

        row = self._conn.execute(
            "SELECT COUNT(*) AS cnt FROM disturbance_events WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        total_dist = row["cnt"] if row else 0

        row = self._conn.execute(
            "SELECT MAX(frame_num) AS mf FROM detections WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        total_frames = (row["mf"] or 0) if row else 0

        self._conn.execute(
            """UPDATE sessions
               SET end_time = ?, total_frames = ?,
                   total_detections = ?, total_disturbances = ?
               WHERE id = ?""",
            (_now_iso(), total_frames, total_det, total_dist, session_id),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Detection logging (batched)
    # ------------------------------------------------------------------

    def log_detection(
        self,
        session_id: int,
        frame_num: int,
        track_id: int,
        class_name: str,
        species: str,
        confidence: float,
        behavior: str,
        x: float,
        y: float,
        disturbed: bool,
    ) -> None:
        """Buffer a single detection row; flushes every *batch_size* rows.

        Parameters
        ----------
        session_id : int
        frame_num : int
        track_id : int
        class_name : str
            YOLO class label.
        species : str
            Classifier species label.
        confidence : float
        behavior : str
            Behaviour state string from Stage 4.
        x, y : float
            Bounding-box centre pixel coordinates.
        disturbed : bool
            Whether this detection is flagged as disturbed.
        """
        self._buffer.append((
            session_id,
            frame_num,
            _now_iso(),
            track_id,
            class_name,
            species,
            confidence,
            behavior,
            x,
            y,
            int(disturbed),
        ))

        if len(self._buffer) >= self.batch_size:
            self.flush()

    # ------------------------------------------------------------------
    # Disturbance logging (immediate)
    # ------------------------------------------------------------------

    def log_disturbance(
        self, session_id: int, event: DisturbanceEvent
    ) -> None:
        """Log a disturbance event immediately (not batched).

        Parameters
        ----------
        session_id : int
        event : DisturbanceEvent
        """
        self._conn.execute(
            """INSERT INTO disturbance_events
               (session_id, timestamp, animal_track_id, animal_species,
                behavior_before, behavior_after, human_track_id,
                distance_px, severity, description)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                datetime.datetime.fromtimestamp(event.timestamp).isoformat(),
                event.animal_track_id,
                event.animal_species,
                event.animal_behavior_before,
                event.animal_behavior_after,
                event.human_track_id,
                event.distance_px,
                event.severity,
                event.description,
            ),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Query helpers (for dashboard / reports)
    # ------------------------------------------------------------------

    def get_session_stats(self, session_id: int) -> Dict:
        """Return summary statistics for a session.

        Parameters
        ----------
        session_id : int

        Returns
        -------
        dict
            Keys: ``session_id``, ``video_source``, ``start_time``,
            ``end_time``, ``total_frames``, ``total_detections``,
            ``total_disturbances``, ``unique_species``, ``unique_tracks``.
        """
        row = self._conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if row is None:
            return {}

        stats: Dict = dict(row)

        # Extra aggregates
        species_row = self._conn.execute(
            "SELECT COUNT(DISTINCT species) AS cnt FROM detections WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        stats["unique_species"] = species_row["cnt"] if species_row else 0

        track_row = self._conn.execute(
            "SELECT COUNT(DISTINCT track_id) AS cnt FROM detections WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        stats["unique_tracks"] = track_row["cnt"] if track_row else 0

        return stats

    def get_all_sessions(self) -> List[Dict]:
        """Return all sessions ordered by start time (most recent first).

        Returns
        -------
        list[dict]
        """
        rows = self._conn.execute(
            "SELECT * FROM sessions ORDER BY start_time DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_detections(
        self,
        session_id: Optional[int] = None,
        limit: int = 500,
    ) -> List[Dict]:
        """Query detection rows.

        Parameters
        ----------
        session_id : int, optional
            Filter to a specific session.  ``None`` returns all.
        limit : int
            Maximum rows to return (default 500).

        Returns
        -------
        list[dict]
        """
        if session_id is not None:
            rows = self._conn.execute(
                "SELECT * FROM detections WHERE session_id = ? "
                "ORDER BY frame_num DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM detections ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_disturbance_events(
        self, session_id: Optional[int] = None
    ) -> List[Dict]:
        """Query disturbance events.

        Parameters
        ----------
        session_id : int, optional
            Filter to a specific session.

        Returns
        -------
        list[dict]
        """
        if session_id is not None:
            rows = self._conn.execute(
                "SELECT * FROM disturbance_events WHERE session_id = ? "
                "ORDER BY timestamp DESC",
                (session_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM disturbance_events ORDER BY id DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_behavior_timeline(
        self, session_id: int, track_id: int
    ) -> List[Dict]:
        """Get behaviour state over time for one track in one session.

        Parameters
        ----------
        session_id : int
        track_id : int

        Returns
        -------
        list[dict]
            Rows with ``frame_num``, ``behavior``, ``x``, ``y``.
        """
        rows = self._conn.execute(
            "SELECT frame_num, behavior, x, y FROM detections "
            "WHERE session_id = ? AND track_id = ? "
            "ORDER BY frame_num",
            (session_id, track_id),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_csv(self, session_id: int, output_path: str) -> None:
        """Export all detections for a session to a CSV file.

        Parameters
        ----------
        session_id : int
        output_path : str
            Destination CSV path.
        """
        rows = self._conn.execute(
            "SELECT * FROM detections WHERE session_id = ? ORDER BY frame_num",
            (session_id,),
        ).fetchall()

        if not rows:
            return

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(rows[0].keys())
            for row in rows:
                writer.writerow(tuple(row))

    # ------------------------------------------------------------------
    # Buffer management
    # ------------------------------------------------------------------

    def flush(self) -> None:
        """Write all buffered detection rows to the database."""
        if not self._buffer:
            return
        self._conn.executemany(
            """INSERT INTO detections
               (session_id, frame_num, timestamp, track_id, class_name,
                species, confidence, behavior, x, y, disturbance_flag)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            self._buffer,
        )
        self._conn.commit()
        self._buffer.clear()

    def close(self) -> None:
        """Flush remaining buffer and close the database connection."""
        try:
            self.flush()
        finally:
            self._conn.close()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "WildlifeDB":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"WildlifeDB(path={self.db_path!s}, "
            f"batch_size={self.batch_size}, "
            f"buffered={len(self._buffer)})"
        )


# ===================================================================
# AlertManager
# ===================================================================

class AlertManager:
    """Console (and future webhook/SMS) alerts for disturbance events.

    Parameters
    ----------
    enable_console : bool
        Print colour-coded alerts to *stderr*.
    """

    def __init__(self, enable_console: bool = ENABLE_CONSOLE_ALERTS) -> None:
        self.enable_console = enable_console

    def send_alert(self, event: DisturbanceEvent) -> None:
        """Emit an alert for a :class:`DisturbanceEvent`.

        Currently prints a coloured line to *stderr*.  Future hooks
        (webhook, SMS, e-mail) can be added here.

        Parameters
        ----------
        event : DisturbanceEvent
        """
        if self.enable_console:
            self._console_alert(event)
        # TODO: webhook / SMS / email integrations

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _console_alert(event: DisturbanceEvent) -> None:
        """Print a colour-coded alert line to stderr."""
        colour = _SEVERITY_COLOURS.get(event.severity, "")
        ts = datetime.datetime.fromtimestamp(event.timestamp).strftime(
            "%H:%M:%S"
        )
        line = (
            f"{colour}{_BOLD}[{event.severity}]{_RESET} "
            f"{colour}{ts} — {event.description}{_RESET}"
        )
        print(line, file=sys.stderr)

    def __repr__(self) -> str:
        return f"AlertManager(console={self.enable_console})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()
