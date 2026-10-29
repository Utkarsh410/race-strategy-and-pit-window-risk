"""Per-run JSON Lines session log for received telemetry (Mac side).

* One file per run, named ``acc_<UTC timestamp>_<random>.jsonl``; opened with
  mode ``"x"`` so an existing log is never overwritten.
* The first line is a ``session_start`` record, the last a ``session_end``
  record (absent if the process was killed - :func:`read_session_log`
  tolerates a truncated final line).
* Lines are flushed every ``flush_every_records`` records or
  ``flush_interval_s`` seconds, and on close.
* No secrets are ever written; only the envelope (which carries an HMAC, not
  the secret) and receiver-side quality flags.
"""

from __future__ import annotations

import csv
import json
import logging
import secrets
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Union

logger = logging.getLogger(__name__)

CSV_COLUMNS = [
    "received_at_utc", "source_timestamp_utc", "sequence", "message_type", "mock", "session_id", "track_name",
    "session_type", "session_phase", "current_lap", "laps_completed", "position", "last_lap_time_s", "best_lap_time_s",
    "car_location", "acc_status", "event_type", "quality",
]


class SessionLogger:
    def __init__(self, output_dir: Union[str, Path], *, prefix: str = "acc", flush_every_records: int = 20,
                 flush_interval_s: float = 2.0, metadata: Optional[dict] = None):
        self.output_dir = Path(output_dir)
        self.prefix = prefix
        self.flush_every = flush_every_records
        self.flush_interval = flush_interval_s
        self.metadata = metadata or {}
        self.path: Optional[Path] = None
        self._fh = None
        self._lock = threading.Lock()
        self._pending = 0
        self._last_flush = time.monotonic()
        self.records_written = 0

    def open(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        for _ in range(5):
            path = self.output_dir / f"{self.prefix}_{stamp}_{secrets.token_hex(3)}.jsonl"
            try:
                self._fh = open(path, "x", encoding="utf-8")
                break
            except FileExistsError:
                continue
        else:
            raise RuntimeError("could not create a unique session log file")
        self.path = path
        self._write({"record_type": "session_start", "started_at_utc": datetime.now(timezone.utc).isoformat(),
                     "metadata": self.metadata})
        self.flush()
        logger.info("Session log: %s", path)
        return path

    def __enter__(self) -> "SessionLogger":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _write(self, record: dict[str, Any]) -> None:
        self._fh.write(json.dumps(record, default=str, separators=(",", ":")) + "\n")
        self.records_written += 1
        self._pending += 1

    def log(self, envelope: dict[str, Any], *, received_at_utc: datetime, quality: Iterable[str] = ()) -> None:
        """Append one received message with receiver-side quality flags."""
        with self._lock:
            if self._fh is None:
                return
            self._write({"record_type": "message", "received_at_utc": received_at_utc.isoformat(),
                         "quality": list(quality), "envelope": envelope})
            if self._pending >= self.flush_every or time.monotonic() - self._last_flush >= self.flush_interval:
                self.flush()

    def flush(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            self._pending = 0
            self._last_flush = time.monotonic()

    def close(self, summary: Optional[dict] = None) -> None:
        with self._lock:
            if self._fh is None:
                return
            try:
                self._write({"record_type": "session_end", "ended_at_utc": datetime.now(timezone.utc).isoformat(),
                             "summary": summary or {}})
                self.flush()
            finally:
                self._fh.close()
                self._fh = None


def read_session_log(path: Union[str, Path]) -> tuple[list[dict], int]:
    """Parse a session log; returns ``(records, unreadable_lines)`` (e.g. a line cut by a crash)."""
    records, bad = [], 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    return records, bad


def flatten_record(rec: dict) -> Optional[dict]:
    if rec.get("record_type") != "message":
        return None
    env = rec["envelope"]
    p = env.get("payload", {})
    s, v = p.get("session", {}) or {}, p.get("vehicle", {}) or {}
    return {
        "received_at_utc": rec.get("received_at_utc"), "source_timestamp_utc": p.get("timestamp_utc"),
        "sequence": env.get("sequence"), "message_type": env.get("message_type"), "mock": env.get("mock"),
        "session_id": p.get("session_id"), "track_name": s.get("track_name") or p.get("track_name"),
        "session_type": s.get("session_type") or p.get("session_type"), "session_phase": s.get("session_phase"),
        "current_lap": v.get("current_lap"), "laps_completed": v.get("laps_completed"), "position": v.get("position"),
        "last_lap_time_s": v.get("last_lap_time_s"), "best_lap_time_s": v.get("best_lap_time_s"),
        "car_location": v.get("car_location"), "acc_status": p.get("acc_status"), "event_type": p.get("event_type"),
        "quality": "|".join(rec.get("quality", [])),
    }


def export_session_csv(log_path: Union[str, Path], csv_path: Union[str, Path]) -> Path:
    """Flatten a JSONL session log into an analysis-friendly CSV."""
    records, _ = read_session_log(log_path)
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for rec in records:
            row = flatten_record(rec)
            if row:
                w.writerow(row)
    return csv_path
