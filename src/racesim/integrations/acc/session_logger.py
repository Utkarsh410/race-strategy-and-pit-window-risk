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
* Shared-memory physics messages are logged like every other message: the raw
  envelope keeps the session id, sample sequence, the Windows sampling time
  (``payload.timestamp_utc``), available / missing / invalid channels, and the
  Mac receive time sits beside it. :func:`export_physics_csv` flattens selected
  channels; it streams the log, so memory use does not grow with its length.
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
from typing import Any, Iterable, Iterator, Optional, Sequence, Union

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

    def log_record(self, record_type: str, data: dict[str, Any]) -> None:
        """Append a derived record (completed lap, stint, session metadata, strategy snapshot, ...)."""
        with self._lock:
            if self._fh is None:
                return
            self._write({"record_type": record_type, "logged_at_utc": datetime.now(timezone.utc).isoformat(), "data": data})
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


def iter_session_log(path: Union[str, Path]) -> Iterator[dict]:
    """Stream records from a session log, skipping unreadable lines (constant memory)."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


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


# Channels exported by default (the physics fields used for analysis); any channel name can be requested.
PHYSICS_CSV_CHANNELS = (
    "speed_kmh", "rpm", "gear", "throttle", "brake", "steering_input", "clutch", "fuel_remaining", "velocity",
    "acceleration", "angular_velocity", "wheel_slip", "slip_ratio", "slip_angle", "wheel_pressure", "tyre_core_temperature",
    "suspension_travel", "brake_pressure", "brake_temperature", "abs_activity", "tc_activity",
)
PHYSICS_CSV_META = [
    "received_at_utc", "sampled_at_utc", "sent_at_utc", "sample_age_at_send_s", "session_id", "sender_run_id", "sequence",
    "sample_sequence", "physics_packet_id", "graphics_packet_id", "reader_status", "source", "mock", "physics_schema_version",
    "available_count", "missing", "invalid", "quality",
]


def _channel_columns(name: str) -> list[str]:
    from racesim.integrations.acc.physics_channels import AXES, BY_NAME, DAMAGE_ZONES, WHEELS

    ch = BY_NAME.get(name)
    if ch is None:
        raise ValueError(f"unknown physics channel {name!r}")
    if ch.kind == "wheels":
        return [f"{name}_{w}" for w in WHEELS]
    if ch.kind == "vector":
        return [f"{name}_{a}" for a in AXES]
    if ch.kind == "damage":
        return [f"{name}_{z}" for z in DAMAGE_ZONES]
    if ch.kind == "wheel_vectors":
        return [f"{name}_{w}_{a}" for w in WHEELS for a in AXES]
    return [name]


def _channel_cells(name: str, value: Any) -> list[Any]:
    """CSV cells for one channel; an empty cell means unavailable (never 0)."""
    n = len(_channel_columns(name))
    if value is None:
        return [None] * n
    if not isinstance(value, list):
        return [value]
    if n == 12:  # per-wheel vectors: a missing wheel is three empty cells
        value = [y for x in value for y in (x if isinstance(x, list) else [None, None, None])]
    return list(value) if len(value) == n else [None] * n


def flatten_physics_record(rec: dict, channels: Sequence[str] = PHYSICS_CSV_CHANNELS) -> Optional[dict]:
    """One CSV row for a logged physics *sample* (status-only messages return None)."""
    from racesim.integrations.acc.physics_channels import LEGACY_NAMES

    if rec.get("record_type") != "message":
        return None
    env = rec.get("envelope") or {}
    if env.get("message_type") != "physics":
        return None
    p = env.get("payload") or {}
    values = p.get("channels") or {}
    if not values:
        return None
    if p.get("physics_schema_version") == "1.0":  # logs written before the 1.1 renames
        values = {LEGACY_NAMES.get(k, k): v for k, v in values.items()}
    row = {
        "received_at_utc": rec.get("received_at_utc"), "sampled_at_utc": p.get("timestamp_utc"),
        "sent_at_utc": env.get("sent_at_utc"), "sample_age_at_send_s": p.get("sample_age_s"),
        "session_id": p.get("session_id"), "sender_run_id": env.get("sender_run_id"), "sequence": env.get("sequence"),
        "sample_sequence": p.get("sample_sequence"), "physics_packet_id": p.get("physics_packet_id"),
        "graphics_packet_id": p.get("graphics_packet_id"), "reader_status": p.get("reader_status"),
        "source": p.get("source"), "mock": env.get("mock"), "physics_schema_version": p.get("physics_schema_version"),
        "available_count": len(p.get("available") or []), "missing": "|".join(p.get("missing") or []),
        "invalid": "|".join(f"{k}:{v}" for k, v in (p.get("invalid") or {}).items()),
        "quality": "|".join(rec.get("quality", [])),
    }
    for name in channels:
        row.update(zip(_channel_columns(name), _channel_cells(name, values.get(name))))
    return row


def export_physics_csv(log_path: Union[str, Path], csv_path: Union[str, Path],
                       channels: Sequence[str] = PHYSICS_CSV_CHANNELS) -> tuple[Path, int]:
    """Write one row per logged physics sample with the selected channels; returns ``(path, rows)``.

    Refuses to overwrite an existing CSV. Wheels are ``_fl/_fr/_rl/_rr``,
    vectors ``_x/_y/_z``; values are in native ACC units (see physics_channels).
    """
    csv_path = Path(csv_path)
    columns = PHYSICS_CSV_META + [c for name in channels for c in _channel_columns(name)]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with open(csv_path, "x", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columns)
        w.writeheader()
        for rec in iter_session_log(log_path):
            row = flatten_physics_record(rec, channels)
            if row:
                w.writerow(row)
                rows += 1
    return csv_path, rows
