"""Thread-safe live analysis of received telemetry, with persistence and exports.

The receiver thread calls :meth:`LiveAnalysis.on_message`; dashboards call
:meth:`LiveAnalysis.snapshot`, which returns plain data and never shares mutable
state. One :class:`SessionAnalysis` exists per continuous session, so laps from
separate sessions never mix. Stale data are labelled from the time of the last
accepted message, and a log can be replayed through the same code.
"""

from __future__ import annotations

import csv
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

from racesim.integrations.acc.config import HealthConfig
from racesim.integrations.acc.lap_tracker import CompletedLap, LapTracker
from racesim.integrations.acc.monitoring import fuel_status, missing_channels, tyre_status
from racesim.integrations.acc.pace_analysis import analyse_pace
from racesim.integrations.acc.race_state import Decision, Observed, Provenance, SessionIdentity, SessionTimeRemaining, label_for
from racesim.integrations.acc.session_logger import SessionLogger, read_session_log

logger = logging.getLogger(__name__)

MAX_SESSIONS_KEPT = 20
OVERVIEW_LAP_TIME_ROWS = ("Last lap", "Best lap (session)")  # shown as m:ss.mmm; stored in seconds
LAP_CSV_COLUMNS = ["session_key", "lap_number", "lap_time_s", "classification", "sector_1_s", "sector_2_s", "sector_3_s", "sector_source",
                   "position", "in_pit", "stint_id", "stint_lap", "tyre_compound", "tyre_age_laps", "fuel_start_kg", "fuel_end_kg",
                   "session_time_end_s", "flags"]
STINT_CSV_COLUMNS = ["session_key", "stint_id", "start_reason", "end_reason", "laps", "first_lap", "last_lap", "start_session_time_s",
                     "end_session_time_s", "left_pits_at_start", "pit_entry_session_time_s", "pit_exit_session_time_s", "stationary_observed",
                     "tyre_compound", "tyre_age_start_laps", "fuel_start_kg", "fuel_end_kg"]


@dataclass
class SessionAnalysis:
    session_key: str
    source: Optional[str]
    mock: bool
    tracker: LapTracker
    remaining: SessionTimeRemaining = field(default_factory=SessionTimeRemaining)
    latest: Optional[dict] = None
    latest_received_t: Optional[float] = None
    remaining_s: Optional[float] = None
    messages: int = 0
    events: dict[str, int] = field(default_factory=dict)
    strategy_snapshots: list[dict] = field(default_factory=list)


class LiveAnalysis:
    def __init__(self, *, thresholds: Optional[HealthConfig] = None, sink: Optional[SessionLogger] = None, replay: bool = False):
        self.t = thresholds or HealthConfig()
        self.sink = sink
        self.replay = replay
        self.identity = SessionIdentity()
        self.sessions: dict[str, SessionAnalysis] = {}
        self.current: Optional[SessionAnalysis] = None
        self.counters = {"out_of_order_ignored": 0, "late_other_session_ignored": 0, "duplicate_ignored": 0, "replay_frames_ignored": 0,
                         "reconnections": 0, "session_restarts": 0, "duplicate_events_ignored": 0}
        self._seen_events: set[tuple] = set()
        self._replay_playing = False
        self._lock = threading.RLock()

    # -- input ----------------------------------------------------------------
    def on_message(self, message_type: str, payload: dict, status: str = "new", received_t: Optional[float] = None,
                   received_utc: Optional[datetime] = None, mock: bool = False) -> None:
        received_t = time.monotonic() if received_t is None else received_t
        with self._lock:
            if message_type == "telemetry":
                self._on_telemetry(payload, status, received_t, mock)
            elif message_type == "event":
                self._on_event(payload)

    def _on_telemetry(self, payload: dict, status: str, received_t: float, mock: bool) -> None:
        if status == "duplicate":
            self.counters["duplicate_ignored"] += 1
            return
        if status == "out_of_order":
            self.counters["out_of_order_ignored"] += 1  # telemetry is cumulative: the next message carries this state too
            return
        self._replay_playing = bool((payload.get("session") or {}).get("is_replay_playing"))
        if self._replay_playing:
            self.counters["replay_frames_ignored"] += 1  # the in-game replay is not the live race
            return
        decision = self.identity.resolve(payload, received_t)
        if decision == Decision.OUT_OF_ORDER:
            self.counters["out_of_order_ignored"] += 1
            return
        if decision == Decision.LATE_OTHER_SESSION:
            self.counters["late_other_session_ignored"] += 1
            return
        if decision == Decision.RECONNECTED:
            self.counters["reconnections"] += 1
        if decision in (Decision.NEW_SESSION, Decision.RESTART):
            self.counters["session_restarts"] += decision == Decision.RESTART
            self._begin_session(payload, mock, decision)
        session = self.current
        session.messages += 1
        session.latest, session.latest_received_t = payload, received_t
        s = payload.get("session") or {}
        measured = s.get("session_time_remaining_s")
        session.remaining_s = measured if measured is not None else session.remaining.update(s.get("session_time_elapsed_s"), s.get("session_end_time_raw_s"), s.get("session_phase"))
        for lap in session.tracker.update(payload, received_t):
            self._log("completed_lap", lap.as_dict())
        for stint in session.tracker.pop_closed_stints():
            self._log("stint_summary", stint.as_row())

    def _on_event(self, payload: dict) -> None:
        """Own-car event counts only: real logs repeat each event many times, so they are de-duplicated and never count laps."""
        session = self.current
        if session is None or session.latest is None:
            return
        if payload.get("car_id") != (session.latest.get("vehicle") or {}).get("car_id"):
            return
        key = (session.session_key, payload.get("car_id"), payload.get("event_type"), payload.get("time_ms"), payload.get("message"))
        if key in self._seen_events:
            self.counters["duplicate_events_ignored"] += 1
            return
        self._seen_events.add(key)
        kind = str(payload.get("event_type"))
        session.events[kind] = session.events.get(kind, 0) + 1

    def _begin_session(self, payload: dict, mock: bool, decision: Decision) -> None:
        self._close_current("session_changed")
        key = self.identity.session_key
        self.current = SessionAnalysis(key, payload.get("source"), mock, LapTracker(key))
        self.sessions[key] = self.current
        while len(self.sessions) > MAX_SESSIONS_KEPT:
            self.sessions.pop(next(iter(self.sessions)))
        s = payload.get("session") or {}
        self._log("session_metadata", {"session_key": key, "bridge_session_id": payload.get("session_id"), "decision": decision.value,
                                       "source": payload.get("source"), "mock": mock, "track_name": s.get("track_name"),
                                       "track_length_m": s.get("track_length_m"), "session_type": s.get("session_type"),
                                       "session_index": s.get("session_index"), "car_id": (payload.get("vehicle") or {}).get("car_id"),
                                       "race_number": (payload.get("vehicle") or {}).get("race_number"),
                                       "car_model_id": (payload.get("vehicle") or {}).get("car_model_id")})

    def _close_current(self, reason: str) -> None:
        session = self.current
        if session is None:
            return
        elapsed = ((session.latest or {}).get("session") or {}).get("session_time_elapsed_s")
        session.tracker.close(reason, elapsed)
        for stint in session.tracker.pop_closed_stints():
            self._log("stint_summary", stint.as_row())
        self._log("data_quality", {"session_key": session.session_key, "flags": dict(session.tracker.flags), "counters": dict(self.counters)})

    def finish(self) -> None:
        """Close the open stint and log the quality summary. Safe to call twice."""
        with self._lock:
            self._close_current("logging_stopped")

    def _log(self, record_type: str, data: dict) -> None:
        if self.sink is not None:
            self.sink.log_record(record_type, data)

    def record_strategy(self, result: dict) -> None:
        """Store an on-demand strategy result with its session and log it."""
        with self._lock:
            session = self.sessions.get(result.get("session_key")) or self.current
            if session is not None:
                session.strategy_snapshots.append(result)
            self._log("strategy_snapshot", result)

    # -- output ---------------------------------------------------------------
    def data_state(self, session: SessionAnalysis, now: float) -> tuple[str, Optional[float]]:
        if session.latest_received_t is None:
            return "WAITING_FOR_DATA", None
        age = max(0.0, now - session.latest_received_t)
        if self.replay:
            return "REPLAY", age
        if age <= self.t.stale_after_seconds:
            return "LIVE", age
        return ("STALE" if age <= self.t.disconnect_after_seconds else "DISCONNECTED"), age

    def snapshot(self, now: Optional[float] = None, session_key: Optional[str] = None) -> dict:
        now = time.monotonic() if now is None else now
        with self._lock:
            session = self.sessions.get(session_key) if session_key else self.current
            if session is None or session.latest is None:
                return {"available": False, "session_key": None, "stale": False, "data_state": "WAITING_FOR_DATA", "sessions": list(self.sessions),
                        "laps": [], "stints": [], "transitions": [], "counters": dict(self.counters), "mock": False, "replay": self.replay,
                        "overview": [], "pace": analyse_pace([]), "fuel": fuel_status([], None), "tyres": tyre_status({}, []),
                        "missing_channels": missing_channels({}), "flags": {}, "strategy_snapshots": [], "events": {}, "last_lap_flags": []}
            return self._snapshot(session, now)

    def _snapshot(self, session: SessionAnalysis, now: float) -> dict:
        state, age = self.data_state(session, now)
        stale = state in ("STALE", "DISCONNECTED")
        payload, tracker = session.latest, session.tracker
        s, v = payload.get("session") or {}, payload.get("vehicle") or {}
        laps: list[CompletedLap] = list(tracker.laps)
        pace = analyse_pace(laps)
        pace_s, pace_n = pace["median"]["value"], pace["median"]["n"]
        return {
            "available": True, "session_key": session.session_key, "sessions": list(self.sessions), "source": session.source,
            "mock": session.mock, "replay": self.replay, "data_state": state, "stale": stale, "age_s": age,
            "overview": [self._row(c, session, stale) for c in self._overview_channels(session)],
            "session_type": s.get("session_type"), "session_phase": s.get("session_phase"), "laps_completed": v.get("laps_completed"),
            "session_time_remaining_s": session.remaining_s, "track_temperature_c": s.get("track_temperature_c"),
            "ambient_temperature_c": s.get("ambient_temperature_c"), "rain_level_raw": s.get("rain_level_raw"),
            "stops_completed": sum(1 for st in tracker.stints if st.end_reason == "pit_visit"),
            "laps": [lap.as_row() for lap in laps], "stints": [st.as_row() for st in tracker.stints], "transitions": list(tracker.transitions),
            "pace": pace, "fuel": fuel_status(laps, v.get("fuel_kg"), pace_s=pace_s, pace_n=pace_n, stale=stale),
            "tyres": tyre_status(v, tracker.stints), "missing_channels": missing_channels(v), "flags": dict(tracker.flags),
            "counters": dict(self.counters), "events": dict(session.events), "strategy_snapshots": list(session.strategy_snapshots),
            "last_lap_flags": list(laps[-1].flags) if laps else [],
        }

    @staticmethod
    def _overview_channels(session: SessionAnalysis) -> list[tuple[str, Any, str, str]]:
        payload, tracker = session.latest, session.tracker
        s, v, src = payload.get("session") or {}, payload.get("vehicle") or {}, payload.get("field_sources") or {}
        def field_row(label, section, key, unit=""):
            return label, (s if section == "session" else v).get(key), unit, src.get(f"{section}.{key}", "")
        last = tracker.laps[-1] if tracker.laps else None
        stint = tracker.current_stint
        return [
            field_row("Track", "session", "track_name"), field_row("Session type", "session", "session_type"),
            field_row("Session phase", "session", "session_phase"), field_row("Session time elapsed", "session", "session_time_elapsed_s", "s"),
            ("Session time remaining", session.remaining_s, "s", src.get("session.session_time_remaining_s") or "DERIVED: sessionEndTime, checked against elapsed"),
            field_row("Current lap", "vehicle", "current_lap"), field_row("Completed laps", "vehicle", "laps_completed"),
            field_row("Last lap", "vehicle", "last_lap_time_s", "s"), field_row("Best lap (session)", "vehicle", "best_lap_time_s", "s"),
            ("Last lap sectors", None if last is None or not last.sector_times_s else ", ".join("-" if x is None else f"{x:.3f}" for x in last.sector_times_s),
             "s", "" if last is None or not last.sector_source else last.sector_source),
            field_row("Position", "vehicle", "position"), field_row("Car location", "vehicle", "car_location"),
            ("Current stint", None if stint is None else stint.stint_id, "", "DERIVED: pit exits observed"),
            field_row("Car index", "vehicle", "car_id"), field_row("Race number", "vehicle", "race_number"), field_row("Car model id", "vehicle", "car_model_id"),
            field_row("Driver", "vehicle", "driver_name"), field_row("Track temperature", "session", "track_temperature_c", "degC"),
            field_row("Air temperature", "session", "ambient_temperature_c", "degC"),
        ]

    def _row(self, channel: tuple, session: SessionAnalysis, stale: bool) -> dict:
        name, value, unit, source = channel
        obs = Observed(value, label_for(value, source, mock=session.mock, stale=stale, replay=self.replay), source)
        return {"channel": name, "value": obs.value, "unit": unit, "label": obs.label.value, "source": obs.source or "-"}


# ---------------------------------------------------------------------------
# Replay and exports
# ---------------------------------------------------------------------------

def replay_session_log(path: Union[str, Path], *, thresholds: Optional[HealthConfig] = None) -> tuple[LiveAnalysis, float]:
    """Run a recorded log through the live code. Returns the analysis and the log's last time, for ``snapshot(now=...)``."""
    records, _ = read_session_log(path)
    analysis = LiveAnalysis(thresholds=thresholds, replay=True)
    last = 0.0
    for rec in records:
        if rec.get("record_type") != "message":
            continue
        env = rec["envelope"]
        last = datetime.fromisoformat(rec["received_at_utc"]).astimezone(timezone.utc).timestamp()
        status = "out_of_order" if "out_of_order" in rec.get("quality", []) else "new"
        analysis.on_message(env["message_type"], env["payload"], status, last, mock=env.get("mock", False))
    return analysis, last


def _unique(path: Path) -> Path:
    """Never overwrite an earlier export: add a numeric suffix if the name is taken."""
    if not path.exists():
        return path
    for i in range(2, 10_000):
        candidate = path.with_name(f"{path.stem}_{i}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not find a free name for {path}")


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path = _unique(path)
    with open(path, "x", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


def all_rows(analysis: LiveAnalysis) -> tuple[list[dict], list[dict]]:
    with analysis._lock:
        sessions = list(analysis.sessions.values())
        return ([lap.as_row() for s in sessions for lap in s.tracker.laps], [st.as_row() for s in sessions for st in s.tracker.stints])


def write_session_exports(analysis: LiveAnalysis, stem: Union[str, Path], *, html: bool = True, now: Optional[float] = None) -> dict[str, Path]:
    """Write ``<stem>_laps.csv``, ``<stem>_stints.csv`` and optionally ``<stem>_summary.html`` (never overwriting)."""
    stem = Path(stem)
    laps, stints = all_rows(analysis)
    out = {"laps": _write_csv(stem.with_name(stem.name + "_laps.csv"), LAP_CSV_COLUMNS, laps),
           "stints": _write_csv(stem.with_name(stem.name + "_stints.csv"), STINT_CSV_COLUMNS, stints)}
    if html:
        from racesim.reporting import generate_live_session_report

        path = _unique(stem.with_name(stem.name + "_summary.html"))
        with analysis._lock:
            keys = list(analysis.sessions)
        generate_live_session_report([analysis.snapshot(now, key) for key in keys], path)
        out["summary"] = path
    return out
