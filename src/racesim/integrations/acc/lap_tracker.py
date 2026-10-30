"""Completed-lap detection, lap classification and stint management (Mac side, standard library only).

Evidence from real ACC logs that shapes this module:

* ``laps_completed`` and ``lastLap`` do **not** change when an out-lap crosses the
  line, so "pit activity since the last lap increment" would wrongly mark the
  first timed lap as an out-lap. Pit evidence is taken only from the lap's own
  window ``[end - lap_time, end]`` on the session clock.
* Broadcasting events repeat many times, so laps are counted from
  ``laps_completed`` in telemetry and never from events.

A slow lap is never discarded. Without evidence of the cause it becomes
``UNCERTAIN`` and stays visible in every table.
"""

from __future__ import annotations

import statistics
from collections import deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional

PIT_LOCATIONS = {"PITLANE", "PIT_ENTRY"}
TRACK_LOCATIONS = {"TRACK", "PIT_EXIT"}
LAP_TIME_WAIT_S = 3.0             # wait this long for lastLap to update after the lap counter moved
MAX_TELEMETRY_GAP_S = 5.0         # a longer hole inside a lap leaves its classification uncertain
SLOW_LAP_RATIO = 1.07
FAST_LAP_RATIO = 0.90
OUTLIER_MIN_CLEAN_LAPS = 3
SECTOR_SUM_TOLERANCE_S = 0.1
LINE_CROSSING_DROP_S = 10.0
LAP_CLOCK_TOLERANCE_S = 1.0       # lastLap vs the session clock between two lap-counter changes
MAX_SAMPLES = 50_000


class LapClass(str, Enum):
    CLEAN = "CLEAN_CANDIDATE"
    INVALID = "INVALID"
    INCOMPLETE = "INCOMPLETE"
    OUT_LAP = "OUT_LAP"
    IN_LAP = "IN_LAP"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class CompletedLap:
    session_key: str
    lap_number: int
    lap_time_s: Optional[float]
    classification: LapClass
    stint_id: Optional[int]
    stint_lap: Optional[int]
    position: Optional[int]
    in_pit: bool
    sector_times_s: Optional[list[Optional[float]]] = None
    sector_source: Optional[str] = None
    tyre_compound: Optional[str] = None
    tyre_age_laps: Optional[int] = None
    fuel_start_kg: Optional[float] = None
    fuel_end_kg: Optional[float] = None
    session_time_end_s: Optional[float] = None
    flags: list[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return self.classification == LapClass.CLEAN and self.lap_time_s is not None

    def as_dict(self) -> dict:
        d = asdict(self)
        d["classification"] = self.classification.value
        return d

    def as_row(self) -> dict:
        d = self.as_dict()
        sectors = d.pop("sector_times_s") or []
        for i in range(3):
            d[f"sector_{i + 1}_s"] = sectors[i] if i < len(sectors) else None
        d["flags"] = "|".join(d["flags"])
        return d


@dataclass
class Stint:
    session_key: str
    stint_id: int
    start_reason: str
    start_session_time_s: Optional[float] = None
    end_session_time_s: Optional[float] = None
    end_reason: Optional[str] = None
    lap_numbers: list[int] = field(default_factory=list)
    pit_entry_session_time_s: Optional[float] = None
    pit_exit_session_time_s: Optional[float] = None
    stationary_observed: Optional[bool] = None
    tyre_compound: Optional[str] = None
    tyre_age_start_laps: Optional[int] = None
    fuel_start_kg: Optional[float] = None
    fuel_end_kg: Optional[float] = None
    left_pits_at_start: bool = False
    track_samples: int = 0

    def as_row(self) -> dict:
        d = asdict(self)
        d["laps"] = len(d.pop("lap_numbers"))
        d["first_lap"] = self.lap_numbers[0] if self.lap_numbers else None
        d["last_lap"] = self.lap_numbers[-1] if self.lap_numbers else None
        d.pop("track_samples")
        return d


@dataclass(frozen=True)
class _Sample:
    elapsed: Optional[float]
    location: Optional[str]
    fuel: Optional[float]
    stint_id: Optional[int]
    phase: Optional[str]
    received_t: float


@dataclass
class _Pending:
    lap: int
    elapsed: Optional[float]
    received_t: float
    previous_time: Optional[float]
    current_splits: Optional[list]
    previous_elapsed: Optional[float] = None


class LapTracker:
    """Feed telemetry in arrival order; ``update`` returns laps that were finalised by that message."""

    def __init__(self, session_key: str):
        self.session_key = session_key
        self.laps: list[CompletedLap] = []
        self.stints: list[Stint] = []
        self.transitions: list[dict] = []
        self.flags: dict[str, int] = {}
        self.closed_stints: list[Stint] = []  # closed since the last call to pop_closed_stints
        self._seen: set[int] = set()
        self._samples: deque[_Sample] = deque(maxlen=MAX_SAMPLES)
        self._laps_value: Optional[int] = None
        self._counter_elapsed: Optional[float] = None
        self._last_time: Optional[float] = None
        self._pending: Optional[_Pending] = None
        self._in_pit: Optional[bool] = None
        self._location: Optional[str] = None
        self._stationary = False
        self._current_splits: Optional[list] = None
        self._current_lap_time: Optional[float] = None
        self._session_type: Optional[str] = None

    @property
    def current_stint(self) -> Optional[Stint]:
        return self.stints[-1] if self.stints else None

    def _flag(self, name: str) -> None:
        self.flags[name] = self.flags.get(name, 0) + 1

    # -- input ----------------------------------------------------------------
    def update(self, payload: dict, received_t: float) -> list[CompletedLap]:
        s, v = payload.get("session") or {}, payload.get("vehicle") or {}
        elapsed, laps = s.get("session_time_elapsed_s"), v.get("laps_completed")
        self._session_type = s.get("session_type") or self._session_type
        self._update_pit(v, elapsed)
        stint = self.current_stint
        if stint is not None:
            stint.fuel_start_kg = stint.fuel_start_kg if stint.fuel_start_kg is not None else v.get("fuel_kg")
            stint.fuel_end_kg = v.get("fuel_kg") if v.get("fuel_kg") is not None else stint.fuel_end_kg
            if v.get("tyre_compound") is not None:
                stint.tyre_compound = v["tyre_compound"]
            if stint.tyre_age_start_laps is None:
                stint.tyre_age_start_laps = v.get("tyre_age_laps")
        self._samples.append(_Sample(elapsed, v.get("car_location"), v.get("fuel_kg"), stint.stint_id if stint else None,
                                     s.get("session_phase"), received_t))
        out: list[CompletedLap] = []
        if self._pending is not None:
            if laps is not None and laps > self._pending.lap:  # lastLap now describes a later lap, not the pending one
                out.append(self._finalise({**v, "last_lap_time_s": None}, timed_out=True))
            else:
                out += self._try_finalise(v, received_t)
        if laps is not None:
            out += self._update_lap_counter(laps, v, elapsed, received_t)
        cur = v.get("current_lap_time_s")
        if (cur is not None and self._current_lap_time is not None and cur < self._current_lap_time - LINE_CROSSING_DROP_S
                and not (self._pending is not None or (out and out[-1].lap_number == laps))):
            self._flag("line_crossing_without_lap_count")  # e.g. the out-lap, which ACC does not count
        self._current_lap_time = cur if cur is not None else self._current_lap_time
        if v.get("current_lap_splits_s"):
            self._current_splits = v["current_lap_splits_s"]
        return out

    def _update_lap_counter(self, laps: int, v: dict, elapsed: Optional[float], received_t: float) -> list[CompletedLap]:
        out: list[CompletedLap] = []
        if self._laps_value is None:
            self._laps_value, self._last_time, self._counter_elapsed = laps, v.get("last_lap_time_s"), elapsed
            if laps > 0:
                self._flag("joined_mid_session")  # laps before this point were not observed, so none are recorded
            return out
        if laps <= self._laps_value:
            if self._pending is None:
                self._last_time = v.get("last_lap_time_s") if v.get("last_lap_time_s") is not None else self._last_time
            return out
        for missed in range(self._laps_value + 1, laps):
            out.append(self._missed_lap(missed, elapsed))
        self._laps_value = laps
        self._pending = _Pending(laps, elapsed, received_t, self._last_time, self._current_splits, self._counter_elapsed)
        self._counter_elapsed = elapsed
        self._current_splits = None
        out += self._try_finalise(v, received_t)
        return out

    def _try_finalise(self, v: dict, received_t: float) -> list[CompletedLap]:
        p = self._pending
        value = v.get("last_lap_time_s")
        if value is not None and (value != p.previous_time or self._matches_clock(value, p)):
            return [self._finalise(v, timed_out=False)]
        if received_t - p.received_t >= LAP_TIME_WAIT_S:
            return [self._finalise(v, timed_out=True)]
        return []

    @staticmethod
    def _matches_clock(value: float, p: _Pending) -> bool:
        """Two laps can share a time: an unchanged lastLap still describes this lap if the session clock agrees."""
        return (p.elapsed is not None and p.previous_elapsed is not None
                and abs((p.elapsed - p.previous_elapsed) - value) <= LAP_CLOCK_TOLERANCE_S)

    # -- stint / pit state ----------------------------------------------------
    def _update_pit(self, v: dict, elapsed: Optional[float]) -> None:
        loc = v.get("car_location")
        if loc is None or (loc not in PIT_LOCATIONS and loc not in TRACK_LOCATIONS):
            return
        in_pit = loc in PIT_LOCATIONS
        if self._in_pit is None:
            self.stints.append(Stint(self.session_key, 1, "session_start_in_pit" if in_pit else "first_seen_on_track", elapsed))
            self._in_pit, self._location = in_pit, loc
            self._stationary = in_pit and v.get("speed_kmh") == 0
            return
        stint = self.stints[-1]
        if in_pit and v.get("speed_kmh") == 0:
            self._stationary = True
        if not in_pit:
            stint.track_samples += 1
        if in_pit and not self._in_pit and self._location != "PIT_EXIT":  # PIT_EXIT -> PITLANE is location flicker, not an entry
            self._in_pit, self._stationary = True, v.get("speed_kmh") == 0
            stint.pit_entry_session_time_s = elapsed
            self.transitions.append({"session_key": self.session_key, "event": "PIT_ENTRY", "session_time_s": elapsed,
                                     "laps_completed": v.get("laps_completed"), "stint_id": stint.stint_id, "stationary_observed": None})
        elif not in_pit and self._in_pit:
            self._in_pit = False
            self.transitions.append({"session_key": self.session_key, "event": "PIT_EXIT", "session_time_s": elapsed,
                                     "laps_completed": v.get("laps_completed"), "stint_id": stint.stint_id,
                                     "stationary_observed": self._stationary})
            if stint.pit_entry_session_time_s is None and not stint.lap_numbers and stint.track_samples <= 1:
                stint.left_pits_at_start = True  # leaving the garage at the start is not a stint transition
            else:
                stint.pit_exit_session_time_s, stint.stationary_observed = elapsed, self._stationary
                stint.end_session_time_s, stint.end_reason = elapsed, "pit_visit"
                self.closed_stints.append(stint)
                self.stints.append(Stint(self.session_key, stint.stint_id + 1, "pit_exit", elapsed))
        self._location = loc

    def close(self, reason: str, elapsed: Optional[float] = None) -> None:
        stint = self.current_stint
        if stint is not None and stint.end_reason is None:
            stint.end_session_time_s, stint.end_reason = elapsed if elapsed is not None else stint.end_session_time_s, reason
            self.closed_stints.append(stint)

    def pop_closed_stints(self) -> list[Stint]:
        closed, self.closed_stints = self.closed_stints, []
        return closed

    # -- lap finalisation -----------------------------------------------------
    def _missed_lap(self, lap: int, elapsed: Optional[float]) -> CompletedLap:
        self._flag("telemetry_missed_lap")
        stint = self.current_stint
        return self._record(CompletedLap(self.session_key, lap, None, LapClass.INCOMPLETE, stint.stint_id if stint else None, None, None,
                                         False, session_time_end_s=None, flags=["telemetry_missed_lap"]))

    def _finalise(self, v: dict, *, timed_out: bool) -> CompletedLap:
        p, self._pending = self._pending, None
        value = v.get("last_lap_time_s")
        fresh = value is not None and (value != p.previous_time or not timed_out)
        lap_time = value if fresh else None
        if lap_time is not None:
            self._last_time = lap_time
        window = self._window(p.elapsed, lap_time)
        flags: list[str] = []
        if lap_time is None:
            flags.append("lap_time_not_updated")
        classification = self._classify(lap_time, v, window, p, flags)
        sectors, source = self._sectors(lap_time, v, p, flags)
        stint = self.current_stint
        stint_id = window[-1].stint_id if window else (stint.stint_id if stint else None)
        owner = next((st for st in self.stints if st.stint_id == stint_id), None)
        lap = CompletedLap(
            self.session_key, p.lap, lap_time, classification, stint_id,
            None if owner is None else len(owner.lap_numbers) + 1, v.get("position"), any(x.location in PIT_LOCATIONS for x in window),
            sectors, source, v.get("tyre_compound"), v.get("tyre_age_laps"),
            next((x.fuel for x in window if x.fuel is not None), None),
            next((x.fuel for x in reversed(window) if x.fuel is not None), None), p.elapsed, flags)
        if owner is not None:
            owner.lap_numbers.append(p.lap)
        self._trim(p.elapsed)
        return self._record(lap)

    def _record(self, lap: CompletedLap) -> CompletedLap:
        if lap.lap_number in self._seen:  # a lap is processed once, whatever arrives later
            self._flag("duplicate_lap_ignored")
            return lap
        self._seen.add(lap.lap_number)
        self.laps.append(lap)
        for f in lap.flags:
            self._flag(f)
        return lap

    def _window(self, end: Optional[float], lap_time: Optional[float]) -> list[_Sample]:
        if end is None or lap_time is None:
            return list(self._samples)
        return [x for x in self._samples if x.elapsed is not None and end - lap_time - 0.5 <= x.elapsed <= end]

    def _trim(self, end: Optional[float]) -> None:
        if end is None:
            self._samples.clear()
            return
        while self._samples and self._samples[0].elapsed is not None and self._samples[0].elapsed < end - 1.0:
            self._samples.popleft()

    def _classify(self, lap_time: Optional[float], v: dict, window: list[_Sample], p: _Pending, flags: list[str]) -> LapClass:
        if lap_time is None:
            return LapClass.INCOMPLETE
        if p.elapsed is None:
            flags.append("session_clock_unavailable")
        if v.get("last_lap_valid") is False:
            flags.append("acc_lap_invalid")
            return LapClass.INVALID
        pit = [x.location in PIT_LOCATIONS for x in window]
        if v.get("last_lap_is_out_lap") or (pit and pit[0]) or any(x.location == "PIT_EXIT" for x in window):
            flags.append("pit_exit_in_lap" if not v.get("last_lap_is_out_lap") else "acc_out_lap")
            if any(pit) and not pit[0]:
                flags.append("pit_entry_in_lap")
            return LapClass.OUT_LAP
        if v.get("last_lap_is_in_lap") or any(pit):
            flags.append("pit_entry_in_lap" if not v.get("last_lap_is_in_lap") else "acc_in_lap")
            return LapClass.IN_LAP
        uncertain = []
        if p.elapsed is None:
            uncertain.append("lap_window_unknown")
        if v.get("last_lap_valid") is None:
            uncertain.append("validity_unknown")
        times = [x.elapsed for x in window if x.elapsed is not None]
        if p.elapsed is not None:
            if not times or times[0] > p.elapsed - lap_time + MAX_TELEMETRY_GAP_S:
                uncertain.append("lap_start_not_observed")
            if any(b - a > MAX_TELEMETRY_GAP_S for a, b in zip(times, times[1:])):
                uncertain.append("telemetry_gap_in_lap")
        if any(x.phase not in (None, "SESSION") for x in window):
            uncertain.append("session_phase_not_green")
        if any(x.location not in PIT_LOCATIONS | TRACK_LOCATIONS for x in window):
            uncertain.append("car_location_unknown")
        if self._session_type == "RACE" and p.lap == 1:
            uncertain.append("race_start_lap")
        clean = [lap.lap_time_s for lap in self.laps if lap.is_clean]
        if len(clean) >= OUTLIER_MIN_CLEAN_LAPS:
            median = statistics.median(clean)
            if lap_time > median * SLOW_LAP_RATIO:
                uncertain.append("slow_lap_cause_unknown")
            elif lap_time < median * FAST_LAP_RATIO:
                uncertain.append("implausibly_fast_lap")
        flags.extend(uncertain)
        return LapClass.UNCERTAIN if uncertain else LapClass.CLEAN

    def _sectors(self, lap_time: Optional[float], v: dict, p: _Pending, flags: list[str]) -> tuple[Optional[list], Optional[str]]:
        """Sector times only when they add up to the lap: the wire format's per-sector meaning is unverified on real data."""
        sectors, source = v.get("last_lap_splits_s"), "REALTIME_CAR_UPDATE.lastLap.splits"
        if not sectors and p.current_splits and lap_time is not None:
            known = list(p.current_splits)
            if len(known) == 2 and None not in known or (len(known) == 3 and known[2] is None and None not in known[:2]):
                sectors, source = [known[0], known[1], round(lap_time - known[0] - known[1], 3)], "DERIVED: currentLap splits; last sector = lap - others"
        if not sectors:
            return None, None
        if lap_time is not None and None not in sectors and abs(sum(sectors) - lap_time) > SECTOR_SUM_TOLERANCE_S:
            flags.append("sector_sum_mismatch")
            return None, None
        return list(sectors), source
