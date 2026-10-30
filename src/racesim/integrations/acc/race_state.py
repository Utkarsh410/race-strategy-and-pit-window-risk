"""Live race state: session identity and provenance labels (Mac side, standard library only).

Session identity cannot rest on ``session_id`` alone. Real ACC logs show that a
session restart keeps the bridge's identifier while the session clock returns
to zero, and that a restarted bridge invents a new identifier for a session
that simply continues. :class:`SessionIdentity` therefore decides from the
identifier, the session clock, lap count, track, session type and timestamps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

RESTART_REGRESSION_S = 5.0        # a smaller backwards step in session time is treated as packet reordering
RECONNECT_CLOCK_TOLERANCE_S = 30.0
END_TIME_SUM_TOLERANCE_S = 2.0


class Provenance(str, Enum):
    LIVE = "LIVE MEASUREMENT"
    DERIVED = "DERIVED FROM TELEMETRY"
    MODEL = "MODEL PREDICTION"
    CONFIGURED = "CONFIGURED ASSUMPTION"
    UNAVAILABLE = "UNAVAILABLE"
    STALE = "STALE DATA"
    MOCK = "MOCK - NOT LIVE"
    REPLAY = "REPLAY - NOT LIVE"


def label_for(value: Any, source: str = "", *, mock: bool = False, stale: bool = False, replay: bool = False) -> Provenance:
    """Display label of one value. Missing always wins: an absent value is never shown as stale or live."""
    if value is None:
        return Provenance.UNAVAILABLE
    if mock:
        return Provenance.MOCK
    if replay:
        return Provenance.REPLAY
    if stale:
        return Provenance.STALE
    return Provenance.DERIVED if source.startswith("DERIVED") else Provenance.LIVE


@dataclass(frozen=True)
class Observed:
    """One value with its provenance; ``value is None`` means unavailable, never zero."""

    value: Any
    label: Provenance
    source: str = ""


class Decision(str, Enum):
    NEW_SESSION = "new_session"
    SAME = "same"
    RESTART = "restart"
    RECONNECTED = "reconnected"
    OUT_OF_ORDER = "out_of_order"
    LATE_OTHER_SESSION = "late_other_session"


def parse_ts(v: Any) -> Optional[datetime]:
    if isinstance(v, datetime):
        return v
    try:
        return datetime.fromisoformat(str(v))
    except ValueError:
        return None


@dataclass
class SessionIdentity:
    """Maps telemetry onto a stable ``session_key``; one key means one continuous session."""

    session_id: Optional[str] = None
    session_key: Optional[str] = None
    restarts: int = 0
    retired: set[str] = field(default_factory=set)
    _ts: Optional[datetime] = None
    _elapsed: Optional[float] = None
    _laps: Optional[int] = None
    _received: Optional[float] = None
    _context: Optional[tuple] = None
    _source: Optional[str] = None

    def resolve(self, payload: dict, received_t: float) -> Decision:
        sid = payload["session_id"]
        s, v = payload.get("session") or {}, payload.get("vehicle") or {}
        ts, elapsed, laps = parse_ts(payload.get("timestamp_utc")), s.get("session_time_elapsed_s"), v.get("laps_completed")
        context = (s.get("track_name"), s.get("session_type"), s.get("session_index"))
        if sid in self.retired:
            return Decision.LATE_OTHER_SESSION
        if self.session_id is None:
            decision = self._begin(sid)
        elif sid == self.session_id:
            if ts is not None and self._ts is not None and ts < self._ts:
                return Decision.OUT_OF_ORDER
            if self._is_restart(context, elapsed, laps):
                self.restarts += 1
                self.session_key = f"{sid}#r{self.restarts}"
                decision = Decision.RESTART
            else:
                decision = Decision.SAME
        elif self._continues(payload, context, elapsed, laps, received_t):
            self.retired.add(self.session_id)
            self.session_id = sid
            decision = Decision.RECONNECTED
        else:
            self.retired.add(self.session_id)
            decision = self._begin(sid)
        self._ts, self._elapsed, self._laps, self._received = ts, elapsed, laps, received_t
        self._context, self._source = context, payload.get("source")
        return decision

    def _begin(self, sid: str) -> Decision:
        self.session_id, self.session_key, self.restarts = sid, sid, 0
        return Decision.NEW_SESSION

    def _is_restart(self, context: tuple, elapsed: Optional[float], laps: Optional[int]) -> bool:
        if context != self._context and None not in context and None not in (self._context or (None,)):
            return True
        if elapsed is not None and self._elapsed is not None and elapsed < self._elapsed - RESTART_REGRESSION_S:
            return True
        return laps is not None and self._laps is not None and laps < self._laps

    def _continues(self, payload: dict, context: tuple, elapsed: Optional[float], laps: Optional[int], received_t: float) -> bool:
        """A restarted bridge reports a new id; the session is the same only if its clock kept running."""
        if payload.get("source") != self._source or context != self._context or None in context:
            return False
        if None in (elapsed, self._elapsed, laps, self._laps, self._received) or laps < self._laps or elapsed < self._elapsed:
            return False
        return abs((elapsed - self._elapsed) - (received_t - self._received)) <= RECONNECT_CLOCK_TOLERANCE_S


@dataclass
class SessionTimeRemaining:
    """Derive time remaining from ``sessionEndTime``.

    In real logs ``elapsed + sessionEndTime`` stays constant (900.0 s in a 15-minute
    session), so the raw value counts down. That holds only in the ``SESSION`` phase:
    in ``PRE_FORMATION`` the raw value falls while elapsed stays at 0, then jumps back.
    A value is produced only in ``SESSION`` and only while the sum stays stable.
    """

    _reference: Optional[float] = None
    _consistent: int = 0
    inconsistent: bool = False

    def update(self, elapsed: Optional[float], raw_end: Optional[float], phase: Optional[str]) -> Optional[float]:
        if elapsed is None or raw_end is None or raw_end < 0 or phase != "SESSION":
            return None
        total = elapsed + raw_end
        if self._reference is None:
            self._reference = total
        if abs(total - self._reference) > END_TIME_SUM_TOLERANCE_S:
            self.inconsistent = True
        self._consistent += 1
        return raw_end if self._consistent >= 2 and not self.inconsistent else None
