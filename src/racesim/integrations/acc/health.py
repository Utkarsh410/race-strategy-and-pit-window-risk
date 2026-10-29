"""Connection health for the Mac receiver (thread-safe).

Four components are tracked separately, because a running UDP receiver proves
nothing about ACC itself:

1. **receiver** - is this process listening on its socket?
2. **bridge** - are heartbeats arriving from the Windows bridge (or mock sender)?
3. **acc** - what does the bridge report about its own ACC connection?
4. **telemetry** - how fresh is the latest valid telemetry message?

The overall state is ``CONNECTED`` only when all four are ``CONNECTED``.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional

from racesim.integrations.acc.config import HealthConfig


class HealthState(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    WAITING_FOR_DATA = "WAITING_FOR_DATA"
    CONNECTED = "CONNECTED"
    STALE = "STALE"
    DISCONNECTED = "DISCONNECTED"
    ERROR = "ERROR"


_SEVERITY = [HealthState.CONNECTED, HealthState.WAITING_FOR_DATA, HealthState.STALE, HealthState.DISCONNECTED,
             HealthState.NOT_STARTED, HealthState.ERROR]


@dataclass
class HealthSnapshot:
    overall: HealthState
    receiver: HealthState
    bridge: HealthState
    acc: HealthState
    telemetry: HealthState
    listening: bool
    packets_received: int
    valid_messages: int
    invalid_packets: int
    duplicate_packets: int
    out_of_order_packets: int
    sequence_gaps: int
    last_sequence: Optional[int]
    last_valid_age_s: Optional[float]
    last_telemetry_age_s: Optional[float]
    last_heartbeat_age_s: Optional[float]
    session_id: Optional[str]
    acc_status: Optional[str]
    acc_last_packet_age_s: Optional[float]
    mock: Optional[bool]
    last_error: Optional[str]
    invalid_reasons: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, HealthState):
                d[k] = v.value
        return d


def _age(now: float, t: Optional[float]) -> Optional[float]:
    return None if t is None else max(0.0, now - t)


class ConnectionHealth:
    """Counters and freshness tracking. Times are ``time.monotonic()`` seconds."""

    WINDOW = 1024  # recent sequence numbers remembered for duplicate detection

    def __init__(self, thresholds: Optional[HealthConfig] = None):
        self.t = thresholds or HealthConfig()
        self._lock = threading.Lock()
        self._started = False
        self._listening = False
        self._error: Optional[str] = None
        self.packets_received = 0
        self.valid_messages = 0
        self.invalid_packets = 0
        self.invalid_reasons: dict[str, int] = {}
        self.duplicates = 0
        self.out_of_order = 0
        self.gaps = 0
        self.last_sequence: Optional[int] = None
        self._run: Optional[tuple[str, str]] = None
        self._recent: deque[int] = deque(maxlen=self.WINDOW)
        self._recent_set: set[int] = set()
        self.last_valid_t: Optional[float] = None
        self.last_telemetry_t: Optional[float] = None
        self.last_heartbeat_t: Optional[float] = None
        self.session_id: Optional[str] = None
        self.acc_status: Optional[str] = None
        self.acc_packet_age_at_heartbeat: Optional[float] = None
        self.mock: Optional[bool] = None

    # -- receiver lifecycle -------------------------------------------------
    def mark_listening(self) -> None:
        with self._lock:
            self._started, self._listening, self._error = True, True, None

    def mark_stopped(self) -> None:
        with self._lock:
            self._listening = False

    def mark_error(self, message: str) -> None:
        with self._lock:
            self._error = message[:300]

    # -- packets --------------------------------------------------------------
    def record_packet(self) -> None:
        with self._lock:
            self.packets_received += 1

    def record_invalid(self, reason: str) -> None:
        with self._lock:
            self.invalid_packets += 1
            key = reason.split(":")[0][:60]
            self.invalid_reasons[key] = self.invalid_reasons.get(key, 0) + 1

    def check_sequence(self, sender_id: str, run_id: str, sequence: int) -> str:
        """Classify a sequence number: ``new``, ``duplicate`` or ``out_of_order`` (gaps are counted)."""
        with self._lock:
            run = (sender_id, run_id)
            if run != self._run:  # new sender process: start a fresh sequence space
                self._run, self.last_sequence = run, None
                self._recent.clear()
                self._recent_set.clear()
            if sequence in self._recent_set:
                self.duplicates += 1
                return "duplicate"
            if len(self._recent) == self._recent.maxlen:
                self._recent_set.discard(self._recent[0])
            self._recent.append(sequence)
            self._recent_set.add(sequence)
            if self.last_sequence is not None and sequence < self.last_sequence:
                self.out_of_order += 1
                self.gaps = max(0, self.gaps - 1)  # a late packet fills a previously counted gap
                return "out_of_order"
            if self.last_sequence is not None and sequence > self.last_sequence + 1:
                self.gaps += sequence - self.last_sequence - 1
            self.last_sequence = sequence
            return "new"

    def record_valid(self, message_type: str, now: float, *, session_id: Optional[str] = None, mock: bool = False,
                     acc_status: Optional[str] = None, acc_packet_age_s: Optional[float] = None) -> None:
        with self._lock:
            self.valid_messages += 1
            self.last_valid_t = now
            self.mock = mock
            if session_id:
                self.session_id = session_id
            if message_type == "telemetry":
                self.last_telemetry_t = now
            elif message_type == "heartbeat":
                self.last_heartbeat_t = now
                self.acc_status = acc_status
                self.acc_packet_age_at_heartbeat = acc_packet_age_s

    # -- state ----------------------------------------------------------------
    def _freshness(self, age: Optional[float]) -> HealthState:
        if age is None:
            return HealthState.WAITING_FOR_DATA
        if age <= self.t.stale_after_seconds:
            return HealthState.CONNECTED
        if age <= self.t.disconnect_after_seconds:
            return HealthState.STALE
        return HealthState.DISCONNECTED

    def snapshot(self, now: Optional[float] = None) -> HealthSnapshot:
        now = time.monotonic() if now is None else now
        with self._lock:
            if self._error:
                receiver = HealthState.ERROR
            elif not self._started:
                receiver = HealthState.NOT_STARTED
            elif not self._listening:
                receiver = HealthState.DISCONNECTED
            else:
                receiver = HealthState.CONNECTED if self.valid_messages else HealthState.WAITING_FOR_DATA
            hb_age = _age(now, self.last_heartbeat_t)
            bridge = HealthState.NOT_STARTED if hb_age is None else self._freshness(hb_age)
            # ACC state as last reported by the bridge; unknown once the bridge itself is not fresh.
            if hb_age is None:
                acc = HealthState.NOT_STARTED
            elif bridge != HealthState.CONNECTED:
                acc = HealthState.STALE if bridge == HealthState.STALE else HealthState.DISCONNECTED
            elif self.acc_status in ("AUTH_FAILED", "ERROR"):
                acc = HealthState.ERROR
            elif self.acc_status in ("REGISTERED", "MOCK"):
                acc_age = None if self.acc_packet_age_at_heartbeat is None else self.acc_packet_age_at_heartbeat + hb_age
                acc = self._freshness(acc_age)
            else:  # NOT_REGISTERED / REGISTERING / NO_DATA
                acc = HealthState.WAITING_FOR_DATA
            tel_age = _age(now, self.last_telemetry_t)
            telemetry = self._freshness(tel_age)
            parts = [receiver, bridge, acc, telemetry]
            if receiver in (HealthState.NOT_STARTED, HealthState.ERROR):
                overall = receiver
            elif all(p == HealthState.CONNECTED for p in parts):
                overall = HealthState.CONNECTED
            elif self.valid_messages == 0:
                overall = HealthState.WAITING_FOR_DATA
            else:
                overall = max(parts, key=_SEVERITY.index)
                if overall in (HealthState.NOT_STARTED, HealthState.CONNECTED):
                    overall = HealthState.WAITING_FOR_DATA
            return HealthSnapshot(
                overall, receiver, bridge, acc, telemetry, self._listening, self.packets_received, self.valid_messages,
                self.invalid_packets, self.duplicates, self.out_of_order, self.gaps, self.last_sequence,
                _age(now, self.last_valid_t), tel_age, hb_age, self.session_id, self.acc_status,
                None if self.acc_packet_age_at_heartbeat is None or hb_age is None else self.acc_packet_age_at_heartbeat + hb_age,
                self.mock, self._error, dict(self.invalid_reasons))
