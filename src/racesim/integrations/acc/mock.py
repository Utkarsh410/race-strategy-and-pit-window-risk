"""Deterministic MOCK telemetry for testing the pipeline without ACC.

Every payload has ``source = "mock"`` and every envelope ``mock = true``; the
track is named ``MOCK TRACK (synthetic)``. The receiver and dashboard label
these as mock data. They are never presented as live ACC telemetry.
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from typing import Optional

from racesim.integrations.acc.models import EventPayload, HeartbeatPayload, TelemetryPayload, utc_now

MOCK_TRACK = "MOCK TRACK (synthetic)"


@dataclass
class MockTelemetryGenerator:
    seed: int = 0
    base_lap_time_s: float = 100.0
    lap_samples: int = 10           # telemetry messages per simulated lap
    session_id: str = field(default_factory=lambda: "mock-" + uuid.uuid4().hex[:8])
    _rng: random.Random = field(init=False)
    _i: int = field(default=0, init=False)
    _laps: int = field(default=0, init=False)
    _last: Optional[float] = field(default=None, init=False)
    _best: Optional[float] = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def telemetry(self, sequence: int = 0) -> TelemetryPayload:
        """Next mock telemetry sample (completes a lap every ``lap_samples`` calls)."""
        self._i += 1
        events = []
        if self._i % self.lap_samples == 0:
            self._laps += 1
            self._last = round(self.base_lap_time_s + self._rng.gauss(0, 0.4), 3)
            self._best = self._last if self._best is None else min(self._best, self._last)
            events.append("lap_completed")
        frac = (self._i % self.lap_samples) / self.lap_samples
        src = "MOCK generator (not ACC)"
        vehicle = {"car_id": 1, "position": 1 + (self._laps % 3), "laps_completed": self._laps, "current_lap": self._laps + 1,
                   "current_lap_time_s": round(frac * self.base_lap_time_s, 3), "car_location": "TRACK",
                   "speed_kmh": int(180 + 40 * self._rng.random()), "spline_position": round(frac, 4), "is_focused_car": True}
        if self._last is not None:
            vehicle.update(last_lap_time_s=self._last, best_lap_time_s=self._best, last_lap_valid=True)
        session = {"track_name": MOCK_TRACK, "session_type": "PRACTICE", "session_phase": "SESSION",
                   "session_time_elapsed_s": round(self._i * self.base_lap_time_s / self.lap_samples, 3),
                   "track_temperature_c": 30, "ambient_temperature_c": 22}
        sources = {f"session.{k}": src for k in session} | {f"vehicle.{k}": src for k in vehicle}
        return TelemetryPayload(source="mock", session_id=self.session_id, timestamp_utc=utc_now(), sequence=sequence,
                                native_message_type="MOCK", session=session, vehicle=vehicle,
                                connection={"connected": True, "last_packet_age_s": 0.0}, field_sources=sources,
                                data_quality=["MOCK DATA - not from ACC"] + events)

    def heartbeat(self, *, forwarded: int, uptime_s: float) -> HeartbeatPayload:
        return HeartbeatPayload(source="mock", timestamp_utc=utc_now(), acc_status="MOCK", acc_last_packet_age_s=0.0,
                                native_messages_received=self._i, messages_forwarded=forwarded, session_id=self.session_id,
                                track_name=MOCK_TRACK, session_type="PRACTICE", uptime_s=uptime_s)

    def lap_event(self) -> EventPayload:
        return EventPayload(source="mock", timestamp_utc=utc_now(), session_id=self.session_id, event_type="LAP_COMPLETED",
                            message=f"MOCK lap {self._laps} completed", car_id=1)
