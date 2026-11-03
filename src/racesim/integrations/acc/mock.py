"""Deterministic MOCK telemetry for testing the pipeline without ACC.

Every payload has ``source = "mock"`` and every envelope ``mock = true``; the
track is named ``MOCK TRACK (synthetic)``. The receiver and dashboard label
these as mock data. They are never presented as live ACC telemetry.
"""

from __future__ import annotations

import math
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


# ---------------------------------------------------------------------------
# Mock shared-memory source (same attribute structure as pyaccsharedmemory 1.0.0)
# ---------------------------------------------------------------------------

from enum import Enum as _Enum
from types import SimpleNamespace as _NS


class MockAccStatus(_Enum):
    """Mirrors pyaccsharedmemory's ACC_STATUS member names."""

    ACC_OFF = 0
    ACC_REPLAY = 1
    ACC_LIVE = 2
    ACC_PAUSE = 3


def _wheels(fl, fr, rl, rr):
    return _NS(front_left=fl, front_right=fr, rear_left=rl, rear_right=rr)


def _vec(x, y, z):
    return _NS(x=x, y=y, z=z)


_contact = _wheels  # ContactPoint has the same per-wheel attributes, holding vectors


@dataclass
class MockSharedMemorySource:
    """Deterministic MOCK snapshots shaped like ``pyaccsharedmemory``'s ``ACC_map``.

    Values are synthetic and plausible-looking; they are NOT ACC data.
    ``drop`` removes attributes (to test missing-channel detection) and
    ``status`` simulates ACC being off, paused or in a replay.
    """

    seed: int = 0
    drop: tuple[str, ...] = ()
    status: str = "ACC_LIVE"
    freeze_packet: bool = False
    raise_every: int = 0          # simulate the library's ValueError on unknown enums every N reads
    source_name: str = "mock"
    _i: int = field(default=0, init=False)
    closed: bool = field(default=False, init=False)

    def read(self):
        self._i += 1
        if self.raise_every and self._i % self.raise_every == 0:
            raise ValueError("99 is not a valid ACC_FLAG_TYPE")
        rng = random.Random(self.seed * 100_003 + self._i)
        t = self._i * 0.05
        speed = 180 + 40 * math.sin(t / 3)
        phys = _NS(packed_id=1 if self.freeze_packet else self._i, gas=round(0.5 + 0.5 * math.sin(t), 3),
                   brake=round(max(0.0, -math.sin(t)), 3), clutch=0.0, steer_angle=round(0.3 * math.sin(t / 2), 3), gear=4,
                   rpm=int(6500 + 800 * math.sin(t)), speed_kmh=round(speed, 2), fuel=round(60 - 0.001 * self._i, 3),
                   velocity=_vec(speed / 3.6, 0.0, 0.1), local_velocity=_vec(0.1, 0.0, speed / 3.6), g_force=_vec(0.1, 1.0, -0.4),
                   local_angular_vel=_vec(0.0, 0.05, 0.0), heading=0.5, pitch=0.01, roll=0.0,
                   wheel_slip=_wheels(0.1, 0.1, 0.2, 0.2), slip_ratio=_wheels(0.01, 0.01, 0.02, 0.02),
                   slip_angle=_wheels(0.02, 0.02, 0.01, 0.01), wheel_angular_s=_wheels(150.0, 150.0, 148.0, 148.0),
                   wheel_pressure=_wheels(*(round(27.5 + rng.gauss(0, 0.05), 2) for _ in range(4))),
                   tyre_core_temp=_wheels(*(round(85 + rng.gauss(0, 0.5), 1) for _ in range(4))),
                   brake_temp=_wheels(450.0, 452.0, 380.0, 381.0), brake_pressure=_wheels(0.0, 0.0, 0.0, 0.0),
                   pad_life=_wheels(28.5, 28.5, 28.7, 28.7), disc_life=_wheels(31.8, 31.8, 31.9, 31.9),
                   suspension_travel=_wheels(0.031, 0.030, 0.042, 0.041), suspension_damage=_wheels(0, 0, 0, 0),
                   tc=0.0, abs=0.0, brake_bias=0.56, pit_limiter_on=False, is_ai_controlled=False, air_temp=22.0, road_temp=30.0,
                   water_temp=88.0, turbo_boost=0.0, is_engine_running=True,
                   tyre_contact_point=_contact(*(_vec(x, 0.0, z) for x, z in ((0.8, 1.3), (-0.8, 1.3), (0.8, -1.3), (-0.8, -1.3)))),
                   tyre_contact_normal=_contact(*(_vec(0.0, 1.0, 0.0) for _ in range(4))),
                   tyre_contact_heading=_contact(*(_vec(0.0, 0.0, 1.0) for _ in range(4))),
                   car_damage=_NS(front=0.0, rear=0.0, left=0.0, right=0.0, center=0.0))
        gfx = _NS(packed_id=self._i, status=MockAccStatus[self.status], session_type=_NS(name="ACC_PRACTICE"), completed_lap=self._i // 1000,
                  position=1, current_time=int(self._i * 50 % 100_000), last_time=101_234, best_time=100_987,
                  current_sector_index=0, is_in_pit=False, is_in_pit_lane=False, tyre_compound="dry_compound\x00\x00",
                  current_tyre_set=1, track_grip_status=_NS(name="ACC_OPTIMUM"), rain_intensity=_NS(name="ACC_NO_RAIN"),
                  flag=_NS(name="ACC_NO_FLAG"), session_time_left=1_800_000.0, fuel_per_lap=2.9, fuel_estimated_laps=20.5,
                  used_fuel=0.5)
        static = _NS(car_model="mock_car (synthetic)", track="MOCK TRACK (synthetic)", max_rpm=8000, max_fuel=120.0,
                     sm_version="1.8\x00", ac_version="mock\x00")
        for name in self.drop:
            for obj in (phys, gfx, static):
                if hasattr(obj, name):
                    delattr(obj, name)
        return _NS(Physics=phys, Graphics=gfx, Static=static)

    def close(self) -> None:
        self.closed = True
