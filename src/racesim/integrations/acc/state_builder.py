"""Build normalised telemetry from native ACC messages (bridge side) and
label channels for display (receiver side).

Bridge side: :class:`AccStateBuilder` keeps the latest session, track,
entry-list and car state, detects session changes and produces
:class:`TelemetryPayload` objects for one car (the focused car by default).
Nothing is invented: a field is populated only when the native message that
supplies it has been received.

Receiver side: :func:`channel_table` turns a payload into rows labelled
``LIVE - MEASURED``, ``DERIVED``, ``UNAVAILABLE``, ``STALE`` or ``MOCK``.
:class:`LiveTelemetryFeed` is the read-only interface a future simulation
integration can use; it does not feed the strategy engine in this phase.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional, Protocol

from racesim.integrations.acc.models import EventPayload, SessionInfo, TelemetryPayload, VehicleInfo, utc_now
from racesim.integrations.acc.parser import (
    BroadcastingEvent, EntryList, EntryListCar, LapInfo, NativeMessage, RealtimeCarUpdate, RealtimeUpdate, TrackData,
)
from racesim.integrations.acc.protocol import BroadcastingEventType, CarLocation, SessionPhase, SessionType, enum_name


SESSION_RESTART_REGRESSION_MS = 5000.0


def _lap_s(lap: LapInfo) -> Optional[float]:
    """Lap time in seconds; None for the int32-max sentinel or non-positive placeholders."""
    if lap.lap_time_ms is None or lap.lap_time_ms <= 0:
        return None
    return lap.lap_time_ms / 1000.0


@dataclass
class AccStateBuilder:
    include_driver_names: bool = False
    car_index: Optional[int] = None  # None = follow ACC's focused car
    realtime: Optional[RealtimeUpdate] = None
    track: Optional[TrackData] = None
    entry_ids: tuple[int, ...] = ()
    entries: dict[int, EntryListCar] = field(default_factory=dict)
    cars: dict[int, RealtimeCarUpdate] = field(default_factory=dict)
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    session_changes: int = 0
    _session_key: Optional[tuple] = None
    unknown_car_seen: bool = False

    def target_car(self) -> Optional[int]:
        if self.car_index is not None:
            return self.car_index
        return None if self.realtime is None else self.realtime.focused_car_index

    def update(self, msg: NativeMessage) -> list[str]:
        """Apply a native message; returns notes such as ``session_changed``."""
        notes: list[str] = []
        if isinstance(msg, RealtimeUpdate):
            key = (msg.event_index, msg.session_index, msg.session_type, self.track.track_name if self.track else None)
            # A restart keeps every identifier above; only the session clock shows it (seen in real ACC logs).
            restarted = (self.realtime is not None and not msg.is_replay_playing and not self.realtime.is_replay_playing
                         and msg.session_time_ms < self.realtime.session_time_ms - SESSION_RESTART_REGRESSION_MS)
            if self._session_key is not None and (key != self._session_key or restarted):
                self.session_id = uuid.uuid4().hex[:12]
                self.session_changes += 1
                self.cars.clear()
                notes.append("session_changed")
            self._session_key = key
            self.realtime = msg
        elif isinstance(msg, TrackData):
            if self.track is not None and self.track.track_name != msg.track_name:
                self.session_id = uuid.uuid4().hex[:12]
                self.session_changes += 1
                notes.append("session_changed")
            self.track = msg
            if self.realtime is not None:
                r = self.realtime
                self._session_key = (r.event_index, r.session_index, r.session_type, msg.track_name)
        elif isinstance(msg, EntryList):
            self.entry_ids = msg.car_indices
            self.entries = {k: v for k, v in self.entries.items() if k in msg.car_indices}
        elif isinstance(msg, EntryListCar):
            self.entries[msg.car_index] = msg
        elif isinstance(msg, RealtimeCarUpdate):
            self.cars[msg.car_index] = msg
            if msg.car_index not in self.entries:
                self.unknown_car_seen = True
                notes.append("unknown_car")
        return notes

    def session_info(self) -> tuple[SessionInfo, dict[str, str]]:
        src: dict[str, str] = {}
        kw: dict = {}
        if self.track is not None:
            kw.update(track_name=self.track.track_name, track_length_m=self.track.track_meters)
            src.update({"session.track_name": "TRACK_DATA.trackName", "session.track_length_m": "TRACK_DATA.trackMeters"})
        r = self.realtime
        if r is not None:
            kw.update(session_type=enum_name(SessionType, r.session_type), session_phase=enum_name(SessionPhase, r.session_phase),
                      session_index=r.session_index, session_time_elapsed_s=r.session_time_ms / 1000.0,
                      session_end_time_raw_s=r.session_end_time_ms / 1000.0, time_of_day_s=r.time_of_day_ms / 1000.0,
                      is_replay_playing=r.is_replay_playing, ambient_temperature_c=r.ambient_temp_raw,
                      track_temperature_c=r.track_temp_raw, clouds_raw=r.clouds_raw, rain_level_raw=r.rain_level_raw,
                      wetness_raw=r.wetness_raw)
            for f, n in (("session_type", "sessionType"), ("session_phase", "sessionPhase"), ("session_index", "sessionIndex"),
                         ("session_time_elapsed_s", "sessionTime"), ("session_end_time_raw_s", "sessionEndTime"),
                         ("time_of_day_s", "timeOfDay"), ("is_replay_playing", "isReplayPlaying"),
                         ("ambient_temperature_c", "ambientTemp"), ("track_temperature_c", "trackTemp"),
                         ("clouds_raw", "clouds"), ("rain_level_raw", "rainLevel"), ("wetness_raw", "wetness")):
                src[f"session.{f}"] = f"REALTIME_UPDATE.{n}"
        return SessionInfo(**kw), src

    def vehicle_info(self, car_index: int) -> tuple[VehicleInfo, dict[str, str]]:
        src: dict[str, str] = {}
        kw: dict = {"car_id": car_index, "is_focused_car": self.realtime is not None and self.realtime.focused_car_index == car_index}
        src["vehicle.car_id"] = "REALTIME_CAR_UPDATE.carIndex"
        e = self.entries.get(car_index)
        if e is not None:
            kw.update(race_number=e.race_number, car_model_id=e.car_model_type)
            src.update({"vehicle.race_number": "ENTRY_LIST_CAR.raceNumber", "vehicle.car_model_id": "ENTRY_LIST_CAR.carModelType"})
            if self.include_driver_names and 0 <= e.current_driver_index < len(e.drivers):
                d = e.drivers[e.current_driver_index]
                kw["driver_name"] = (d.short_name or f"{d.first_name} {d.last_name}").strip()[:64] or None
                src["vehicle.driver_name"] = "ENTRY_LIST_CAR.drivers[currentDriverIndex]"
        c = self.cars.get(car_index)
        if c is not None:
            kw.update(position=c.position, cup_position=c.cup_position, track_position=c.track_position,
                      laps_completed=c.laps, current_lap=c.laps + 1, car_location=enum_name(CarLocation, c.car_location),
                      speed_kmh=c.speed_kmh, delta_s=c.delta_ms / 1000.0)
            if 0.0 <= c.spline_position <= 1.0:
                kw["spline_position"] = c.spline_position
                src["vehicle.spline_position"] = "REALTIME_CAR_UPDATE.splinePosition"
            for f, n in (("position", "position"), ("cup_position", "cupPosition"), ("track_position", "trackPosition"),
                         ("laps_completed", "laps"), ("car_location", "carLocation"), ("speed_kmh", "kmh"), ("delta_s", "delta")):
                src[f"vehicle.{f}"] = f"REALTIME_CAR_UPDATE.{n}"
            src["vehicle.current_lap"] = "DERIVED: laps_completed + 1"
            for f, lap, n in (("last_lap_time_s", c.last_lap, "lastLap"), ("best_lap_time_s", c.best_session_lap, "bestSessionLap"),
                              ("current_lap_time_s", c.current_lap, "currentLap")):
                v = _lap_s(lap)
                if v is not None:
                    kw[f] = v
                    src[f"vehicle.{f}"] = f"REALTIME_CAR_UPDATE.{n}.laptimeMs"
            if _lap_s(c.last_lap) is not None:
                kw.update(last_lap_valid=not c.last_lap.is_invalid, last_lap_is_out_lap=c.last_lap.is_out_lap,
                          last_lap_is_in_lap=c.last_lap.is_in_lap)
                src.update({"vehicle.last_lap_valid": "REALTIME_CAR_UPDATE.lastLap.isInvalid",
                            "vehicle.last_lap_is_out_lap": "REALTIME_CAR_UPDATE.lastLap.isOutLap",
                            "vehicle.last_lap_is_in_lap": "REALTIME_CAR_UPDATE.lastLap.isInLap"})
                last_splits = [None if s is None or s <= 0 else s / 1000.0 for s in c.last_lap.splits_ms]
                if any(s is not None for s in last_splits):
                    kw["last_lap_splits_s"] = last_splits
                    src["vehicle.last_lap_splits_s"] = "REALTIME_CAR_UPDATE.lastLap.splits"
            kw["current_lap_is_invalid"] = c.current_lap.is_invalid
            src["vehicle.current_lap_is_invalid"] = "REALTIME_CAR_UPDATE.currentLap.isInvalid"
            splits = [None if s is None or s <= 0 else s / 1000.0 for s in c.current_lap.splits_ms]
            if any(s is not None for s in splits):
                kw["current_lap_splits_s"] = splits
                kw["current_sector"] = sum(s is not None for s in splits) + 1
                src["vehicle.current_lap_splits_s"] = "REALTIME_CAR_UPDATE.currentLap.splits"
                src["vehicle.current_sector"] = "DERIVED: completed splits in current lap + 1"
        return VehicleInfo(**kw), src

    def telemetry_payload(self, sequence: int = 0, *, last_packet_age_s: Optional[float] = None) -> Optional[TelemetryPayload]:
        """Normalised snapshot of the target car, or None until a realtime car update for it has arrived."""
        car = self.target_car()
        if car is None or car not in self.cars:
            return None
        session, s_src = self.session_info()
        vehicle, v_src = self.vehicle_info(car)
        quality = []
        if self.track is None:
            quality.append("track_data_not_received")
        if car not in self.entries:
            quality.append("entry_list_car_not_received")
        if self.realtime is not None and self.realtime.is_replay_playing:
            quality.append("replay_playing")
        return TelemetryPayload(source="acc_udp", session_id=self.session_id, timestamp_utc=utc_now(), sequence=sequence,
                                native_message_type="REALTIME_CAR_UPDATE", session=session, vehicle=vehicle,
                                connection={"connected": True, "last_packet_age_s": last_packet_age_s},
                                field_sources={**s_src, **v_src}, data_quality=quality)

    def event_payload(self, ev: BroadcastingEvent) -> EventPayload:
        return EventPayload(source="acc_udp", timestamp_utc=utc_now(), session_id=self.session_id,
                            event_type=enum_name(BroadcastingEventType, ev.event_type), message=ev.message[:300],
                            time_ms=ev.time_ms, car_id=ev.car_index if ev.car_index >= 0 else None)


# ---------------------------------------------------------------------------
# Receiver / dashboard side
# ---------------------------------------------------------------------------

LIVE, DERIVED, UNAVAILABLE, STALE, MOCK = "LIVE - MEASURED", "DERIVED", "UNAVAILABLE", "STALE", "MOCK - NOT LIVE"

DISPLAY_CHANNELS = [
    ("session", "track_name", "Track", ""), ("session", "session_type", "Session type", ""),
    ("session", "session_phase", "Session phase", ""), ("session", "session_time_elapsed_s", "Session time elapsed", "s"),
    ("session", "session_time_remaining_s", "Session time remaining", "s"),
    ("session", "session_end_time_raw_s", "sessionEndTime (raw, meaning unverified)", "s"),
    ("session", "track_temperature_c", "Track temperature", "degC"), ("session", "ambient_temperature_c", "Air temperature", "degC"),
    ("vehicle", "car_id", "Car index", ""), ("vehicle", "position", "Position", ""), ("vehicle", "current_lap", "Current lap", ""),
    ("vehicle", "laps_completed", "Laps completed", ""), ("vehicle", "last_lap_time_s", "Last lap", "s"),
    ("vehicle", "best_lap_time_s", "Best lap (session)", "s"), ("vehicle", "current_lap_time_s", "Current lap time", "s"),
    ("vehicle", "current_sector", "Current sector", ""), ("vehicle", "car_location", "Car location", ""),
    ("vehicle", "speed_kmh", "Speed", "km/h"), ("vehicle", "driver_name", "Driver", ""),
    ("vehicle", "fuel_kg", "Fuel (not in UDP broadcast API)", "kg"),
]


def channel_table(payload: Optional[dict], *, mock: bool, stale: bool) -> list[dict]:
    """Rows for the dashboard: every channel with its value and provenance label."""
    rows = []
    sources = (payload or {}).get("field_sources", {})
    for section, key, label, unit in DISPLAY_CHANNELS:
        value = ((payload or {}).get(section) or {}).get(key) if payload else None
        src = sources.get(f"{section}.{key}", "")
        if value is None:
            tag = UNAVAILABLE
        elif mock:
            tag = MOCK
        elif stale:
            tag = STALE
        elif src.startswith("DERIVED"):
            tag = DERIVED
        else:
            tag = LIVE
        rows.append({"channel": label, "value": value, "unit": unit, "status": tag, "native_source": src or "-"})
    return rows


class LiveTelemetryFeed(Protocol):
    """Read-only interface for future simulation integration (not wired to the optimiser in this phase)."""

    def latest_telemetry(self) -> Optional[tuple[dict, float]]: ...
    def health_snapshot(self): ...
