"""Normalised RaceSim telemetry schema (version 1.0) and the forwarding envelope.

Every field is optional unless it is always known. ``None`` means *unknown or
not supplied*; ``0`` is a real value. Units are in the field names (``_s``,
``_m``, ``_kmh``, ``_c``). ``field_sources`` records, for every populated field,
the native ACC message it came from or the rule used to derive it.

Native source of each field (UDP broadcasting API)
--------------------------------------------------
=====================================  ==============================================
session.track_name, track_length_m     TRACK_DATA.trackName / trackMeters
session.session_type / session_phase   REALTIME_UPDATE.sessionType / sessionPhase
session.session_time_elapsed_s         REALTIME_UPDATE.sessionTime (ms / 1000)
session.session_end_time_raw_s         REALTIME_UPDATE.sessionEndTime (ms / 1000)
session.session_time_remaining_s       not populated by the bridge; the receiver derives it (race_state.SessionTimeRemaining)
session.ambient/track_temperature_c    REALTIME_UPDATE.ambientTemp / trackTemp (integer degC)
session.*_raw weather bytes            REALTIME_UPDATE.clouds / rainLevel / wetness (raw bytes)
vehicle.car_id                         REALTIME_CAR_UPDATE.carIndex (focused car)
vehicle.race_number, driver_name       ENTRY_LIST_CAR (driver name only if enabled)
vehicle.position / cup / track pos.    REALTIME_CAR_UPDATE.position / cupPosition / trackPosition
vehicle.laps_completed                 REALTIME_CAR_UPDATE.laps
vehicle.current_lap                    DERIVED: laps_completed + 1
vehicle.last/best/current_lap_time_s   REALTIME_CAR_UPDATE.lastLap / bestSessionLap / currentLap
vehicle.current_sector                 DERIVED: completed splits in currentLap + 1 (only if splits present)
vehicle.last_lap_splits_s              REALTIME_CAR_UPDATE.lastLap.splits (ms / 1000)
vehicle.last_lap_is_out_lap / _in_lap  REALTIME_CAR_UPDATE.lastLap.isOutLap / isInLap
vehicle.current_lap_is_invalid         REALTIME_CAR_UPDATE.currentLap.isInvalid
vehicle.car_location, speed_kmh,       REALTIME_CAR_UPDATE.carLocation / kmh / splinePosition / delta
  spline_position, delta_s
=====================================  ==============================================

The broadcasting API does not provide physics channels (tyre compound, age,
temperatures, pressures, wear, fuel, throttle). Those fields exist only as
reserved optional slots for a future Windows shared-memory reader and are
always ``None`` from ``acc_udp``.

Fields added after the first release are optional, so logs and bridges without
them stay valid. A receiver older than the bridge rejects them (``extra="forbid"``):
update the Mac first, or both together.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION = "1.0"
SUPPORTED_SCHEMA_VERSIONS = frozenset({"1.0"})
MAX_DATAGRAM_BYTES = 8192
MAX_FUTURE_SKEW_S = 300.0  # clocks on two PCs differ; reject only absurd timestamps


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _check_utc(v: datetime) -> datetime:
    if v.tzinfo is None or v.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware (UTC)")
    v = v.astimezone(timezone.utc)
    if v > utc_now() + timedelta(seconds=MAX_FUTURE_SKEW_S):
        raise ValueError("timestamp is implausibly far in the future")
    if v.year < 2020:
        raise ValueError("timestamp is implausibly old")
    return v


class SessionInfo(_Model):
    track_name: Optional[str] = Field(default=None, max_length=128)
    track_length_m: Optional[int] = Field(default=None, ge=0, le=100_000)
    session_type: Optional[str] = Field(default=None, max_length=32)
    session_phase: Optional[str] = Field(default=None, max_length=32)
    session_index: Optional[int] = Field(default=None, ge=0)
    session_time_elapsed_s: Optional[float] = Field(default=None, ge=-1.0, le=7 * 24 * 3600)
    session_time_remaining_s: Optional[float] = Field(default=None, ge=0, le=7 * 24 * 3600)
    session_end_time_raw_s: Optional[float] = Field(default=None, le=7 * 24 * 3600)
    time_of_day_s: Optional[float] = Field(default=None, ge=0, le=2 * 86400)
    is_replay_playing: Optional[bool] = None
    ambient_temperature_c: Optional[int] = Field(default=None, ge=-60, le=255)
    track_temperature_c: Optional[int] = Field(default=None, ge=-60, le=255)
    clouds_raw: Optional[int] = Field(default=None, ge=0, le=255)
    rain_level_raw: Optional[int] = Field(default=None, ge=0, le=255)
    wetness_raw: Optional[int] = Field(default=None, ge=0, le=255)


class VehicleInfo(_Model):
    car_id: Optional[int] = Field(default=None, ge=0, le=65535)
    race_number: Optional[int] = None
    car_model_id: Optional[int] = Field(default=None, ge=0, le=255)
    driver_name: Optional[str] = Field(default=None, max_length=64)
    position: Optional[int] = Field(default=None, ge=0, le=500)
    cup_position: Optional[int] = Field(default=None, ge=0, le=500)
    track_position: Optional[int] = Field(default=None, ge=0, le=500)
    laps_completed: Optional[int] = Field(default=None, ge=0, le=10_000)
    current_lap: Optional[int] = Field(default=None, ge=1, le=10_001)
    last_lap_time_s: Optional[float] = Field(default=None, gt=0, le=3600)
    best_lap_time_s: Optional[float] = Field(default=None, gt=0, le=3600)
    current_lap_time_s: Optional[float] = Field(default=None, ge=0, le=3600)
    last_lap_valid: Optional[bool] = None
    current_sector: Optional[int] = Field(default=None, ge=1, le=16)
    current_lap_splits_s: Optional[list[Optional[float]]] = None
    last_lap_splits_s: Optional[list[Optional[float]]] = None
    last_lap_is_out_lap: Optional[bool] = None
    last_lap_is_in_lap: Optional[bool] = None
    current_lap_is_invalid: Optional[bool] = None
    car_location: Optional[str] = Field(default=None, max_length=32)
    speed_kmh: Optional[int] = Field(default=None, ge=0, le=600)
    spline_position: Optional[float] = Field(default=None, ge=0, le=1.0)
    delta_s: Optional[float] = Field(default=None, ge=-3600, le=3600)
    is_focused_car: Optional[bool] = None
    # Reserved for a future shared-memory source; never populated by acc_udp.
    fuel_kg: Optional[float] = Field(default=None, ge=0, le=500)
    tyre_wear: Optional[list[float]] = None
    tyre_compound: Optional[str] = Field(default=None, max_length=32)
    tyre_age_laps: Optional[int] = Field(default=None, ge=0, le=10_000)
    tyre_pressures_psi: Optional[list[float]] = None      # FL, FR, RL, RR
    tyre_temperatures_c: Optional[list[float]] = None     # FL, FR, RL, RR


class ConnectionInfo(_Model):
    connected: bool = True
    last_packet_age_s: Optional[float] = Field(default=None, ge=0)


class TelemetryPayload(_Model):
    schema_version: str = SCHEMA_VERSION
    source: Literal["acc_udp", "mock"]
    session_id: str = Field(min_length=1, max_length=64)
    timestamp_utc: datetime
    sequence: int = Field(ge=0)
    native_message_type: str = Field(max_length=32)
    session: SessionInfo = Field(default_factory=SessionInfo)
    vehicle: VehicleInfo = Field(default_factory=VehicleInfo)
    connection: ConnectionInfo = Field(default_factory=ConnectionInfo)
    field_sources: dict[str, str] = Field(default_factory=dict)
    data_quality: list[str] = Field(default_factory=list)

    @field_validator("timestamp_utc")
    @classmethod
    def _utc(cls, v: datetime) -> datetime:
        return _check_utc(v)

    @field_validator("schema_version")
    @classmethod
    def _version(cls, v: str) -> str:
        if v not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(f"unsupported schema version {v!r}")
        return v


AccStatus = Literal["NOT_REGISTERED", "REGISTERING", "REGISTERED", "NO_DATA", "AUTH_FAILED", "ERROR", "MOCK"]


class HeartbeatPayload(_Model):
    """Bridge self-report. The receiver uses it to separate bridge health from ACC health."""

    source: Literal["acc_udp", "mock"]
    timestamp_utc: datetime
    acc_status: AccStatus
    acc_last_packet_age_s: Optional[float] = Field(default=None, ge=0)
    native_messages_received: int = Field(ge=0)
    native_messages_invalid: int = Field(default=0, ge=0)
    messages_forwarded: int = Field(ge=0)
    session_id: Optional[str] = Field(default=None, max_length=64)
    track_name: Optional[str] = Field(default=None, max_length=128)
    session_type: Optional[str] = Field(default=None, max_length=32)
    last_error: Optional[str] = Field(default=None, max_length=300)
    uptime_s: float = Field(ge=0)

    @field_validator("timestamp_utc")
    @classmethod
    def _utc(cls, v: datetime) -> datetime:
        return _check_utc(v)


class EventPayload(_Model):
    source: Literal["acc_udp", "mock"]
    timestamp_utc: datetime
    session_id: str = Field(min_length=1, max_length=64)
    event_type: str = Field(max_length=32)
    message: str = Field(default="", max_length=300)
    time_ms: Optional[int] = None
    car_id: Optional[int] = Field(default=None, ge=0)

    @field_validator("timestamp_utc")
    @classmethod
    def _utc(cls, v: datetime) -> datetime:
        return _check_utc(v)


PHYSICS_SCHEMA_VERSION = "1.0"
ReaderStatus = Literal["NOT_AVAILABLE", "WAITING_FOR_ACC", "LIVE", "PAUSED", "REPLAY", "ERROR", "MOCK"]
ChannelValue = Union[None, bool, int, float, str, list[Optional[float]]]


class PhysicsPayload(_Model):
    """Shared-memory sample from the Windows reader (message type ``physics``).

    Added compatibly: broadcast-only bridges never send it, and receivers that
    predate it reject it as an unknown message type without affecting
    telemetry, heartbeats or events. Channel names, units and their
    verification status are defined in :mod:`racesim.integrations.acc.physics_channels`;
    values are in native ACC units. ``available`` / ``missing`` list which
    channels the reader actually obtained in this sample.
    """

    physics_schema_version: str = PHYSICS_SCHEMA_VERSION
    source: Literal["acc_shared_memory", "mock"]
    timestamp_utc: datetime
    sample_sequence: int = Field(ge=0)
    physics_packet_id: Optional[int] = None
    graphics_packet_id: Optional[int] = None
    reader_status: ReaderStatus
    session_id: Optional[str] = Field(default=None, max_length=64)
    channels: dict[str, ChannelValue] = Field(default_factory=dict)
    available: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    samples_read: int = Field(default=0, ge=0)
    read_errors: int = Field(default=0, ge=0)
    sample_age_s: Optional[float] = Field(default=None, ge=0, le=3600)
    last_error: Optional[str] = Field(default=None, max_length=300)

    @field_validator("timestamp_utc")
    @classmethod
    def _utc(cls, v: datetime) -> datetime:
        return _check_utc(v)

    @field_validator("physics_schema_version")
    @classmethod
    def _pversion(cls, v: str) -> str:
        if v != PHYSICS_SCHEMA_VERSION:
            raise ValueError(f"unsupported physics schema version {v!r}")
        return v

    @field_validator("channels")
    @classmethod
    def _channels(cls, v: dict) -> dict:
        import math

        from racesim.integrations.acc.physics_channels import BY_NAME

        for name, value in v.items():
            ch = BY_NAME.get(name)
            if ch is None:
                raise ValueError(f"unknown physics channel {name!r}")
            if value is None:
                continue
            if ch.kind in ("wheels", "vector"):
                n = 4 if ch.kind == "wheels" else 3
                if not isinstance(value, list) or len(value) != n:
                    raise ValueError(f"{name} must be a list of {n} values")
                if any(x is not None and (isinstance(x, bool) or not math.isfinite(x)) for x in value):
                    raise ValueError(f"{name} contains a non-finite value")
            elif ch.kind in ("text", "enum"):
                if not isinstance(value, str) or len(value) > 64:
                    raise ValueError(f"{name} must be a short string")
            elif ch.kind == "bool":
                if not isinstance(value, bool):
                    raise ValueError(f"{name} must be a boolean")
            elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        return v


PAYLOAD_MODELS: dict[str, type[_Model]] = {"telemetry": TelemetryPayload, "heartbeat": HeartbeatPayload, "event": EventPayload,
                                           "physics": PhysicsPayload}


class Envelope(_Model):
    """Forwarding envelope sent from the Windows bridge (or mock sender) to the Mac receiver.

    ``mac`` is HMAC-SHA256 (hex) over the canonical JSON of every other field.
    ``sender_run_id`` changes each time a sender starts, so a sequence reset after
    a restart is not mistaken for out-of-order delivery.
    """

    message_type: Literal["telemetry", "heartbeat", "event", "physics"]
    schema_version: str
    sequence: int = Field(ge=0)
    sent_at_utc: datetime
    sender_id: str = Field(min_length=1, max_length=64)
    sender_run_id: str = Field(min_length=1, max_length=64)
    mock: bool
    payload: dict[str, Any]
    mac: Optional[str] = Field(default=None, min_length=64, max_length=64)

    @field_validator("sent_at_utc")
    @classmethod
    def _utc(cls, v: datetime) -> datetime:
        return _check_utc(v)

    @field_validator("schema_version")
    @classmethod
    def _version(cls, v: str) -> str:
        if v not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(f"unsupported schema version {v!r}")
        return v

    def typed_payload(self):
        return PAYLOAD_MODELS[self.message_type].model_validate(self.payload)
