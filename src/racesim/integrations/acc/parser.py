"""Decode native ACC broadcasting messages into typed records.

Field order follows :mod:`racesim.integrations.acc.protocol` (see its
verification notes). Times on the wire are milliseconds; the records keep the
native units and the state builder converts to seconds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

from racesim.integrations.acc.protocol import INT32_NULL, Inbound, ProtocolError, Reader, UnsupportedMessageType

MAX_SPLITS = 16  # sanity bound; ACC uses 3 sectors


def _opt_ms(v: int) -> Optional[int]:
    """int32 lap/split time with the int32-max 'no value' sentinel mapped to None."""
    return None if v == INT32_NULL else v


@dataclass(frozen=True)
class LapInfo:
    lap_time_ms: Optional[int]
    car_index: int
    driver_index: int
    splits_ms: tuple[Optional[int], ...]
    is_invalid: bool
    is_valid_for_best: bool
    is_out_lap: bool
    is_in_lap: bool


@dataclass(frozen=True)
class RegistrationResult:
    connection_id: int
    success: bool
    second_flag: bool  # 'read_only' (acbc) vs 'AllowCommands' (broawp): semantics unverified, kept raw
    error_message: str
    native_type: str = "REGISTRATION_RESULT"


@dataclass(frozen=True)
class RealtimeUpdate:
    event_index: int
    session_index: int
    session_type: int
    session_phase: int
    session_time_ms: float
    session_end_time_ms: float
    focused_car_index: int
    active_camera_set: str
    active_camera: str
    current_hud_page: str
    is_replay_playing: bool
    replay_session_time_ms: Optional[float]
    replay_remaining_time_ms: Optional[float]
    time_of_day_ms: float
    ambient_temp_raw: int
    track_temp_raw: int
    clouds_raw: int
    rain_level_raw: int
    wetness_raw: int
    best_session_lap: LapInfo
    native_type: str = "REALTIME_UPDATE"


@dataclass(frozen=True)
class RealtimeCarUpdate:
    car_index: int
    driver_index: int
    driver_count: int
    gear_raw: int  # encoding disagrees between sources; not normalised
    world_pos_x: float
    world_pos_y: float
    yaw: float
    car_location: int
    speed_kmh: int
    position: int
    cup_position: int
    track_position: int
    spline_position: float
    laps: int
    delta_ms: int
    best_session_lap: LapInfo
    last_lap: LapInfo
    current_lap: LapInfo
    native_type: str = "REALTIME_CAR_UPDATE"


@dataclass(frozen=True)
class EntryList:
    connection_id: int
    car_indices: tuple[int, ...]
    native_type: str = "ENTRY_LIST"


@dataclass(frozen=True)
class DriverInfo:
    first_name: str
    last_name: str
    short_name: str
    category: int
    nationality: int


@dataclass(frozen=True)
class EntryListCar:
    car_index: int
    car_model_type: int
    team_name: str
    race_number: int
    cup_category: int
    current_driver_index: int
    nationality: int
    drivers: tuple[DriverInfo, ...]
    native_type: str = "ENTRY_LIST_CAR"


@dataclass(frozen=True)
class TrackData:
    connection_id: int
    track_name: str
    track_id: int
    track_meters: int
    camera_sets: dict[str, tuple[str, ...]] = field(default_factory=dict)
    hud_pages: tuple[str, ...] = ()
    native_type: str = "TRACK_DATA"


@dataclass(frozen=True)
class BroadcastingEvent:
    event_type: int
    message: str
    time_ms: int
    car_index: int
    native_type: str = "BROADCASTING_EVENT"


NativeMessage = Union[RegistrationResult, RealtimeUpdate, RealtimeCarUpdate, EntryList, EntryListCar, TrackData, BroadcastingEvent]


def read_lap(r: Reader) -> LapInfo:
    lap_time = _opt_ms(r.i32())
    car_index = r.u16()
    driver_index = r.u16()
    n = r.u8()
    if n > MAX_SPLITS:
        raise ProtocolError(f"implausible split count {n}")
    splits = tuple(_opt_ms(r.i32()) for _ in range(n))
    return LapInfo(lap_time, car_index, driver_index, splits, r.boolean(), r.boolean(), r.boolean(), r.boolean())


def _registration(r: Reader) -> RegistrationResult:
    return RegistrationResult(r.u32(), r.boolean(), r.boolean(), r.string())


def _realtime(r: Reader) -> RealtimeUpdate:
    event_index, session_index = r.u16(), r.u16()
    session_type, phase = r.u8(), r.u8()
    session_time, session_end = r.f32(), r.f32()
    focused = r.u32()
    cam_set, cam, hud = r.string(), r.string(), r.string()
    replay = r.boolean()
    replay_time = replay_remaining = None
    if replay:
        replay_time, replay_remaining = r.f32(), r.f32()
    time_of_day = r.f32()
    ambient, track, clouds, rain, wet = r.u8(), r.u8(), r.u8(), r.u8(), r.u8()
    return RealtimeUpdate(event_index, session_index, session_type, phase, session_time, session_end, focused, cam_set, cam, hud,
                          replay, replay_time, replay_remaining, time_of_day, ambient, track, clouds, rain, wet, read_lap(r))


def _car(r: Reader) -> RealtimeCarUpdate:
    car_index, driver_index, driver_count = r.u16(), r.u16(), r.u8()
    gear = r.u8()
    x, y, yaw = r.f32(), r.f32(), r.f32()
    location = r.u8()
    speed, pos, cup_pos, track_pos = r.u16(), r.u16(), r.u16(), r.u16()
    spline = r.f32()
    laps = r.u16()
    delta = r.i32()
    return RealtimeCarUpdate(car_index, driver_index, driver_count, gear, x, y, yaw, location, speed, pos, cup_pos, track_pos,
                             spline, laps, delta, read_lap(r), read_lap(r), read_lap(r))


def _entry_list(r: Reader) -> EntryList:
    conn = r.u32()
    n = r.u16()
    return EntryList(conn, tuple(r.u16() for _ in range(n)))


def _entry_list_car(r: Reader) -> EntryListCar:
    car_index, model = r.u16(), r.u8()
    team = r.string()
    number = r.i32()
    cup, current_driver, nationality = r.u8(), r.u8(), r.u16()
    n = r.u8()
    drivers = tuple(DriverInfo(r.string(), r.string(), r.string(), r.u8(), r.u16()) for _ in range(n))
    return EntryListCar(car_index, model, team, number, cup, current_driver, nationality, drivers)


def _track(r: Reader) -> TrackData:
    conn = r.i32()
    name = r.string()
    track_id, meters = r.i32(), r.i32()
    sets: dict[str, tuple[str, ...]] = {}
    for _ in range(r.u8()):
        set_name = r.string()
        sets[set_name] = tuple(r.string() for _ in range(r.u8()))
    pages = tuple(r.string() for _ in range(r.u8()))
    return TrackData(conn, name, track_id, meters, sets, pages)


def _event(r: Reader) -> BroadcastingEvent:
    return BroadcastingEvent(r.u8(), r.string(), r.i32(), r.i32())


_DECODERS = {
    Inbound.REGISTRATION_RESULT: _registration,
    Inbound.REALTIME_UPDATE: _realtime,
    Inbound.REALTIME_CAR_UPDATE: _car,
    Inbound.ENTRY_LIST: _entry_list,
    Inbound.TRACK_DATA: _track,
    Inbound.ENTRY_LIST_CAR: _entry_list_car,
    Inbound.BROADCASTING_EVENT: _event,
}


@dataclass(frozen=True)
class ParseResult:
    message: NativeMessage
    trailing_bytes: int


def parse_native_message(data: bytes) -> ParseResult:
    """Decode one datagram. Raises :class:`ProtocolError` (or :class:`UnsupportedMessageType`).

    Trailing bytes are tolerated and counted (a newer game version may append
    fields); truncated messages are rejected.
    """
    if not data:
        raise ProtocolError("empty datagram")
    r = Reader(data)
    type_id = r.u8()
    try:
        decoder = _DECODERS[Inbound(type_id)]
    except ValueError:
        raise UnsupportedMessageType(type_id) from None
    msg = decoder(r)
    return ParseResult(msg, r.remaining)
