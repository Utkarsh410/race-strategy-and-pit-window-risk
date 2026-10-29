import pytest

import acc_fixtures as F
from racesim.integrations.acc.parser import (
    BroadcastingEvent, EntryList, EntryListCar, RealtimeCarUpdate, RealtimeUpdate, RegistrationResult, TrackData,
    parse_native_message,
)
from racesim.integrations.acc.protocol import ProtocolError, UnsupportedMessageType


def test_registration_result():
    m = parse_native_message(F.registration_result(9, True, False, "")).message
    assert isinstance(m, RegistrationResult) and m.connection_id == 9 and m.success
    bad = parse_native_message(F.registration_result(0, False, False, "Password is wrong")).message
    assert not bad.success and bad.error_message == "Password is wrong"


def test_realtime_update_live_and_replay():
    m = parse_native_message(F.realtime_update()).message
    assert isinstance(m, RealtimeUpdate) and m.session_type == 10 and m.focused_car_index == 3
    assert m.session_time_ms == pytest.approx(125_000.0) and not m.is_replay_playing and m.replay_session_time_ms is None
    assert m.track_temp_raw == 31 and m.best_session_lap.lap_time_ms == 101_234
    r = parse_native_message(F.realtime_update(replay=True)).message
    assert r.is_replay_playing and r.replay_session_time_ms == pytest.approx(1000.0)


def test_car_update_and_null_lap():
    m = parse_native_message(F.car_update(last_ms=2**31 - 1)).message
    assert isinstance(m, RealtimeCarUpdate) and m.position == 2 and m.laps == 4 and m.speed_kmh == 212
    assert m.last_lap.lap_time_ms is None  # int32-max sentinel -> unknown
    assert m.last_lap.splits_ms == (33_000, 35_000, 34_345)


def test_entry_list_track_event():
    assert parse_native_message(F.entry_list(cars=(3, 5, 9))).message.car_indices == (3, 5, 9)
    car = parse_native_message(F.entry_list_car()).message
    assert isinstance(car, EntryListCar) and car.race_number == 88 and car.drivers[0].short_name == "TES"
    t = parse_native_message(F.track_data()).message
    assert isinstance(t, TrackData) and t.track_name == "Test Track" and t.track_meters == 5793
    assert t.camera_sets == {"set1": ("cam1", "cam2")} and t.hud_pages == ("Basic HUD",)
    e = parse_native_message(F.broadcasting_event()).message
    assert isinstance(e, BroadcastingEvent) and e.event_type == 5 and e.car_index == 3


def test_unknown_truncated_and_trailing():
    with pytest.raises(UnsupportedMessageType) as exc:
        parse_native_message(bytes([42, 1, 2]))
    assert exc.value.type_id == 42
    with pytest.raises(ProtocolError):
        parse_native_message(F.car_update()[:-5])
    with pytest.raises(ProtocolError):
        parse_native_message(b"")
    assert parse_native_message(F.entry_list() + b"\x00\x00").trailing_bytes == 2
