"""Session identity, stale and missing data, provenance labels and bridge restart detection."""

from live_fixtures import Script, feed

from acc_fixtures import car_update, entry_list_car, realtime_update, track_data
from racesim.integrations.acc.config import HealthConfig
from racesim.integrations.acc.live_analysis import LiveAnalysis
from racesim.integrations.acc.models import TelemetryPayload
from racesim.integrations.acc.parser import parse_native_message
from racesim.integrations.acc.race_state import Decision, Provenance, SessionIdentity, SessionTimeRemaining, label_for
from racesim.integrations.acc.state_builder import AccStateBuilder


def _row(snap, name):
    return next(r for r in snap["overview"] if r["channel"] == name)


def test_fixture_payload_matches_the_wire_schema():
    TelemetryPayload.model_validate(Script().tel(1.0, laps=1, last=120.0, tyre_compound="DRY", fuel_kg=40.0))


def test_clock_regression_with_same_id_is_a_restart():
    s, ident = Script(), SessionIdentity()
    assert ident.resolve(s.tel(300.0), 1.0) == Decision.NEW_SESSION
    b = s.tel(300.5)
    b["session"]["session_time_elapsed_s"] = 0.0
    assert ident.resolve(b, 2.0) == Decision.RESTART
    assert ident.session_key == "sess-1#r1"


def test_small_clock_step_back_is_not_a_restart():
    s, ident = Script(), SessionIdentity()
    ident.resolve(s.tel(300.0), 1.0)
    late = s.tel(301.0)
    late["session"]["session_time_elapsed_s"] = 299.0
    assert ident.resolve(late, 2.0) == Decision.SAME


def test_older_timestamp_is_out_of_order():
    s, ident = Script(), SessionIdentity()
    old, new = s.tel(10.0), s.tel(11.0)
    ident.resolve(new, 1.0)
    assert ident.resolve(old, 2.0) == Decision.OUT_OF_ORDER


def test_new_id_on_different_track_is_a_new_session_and_old_id_is_retired():
    a, b, ident = Script("a"), Script("b", track="Other"), SessionIdentity()
    ident.resolve(a.tel(100.0), 1.0)
    assert ident.resolve(b.tel(5.0), 2.0) == Decision.NEW_SESSION
    assert ident.resolve(a.tel(101.0), 3.0) == Decision.LATE_OTHER_SESSION


def test_new_id_with_running_clock_is_a_reconnection():
    a, b, ident = Script("a"), Script("b"), SessionIdentity()
    ident.resolve(a.tel(100.0, laps=3), 1000.0)
    assert ident.resolve(b.tel(120.0, laps=3), 1020.0) == Decision.RECONNECTED
    assert ident.session_key == "a"


def test_new_id_where_clock_disagrees_is_a_new_session():
    a, b, ident = Script("a"), Script("b"), SessionIdentity()
    ident.resolve(a.tel(100.0, laps=3), 1000.0)
    assert ident.resolve(b.tel(900.0, laps=3), 1020.0) == Decision.NEW_SESSION


def test_reconnection_keeps_laps_in_one_session():
    a, b, an = Script("a"), Script("b"), LiveAnalysis()
    feed(an, [(1.0, a.tel(1.0)), (120.0, a.tel(120.0, laps=1, last=119.0))])
    feed(an, [(140.0, b.tel(140.0, laps=1, last=119.0)), (240.0, b.tel(240.0, laps=2, last=118.0))])
    snap = an.snapshot(240.0)
    assert list(an.sessions) == ["a"] and an.counters["reconnections"] == 1
    assert [lap["lap_number"] for lap in snap["laps"]] == [1, 2]


def test_restart_separates_laps_between_sessions():
    s, an = Script(), LiveAnalysis()
    feed(an, [(1.0, s.tel(1.0)), (120.0, s.tel(120.0, laps=1, last=119.0))])
    s.clock_offset = 129.5
    feed(an, [(130.0, s.tel(0.5)), (250.0, s.tel(120.0, laps=1, last=118.5))])
    assert list(an.sessions) == ["sess-1", "sess-1#r1"]
    assert [lap["lap_time_s"] for lap in an.snapshot(250.0, "sess-1")["laps"]] == [119.0]
    assert [lap["lap_time_s"] for lap in an.snapshot(250.0)["laps"]] == [118.5]


def test_stale_and_disconnected_labels_from_age():
    s, an = Script(), LiveAnalysis(thresholds=HealthConfig(stale_after_seconds=3, disconnect_after_seconds=10))
    feed(an, [(100.0, s.tel(100.0, laps=1, last=119.0))])
    assert an.snapshot(101.0)["data_state"] == "LIVE"
    assert _row(an.snapshot(101.0), "Last lap")["label"] == Provenance.LIVE.value
    stale = an.snapshot(105.0)
    assert stale["data_state"] == "STALE" and stale["stale"]
    assert _row(stale, "Last lap")["label"] == Provenance.STALE.value
    assert an.snapshot(120.0)["data_state"] == "DISCONNECTED"


def test_missing_values_are_unavailable_never_zero_or_stale():
    s, an = Script(), LiveAnalysis()
    feed(an, [(1.0, s.tel(1.0))])
    for snap in (an.snapshot(1.0), an.snapshot(100.0)):
        row = _row(snap, "Last lap")
        assert row["value"] is None and row["label"] == Provenance.UNAVAILABLE.value


def test_derived_values_are_labelled_derived():
    s, an = Script(), LiveAnalysis()
    feed(an, [(1.0, s.tel(1.0))])
    assert _row(an.snapshot(1.0), "Current lap")["label"] == Provenance.DERIVED.value


def test_mock_is_never_labelled_live():
    assert label_for(1.0, "x", mock=True) == Provenance.MOCK
    s, an = Script(source="mock"), LiveAnalysis()
    an.on_message("telemetry", s.tel(1.0, laps=1, last=99.0), "new", 1.0, mock=True)
    snap = an.snapshot(1.0)
    assert snap["mock"] and _row(snap, "Last lap")["label"] == Provenance.MOCK.value


def test_time_remaining_only_in_green_phase_and_while_consistent():
    r = SessionTimeRemaining()
    assert r.update(0.0, 899.0, "PRE_FORMATION") is None  # the raw value counts down here while elapsed stays 0
    assert r.update(1.0, 899.0, "SESSION") is None         # one sample cannot show consistency
    assert r.update(2.0, 898.0, "SESSION") == 898.0
    assert r.update(3.0, 500.0, "SESSION") is None
    assert r.inconsistent
    assert SessionTimeRemaining().update(1.0, -1.0, "SESSION") is None


def test_time_remaining_is_derived_in_the_snapshot():
    s, an = Script(end_total=900.0), LiveAnalysis()
    feed(an, [(1.0, s.tel(1.0)), (2.0, s.tel(2.0))])
    row = _row(an.snapshot(2.0), "Session time remaining")
    assert row["value"] == 898.0 and row["label"] == Provenance.DERIVED.value


def test_in_game_replay_frames_are_not_analysed():
    s, an = Script(), LiveAnalysis()
    frame = s.tel(1.0, laps=5, last=100.0)
    frame["session"]["is_replay_playing"] = True
    feed(an, [(1.0, frame)])
    assert not an.snapshot(1.0)["available"] and an.counters["replay_frames_ignored"] == 1


def test_receiver_status_duplicate_and_out_of_order_are_ignored():
    s, an = Script(), LiveAnalysis()
    an.on_message("telemetry", s.tel(10.0, laps=1, last=100.0), "new", 10.0)
    an.on_message("telemetry", s.tel(11.0, laps=2, last=101.0), "duplicate", 11.0)
    an.on_message("telemetry", s.tel(12.0, laps=2, last=101.0), "out_of_order", 12.0)
    assert an.snapshot(12.0)["laps_completed"] == 1
    assert an.counters["duplicate_ignored"] == 1 and an.counters["out_of_order_ignored"] == 1


def _builder():
    b = AccStateBuilder()
    for raw in (track_data(), entry_list_car(), realtime_update(session_time_ms=300_000.0)):
        b.update(parse_native_message(raw).message)
    return b


def test_bridge_detects_restart_that_keeps_identifiers():
    b = _builder()
    first = b.session_id
    notes = b.update(parse_native_message(realtime_update(session_time_ms=0.0)).message)
    assert "session_changed" in notes and b.session_id != first


def test_bridge_keeps_session_when_clock_advances():
    b = _builder()
    first = b.session_id
    assert "session_changed" not in b.update(parse_native_message(realtime_update(session_time_ms=301_000.0)).message)
    assert b.session_id == first


def test_bridge_forwards_last_lap_sectors_and_flags():
    b = _builder()
    b.update(parse_native_message(car_update()).message)
    payload = b.telemetry_payload()
    assert payload.vehicle.last_lap_splits_s == [33.0, 35.0, 34.345]
    assert payload.vehicle.last_lap_is_out_lap is False and payload.vehicle.last_lap_is_in_lap is False
    assert payload.vehicle.fuel_kg is None and payload.vehicle.tyre_compound is None
