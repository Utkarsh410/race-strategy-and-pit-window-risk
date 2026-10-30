"""Persistence, exports, replay, dashboard data preparation and receiver integration."""

import csv
import threading
from datetime import datetime
from pathlib import Path

import pytest
from live_fixtures import Script, drive, feed

from racesim.integrations.acc.live_analysis import LiveAnalysis, replay_session_log, write_session_exports
from racesim.integrations.acc.models import Envelope, utc_now
from racesim.integrations.acc.race_state import Provenance as P
from racesim.integrations.acc.receiver import TelemetryReceiver
from racesim.integrations.acc.session_logger import SessionLogger, flatten_record, read_session_log

from dashboard import acc_performance_data as data


def _analysis(sink=None):
    s, an = Script(), LiveAnalysis(sink=sink)
    feed(an, [(0.5, s.tel(0.5))] + drive(s, 0.5, 0, [100.0, 101.0, 102.0, 103.0]))
    return s, an, 404.5


def test_derived_records_are_logged(tmp_path):
    with SessionLogger(tmp_path) as log:
        path = log.path
        _, an, _ = _analysis(log)
        an.finish()
    records, bad = read_session_log(path)
    kinds = [r["record_type"] for r in records]
    assert bad == 0
    assert kinds.count("session_metadata") == 1 and kinds.count("completed_lap") == 4
    assert "stint_summary" in kinds and "data_quality" in kinds
    assert next(r for r in records if r["record_type"] == "completed_lap")["data"]["classification"] == "CLEAN_CANDIDATE"


def test_derived_records_do_not_break_the_message_flattener(tmp_path):
    with SessionLogger(tmp_path) as log:
        path = log.path
        log.log_record("completed_lap", {"lap_number": 1})
    records, _ = read_session_log(path)
    assert [flatten_record(r) for r in records] == [None, None, None]


def test_strategy_snapshot_is_logged_and_kept(tmp_path):
    with SessionLogger(tmp_path) as log:
        path = log.path
        _, an, now = _analysis(log)
        an.record_strategy({"status": "SCENARIO_ONLY", "session_key": "sess-1"})
    assert an.snapshot(now)["strategy_snapshots"][0]["status"] == "SCENARIO_ONLY"
    assert any(r["record_type"] == "strategy_snapshot" for r in read_session_log(path)[0])


def test_csv_exports_contain_laps_and_stints(tmp_path):
    _, an, now = _analysis()
    out = write_session_exports(an, tmp_path / "run", now=now)
    laps = list(csv.DictReader(open(out["laps"])))
    assert [r["lap_number"] for r in laps] == ["1", "2", "3", "4"] and laps[0]["session_key"] == "sess-1"
    assert list(csv.DictReader(open(out["stints"])))[0]["laps"] == "4"
    assert "ACC session summary" in out["summary"].read_text() and "Laps" in out["summary"].read_text()


def test_exports_never_overwrite_earlier_sessions(tmp_path):
    _, an, now = _analysis()
    first = write_session_exports(an, tmp_path / "run", now=now)
    before = first["laps"].read_text()
    second = write_session_exports(an, tmp_path / "run", now=now)
    assert second["laps"] != first["laps"] and second["summary"] != first["summary"]
    assert first["laps"].read_text() == before


def test_session_logs_are_separate_files(tmp_path):
    with SessionLogger(tmp_path) as a, SessionLogger(tmp_path) as b:
        assert a.path != b.path


def test_events_never_count_laps_and_repeats_are_ignored():
    s, an, now = _analysis()
    for _ in range(50):
        an.on_message("event", s.event(time_ms=1234), "new", now)
    an.on_message("event", s.event(time_ms=9, car_id=7), "new", now)
    snap = an.snapshot(now)
    assert len(snap["laps"]) == 4 and snap["events"] == {"LAP_COMPLETED": 1}
    assert an.counters["duplicate_events_ignored"] == 49


def _write_log(tmp_path, frames, mock=False):
    with SessionLogger(tmp_path) as log:
        path = log.path
        for i, (t, payload) in enumerate(frames):
            env = {"message_type": "telemetry", "schema_version": "1.0", "sequence": i, "sender_id": "x", "sender_run_id": "y", "mock": mock,
                   "sent_at_utc": utc_now().isoformat(), "payload": payload}
            log.log(env, received_at_utc=datetime.fromisoformat(payload["timestamp_utc"]))
    return path


def test_replay_matches_live_and_is_labelled_replay(tmp_path):
    s = Script()
    frames = [(0.5, s.tel(0.5))] + drive(s, 0.5, 0, [100.0, 101.0])
    live = LiveAnalysis()
    feed(live, frames)
    replayed, last = replay_session_log(_write_log(tmp_path, frames))
    a, b = live.snapshot(frames[-1][0]), replayed.snapshot(last)
    assert [(x["lap_number"], x["lap_time_s"], x["classification"]) for x in a["laps"]] == [(x["lap_number"], x["lap_time_s"], x["classification"]) for x in b["laps"]]
    assert b["replay"] and b["data_state"] == "REPLAY"
    assert next(r for r in b["overview"] if r["channel"] == "Last lap")["label"] == P.REPLAY.value


def test_replay_of_a_cut_log_uses_what_is_readable(tmp_path):
    s = Script()
    path = _write_log(tmp_path, [(0.5, s.tel(0.5))] + drive(s, 0.5, 0, [100.0]))
    with open(path, "a") as fh:
        fh.write('{"record_type":"message","envel')
    analysis, last = replay_session_log(path)
    assert len(analysis.snapshot(last)["laps"]) == 1


def test_snapshot_is_detached_from_live_state():
    _, an, now = _analysis()
    snap = an.snapshot(now)
    snap["laps"].clear()
    snap["flags"]["x"] = 1
    assert len(an.snapshot(now)["laps"]) == 4 and "x" not in an.snapshot(now)["flags"]


def test_concurrent_reads_during_ingest():
    s, an = Script(), LiveAnalysis()
    frames = [(0.5, s.tel(0.5))] + drive(s, 0.5, 0, [100.0] * 20)
    errors, stop = [], threading.Event()
    def reader():
        while not stop.is_set():
            try:
                an.snapshot(0.0)
            except Exception as exc:  # noqa: BLE001 - any error here is the failure under test
                errors.append(exc)
    t = threading.Thread(target=reader)
    t.start()
    feed(an, frames)
    stop.set()
    t.join()
    assert not errors and len(an.snapshot(0.0)["laps"]) == 20


def test_receiver_feeds_listener_and_survives_a_failing_one():
    s, an = Script(), LiveAnalysis()
    rx = TelemetryReceiver(secret=None, require_auth=False)
    def boom(*_):
        raise RuntimeError("listener failure")
    rx.add_listener(boom)
    rx.add_listener(an.on_message)
    payload = s.tel(0.5)
    payload["timestamp_utc"] = utc_now().isoformat()
    env = Envelope(message_type="telemetry", schema_version="1.0", sequence=1, sent_at_utc=utc_now(), sender_id="x", sender_run_id="y", mock=False, payload=payload)
    assert rx.handle_datagram(env.model_dump_json().encode()) == "accepted"
    assert rx.listener_errors == 1 and an.snapshot()["available"]


# -- dashboard data preparation ---------------------------------------------------

def test_empty_snapshot_prepares_empty_tables():
    snap = LiveAnalysis().snapshot(0.0)
    assert data.overview_table(snap).empty and data.recent_laps_table(snap).empty and data.stint_table(snap).empty
    assert set(data.fuel_table(snap)["label"]) == {P.UNAVAILABLE.value}
    assert set(data.tyre_table(snap)["label"]) == {P.UNAVAILABLE.value}


def test_pace_table_shows_sample_sizes_and_insufficient_rows():
    _, an, now = _analysis()
    table = data.pace_table(an.snapshot(now)).set_index("statistic")
    assert table.loc["Median lap time", "valid_laps_used"] == 4
    assert table.loc["Rolling pace, last 3 valid laps", "value_s"] == (101.0 + 102.0 + 103.0) / 3
    row = table.loc["Rolling pace, last 5 valid laps"]
    assert row["value_s"] is None or row["value_s"] != row["value_s"]
    assert "4 of 5" in row["label"] and P.UNAVAILABLE.value in row["label"]


def test_recent_laps_newest_first_and_limited():
    _, an, now = _analysis()
    assert list(data.recent_laps_table(an.snapshot(now), n=2)["lap_number"]) == [4, 3]


def test_stint_table_combines_pace_and_history():
    _, an, now = _analysis()
    row = data.stint_table(an.snapshot(now)).iloc[0]
    assert row["laps"] == 4 and row["clean_laps"] == 4 and row["ended"] == "in progress" and row["tyre_compound"] == P.UNAVAILABLE.value


def test_fuel_and_tyre_labels_separate_measured_from_derived():
    s, an = Script(), LiveAnalysis()
    feed(an, [(0.5, s.tel(0.5, fuel_kg=50.0, tyre_compound="DRY"))])
    snap = an.snapshot(0.5)
    fuel = data.fuel_table(snap).set_index("channel")["label"]
    assert fuel["Latest fuel"] == P.LIVE.value and fuel["Consumption"] == P.UNAVAILABLE.value
    assert data.tyre_table(snap).set_index("channel")["label"]["Compound"] == P.LIVE.value


def test_stale_snapshot_labels_measurements_stale():
    s, an = Script(), LiveAnalysis()
    feed(an, [(0.5, s.tel(0.5, fuel_kg=50.0))])
    assert data.fuel_table(an.snapshot(50.0)).set_index("channel")["label"]["Latest fuel"] == P.STALE.value


def test_strategy_inputs_table_is_arrow_safe_text():
    from racesim.integrations.acc.live_strategy import build_inputs

    table = data.strategy_inputs_table(build_inputs(LiveAnalysis().snapshot(0.0)).rows())
    assert all(isinstance(v, str) for v in table["value"])


def _page():
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    return AppTest.from_file(str(Path(__file__).resolve().parents[3] / "pages" / "2_ACC_Performance.py"), default_timeout=120)


@pytest.fixture
def manager():
    """The process-wide receiver manager, emptied so other tests' receivers do not leak in."""
    from dashboard.acc_live_view import receiver_manager

    receiver_manager.clear()
    yield receiver_manager()
    receiver_manager.clear()


def test_page_renders_without_a_receiver(manager):
    at = _page().run()
    assert not at.exception and not at.error, [e.value for e in at.error]
    assert any("Receiver not started" in i.value for i in at.info)


def test_page_renders_every_panel_and_strategy_runs_only_on_click(manager, monkeypatch):
    import racesim.integrations.acc.live_strategy as ls

    s, an = Script(session_type="RACE"), LiveAnalysis()
    frames = [(0.5, s.tel(0.5))] + drive(s, 0.5, 0, [100.0] * 6)
    feed(an, frames)
    manager.analysis = an
    calls = []
    real = ls.run_live_strategy
    monkeypatch.setattr(ls, "run_live_strategy", lambda *a, **k: calls.append(1) or real(*a, **k))
    at = _page().run()
    assert not at.exception, [e.value for e in at.exception]
    assert [x.label for x in at.tabs] == ["Session overview", "Lap performance", "Stint analysis", "Fuel and tyres"]
    assert not calls
    run = next(b for b in at.button if b.label == "Run strategy analysis")
    assert run.disabled  # stale, and tyres and fuel are unavailable
    assert any("BLOCKED" in m.value for m in at.markdown)


def test_receiver_manager_starts_one_receiver_and_writes_exports(manager, tmp_path):
    from racesim.integrations.acc.config import AccIntegrationConfig

    cfg = AccIntegrationConfig()
    manager.start("127.0.0.1", 0, secret=None, log_dir=tmp_path, cfg=cfg)
    first = manager.receiver
    manager.start("127.0.0.1", 0, secret=None, log_dir=tmp_path, cfg=cfg)
    assert manager.receiver is first and first.running and manager.analysis is not None
    manager.stop()
    assert not first.running and set(manager.exports) == {"laps", "stints", "summary"}
    assert all(p.exists() for p in manager.exports.values())
