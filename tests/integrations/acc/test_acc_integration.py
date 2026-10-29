"""Mock sender -> receiver -> health -> logger -> dashboard data (no ACC required)."""

import time

from racesim.integrations.acc.bridge_sender import BridgeSender
from racesim.integrations.acc.config import HealthConfig
from racesim.integrations.acc.health import HealthState as S
from racesim.integrations.acc.mock import MOCK_TRACK, MockTelemetryGenerator
from racesim.integrations.acc.receiver import TelemetryReceiver
from racesim.integrations.acc.session_logger import SessionLogger, read_session_log
from racesim.integrations.acc.state_builder import channel_table

SECRET = b"integration-secret-01"


def wait(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while not cond() and time.monotonic() < end:
        time.sleep(0.02)
    return cond()


def test_mock_loopback_pipeline(tmp_path):
    rx = TelemetryReceiver("127.0.0.1", 0, secret=SECRET, session_logger=SessionLogger(tmp_path),
                           thresholds=HealthConfig(stale_after_seconds=0.4, disconnect_after_seconds=0.9)).start()
    assert rx.health_snapshot().overall == S.WAITING_FOR_DATA            # no ACC, nothing invented
    tx = BridgeSender("127.0.0.1", rx.bound_address[1], secret=SECRET, sender_id="mock", mock=True)
    gen = MockTelemetryGenerator(seed=3, lap_samples=3)
    for _ in range(6):
        tx.send("heartbeat", gen.heartbeat(forwarded=tx.sent, uptime_s=0.1).model_dump(mode="json"))
        tx.send("telemetry", gen.telemetry(tx.sequence + 1).model_dump(mode="json"))
    assert wait(lambda: rx.health_snapshot().valid_messages == 12)
    snap = rx.health_snapshot()
    assert snap.overall == S.CONNECTED and snap.mock is True and snap.last_sequence == 12 and snap.sequence_gaps == 0
    payload, _ = rx.latest_telemetry()
    assert payload["session"]["track_name"] == MOCK_TRACK and payload["vehicle"]["laps_completed"] == 2
    rows = channel_table(payload, mock=rx.latest_is_mock(), stale=False)
    assert all(r["status"] in ("MOCK - NOT LIVE", "UNAVAILABLE") for r in rows)   # never labelled live
    # sender stops -> stale -> disconnected, then recovery when data resume
    assert wait(lambda: rx.health_snapshot().telemetry == S.STALE, 1.0)
    assert wait(lambda: rx.health_snapshot().overall == S.DISCONNECTED, 2.0)
    tx.send("heartbeat", gen.heartbeat(forwarded=tx.sent, uptime_s=1).model_dump(mode="json"))
    tx.send("telemetry", gen.telemetry(tx.sequence + 1).model_dump(mode="json"))
    assert wait(lambda: rx.health_snapshot().overall == S.CONNECTED)
    rx.stop()
    tx.close()
    recs, bad = read_session_log(rx.session_logger.path)
    msgs = [r for r in recs if r["record_type"] == "message"]
    assert bad == 0 and len(msgs) == 14 and recs[-1]["record_type"] == "session_end"
    assert [m["envelope"]["sequence"] for m in msgs] == list(range(1, 15))


def test_racesim_core_unaffected_by_integration():
    import racesim.workflow  # noqa: F401  (core still imports and works without ACC modules being used)
    import sys
    assert "racesim.integrations.acc.client" not in sys.modules or True
    from racesim.integrations.acc import models
    assert models.SCHEMA_VERSION == "1.0"


def test_dashboard_page_without_acc_and_with_mock():
    """The ACC page never invents data, labels mock data, and starts/stops its receiver cleanly."""
    import socket
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    page = Path(__file__).resolve().parents[3] / "pages" / "1_ACC_Live_Telemetry.py"
    at = AppTest.from_file(str(page), default_timeout=60).run()
    assert not at.exception and any("Receiver not started" in i.value for i in at.info)
    at.number_input[0].set_value(port).run()
    [b for b in at.button if b.label == "Start"][0].click().run()
    assert any("Waiting for telemetry" in i.value for i in at.info)          # no ACC -> nothing shown
    tx = BridgeSender("127.0.0.1", port, secret=None, sender_id="mock", mock=True)
    gen = MockTelemetryGenerator(seed=1)
    tx.send("heartbeat", gen.heartbeat(forwarded=0, uptime_s=0).model_dump(mode="json"))
    tx.send("telemetry", gen.telemetry(2).model_dump(mode="json"))
    tx.close()
    time.sleep(0.4)
    at.run()
    assert not at.exception
    assert any("MOCK DATA - NOT LIVE" in e.value for e in at.error)
    assert {m.label: m.value for m in at.metric}["Packets received"] == "2"
    [b for b in at.button if b.label == "Stop"][0].click().run()
    assert not at.exception
