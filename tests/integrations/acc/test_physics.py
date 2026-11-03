"""Shared-memory physics: channels, reader, forwarding, receiver health, dashboard rows.

All tests use the MOCK source (attribute names from pyaccsharedmemory 1.0.0) or
a fake pyaccsharedmemory module; none needs ACC or Windows. They do not prove
the real shared-memory integration works - that needs a live ACC session.
"""

import json
import math
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import pytest
from pydantic import ValidationError

from racesim.integrations.acc import shared_memory as sm
from racesim.integrations.acc.bridge_sender import BridgeSender
from racesim.integrations.acc.config import HealthConfig, PhysicsConfig
from racesim.integrations.acc.health import ConnectionHealth, HealthState as S
from racesim.integrations.acc.mock import MockSharedMemorySource
from racesim.integrations.acc.models import Envelope, PhysicsPayload, utc_now
from racesim.integrations.acc.physics_channels import BY_NAME, CHANNELS, physics_table
from racesim.integrations.acc.receiver import TelemetryReceiver
from racesim.integrations.acc.shared_memory import PhysicsForwarder, PhysicsReader, extract_channels

ROOT = Path(__file__).resolve().parents[3]


def test_channel_table_is_consistent():
    assert len(BY_NAME) == len(CHANNELS)
    assert {c.unit_status for c in CHANNELS} <= {"kunos", "conflict", "unstated", "library"}
    assert BY_NAME["speed_kmh"].unit == "km/h" and BY_NAME["speed_kmh"].unit_status == "kunos"
    assert BY_NAME["fuel_remaining"].unit_status == "conflict"           # Kunos kg vs README litres
    assert BY_NAME["tyre_pressure"].unit_status == "unstated"            # psi is only expected
    assert BY_NAME["wheel_angular_speed"].unit == "rad/s"
    assert not any("player" in c.attr for c in CHANNELS)                  # names are never read


def test_extract_channels_and_missing_detection():
    snap = MockSharedMemorySource(drop=("brake_temp", "fuel_per_lap", "track")).read()
    e = extract_channels(snap)
    assert {"brake_temp", "fuel_per_lap_l", "track"} <= set(e.missing)
    assert "brake_temp" not in e.channels and len(e.available) + len(e.missing) == len(CHANNELS)
    assert e.channels["tyre_compound"] == "dry_compound"                  # NULs stripped
    assert e.channels["status"] == "ACC_LIVE" and e.reader_status == "LIVE"
    assert len(e.channels["tyre_pressure"]) == 4 and len(e.channels["velocity"]) == 3
    snap.Physics.rpm = float("nan")
    assert "rpm" in extract_channels(snap).missing                        # non-finite is unavailable, not 0
    snap.Physics.speed_kmh = 0.0
    assert extract_channels(snap).channels["speed_kmh"] == 0.0            # 0 is a value


@pytest.mark.parametrize("status,expected", [("ACC_OFF", "WAITING_FOR_ACC"), ("ACC_PAUSE", "PAUSED"), ("ACC_REPLAY", "REPLAY")])
def test_status_mapping(status, expected):
    assert extract_channels(MockSharedMemorySource(status=status).read()).reader_status == expected


def test_reader_dedup_stale_and_errors():
    r = PhysicsReader(lambda: MockSharedMemorySource(freeze_packet=True), stale_after_s=1.0)
    assert r.poll_once(now=0.0) and not r.poll_once(now=0.1) and not r.poll_once(now=0.2)   # same packet id
    assert r.state.new_samples == 1 and r.state.status == "MOCK"
    r.poll_once(now=2.0)
    assert r.state.status == "WAITING_FOR_ACC"                            # packet id not advancing
    e = PhysicsReader(lambda: MockSharedMemorySource(raise_every=2))
    results = [e.poll_once() for _ in range(6)]
    assert e.state.read_errors == 3 and results.count(True) == 3 and "ValueError" in e.state.last_error


def test_reader_unavailable_off_windows(monkeypatch):
    monkeypatch.setattr(sm.sys, "platform", "darwin")
    r = PhysicsReader(sm.PyAccSharedMemorySource)
    assert not r.poll_once() and r.state.status == "NOT_AVAILABLE" and "Windows" in r.state.last_error


def test_library_dedup_is_bypassed(monkeypatch):
    """Fake pyaccsharedmemory reproducing the verified behaviour: None when suspension_travel is unchanged."""
    class FakeAsm:
        def __init__(self):
            self.physics_old, self.closed = None, False
            self.src = MockSharedMemorySource()

        def read_shared_memory(self):
            snap = self.src.read()
            snap.Physics.suspension_travel = types.SimpleNamespace(front_left=0.03, front_right=0.03, rear_left=0.04, rear_right=0.04)
            if self.physics_old is not None and self.physics_old.suspension_travel == snap.Physics.suspension_travel:
                return None
            self.physics_old = snap.Physics
            return snap

        def close(self):
            self.closed = True

    monkeypatch.setattr(sm.sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "pyaccsharedmemory", types.SimpleNamespace(accSharedMemory=FakeAsm))
    src = sm.PyAccSharedMemorySource()
    assert all(src.read() is not None for _ in range(5))                  # stationary car still produces samples
    src.close()
    assert src._asm.closed


def test_payload_validation():
    now = utc_now()
    ok = PhysicsPayload(source="acc_shared_memory", timestamp_utc=now, sample_sequence=1, reader_status="LIVE",
                        channels={"speed_kmh": 0.0, "tyre_pressure": [27.5, None, 27.4, 27.6], "tyre_compound": "dry_compound"})
    assert ok.channels["speed_kmh"] == 0.0
    bad = [{"made_up_channel": 1.0}, {"tyre_pressure": [1.0, 2.0]}, {"rpm": float("nan")}, {"rpm": True}, {"pit_limiter_on": 1},
           {"velocity": [1.0, float("inf"), 0.0]}]
    for ch in bad:
        with pytest.raises(ValidationError):
            PhysicsPayload(source="acc_shared_memory", timestamp_utc=now, sample_sequence=1, reader_status="LIVE", channels=ch)
    with pytest.raises(ValidationError):
        PhysicsPayload(physics_schema_version="9", source="mock", timestamp_utc=now, sample_sequence=1, reader_status="MOCK")
    env = Envelope(message_type="physics", schema_version="1.0", sequence=1, sent_at_utc=now, sender_id="w", sender_run_id="r",
                   mock=False, payload=ok.model_dump(mode="json"))
    assert env.typed_payload().reader_status == "LIVE"


def test_forwarder_sends_samples_then_status():
    sent = []

    class Rec:
        def send(self, t, p):
            sent.append((t, p))
            return True

    r = PhysicsReader(lambda: MockSharedMemorySource())
    f = PhysicsForwarder(r, Rec(), status_interval_s=1.0)
    assert f.tick(now=0.0) == "status"                                    # nothing read yet -> reader status
    r.poll_once()
    assert f.tick(now=0.1) == "sample" and f.tick(now=0.2) is None       # each sample sent once
    assert f.tick(now=1.2) == "status" and not sent[-1][1]["channels"]
    assert all(t == "physics" for t, _ in sent) and sent[1][1]["available"]


def test_health_physics_component():
    h = ConnectionHealth(HealthConfig(stale_after_seconds=1, disconnect_after_seconds=3))
    h.mark_listening()
    h.record_valid("heartbeat", 0.0, acc_status="REGISTERED", acc_packet_age_s=0.1)
    h.record_valid("telemetry", 0.0)
    snap = h.snapshot(0.2)
    assert snap.physics == S.NOT_STARTED and snap.overall == S.CONNECTED   # broadcast-only unaffected
    h.record_valid("physics", 0.3, physics_reader_status="LIVE", physics_has_sample=True)
    assert h.snapshot(0.5).physics == S.CONNECTED
    assert h.snapshot(2.0).physics == S.STALE and h.snapshot(5.0).physics == S.DISCONNECTED
    h.record_valid("physics", 6.0, physics_reader_status="WAITING_FOR_ACC")
    assert h.snapshot(6.1).physics == S.WAITING_FOR_DATA
    h.record_valid("physics", 7.0, physics_reader_status="NOT_AVAILABLE")
    assert h.snapshot(7.1).physics == S.ERROR


def test_dashboard_rows_labels():
    p = PhysicsReader(lambda: MockSharedMemorySource(drop=("water_temp",)))
    p.poll_once()
    payload = p.build_payload().model_dump(mode="json")
    rows = {r["channel"]: r for r in physics_table(payload, mock=False, stale=False)}
    assert rows["speed_kmh"]["status"] == "LIVE - MEASURED" and rows["water_temp"]["status"] == "UNAVAILABLE"
    assert rows["speed_ms"]["status"] == "DERIVED" and rows["speed_ms"]["value"] == pytest.approx(rows["speed_kmh"]["value"] / 3.6)
    assert rows["tyre_core_temp_spread"]["status"] == "DERIVED"
    assert {r["status"] for r in physics_table(payload, mock=True, stale=False)} <= {"MOCK - NOT LIVE", "UNAVAILABLE"}
    assert rows["speed_kmh"]["status"] != "STALE"
    assert physics_table(payload, mock=False, stale=True)[0]["status"] == "STALE"


def test_receiver_stores_physics_over_udp():
    rx = TelemetryReceiver("127.0.0.1", 0, secret=b"physics-test-secret").start()
    tx = BridgeSender("127.0.0.1", rx.bound_address[1], secret=b"physics-test-secret", sender_id="w", mock=True)
    r = PhysicsReader(lambda: MockSharedMemorySource())
    f = PhysicsForwarder(r, tx)
    r.poll_once()
    f.tick()
    end = time.monotonic() + 3
    while rx.latest_physics() is None and time.monotonic() < end:
        time.sleep(0.02)
    payload, age, mock = rx.latest_physics()
    assert mock and payload["source"] == "mock" and payload["channels"]["rpm"] > 0
    assert rx.health_snapshot().physics == S.CONNECTED
    assert rx.latest_telemetry() is None                                  # physics never masquerades as telemetry
    rx.stop()
    tx.close()


def test_bridge_with_physics_and_shutdown():
    from test_bridge import SECRET, FakeAcc
    from racesim.integrations.acc.bridge import AccBridge
    from racesim.integrations.acc.config import AccBridgeConfig

    fake = FakeAcc()
    rx = TelemetryReceiver("127.0.0.1", 0, secret=SECRET).start()
    cfg = AccBridgeConfig(native_broadcast_port=fake.port, target_host="127.0.0.1", target_port=rx.bound_address[1],
                          heartbeat_interval_s=0.2, forward_min_interval_s=0.0, registration_timeout_s=1.0)
    bridge = AccBridge(cfg, connection_password="asd", secret=SECRET, physics=PhysicsConfig(enabled=True, sample_hz=50, forward_hz=10),
                       # stands in for the real source on loopback; a mock-labelled source in a live bridge is rejected
                       physics_source_factory=lambda: MockSharedMemorySource(source_name="acc_shared_memory"))
    t = threading.Thread(target=bridge.run, daemon=True)
    t.start()
    time.sleep(1.2)
    bridge.stop()
    t.join(3)
    fake.close()
    assert not t.is_alive() and not bridge.physics_reader.running              # clean shutdown of all threads
    assert bridge.physics_forwarder.sent_samples > 3 and "physics" in bridge.status_summary()
    kinds = {h["message_type"] for h in rx.recent(500)}
    assert {"telemetry", "heartbeat", "physics"} <= kinds
    seqs = [h["sequence"] for h in rx.recent(500)]
    assert len(seqs) == len(set(seqs))                                         # one shared, thread-safe counter
    rx.stop()


def test_mock_physics_in_live_bridge_is_rejected():
    """A MOCK physics source inside a live (non-mock) bridge envelope must never be accepted as live."""
    rx = TelemetryReceiver(secret=None, require_auth=False)
    tx = BridgeSender("127.0.0.1", 9, secret=None, sender_id="w", mock=False)
    r = PhysicsReader(lambda: MockSharedMemorySource())
    r.poll_once()
    env = tx.build("physics", r.build_payload().model_dump(mode="json"))
    tx.close()
    assert rx.handle_datagram(json.dumps(env).encode()) == "rejected" and rx.latest_physics() is None


def test_broadcast_only_bridge_sends_no_physics():
    from racesim.integrations.acc.bridge import AccBridge
    from racesim.integrations.acc.config import AccBridgeConfig

    b = AccBridge(AccBridgeConfig(target_host="127.0.0.1"), connection_password="x", secret=None)
    assert b.physics_reader is None and "physics" not in b.status_summary()
    b.sender.close()


def test_config_validation():
    with pytest.raises(ValidationError):
        PhysicsConfig(sample_hz=5, forward_hz=10)
    with pytest.raises(ValidationError):
        PhysicsConfig(channels=["speed_kmh", "warp_drive"])
    assert PhysicsConfig().enabled is False


def test_diagnostic_mock_json():
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "acc_shared_memory_diagnostic.py"), "--mock", "--once", "--json"],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    data = json.loads(out.stdout)
    assert data["source"] == "mock" and data["reader_status"] == "MOCK" and data["channels"]["speed_kmh"] > 0
