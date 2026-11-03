"""Physics schema 1.1: target names, range validation, contact/damage channels, combined
health (DEGRADED), receiver session handling, physics CSV export, reader recovery.

Everything runs against the MOCK source or a fake shared-memory provider; none of it
needs ACC or Windows, and none of it shows that the real shared-memory path works.
"""

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from racesim.integrations.acc.bridge_sender import BridgeSender
from racesim.integrations.acc.config import HealthConfig, PhysicsConfig
from racesim.integrations.acc.health import ConnectionHealth, HealthState as S
from racesim.integrations.acc.mock import MockSharedMemorySource, MockTelemetryGenerator
from racesim.integrations.acc.models import PhysicsPayload, utc_now
from racesim.integrations.acc.physics_channels import (BY_NAME, CHANNELS, GROUPS, LEGACY_NAMES, RESERVED_NAMES,
                                                       TYRE_WHEEL_CHANNELS, physics_table, quality_summary, wheel_rows)
from racesim.integrations.acc.receiver import TelemetryReceiver
from racesim.integrations.acc.session_logger import (PHYSICS_CSV_CHANNELS, SessionLogger, export_physics_csv,
                                                     export_session_csv)
from racesim.integrations.acc.shared_memory import PhysicsReader, extract_channels

ROOT = Path(__file__).resolve().parents[3]
TARGET_NAMES = ("speed_kmh", "rpm", "gear", "throttle", "brake", "steering_input", "clutch", "velocity", "acceleration",
                "angular_velocity", "wheel_slip", "slip_ratio", "slip_angle", "wheel_pressure", "tyre_core_temperature",
                "suspension_travel", "brake_pressure", "brake_temperature", "abs_activity", "tc_activity")


# -- channel definitions ---------------------------------------------------------------
def test_target_names_and_semantics():
    assert set(TARGET_NAMES) <= set(BY_NAME)
    assert "fuel_l" not in BY_NAME and "fuel_l" in RESERVED_NAMES      # unit not verified: not named in litres
    assert set(LEGACY_NAMES.values()) <= set(BY_NAME) and not set(LEGACY_NAMES) & set(BY_NAME)
    assert "not degrees" in BY_NAME["steering_input"].unit and BY_NAME["steering_input"].valid_range == (-1, 1)
    assert BY_NAME["throttle"].valid_range == (0, 1) and "raw" in BY_NAME["gear"].unit
    assert {c.group for c in CHANNELS} <= set(GROUPS)
    assert all(c.unit_status == "unstated" for c in CHANNELS if c.kind in ("vector", "wheel_vectors"))


def test_contact_and_damage_channels():
    e = extract_channels(MockSharedMemorySource().read())
    assert len(e.channels["tyre_contact_point"]) == 4 and len(e.channels["tyre_contact_point"][0]) == 3
    assert e.channels["car_damage"] == [0.0] * 5 and e.channels["is_engine_running"] is True
    snap = MockSharedMemorySource().read()
    snap.Physics.tyre_contact_point.rear_right.z = float("nan")        # partial vector -> that wheel unavailable
    assert extract_channels(snap).channels["tyre_contact_point"][3] is None


def test_out_of_range_values_are_invalid_not_measured():
    snap = MockSharedMemorySource().read()
    snap.Physics.gas, snap.Physics.speed_kmh = 1.5, 0.0
    snap.Physics.wheel_pressure.front_left = -3.0
    e = extract_channels(snap)
    assert "throttle" not in e.channels and "out of range" in e.invalid["throttle"]
    assert "wheel_pressure" in e.invalid and "wheel_pressure" not in e.available
    assert e.channels["speed_kmh"] == 0.0                               # zero is a reading, not "unavailable"
    snap.Physics.gas = 1.0000001                                        # float noise at the bound is accepted
    assert extract_channels(snap).channels["throttle"] == pytest.approx(1.0)


def test_payload_moves_range_errors_but_rejects_protocol_errors():
    now = utc_now()
    p = PhysicsPayload(source="acc_shared_memory", timestamp_utc=now, sample_sequence=1, reader_status="LIVE",
                       channels={"rpm": 7000, "brake": 3.0}, available=["rpm", "brake"])
    assert p.channels == {"rpm": 7000} and "brake" in p.invalid and p.available == ["rpm"]
    for bad in ({"tyre_contact_point": [[0, 0]] * 4}, {"car_damage": [0, 0]}, {"steering": 0.1}):
        with pytest.raises(ValidationError):                            # wrong shape / unknown name in 1.1
            PhysicsPayload(source="acc_shared_memory", timestamp_utc=now, sample_sequence=1, reader_status="LIVE", channels=bad)


def test_schema_1_0_payload_is_renamed():
    p = PhysicsPayload.model_validate({
        "physics_schema_version": "1.0", "source": "acc_shared_memory", "timestamp_utc": utc_now().isoformat(),
        "sample_sequence": 3, "reader_status": "LIVE", "channels": {"steering": 0.2, "tyre_pressure": [27.0] * 4, "rpm": 6000},
        "available": ["steering", "tyre_pressure", "rpm"], "missing": ["brake_temp"]})
    assert p.channels["steering_input"] == 0.2 and p.channels["wheel_pressure"] == [27.0] * 4
    assert p.available == ["steering_input", "wheel_pressure", "rpm"] and p.missing == ["brake_temperature"]
    assert p.physics_schema_version == "1.0"                            # the version received is kept truthfully
    assert PhysicsConfig(channels=["tyre_pressure"]).channels == ["wheel_pressure"]


# -- health ---------------------------------------------------------------------------
def _session_fresh(h, t):
    h.record_valid("heartbeat", t, acc_status="REGISTERED", acc_packet_age_s=0.0)
    h.record_valid("telemetry", t)


def _health(**kw):
    h = ConnectionHealth(HealthConfig(stale_after_seconds=1, disconnect_after_seconds=3, **kw))
    h.mark_listening()
    return h


def test_session_without_physics_keeps_broadcast_rule():
    h = _health()
    _session_fresh(h, 0.0)
    s = h.snapshot(0.1)
    assert s.overall == S.CONNECTED and s.physics == S.NOT_STARTED and not s.physics_expected


def test_stale_physics_degrades_overall():
    h = _health()
    _session_fresh(h, 0.0)
    h.record_valid("physics", 0.0, physics_reader_status="LIVE", physics_has_sample=True)
    assert h.snapshot(0.2).overall == S.CONNECTED
    _session_fresh(h, 2.0)
    h.record_valid("physics", 2.0, physics_reader_status="WAITING_FOR_ACC")   # status messages still arrive
    s = h.snapshot(2.1)
    assert s.physics == S.WAITING_FOR_DATA and s.overall == S.DEGRADED     # never "fully healthy"
    _session_fresh(h, 6.0)
    s = h.snapshot(6.1)                                                     # physics messages stopped entirely
    assert s.physics == S.DISCONNECTED and s.overall == S.DEGRADED


def test_physics_freshness_uses_sample_time_not_receive_time():
    h = _health()
    _session_fresh(h, 10.0)
    h.record_valid("physics", 10.0, physics_reader_status="LIVE", physics_has_sample=True, physics_sample_age_s=2.5)
    s = h.snapshot(10.0)
    assert s.physics == S.STALE and s.last_physics_sample_age_s == pytest.approx(2.5) and s.overall == S.DEGRADED


def test_require_physics_and_physics_without_session():
    h = _health(require_physics=True)
    _session_fresh(h, 0.0)
    assert h.snapshot(0.1).overall == S.DEGRADED                            # expected from the start
    h2 = _health()
    h2.record_valid("physics", 0.0, physics_reader_status="LIVE", physics_has_sample=True)
    s = h2.snapshot(0.1)
    assert s.physics == S.CONNECTED and s.overall not in (S.CONNECTED, S.DEGRADED)


# -- receiver -------------------------------------------------------------------------
class Link:
    """Builds signed-free envelopes the way the bridge does (mock flag consistent with the MOCK sources)."""

    def __init__(self, rx, run="run1"):
        self.rx, self.tx = rx, BridgeSender("127.0.0.1", 9, secret=None, sender_id="win", mock=True)
        self.tx.run_id = run
        self.gen = MockTelemetryGenerator(seed=3, session_id="sess-A")

    def send(self, kind, payload):
        return self.rx.handle_datagram(json.dumps(self.tx.build(kind, payload)).encode())

    def telemetry(self):
        return self.send("telemetry", self.gen.telemetry(self.tx.sequence + 1).model_dump(mode="json"))

    def physics(self, reader, session_id="sess-A", include_sample=True, **over):
        p = reader.build_payload(session_id=session_id, include_sample=include_sample).model_dump(mode="json")
        p.update(over)
        return self.send("physics", p)


def _reader():
    r = PhysicsReader(lambda: MockSharedMemorySource())
    r.poll_once()
    return r


def test_receiver_quality_flags_and_session_change(tmp_path):
    rx = TelemetryReceiver(secret=None, require_auth=False, session_logger=SessionLogger(tmp_path))
    rx.session_logger.open()
    link, r = Link(rx), _reader()
    assert link.physics(r) == "accepted"                                    # physics before any session data
    link.telemetry()
    link.physics(r, session_id="sess-OLD")
    rx.stop()
    quality = [json.loads(line).get("quality") for line in rx.session_logger.path.read_text().splitlines()[1:-1]]
    assert ["physics_without_session"] in quality and ["physics_session_mismatch"] in quality
    rx2 = TelemetryReceiver(secret=None, require_auth=False)
    link2 = Link(rx2)
    link2.telemetry()
    link2.physics(r)
    assert rx2.latest_physics() is not None
    link2.gen.session_id = "sess-B"                                         # new ACC session starts
    link2.telemetry()
    assert rx2.latest_physics() is None                                     # old session's sample is dropped


def test_receiver_keeps_newest_sample_and_clears_on_bridge_restart():
    rx = TelemetryReceiver(secret=None, require_auth=False)
    link, r = Link(rx), _reader()
    r.poll_once()
    link.physics(r, sample_sequence=5)
    link.physics(r, sample_sequence=4)                                      # reader sample older than the one held
    assert rx.latest_physics()[0]["sample_sequence"] == 5 and rx.physics_samples_dropped == 1
    restarted = Link(rx, run="run2")
    assert restarted.physics(_reader(), include_sample=False) == "accepted"   # first status from the new run
    assert rx.latest_physics() is None and rx.latest_physics_status()[0]["reader_status"] == "MOCK"


def test_udp_only_messages_still_accepted():
    rx = TelemetryReceiver(secret=None, require_auth=False)
    link = Link(rx)
    assert link.telemetry() == "accepted" and rx.latest_physics() is None
    assert rx.health_snapshot().physics == S.NOT_STARTED


# -- logging and export -----------------------------------------------------------------
def test_physics_csv_export(tmp_path):
    rx = TelemetryReceiver(secret=None, require_auth=False, session_logger=SessionLogger(tmp_path))
    rx.session_logger.open()
    link = Link(rx)
    link.telemetry()
    r = PhysicsReader(lambda: MockSharedMemorySource(drop=("brake_temp",)))
    for _ in range(3):
        r.poll_once()
        link.physics(r)
    link.physics(r, include_sample=False)                                   # status-only: not a CSV row
    rx.stop()
    log = rx.session_logger.path
    out, rows = export_physics_csv(log, tmp_path / "phys.csv")
    assert rows == 3
    data = list(csv.DictReader(out.open()))
    assert data[0]["session_id"] == "sess-A" and data[0]["sampled_at_utc"] and data[0]["received_at_utc"]
    assert data[0]["sampled_at_utc"] != data[0]["received_at_utc"]          # Windows sample time kept apart
    assert [d["sample_sequence"] for d in data] == ["1", "2", "3"]
    assert data[0]["brake_temperature_fl"] == "" and "brake_temperature" in data[0]["missing"]  # empty, never 0
    assert float(data[0]["wheel_pressure_fl"]) > 0 and data[0]["brake"] != ""
    with pytest.raises(FileExistsError):
        export_physics_csv(log, tmp_path / "phys.csv")                      # never overwrites
    export_session_csv(log, tmp_path / "all.csv")                           # telemetry CSV unaffected
    assert any(row["message_type"] == "physics" for row in csv.DictReader((tmp_path / "all.csv").open()))
    _, n = export_physics_csv(log, tmp_path / "contact.csv", channels=("tyre_contact_point", "car_damage"))
    header = next(csv.reader((tmp_path / "contact.csv").open()))
    assert n == 3 and "tyre_contact_point_rr_z" in header and "car_damage_center" in header


def test_physics_csv_reads_schema_1_0_logs(tmp_path):
    rec = {"record_type": "message", "received_at_utc": "2026-01-01T00:00:01+00:00", "quality": [],
           "envelope": {"message_type": "physics", "sequence": 1, "sender_run_id": "r", "mock": False,
                        "payload": {"physics_schema_version": "1.0", "timestamp_utc": "2026-01-01T00:00:00+00:00",
                                    "sample_sequence": 1, "reader_status": "LIVE", "source": "acc_shared_memory",
                                    "channels": {"tyre_pressure": [27.1, 27.2, 27.3, 27.4], "gear_raw": 3}}}}
    log = tmp_path / "old.jsonl"
    log.write_text("not json\n" + json.dumps(rec) + "\n")
    out, rows = export_physics_csv(log, tmp_path / "old.csv")
    row = next(csv.DictReader(out.open()))
    assert rows == 1 and row["wheel_pressure_rr"] == "27.4" and row["gear"] == "3"


def test_default_csv_channels_cover_targets():
    assert set(TARGET_NAMES) <= set(PHYSICS_CSV_CHANNELS)


# -- dashboard helpers ----------------------------------------------------------------
def test_grouped_rows_and_quality_summary():
    r = PhysicsReader(lambda: MockSharedMemorySource(drop=("water_temp",)))
    r.poll_once()
    payload = r.build_payload().model_dump(mode="json")
    rows = wheel_rows(payload["channels"], TYRE_WHEEL_CHANNELS)
    assert [x["wheel"] for x in rows] == ["FL", "FR", "RL", "RR"] and rows[0]["wheel_pressure [psi (expected)]"] > 0
    q = quality_summary(payload)
    assert "water_temp" in q["missing"] and "fuel_remaining" in q["unverified_units"] and q["defined"] == len(CHANNELS)
    table = {x["channel"]: x for x in physics_table(payload, mock=False, stale=False, sample_age_s=0.4)}
    assert table["rpm"]["source"] == "shared memory physics" and table["rpm"]["age_s"] == 0.4
    assert table["rpm"]["measured_at_utc"] == payload["timestamp_utc"] and table["water_temp"]["measured_at_utc"] is None
    payload["invalid"] = {"rpm": "out of range"}
    payload["channels"].pop("rpm")
    assert {x["channel"]: x for x in physics_table(payload, mock=False, stale=False)}["rpm"]["status"] == "INVALID"


# -- reader recovery ----------------------------------------------------------------------
def test_reader_reopens_after_repeated_failures():
    opened = []

    class Broken(MockSharedMemorySource):
        def read(self):
            raise OSError("mapping gone")

    def factory():
        opened.append(1)
        return Broken() if len(opened) == 1 else MockSharedMemorySource()

    r = PhysicsReader(factory, reopen_after_errors=3)
    assert not any(r.poll_once() for _ in range(3))
    assert r.state.reopens == 1 and r.state.status == "ERROR" and "mapping gone" in r.state.last_error
    assert r.poll_once() and len(opened) == 2 and r.state.status == "MOCK"


def test_diagnostic_brief_mock():
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "acc_shared_memory_diagnostic.py"), "--mock", "--count", "1"],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    for label in ("MOCK - NOT ACC", "Speed", "RPM", "Gear", "Throttle", "Brake", "Fuel", "Tyre pressure", "Tyre core temp"):
        assert label in out.stdout
