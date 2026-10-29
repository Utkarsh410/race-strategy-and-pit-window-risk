"""Windows-bridge tests against a fake ACC endpoint on loopback (no game required).

The fake endpoint answers with manually constructed datagrams (see acc_fixtures);
this verifies the client's protocol handling, not a live ACC session.
"""

import socket
import threading
import time

import pytest

import acc_fixtures as F
from racesim.integrations.acc.bridge import AccBridge
from racesim.integrations.acc.bridge_sender import BridgeSender
from racesim.integrations.acc.client import AccBroadcastClient
from racesim.integrations.acc.config import AccBridgeConfig, AccIntegrationConfig
from racesim.integrations.acc.parser import parse_native_message
from racesim.integrations.acc.protocol import CONTROL_COMMAND_TYPES, Outbound
from racesim.integrations.acc.receiver import TelemetryReceiver
from racesim.integrations.acc.state_builder import AccStateBuilder, channel_table

SECRET = b"bridge-test-secret-01"


class FakeAcc:
    """Minimal stand-in for ACC's broadcasting listener."""

    def __init__(self, accept=True, error="Password is wrong"):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.settimeout(0.2)
        self.port = self.sock.getsockname()[1]
        self.accept, self.error = accept, error
        self.received: list[bytes] = []
        self._stop = threading.Event()
        self.t = threading.Thread(target=self.run, daemon=True)
        self.t.start()

    def run(self):
        while not self._stop.is_set():
            try:
                data, addr = self.sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            self.received.append(data)
            if data[0] == Outbound.REGISTER_COMMAND_APPLICATION:
                self.sock.sendto(F.registration_result(7, self.accept, False, "" if self.accept else self.error), addr)
                if self.accept:
                    for msg in (F.realtime_update(), F.car_update(), F.car_update(car=5, position=1),
                                F.broadcasting_event(), bytes([99, 0])):
                        self.sock.sendto(msg, addr)
            elif data[0] == Outbound.REQUEST_ENTRY_LIST:
                for msg in (F.entry_list(), F.entry_list_car()):
                    self.sock.sendto(msg, addr)
            elif data[0] == Outbound.REQUEST_TRACK_DATA:
                self.sock.sendto(F.track_data(), addr)
                self.sock.sendto(F.car_update(laps=5), addr)

    def close(self):
        self._stop.set()
        self.t.join(1)
        self.sock.close()


def run_bridge(fake, rx_port, seconds=1.5, password="asd"):
    cfg = AccBridgeConfig(native_broadcast_port=fake.port, target_host="127.0.0.1", target_port=rx_port,
                          heartbeat_interval_s=0.2, forward_min_interval_s=0.0, registration_timeout_s=1.0, retry_interval_s=0.3)
    bridge = AccBridge(cfg, connection_password=password, secret=SECRET)
    t = threading.Thread(target=bridge.run, daemon=True)
    t.start()
    time.sleep(seconds)
    t0 = time.monotonic()
    bridge.stop()
    t.join(3)
    return bridge, time.monotonic() - t0, t


def test_bridge_end_to_end_with_fake_acc():
    fake = FakeAcc()
    rx = TelemetryReceiver("127.0.0.1", 0, secret=SECRET).start()
    try:
        bridge, stop_time, t = run_bridge(fake, rx.bound_address[1])
    finally:
        fake.close()
    assert not t.is_alive() and stop_time < 2.0                      # clean shutdown
    sent_types = [d[0] for d in fake.received]
    assert sent_types[0] == Outbound.REGISTER_COMMAND_APPLICATION
    assert fake.received[0] == F.REGISTER_TEST_VECTOR.replace(b"Your name", b"Your name") or fake.received[0][1] == 4
    assert {Outbound.REQUEST_ENTRY_LIST, Outbound.REQUEST_TRACK_DATA, Outbound.UNREGISTER_COMMAND_APPLICATION} <= set(sent_types)
    assert not CONTROL_COMMAND_TYPES & set(sent_types)                # read-only
    assert fake.received[0][-2:] == b"\x00\x00"                       # empty command password
    assert bridge.client.status.unsupported_types == {99: 1}         # unknown native type ignored safely
    deadline = time.monotonic() + 2
    while rx.latest_telemetry() is None and time.monotonic() < deadline:
        time.sleep(0.02)
    payload, _ = rx.latest_telemetry()
    v, s = payload["vehicle"], payload["session"]
    assert payload["source"] == "acc_udp" and v["car_id"] == 3 and v["position"] == 2
    assert v["laps_completed"] in (4, 5) and v["current_lap"] == v["laps_completed"] + 1
    assert v["last_lap_time_s"] == pytest.approx(102.345) and s["session_type"] == "RACE" and s["track_temperature_c"] == 31
    assert payload["field_sources"]["vehicle.current_lap"].startswith("DERIVED")
    assert v["driver_name"] is None                                   # names not forwarded by default
    assert rx.latest_heartbeat()[0]["acc_status"] in ("REGISTERED", "NOT_REGISTERED")
    assert any(e["event_type"] == "LAP_COMPLETED" for e, _ in rx.events)
    rx.stop()


def test_authentication_failure_is_reported_and_retried():
    fake = FakeAcc(accept=False)
    rx = TelemetryReceiver("127.0.0.1", 0, secret=SECRET).start()
    try:
        bridge, _, _ = run_bridge(fake, rx.bound_address[1], seconds=1.2, password="pw-Zq81-not-valid")
    finally:
        fake.close()
    st = bridge.client.status
    assert st.state in ("AUTH_FAILED", "NOT_REGISTERED") and "Password is wrong" in (st.last_error or "")
    assert st.registrations >= 2                                      # retried
    deadline = time.monotonic() + 2
    while rx.latest_heartbeat() is None and time.monotonic() < deadline:
        time.sleep(0.02)
    # the bridge keeps retrying, so the latest heartbeat may show REGISTERING; AUTH_FAILED must have been reported
    assert any(h["payload"].get("acc_status") == "AUTH_FAILED" for h in rx.recent(500) if h["message_type"] == "heartbeat")
    assert rx.latest_telemetry() is None                              # nothing invented
    assert "pw-Zq81-not-valid" not in str(bridge.status_summary()) + str(bridge.heartbeat_payload())               # password never in status/logs
    rx.stop()


def test_acc_not_running_recovers_without_crash():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    dead_port = s.getsockname()[1]
    s.close()
    client = AccBroadcastClient("127.0.0.1", dead_port, connection_password="x", registration_timeout_s=0.3,
                                retry_interval_s=0.1, socket_timeout_s=0.1)
    t = threading.Thread(target=client.run, daemon=True)
    t.start()
    time.sleep(0.8)
    client.stop()
    t.join(2)
    assert not t.is_alive() and client.status.registrations >= 2 and client.status.state != "REGISTERED"


def test_network_errors_are_not_fatal():
    tx = BridgeSender("203.0.113.1", 9, secret=None, sender_id="x")   # TEST-NET-3, typically unroutable
    tx._sock.close()                                                   # force an OSError on send
    assert tx.send("heartbeat", {"x": 1}) is False and tx.errors == 1 and "send failed" in tx.last_error
    big = BridgeSender("127.0.0.1", 9, secret=None, sender_id="x", max_bytes=600)
    assert big.send("event", {"m": "y" * 2000}) is False and "too large" in big.last_error
    big.close()


def test_state_builder_normalisation_and_session_change():
    b = AccStateBuilder(include_driver_names=True)
    for raw in (F.realtime_update(), F.track_data(), F.entry_list_car(), F.car_update(current_splits=(33_000,))):
        b.update(parse_native_message(raw).message)
    p = b.telemetry_payload(1)
    assert p.vehicle.driver_name == "TES" and p.vehicle.race_number == 88 and p.session.track_name == "Test Track"
    assert p.vehicle.current_sector == 2 and p.field_sources["vehicle.current_sector"].startswith("DERIVED")
    assert p.session.session_time_remaining_s is None                  # sessionEndTime meaning unverified
    sid = b.session_id
    assert "session_changed" in b.update(parse_native_message(F.realtime_update(session=1)).message)
    assert b.session_id != sid and b.telemetry_payload(2) is None       # car state cleared on a new session
    rows = channel_table(p.model_dump(mode="json"), mock=False, stale=False)
    labels = {r["channel"]: r["status"] for r in rows}
    assert labels["Position"] == "LIVE - MEASURED" and labels["Current lap"] == "DERIVED"
    assert labels["Fuel (not in UDP broadcast API)"] == "UNAVAILABLE"
    assert {r["status"] for r in channel_table(p.model_dump(mode="json"), mock=True, stale=False)} <= {"MOCK - NOT LIVE", "UNAVAILABLE"}


def test_config_validation():
    with pytest.raises(ValueError):
        AccBridgeConfig(target_host="my-mac.local")
    cfg = AccIntegrationConfig(acc_bridge=AccBridgeConfig(target_host="127.0.0.1", target_port=9000))
    assert cfg.bridge_port_conflicts()
    with pytest.raises(ValueError):
        AccIntegrationConfig.model_validate({"health": {"stale_after_seconds": 5, "disconnect_after_seconds": 2}})
