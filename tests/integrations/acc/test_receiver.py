import json
import socket
import time

import pytest

from racesim.integrations.acc import auth
from racesim.integrations.acc.bridge_sender import BridgeSender
from racesim.integrations.acc.config import HealthConfig
from racesim.integrations.acc.health import HealthState as S
from racesim.integrations.acc.mock import MockTelemetryGenerator
from racesim.integrations.acc.receiver import InvalidMessage, ReceiverBindError, TelemetryReceiver, validate_datagram

SECRET = b"unit-test-secret-0001"


def signed(env: dict, secret=SECRET) -> bytes:
    env = dict(env)
    env["mac"] = auth.sign(env, secret)
    return json.dumps(env).encode()


def mock_envelope(seq=1, run="r1"):
    tx = BridgeSender("127.0.0.1", 9, secret=None, sender_id="t", mock=True)
    tx.run_id, tx.sequence = run, seq - 1
    env = tx.build("telemetry", MockTelemetryGenerator(seed=1).telemetry(seq).model_dump(mode="json"))
    tx.close()
    return env


def test_validate_good_and_bad():
    env, payload = validate_datagram(signed(mock_envelope()), secret=SECRET, require_auth=True)
    assert env.mock and payload.source == "mock"
    cases = {
        b"\xff\xfe": "malformed_json", b"{not json": "malformed_json", b"[1,2]": "malformed_json",
        b"x" * 9000: "oversized",
        signed(mock_envelope(), b"wrong-secret-0000000"): "authentication_failed",
        json.dumps(mock_envelope()).encode(): "authentication_failed",   # unsigned
        signed({**mock_envelope(), "schema_version": "9.9"}): "unsupported_schema_version",
        signed({**mock_envelope(), "message_type": "nonsense"}): "schema_invalid",
        signed({**mock_envelope(), "mock": False}): "schema_invalid",    # mock payload must not claim to be live
    }
    for data, reason in cases.items():
        with pytest.raises(InvalidMessage) as exc:
            validate_datagram(data, secret=SECRET, require_auth=True)
        assert exc.value.reason == reason, (data[:40], exc.value.reason)


def test_sequence_duplicates_gaps_and_staleness():
    rx = TelemetryReceiver(secret=SECRET, thresholds=HealthConfig(stale_after_seconds=1, disconnect_after_seconds=2))
    rx.health.mark_listening()
    results = [rx.handle_datagram(signed(mock_envelope(s)), now=10.0) for s in (1, 2, 2, 5, 4)]
    assert results == ["accepted", "accepted", "duplicate", "accepted", "out_of_order"]
    snap = rx.health_snapshot(now=10.5)
    assert snap.duplicate_packets == 1 and snap.out_of_order_packets == 1 and snap.sequence_gaps == 1
    assert rx.latest_telemetry()[0]["sequence"] == 5  # a late packet never replaces newer telemetry
    assert rx.health_snapshot(now=11.5).telemetry == S.STALE
    assert rx.health_snapshot(now=13.0).telemetry == S.DISCONNECTED
    assert rx.handle_datagram(b"garbage", now=13.0) == "rejected"
    assert rx.health_snapshot(now=13.0).invalid_packets == 1


def test_real_socket_roundtrip_and_shutdown():
    rx = TelemetryReceiver("127.0.0.1", 0, secret=SECRET).start()
    port = rx.bound_address[1]
    tx = BridgeSender("127.0.0.1", port, secret=SECRET, sender_id="w", mock=True)
    gen = MockTelemetryGenerator(seed=0)
    for _ in range(5):
        tx.send("telemetry", gen.telemetry(tx.sequence + 1).model_dump(mode="json"))
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.sendto(b"{malformed", ("127.0.0.1", port))
    s.close()
    deadline = time.monotonic() + 3
    while rx.health_snapshot().valid_messages < 5 and time.monotonic() < deadline:
        time.sleep(0.02)
    snap = rx.health_snapshot()
    assert snap.valid_messages == 5 and snap.last_sequence == 5
    while rx.health_snapshot().invalid_packets < 1 and time.monotonic() < deadline:
        time.sleep(0.02)
    assert rx.running and rx.health_snapshot().invalid_packets == 1  # malformed packet did not stop the receiver
    t0 = time.monotonic()
    rx.stop()
    assert not rx.running and time.monotonic() - t0 < 2.0
    assert rx.health_snapshot().receiver == S.DISCONNECTED
    tx.close()


def test_port_binding_error_and_auth_requirement():
    a = TelemetryReceiver("127.0.0.1", 0, secret=SECRET).start()
    try:
        with pytest.raises(ReceiverBindError):
            TelemetryReceiver("127.0.0.1", a.bound_address[1], secret=SECRET).start()
    finally:
        a.stop()
    with pytest.raises(ValueError):
        TelemetryReceiver(secret=None, require_auth=True)


def test_sender_filter():
    rx = TelemetryReceiver(secret=SECRET, allowed_sender_ips=["192.0.2.5"])
    assert rx.handle_datagram(signed(mock_envelope()), addr=("192.0.2.99", 1)) == "rejected"
    assert rx.handle_datagram(signed(mock_envelope()), addr=("192.0.2.5", 1)) == "accepted"
