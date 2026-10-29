from racesim.integrations.acc.config import HealthConfig
from racesim.integrations.acc.health import ConnectionHealth, HealthState as S


def test_states_progress_with_time():
    h = ConnectionHealth(HealthConfig(stale_after_seconds=3, disconnect_after_seconds=10))
    assert h.snapshot(0).overall == S.NOT_STARTED
    h.mark_listening()
    assert h.snapshot(0).overall == S.WAITING_FOR_DATA
    h.record_valid("heartbeat", 100.0, acc_status="REGISTERED", acc_packet_age_s=0.1)
    h.record_valid("telemetry", 100.0, session_id="s")
    snap = h.snapshot(101.0)
    assert snap.overall == S.CONNECTED and snap.session_id == "s"
    assert h.snapshot(105.0).overall == S.STALE
    assert h.snapshot(120.0).overall == S.DISCONNECTED


def test_components_are_distinguished():
    h = ConnectionHealth()
    h.mark_listening()
    h.record_valid("heartbeat", 0.0, acc_status="NOT_REGISTERED")  # bridge alive, ACC not connected
    s = h.snapshot(0.5)
    assert s.bridge == S.CONNECTED and s.acc == S.WAITING_FOR_DATA and s.telemetry == S.WAITING_FOR_DATA
    assert s.overall != S.CONNECTED
    h.record_valid("heartbeat", 1.0, acc_status="AUTH_FAILED")
    assert h.snapshot(1.2).acc == S.ERROR and h.snapshot(1.2).overall == S.ERROR
    h.mark_error("socket closed")
    assert h.snapshot(1.2).receiver == S.ERROR


def test_sequence_tracking():
    h = ConnectionHealth()
    assert [h.check_sequence("a", "r1", s) for s in (1, 2, 2, 5, 4, 6)] == ["new", "new", "duplicate", "new", "out_of_order", "new"]
    assert h.duplicates == 1 and h.out_of_order == 1 and h.gaps == 1  # 3 still missing
    assert h.check_sequence("a", "r2", 1) == "new"  # restarted sender is a new sequence space
