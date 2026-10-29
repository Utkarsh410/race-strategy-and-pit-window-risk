from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from racesim.integrations.acc.models import Envelope, HeartbeatPayload, TelemetryPayload, VehicleInfo

NOW = datetime.now(timezone.utc)


def base(**kw):
    d = dict(source="acc_udp", session_id="abc", timestamp_utc=NOW, sequence=1, native_message_type="REALTIME_CAR_UPDATE")
    d.update(kw)
    return d


def test_valid_and_optional_fields():
    p = TelemetryPayload(**base())
    assert p.schema_version == "1.0" and p.vehicle.position is None and p.session.track_name is None
    p2 = TelemetryPayload(**base(vehicle={"position": 0, "laps_completed": 0}))
    assert p2.vehicle.position == 0 and p2.vehicle.laps_completed == 0  # 0 is a value, not "unknown"


@pytest.mark.parametrize("bad", [
    {"timestamp_utc": datetime.now()},                                   # naive
    {"timestamp_utc": NOW + timedelta(days=1)},                          # future
    {"timestamp_utc": datetime(2001, 1, 1, tzinfo=timezone.utc)},        # implausibly old
    {"schema_version": "2.0"},
    {"sequence": -1},
    {"source": "telemetry_from_somewhere"},
    {"vehicle": {"position": "first"}},
    {"vehicle": {"speed_kmh": 9999}},
    {"vehicle": {"last_lap_time_s": 0}},
    {"unexpected": 1},
])
def test_invalid(bad):
    with pytest.raises(ValidationError):
        TelemetryPayload(**base(**bad))


def test_envelope_and_heartbeat():
    hb = HeartbeatPayload(source="acc_udp", timestamp_utc=NOW, acc_status="REGISTERED", native_messages_received=5,
                          messages_forwarded=3, uptime_s=1.0)
    env = Envelope(message_type="heartbeat", schema_version="1.0", sequence=1, sent_at_utc=NOW, sender_id="w", sender_run_id="r",
                   mock=False, payload=hb.model_dump(mode="json"))
    assert env.typed_payload().acc_status == "REGISTERED"
    with pytest.raises(ValidationError):
        HeartbeatPayload(source="acc_udp", timestamp_utc=NOW, acc_status="TOTALLY_FINE", native_messages_received=0,
                         messages_forwarded=0, uptime_s=0)


def test_reserved_physics_fields_default_none():
    v = VehicleInfo()
    assert v.fuel_kg is None and v.tyre_wear is None
