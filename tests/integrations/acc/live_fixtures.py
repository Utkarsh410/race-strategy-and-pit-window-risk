"""Deterministic telemetry for the live-analysis tests.

Synthetic, but shaped like recorded ACC sessions: the out-lap does not move the
lap counter, a restart keeps the session id, and events repeat.
"""

from datetime import datetime, timedelta, timezone

BASE = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
UNSET = object()


class Script:
    """Build telemetry payloads on a shared clock; ``t`` is seconds since the session started."""

    def __init__(self, session_id="sess-1", source="acc_udp", session_type="PRACTICE", track="Test Track", end_total=3600.0, session_index=1):
        self.session_id, self.source, self.session_type, self.track = session_id, source, session_type, track
        self.end_total, self.session_index = end_total, session_index
        self.seq = 0
        self.clock_offset = 0.0  # wall-clock seconds added to t; a restart resets t while the wall clock runs on

    def tel(self, t, laps=0, last=None, valid=UNSET, location="TRACK", phase="SESSION", position=1, speed=200, current=None, **vehicle):
        self.seq += 1
        session = {"track_name": self.track, "track_length_m": 5000, "session_type": self.session_type, "session_phase": phase,
                   "session_index": self.session_index, "session_time_elapsed_s": t, "session_end_time_raw_s": self.end_total - t,
                   "track_temperature_c": 30, "ambient_temperature_c": 22, "is_replay_playing": False}
        v = {"car_id": 0, "race_number": 111, "car_model_id": 35, "position": position, "laps_completed": laps, "current_lap": laps + 1,
             "car_location": location, "speed_kmh": speed, "current_lap_time_s": current}
        if last is not None:
            v.update(last_lap_time_s=last, best_lap_time_s=last, last_lap_valid=True if valid is UNSET else valid)
        v.update(vehicle)
        sources = {f"session.{k}": f"REALTIME_UPDATE.{k}" for k in session} | {f"vehicle.{k}": f"REALTIME_CAR_UPDATE.{k}" for k in v}
        sources["vehicle.current_lap"] = "DERIVED: laps_completed + 1"
        return {"schema_version": "1.0", "source": self.source, "session_id": self.session_id, "timestamp_utc": (BASE + timedelta(seconds=t + self.clock_offset)).isoformat(),
                "sequence": self.seq, "native_message_type": "REALTIME_CAR_UPDATE", "session": session, "vehicle": v,
                "connection": {"connected": True}, "field_sources": sources, "data_quality": []}

    def event(self, kind="LAP_COMPLETED", time_ms=1000, car_id=0, message="02:03.052"):
        return {"source": self.source, "timestamp_utc": BASE.isoformat(), "session_id": self.session_id, "event_type": kind,
                "message": message, "time_ms": time_ms, "car_id": car_id}


def drive(script, start_t, laps_from, lap_times, *, step=1.0, valid=UNSET, location="TRACK", **vehicle):
    """Telemetry for consecutive laps; the car crosses the line at the end of each ``lap_times`` entry."""
    out, t, laps = [], start_t, laps_from
    last = vehicle.pop("last", None)
    for lap_time in lap_times:
        end = t + lap_time
        while t + step < end:
            t += step
            out.append((t, script.tel(t, laps=laps, last=last, valid=valid, location=location, **vehicle)))
        t, laps, last = end, laps + 1, lap_time
        out.append((t, script.tel(t, laps=laps, last=last, valid=valid, location=location, **vehicle)))
    return out


def feed(target, frames):
    """Feed ``(t, payload)`` frames to a LapTracker (``update``) or LiveAnalysis (``on_message``); returns finished laps."""
    laps = []
    for t, payload in frames:
        if hasattr(target, "on_message"):
            target.on_message("telemetry", payload, "new", t)
        else:
            laps += target.update(payload, t)
    return laps
