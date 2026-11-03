"""ACC shared-memory channel definitions (physics, graphics, static) and their units.

Sources, in order of authority:

1. Kunos, *ACC Shared Memory Documentation v1.8.12* (2022-03-23; shipped in the
   pyaccsharedmemory repository as ``ACCSharedMemoryDocumentationV1.8.12.pdf``).
2. ``pyaccsharedmemory`` 1.0.0 (MIT, rrennoir/PyAccSharedMemory): attribute
   names and Python types from ``src/pyaccsharedmemory.py``; its README states a
   few units.

``unit_status`` records how far each unit is verified:

* ``kunos``     - stated in the Kunos document.
* ``conflict``  - the Kunos document and the library README disagree, or the
                  Kunos document contradicts itself (e.g. fuel kg vs fuel-per-lap litres).
* ``unstated``  - neither source states a unit; ``unit`` is the conventional
                  expectation and must be checked against the in-game display
                  in a live session (the Windows diagnostic prints raw values for this).
* ``library``   - stated only by the library README.

Values are forwarded **in native units** (no conversion is applied where the
native unit is not verified); field names carry the native unit.
Fields the Kunos document marks "Not used in ACC" / "Not shown in ACC" are not
read. Wheel order is FL, FR, RL, RR (Kunos).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

WHEELS = ("fl", "fr", "rl", "rr")
AXES = ("x", "y", "z")


@dataclass(frozen=True)
class Channel:
    name: str              # normalised RaceSim name
    page: str              # physics | graphics | static
    attr: str              # pyaccsharedmemory attribute
    kind: str              # scalar | wheels | vector | text | enum | bool
    unit: str
    unit_status: str
    meaning: str
    kunos_name: str


C = Channel
CHANNELS: tuple[Channel, ...] = (
    # -- driver inputs ---------------------------------------------------------
    C("throttle", "physics", "gas", "scalar", "fraction 0-1", "kunos", "Gas pedal input", "gas"),
    C("brake", "physics", "brake", "scalar", "fraction 0-1", "kunos", "Brake pedal input", "brake"),
    C("clutch", "physics", "clutch", "scalar", "fraction 0-1", "kunos", "Clutch pedal input", "clutch"),
    C("steering", "physics", "steer_angle", "scalar", "-1 to 1", "conflict",
      "Steering input; Kunos: -1.0 to 1.0, library README: 0.0 to 1.0", "steerAngle"),
    C("gear_raw", "physics", "gear", "scalar", "raw integer", "unstated",
      "Current gear; encoding not stated (AC convention 0=R, 1=N, 2=1st is unverified for ACC)", "gear"),
    # -- engine / speed -----------------------------------------------------------
    C("rpm", "physics", "rpm", "scalar", "rpm", "kunos", "Engine revolutions per minute", "rpm"),
    C("speed_kmh", "physics", "speed_kmh", "scalar", "km/h", "kunos", "Car speed", "speedKmh"),
    C("water_temp", "physics", "water_temp", "scalar", "degC (expected)", "unstated", "Water temperature", "waterTemp"),
    C("turbo_boost", "physics", "turbo_boost", "scalar", "raw", "unstated", "Car turbo level", "turboBoost"),
    # -- fuel -------------------------------------------------------------------------
    C("fuel_remaining", "physics", "fuel", "scalar", "kg (Kunos) / litres (library README)", "conflict",
      "Fuel remaining; Kunos says kg, library README says litres - check against the in-game fuel display", "fuel"),
    C("fuel_per_lap_l", "graphics", "fuel_per_lap", "scalar", "litres/lap", "kunos", "Average fuel consumed per lap", "fuelXLap"),
    C("fuel_estimated_laps", "graphics", "fuel_estimated_laps", "scalar", "laps", "kunos", "Laps possible with current fuel", "fuelEstimatedLaps"),
    C("used_fuel", "graphics", "used_fuel", "scalar", "litres (expected)", "unstated", "Fuel used since last refuelling", "usedFuel"),
    C("max_fuel", "static", "max_fuel", "scalar", "litres (expected)", "unstated", "Maximum fuel tank capacity", "maxFuel"),
    # -- motion -----------------------------------------------------------------------
    C("velocity", "physics", "velocity", "vector", "m/s (expected)", "unstated", "Velocity vector, global coordinates", "velocity"),
    C("local_velocity", "physics", "local_velocity", "vector", "m/s (expected)", "unstated", "Velocity vector, car coordinates", "localVelocity"),
    C("acceleration_g", "physics", "g_force", "vector", "G (expected)", "unstated", "Acceleration vector, global coordinates", "accG"),
    C("local_angular_velocity", "physics", "local_angular_vel", "vector", "rad/s (expected)", "unstated",
      "Angular velocity, car coordinates", "localAngularVel"),
    C("heading", "physics", "heading", "scalar", "rad (expected)", "unstated", "Car yaw orientation", "heading"),
    C("pitch", "physics", "pitch", "scalar", "rad (expected)", "unstated", "Car pitch orientation", "pitch"),
    C("roll", "physics", "roll", "scalar", "rad (expected)", "unstated", "Car roll orientation", "roll"),
    # -- tyres / wheels ------------------------------------------------------------------
    C("wheel_slip", "physics", "wheel_slip", "wheels", "dimensionless (expected)", "unstated", "Tyre slip", "wheelSlip"),
    C("slip_ratio", "physics", "slip_ratio", "wheels", "\"radians\" per Kunos (dimensionless expected)", "conflict",
      "Tyre slip ratio; Kunos labels it 'in radians', which is unusual for a ratio", "slipRatio"),
    C("slip_angle", "physics", "slip_angle", "wheels", "rad (expected)", "unstated", "Tyre slip angle", "slipAngle"),
    C("wheel_angular_speed", "physics", "wheel_angular_s", "wheels", "rad/s", "kunos", "Wheel angular speed", "wheelAngularSpeed"),
    C("tyre_pressure", "physics", "wheel_pressure", "wheels", "psi (expected)", "unstated", "Tyre pressure", "wheelPressure"),
    C("tyre_core_temp", "physics", "tyre_core_temp", "wheels", "degC (expected)", "unstated", "Tyre rubber core temperature", "TyreCoreTemp"),
    C("brake_temp", "physics", "brake_temp", "wheels", "degC (expected)", "unstated", "Brake disc temperature", "brakeTemp"),
    C("brake_pressure", "physics", "brake_pressure", "wheels", "raw (car-specific dash coefficient, Kunos appendix)", "kunos",
      "Brake pressure", "brakePressure"),
    C("pad_life", "physics", "pad_life", "wheels", "mm (library README: starts at 29 mm)", "library", "Brake pad wear", "padLife"),
    C("disc_life", "physics", "disc_life", "wheels", "mm (library README: starts at 32 mm)", "library", "Brake disc wear", "discLife"),
    # -- suspension -----------------------------------------------------------------------
    C("suspension_travel", "physics", "suspension_travel", "wheels", "m (expected)", "unstated", "Suspension travel", "suspensionTravel"),
    C("suspension_damage", "physics", "suspension_damage", "wheels", "raw", "unstated", "Suspension damage level", "suspensionDamage"),
    # -- aids / status -----------------------------------------------------------------------
    C("tc_in_action", "physics", "tc", "scalar", "raw", "unstated", "Traction control in action", "tc"),
    C("abs_in_action", "physics", "abs", "scalar", "raw", "unstated", "ABS in action", "abs"),
    C("brake_bias", "physics", "brake_bias", "scalar", "raw (Kunos appendix 4)", "kunos", "Front brake bias", "brakeBias"),
    C("pit_limiter_on", "physics", "pit_limiter_on", "bool", "", "kunos", "Pit limiter on", "pitLimiterOn"),
    C("is_ai_controlled", "physics", "is_ai_controlled", "bool", "", "kunos", "Car is controlled by the AI", "isAIControlled"),
    C("air_temp", "physics", "air_temp", "scalar", "degC (expected)", "unstated", "Air temperature", "airTemp"),
    C("road_temp", "physics", "road_temp", "scalar", "degC (expected)", "unstated", "Road temperature", "roadTemp"),
    # -- graphics: session / lap ------------------------------------------------------------
    C("status", "graphics", "status", "enum", "", "kunos", "ACC_STATUS (OFF/REPLAY/LIVE/PAUSE)", "status"),
    C("session_type", "graphics", "session_type", "enum", "", "kunos", "ACC_SESSION_TYPE", "session"),
    C("completed_laps", "graphics", "completed_lap", "scalar", "laps", "kunos", "Completed laps", "completedLaps"),
    C("position", "graphics", "position", "scalar", "", "kunos", "Current player position", "position"),
    C("current_lap_time_ms", "graphics", "current_time", "scalar", "ms", "kunos", "Current lap time", "iCurrentTime"),
    C("last_lap_time_ms", "graphics", "last_time", "scalar", "ms", "kunos", "Last lap time", "iLastTime"),
    C("best_lap_time_ms", "graphics", "best_time", "scalar", "ms", "kunos", "Best lap time", "iBestTime"),
    C("current_sector_index", "graphics", "current_sector_index", "scalar", "", "kunos", "Current track sector", "currentSectorIndex"),
    C("is_in_pit", "graphics", "is_in_pit", "bool", "", "kunos", "Car is pitting", "isInPit"),
    C("is_in_pit_lane", "graphics", "is_in_pit_lane", "bool", "", "kunos", "Car is in pit lane", "isInPitLane"),
    C("tyre_compound", "graphics", "tyre_compound", "text", "", "kunos", "Tyre compound used", "tyreCompound"),
    C("current_tyre_set", "graphics", "current_tyre_set", "scalar", "", "kunos", "Tyre set currently in use", "currentTyreSet"),
    C("track_grip_status", "graphics", "track_grip_status", "enum", "", "kunos", "ACC_TRACK_GRIP_STATUS", "trackGripStatus"),
    C("rain_intensity", "graphics", "rain_intensity", "enum", "", "kunos", "ACC_RAIN_INTENSITY", "rainIntensity"),
    C("flag", "graphics", "flag", "enum", "", "kunos", "ACC_FLAG_TYPE", "flag"),
    C("session_time_left", "graphics", "session_time_left", "scalar", "ms (expected)", "unstated", "Session time left", "sessionTimeLeft"),
    # -- static ---------------------------------------------------------------------------------
    C("car_model", "static", "car_model", "text", "", "kunos", "Player car model (Kunos id)", "carModel"),
    C("track", "static", "track", "text", "", "kunos", "Track name", "track"),
    C("max_rpm", "static", "max_rpm", "scalar", "rpm", "kunos", "Maximum rpm", "maxRpm"),
    C("sm_version", "static", "sm_version", "text", "", "kunos", "Shared memory version", "smVersion"),
    C("ac_version", "static", "ac_version", "text", "", "kunos", "ACC version", "acVersion"),
)

BY_NAME = {c.name: c for c in CHANNELS}

# Values derived on the Mac from measured channels (labelled DERIVED, never MEASURED).
DERIVED = {
    "speed_ms": ("m/s", "speed_kmh / 3.6"),
    "tyre_core_temp_spread": ("same unit as tyre_core_temp", "max - min of the four tyre core temperatures"),
    "tyre_pressure_spread": ("same unit as tyre_pressure", "max - min of the four tyre pressures"),
}

# Deliberately NOT read: player names (privacy) and fields Kunos marks unused in ACC.
NOT_READ = ("player_name", "player_surname", "player_nick", "car_coordinates", "car_id",
            "tyreWear", "tyreTempI/M/O", "camberRAD", "rideHeight", "kers*/ers*/drs*", "wheelLoad (absent from library)")


def channel(name: str) -> Optional[Channel]:
    return BY_NAME.get(name)


def derived_values(channels: dict) -> dict[str, float]:
    """Mac-side derived values (see :data:`DERIVED`); only computed when every input is present."""
    out: dict[str, float] = {}
    v = channels.get("speed_kmh")
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        out["speed_ms"] = v / 3.6
    for src, name in (("tyre_core_temp", "tyre_core_temp_spread"), ("tyre_pressure", "tyre_pressure_spread")):
        vals = channels.get(src)
        if isinstance(vals, list) and len(vals) == 4 and all(isinstance(x, (int, float)) for x in vals):
            out[name] = max(vals) - min(vals)
    return out


def physics_table(payload: Optional[dict], *, mock: bool, stale: bool) -> list[dict]:
    """Dashboard rows: every defined channel plus derived values, each with a status label.

    MEASURED (or MOCK / STALE) when the Windows reader supplied it, DERIVED for
    Mac-side calculations, UNAVAILABLE when the reader listed it missing or did
    not send it.
    """
    channels = (payload or {}).get("channels") or {}
    live = "MOCK - NOT LIVE" if mock else ("STALE" if stale else "LIVE - MEASURED")
    rows = []
    for ch in CHANNELS:
        value = channels.get(ch.name)
        rows.append({"channel": ch.name, "value": value, "unit": ch.unit, "unit_status": ch.unit_status,
                     "status": "UNAVAILABLE" if value is None else live, "kunos_field": f"{ch.page}.{ch.kunos_name}",
                     "meaning": ch.meaning})
    derived = derived_values(channels)
    for name, (unit, rule) in DERIVED.items():
        value = derived.get(name)
        rows.append({"channel": name, "value": value, "unit": unit, "unit_status": "derived",
                     "status": "UNAVAILABLE" if value is None else ("MOCK - NOT LIVE" if mock else ("STALE" if stale else "DERIVED")),
                     "kunos_field": "-", "meaning": rule})
    return rows
