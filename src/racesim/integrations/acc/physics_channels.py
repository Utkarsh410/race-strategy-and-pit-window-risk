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
native unit is not verified); a field name only carries a unit when that unit
is verified (hence ``fuel_remaining``, not ``fuel_l``). ``None`` / absent means
unavailable; ``0`` is a real reading. Values outside ``valid_range`` are
reported as invalid, never as measurements. Vectors are ``[x, y, z]``; Kunos
does not state the axis orientation of the global or car coordinate frames.
Fields the Kunos document marks "Not used in ACC" / "Not shown in ACC" are not
read. Wheel order is FL, FR, RL, RR (Kunos).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

WHEELS = ("fl", "fr", "rl", "rr")
AXES = ("x", "y", "z")
DAMAGE_ZONES = ("front", "rear", "left", "right", "center")
KIND_LENGTH = {"wheels": 4, "vector": 3, "wheel_vectors": 4, "damage": 5}

# Dashboard groups, in display order.
GROUPS = ("inputs", "vehicle", "dynamics", "tyres", "brakes", "car", "session")


@dataclass(frozen=True)
class Channel:
    name: str              # normalised RaceSim name
    page: str              # physics | graphics | static
    attr: str              # pyaccsharedmemory attribute
    kind: str              # scalar | wheels | vector | wheel_vectors | damage | text | enum | bool
    unit: str
    unit_status: str
    meaning: str
    kunos_name: str
    group: str = "car"
    # Plausibility range (inclusive) in the native unit. Only set where the bound
    # holds whatever the unverified unit turns out to be; values outside it are
    # moved to ``invalid`` and never shown as measurements.
    valid_range: Optional[tuple[float, float]] = None


GLOBAL = "global (world) coordinates; axis orientation not stated by Kunos"
LOCAL = "car coordinates; axis orientation not stated by Kunos"

C = Channel
CHANNELS: tuple[Channel, ...] = (
    # -- driver inputs ---------------------------------------------------------
    C("throttle", "physics", "gas", "scalar", "fraction 0-1", "kunos", "Gas pedal input (not a percentage)", "gas", "inputs", (0, 1)),
    C("brake", "physics", "brake", "scalar", "fraction 0-1", "kunos", "Brake pedal input (not a percentage)", "brake", "inputs", (0, 1)),
    C("clutch", "physics", "clutch", "scalar", "fraction 0-1", "kunos", "Clutch pedal input", "clutch", "inputs", (0, 1)),
    C("steering_input", "physics", "steer_angle", "scalar", "normalised -1 to 1 (not degrees)", "conflict",
      "Steering input as a fraction of full lock, NOT a wheel angle; Kunos: -1.0 to 1.0, library README: 0.0 to 1.0",
      "steerAngle", "inputs", (-1, 1)),
    # -- vehicle state ------------------------------------------------------------
    C("gear", "physics", "gear", "scalar", "raw ACC gear index", "unstated",
      "Current gear in ACC's raw encoding, not the displayed gear (AC convention 0=R, 1=N, 2=1st is unverified for ACC)",
      "gear", "vehicle"),
    C("rpm", "physics", "rpm", "scalar", "rpm", "kunos", "Engine revolutions per minute", "rpm", "vehicle", (0, 30000)),
    C("speed_kmh", "physics", "speed_kmh", "scalar", "km/h", "kunos", "Car speed", "speedKmh", "vehicle", (0, 500)),
    C("water_temp", "physics", "water_temp", "scalar", "degC (expected)", "unstated", "Water temperature", "waterTemp",
      "vehicle", (-60, 250)),
    C("turbo_boost", "physics", "turbo_boost", "scalar", "raw", "unstated", "Car turbo level", "turboBoost", "vehicle"),
    C("is_engine_running", "physics", "is_engine_running", "bool", "", "kunos", "Engine running", "isEngineRunning", "vehicle"),
    C("pit_limiter_on", "physics", "pit_limiter_on", "bool", "", "kunos", "Pit limiter on", "pitLimiterOn", "vehicle"),
    C("is_ai_controlled", "physics", "is_ai_controlled", "bool", "", "kunos", "Car is controlled by the AI", "isAIControlled", "vehicle"),
    # -- fuel ---------------------------------------------------------------------------
    # Not called fuel_l: Kunos says kg, the library README says litres. Renamed once a live session settles it.
    C("fuel_remaining", "physics", "fuel", "scalar", "kg (Kunos) / litres (library README)", "conflict",
      "Fuel remaining; Kunos says kg, library README says litres - check against the in-game fuel display", "fuel",
      "vehicle", (0, 250)),
    C("fuel_per_lap_l", "graphics", "fuel_per_lap", "scalar", "litres/lap", "kunos", "Average fuel consumed per lap",
      "fuelXLap", "vehicle", (0, 50)),
    C("fuel_estimated_laps", "graphics", "fuel_estimated_laps", "scalar", "laps", "kunos", "Laps possible with current fuel",
      "fuelEstimatedLaps", "vehicle", (0, 10000)),
    C("used_fuel", "graphics", "used_fuel", "scalar", "litres (expected)", "unstated", "Fuel used since last refuelling",
      "usedFuel", "vehicle", (0, 10000)),
    C("max_fuel", "static", "max_fuel", "scalar", "litres (expected)", "unstated", "Maximum fuel tank capacity", "maxFuel",
      "vehicle", (0, 250)),
    # -- dynamics -----------------------------------------------------------------------
    C("velocity", "physics", "velocity", "vector", "m/s (expected)", "unstated", f"Velocity [x, y, z], {GLOBAL}",
      "velocity", "dynamics"),
    C("local_velocity", "physics", "local_velocity", "vector", "m/s (expected)", "unstated", f"Velocity [x, y, z], {LOCAL}",
      "localVelocity", "dynamics"),
    C("acceleration", "physics", "g_force", "vector", "G (expected)", "unstated", f"Acceleration [x, y, z], {GLOBAL}",
      "accG", "dynamics"),
    C("angular_velocity", "physics", "local_angular_vel", "vector", "rad/s (expected)", "unstated",
      f"Angular velocity [x, y, z], {LOCAL}", "localAngularVel", "dynamics"),
    C("heading", "physics", "heading", "scalar", "rad (expected)", "unstated", "Car yaw orientation", "heading", "dynamics"),
    C("pitch", "physics", "pitch", "scalar", "rad (expected)", "unstated", "Car pitch orientation", "pitch", "dynamics"),
    C("roll", "physics", "roll", "scalar", "rad (expected)", "unstated", "Car roll orientation", "roll", "dynamics"),
    C("wheel_slip", "physics", "wheel_slip", "wheels", "dimensionless (expected)", "unstated", "Tyre slip per wheel",
      "wheelSlip", "dynamics"),
    C("slip_ratio", "physics", "slip_ratio", "wheels", "\"radians\" per Kunos (dimensionless expected)", "conflict",
      "Tyre slip ratio; Kunos labels it 'in radians', which is unusual for a ratio", "slipRatio", "dynamics"),
    C("slip_angle", "physics", "slip_angle", "wheels", "rad (expected)", "unstated", "Tyre slip angle", "slipAngle", "dynamics"),
    # -- tyres / wheels ------------------------------------------------------------------
    C("wheel_pressure", "physics", "wheel_pressure", "wheels", "psi (expected)", "unstated", "Tyre pressure", "wheelPressure",
      "tyres", (0, 100)),
    C("tyre_core_temperature", "physics", "tyre_core_temp", "wheels", "degC (expected)", "unstated",
      "Tyre rubber core temperature", "TyreCoreTemp", "tyres", (-60, 400)),
    C("wheel_angular_speed", "physics", "wheel_angular_s", "wheels", "rad/s", "kunos", "Wheel angular speed",
      "wheelAngularSpeed", "tyres"),
    C("suspension_travel", "physics", "suspension_travel", "wheels", "m (expected)", "unstated", "Suspension travel",
      "suspensionTravel", "tyres", (-1, 1)),
    C("tyre_contact_point", "physics", "tyre_contact_point", "wheel_vectors", "m (expected)", "unstated",
      f"Tyre contact point [x, y, z] per wheel, {GLOBAL}", "tyreContactPoint", "tyres"),
    C("tyre_contact_normal", "physics", "tyre_contact_normal", "wheel_vectors", "unit vector (expected)", "unstated",
      f"Tyre contact normal [x, y, z] per wheel, {GLOBAL}", "tyreContactNormal", "tyres"),
    C("tyre_contact_heading", "physics", "tyre_contact_heading", "wheel_vectors", "unit vector (expected)", "unstated",
      f"Tyre contact heading [x, y, z] per wheel, {GLOBAL}", "tyreContactHeading", "tyres"),
    # -- brakes / aids ---------------------------------------------------------------------
    C("brake_pressure", "physics", "brake_pressure", "wheels", "raw (car-specific dash coefficient, Kunos appendix)", "kunos",
      "Brake pressure", "brakePressure", "brakes"),
    C("brake_temperature", "physics", "brake_temp", "wheels", "degC (expected)", "unstated", "Brake disc temperature",
      "brakeTemp", "brakes", (-60, 2000)),
    C("brake_bias", "physics", "brake_bias", "scalar", "raw (Kunos appendix 4)", "kunos", "Front brake bias", "brakeBias", "brakes"),
    C("pad_life", "physics", "pad_life", "wheels", "mm (library README: starts at 29 mm)", "library", "Brake pad wear",
      "padLife", "brakes"),
    C("disc_life", "physics", "disc_life", "wheels", "mm (library README: starts at 32 mm)", "library", "Brake disc wear",
      "discLife", "brakes"),
    C("abs_activity", "physics", "abs", "scalar", "raw", "unstated", "ABS in action (raw level, not a setting)", "abs", "brakes"),
    C("tc_activity", "physics", "tc", "scalar", "raw", "unstated", "Traction control in action (raw level, not a setting)",
      "tc", "brakes"),
    # -- damage / environment ------------------------------------------------------------
    C("car_damage", "physics", "car_damage", "damage", "raw", "unstated",
      "Car damage [front, rear, left, right, center]; library notes 'center' is the sum of all", "carDamage", "car"),
    C("suspension_damage", "physics", "suspension_damage", "wheels", "raw", "unstated", "Suspension damage level",
      "suspensionDamage", "car"),
    C("air_temp", "physics", "air_temp", "scalar", "degC (expected)", "unstated", "Air temperature", "airTemp", "car", (-60, 100)),
    C("road_temp", "physics", "road_temp", "scalar", "degC (expected)", "unstated", "Road temperature", "roadTemp", "car", (-60, 100)),
    # -- graphics: session / lap ------------------------------------------------------------
    C("status", "graphics", "status", "enum", "", "kunos", "ACC_STATUS (OFF/REPLAY/LIVE/PAUSE)", "status", "session"),
    C("session_type", "graphics", "session_type", "enum", "", "kunos", "ACC_SESSION_TYPE", "session", "session"),
    C("completed_laps", "graphics", "completed_lap", "scalar", "laps", "kunos", "Completed laps", "completedLaps", "session",
      (0, 10000)),
    C("position", "graphics", "position", "scalar", "", "kunos", "Current player position", "position", "session", (0, 1000)),
    C("current_lap_time_ms", "graphics", "current_time", "scalar", "ms", "kunos", "Current lap time", "iCurrentTime", "session"),
    C("last_lap_time_ms", "graphics", "last_time", "scalar", "ms", "kunos", "Last lap time", "iLastTime", "session"),
    C("best_lap_time_ms", "graphics", "best_time", "scalar", "ms", "kunos", "Best lap time", "iBestTime", "session"),
    C("current_sector_index", "graphics", "current_sector_index", "scalar", "", "kunos", "Current track sector",
      "currentSectorIndex", "session"),
    C("is_in_pit", "graphics", "is_in_pit", "bool", "", "kunos", "Car is pitting", "isInPit", "session"),
    C("is_in_pit_lane", "graphics", "is_in_pit_lane", "bool", "", "kunos", "Car is in pit lane", "isInPitLane", "session"),
    C("tyre_compound", "graphics", "tyre_compound", "text", "", "kunos", "Tyre compound used", "tyreCompound", "tyres"),
    C("current_tyre_set", "graphics", "current_tyre_set", "scalar", "", "kunos", "Tyre set currently in use", "currentTyreSet", "tyres"),
    C("track_grip_status", "graphics", "track_grip_status", "enum", "", "kunos", "ACC_TRACK_GRIP_STATUS", "trackGripStatus", "session"),
    C("rain_intensity", "graphics", "rain_intensity", "enum", "", "kunos", "ACC_RAIN_INTENSITY", "rainIntensity", "session"),
    C("flag", "graphics", "flag", "enum", "", "kunos", "ACC_FLAG_TYPE", "flag", "session"),
    C("session_time_left", "graphics", "session_time_left", "scalar", "ms (expected)", "unstated", "Session time left",
      "sessionTimeLeft", "session"),
    # -- static ---------------------------------------------------------------------------------
    C("car_model", "static", "car_model", "text", "", "kunos", "Player car model (Kunos id)", "carModel", "session"),
    C("track", "static", "track", "text", "", "kunos", "Track name", "track", "session"),
    C("max_rpm", "static", "max_rpm", "scalar", "rpm", "kunos", "Maximum rpm", "maxRpm", "vehicle", (0, 30000)),
    C("sm_version", "static", "sm_version", "text", "", "kunos", "Shared memory version", "smVersion", "session"),
    C("ac_version", "static", "ac_version", "text", "", "kunos", "ACC version", "acVersion", "session"),
)

# Physics schema 1.0 names -> 1.1 names (1.0 messages are renamed on receipt).
LEGACY_NAMES = {"steering": "steering_input", "gear_raw": "gear", "acceleration_g": "acceleration",
                "local_angular_velocity": "angular_velocity", "tyre_pressure": "wheel_pressure",
                "tyre_core_temp": "tyre_core_temperature", "brake_temp": "brake_temperature",
                "abs_in_action": "abs_activity", "tc_in_action": "tc_activity"}

# Target names from the integration brief that are deliberately not used yet, and why.
RESERVED_NAMES = {"fuel_l": "fuel_remaining carries the raw value until a live session shows whether ACC reports kg or litres"}

BY_NAME = {c.name: c for c in CHANNELS}

# Values derived on the Mac from measured channels (labelled DERIVED, never MEASURED).
DERIVED = {
    "speed_ms": ("m/s", "speed_kmh / 3.6"),
    "tyre_core_temperature_spread": ("same unit as tyre_core_temperature", "max - min of the four tyre core temperatures"),
    "wheel_pressure_spread": ("same unit as wheel_pressure", "max - min of the four tyre pressures"),
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
    for src, name in (("tyre_core_temperature", "tyre_core_temperature_spread"), ("wheel_pressure", "wheel_pressure_spread")):
        vals = channels.get(src)
        if isinstance(vals, list) and len(vals) == 4 and all(isinstance(x, (int, float)) for x in vals):
            out[name] = max(vals) - min(vals)
    return out


def _finite_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def check_value(ch: Channel, value: Any) -> Optional[str]:
    """Why ``value`` is not acceptable for ``ch`` (shape, type, finiteness, range), or None if it is.

    ``None`` elements inside per-wheel / vector lists are allowed (one missing wheel).
    """
    if ch.kind in KIND_LENGTH:
        n = KIND_LENGTH[ch.kind]
        if not isinstance(value, list) or len(value) != n:
            return f"expected a list of {n} values"
        items = value
        if ch.kind == "wheel_vectors":
            if any(x is not None and (not isinstance(x, list) or len(x) != 3) for x in value):
                return "expected 4 lists of 3 values"
            items = [y for x in value if x is not None for y in x]
        if any(x is not None and not _finite_number(x) for x in items):
            return "non-finite or non-numeric element"
        nums = [x for x in items if x is not None]
    elif ch.kind in ("text", "enum"):
        return None if isinstance(value, str) and len(value) <= 64 else "expected a short string"
    elif ch.kind == "bool":
        return None if isinstance(value, bool) else "expected a boolean"
    else:
        if not _finite_number(value):
            return "non-finite or non-numeric value"
        nums = [value]
    if ch.valid_range is not None:
        lo, hi = ch.valid_range
        eps = 1e-6 * max(1.0, hi - lo)  # float noise at the bounds (e.g. throttle 1.0000001)
        bad = [x for x in nums if not lo - eps <= x <= hi + eps]
        if bad:
            return f"out of range [{lo:g}, {hi:g}]: {bad[0]:g}"
    return None


def split_invalid(channels: dict) -> tuple[dict, dict[str, str]]:
    """Separate acceptable channel values from invalid ones (``{name: reason}``). Unknown names are invalid."""
    valid, invalid = {}, {}
    for name, value in channels.items():
        ch = BY_NAME.get(name)
        if ch is None:
            invalid[name] = "unknown channel"
            continue
        if value is None:
            continue
        reason = check_value(ch, value)
        if reason is None:
            valid[name] = value
        else:
            invalid[name] = reason
    return valid, invalid


# Dashboard / export status labels.
LIVE_MEASUREMENT, DERIVED_LABEL, STALE, UNAVAILABLE, INVALID, MOCK = (
    "LIVE MEASUREMENT", "DERIVED", "STALE", "UNAVAILABLE", "INVALID", "MOCK")


def physics_table(payload: Optional[dict], *, mock: bool, stale: bool, sample_age_s: Optional[float] = None) -> list[dict]:
    """Dashboard rows: every defined channel plus derived values, each with a status label.

    LIVE MEASUREMENT (or MOCK / STALE) when the Windows reader supplied it,
    DERIVED for Mac-side calculations, INVALID when it failed validation,
    UNAVAILABLE when the reader listed it missing or did not send it. Every
    row carries its source (shared-memory page) and the sample's measurement
    time and age: all channels in one sample were read together.
    """
    payload = payload or {}
    channels = payload.get("channels") or {}
    invalid = payload.get("invalid") or {}
    measured_at = payload.get("timestamp_utc")
    live = MOCK if mock else (STALE if stale else LIVE_MEASUREMENT)
    rows = []
    for ch in CHANNELS:
        value = channels.get(ch.name)
        status = INVALID if ch.name in invalid else (UNAVAILABLE if value is None else live)
        rows.append({"group": ch.group, "channel": ch.name, "value": value, "unit": ch.unit, "unit_status": ch.unit_status,
                     "status": status, "source": f"shared memory {ch.page}" + (" (MOCK)" if mock else ""),
                     "measured_at_utc": None if value is None else measured_at,
                     "age_s": None if value is None else sample_age_s, "kunos_field": f"{ch.page}.{ch.kunos_name}",
                     "meaning": ch.meaning + (f" [invalid: {invalid[ch.name]}]" if ch.name in invalid else "")})
    derived = derived_values(channels)
    for name, (unit, rule) in DERIVED.items():
        value = derived.get(name)
        rows.append({"group": "derived", "channel": name, "value": value, "unit": unit, "unit_status": "derived",
                     "status": UNAVAILABLE if value is None else (MOCK if mock else (STALE if stale else DERIVED_LABEL)),
                     "source": "Mac (calculated)", "measured_at_utc": None if value is None else measured_at,
                     "age_s": None if value is None else sample_age_s, "kunos_field": "-", "meaning": rule})
    return rows


TYRE_WHEEL_CHANNELS = ("wheel_pressure", "tyre_core_temperature", "wheel_angular_speed", "suspension_travel",
                       "wheel_slip", "slip_ratio", "slip_angle")
BRAKE_WHEEL_CHANNELS = ("brake_temperature", "brake_pressure", "pad_life", "disc_life")


def wheel_rows(channels: dict, names: tuple[str, ...]) -> list[dict]:
    """One row per wheel (FL, FR, RL, RR) with a column per channel; None = unavailable for that wheel."""
    rows = []
    for i, wheel in enumerate(WHEELS):
        row: dict[str, Any] = {"wheel": wheel.upper()}
        for name in names:
            vals = channels.get(name)
            row[f"{name} [{BY_NAME[name].unit}]"] = vals[i] if isinstance(vals, list) and len(vals) == 4 else None
        rows.append(row)
    return rows


def quality_summary(payload: Optional[dict]) -> dict[str, Any]:
    """Counts and lists for the dashboard's data-quality group."""
    p = payload or {}
    return {"available": len(p.get("available") or []), "missing": list(p.get("missing") or []),
            "invalid": dict(p.get("invalid") or {}), "defined": len(CHANNELS),
            "unverified_units": [c.name for c in CHANNELS if c.unit_status in ("conflict", "unstated")
                                 and c.name in (p.get("channels") or {})]}
