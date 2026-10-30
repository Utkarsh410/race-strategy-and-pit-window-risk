"""Fuel and tyre monitoring from the channels that actually arrive (standard library only).

The UDP broadcasting API supplies no fuel or tyre channels, so with today's
bridge every value here is ``None`` and the channel is listed as requiring a
future Windows-side shared-memory reader. macOS cannot read Windows shared
memory, and nothing here substitutes a model value for a missing measurement.
"""

from __future__ import annotations

import statistics
from typing import Optional, Sequence

from racesim.integrations.acc.lap_tracker import CompletedLap, LapClass, Stint

MIN_FUEL_LAPS = 3
MAX_PLAUSIBLE_BURN_KG_PER_LAP = 15.0
SUDDEN_BURN_RATIO = 1.5
SHARED_MEMORY = "Future Windows shared-memory reader (not in the UDP broadcasting API)"
FUTURE_CHANNELS = (
    ("fuel_kg", "Fuel level", SHARED_MEMORY), ("tyre_compound", "Tyre compound", SHARED_MEMORY),
    ("tyre_age_laps", "Tyre age", SHARED_MEMORY), ("tyre_pressures_psi", "Tyre pressures", SHARED_MEMORY),
    ("tyre_temperatures_c", "Tyre temperatures", SHARED_MEMORY), ("tyre_wear", "Tyre wear", SHARED_MEMORY),
)


def lap_fuel_burns(laps: Sequence[CompletedLap]) -> tuple[list[dict], list[str]]:
    """One row per lap with a usable pair of readings, plus flags for implausible readings."""
    rows, flags = [], []
    for lap in laps:
        if lap.fuel_start_kg is None or lap.fuel_end_kg is None:
            continue
        burn = lap.fuel_start_kg - lap.fuel_end_kg
        reason = None
        if lap.classification in (LapClass.IN_LAP, LapClass.OUT_LAP, LapClass.INCOMPLETE) or lap.in_pit:
            reason = "pit_activity_in_lap"
        elif burn <= 0:
            reason = "fuel_not_decreasing"
        elif burn > MAX_PLAUSIBLE_BURN_KG_PER_LAP:
            reason = "implausible_fuel_drop"
        if reason and reason != "pit_activity_in_lap":
            flags.append(f"lap {lap.lap_number}: {reason} ({burn:+.2f} kg)")
        rows.append({"lap_number": lap.lap_number, "burn_kg": burn, "reliable": reason is None, "excluded_reason": reason})
    reliable = [r["burn_kg"] for r in rows if r["reliable"]]
    if len(reliable) >= MIN_FUEL_LAPS:
        median = statistics.median(reliable)
        for r in rows:
            if r["reliable"] and r["burn_kg"] > median * SUDDEN_BURN_RATIO:
                r["reliable"], r["excluded_reason"] = False, "sudden_fuel_change"
                flags.append(f"lap {r['lap_number']}: sudden_fuel_change ({r['burn_kg']:.2f} kg vs median {median:.2f} kg)")
    return rows, flags


def fuel_status(laps: Sequence[CompletedLap], latest_fuel_kg: Optional[float], *, pace_s: Optional[float] = None,
                pace_n: int = 0, stale: bool = False) -> dict:
    rows, flags = lap_fuel_burns(laps)
    reliable = [r["burn_kg"] for r in rows if r["reliable"]]
    burn = statistics.median(reliable) if len(reliable) >= MIN_FUEL_LAPS else None
    laps_left = latest_fuel_kg / burn if burn and latest_fuel_kg is not None and not stale else None
    return {"latest_fuel_kg": latest_fuel_kg, "burn_kg_per_lap": burn, "burn_laps_used": len(reliable), "burn_laps_needed": MIN_FUEL_LAPS,
            "laps_remaining_estimate": laps_left, "time_remaining_estimate_s": laps_left * pace_s if laps_left is not None and pace_s and pace_n >= 3 else None,
            "history": rows, "flags": flags}


def tyre_status(vehicle: dict, stints: Sequence[Stint]) -> dict:
    return {"compound": vehicle.get("tyre_compound"), "age_laps": vehicle.get("tyre_age_laps"),
            "pressures_psi": vehicle.get("tyre_pressures_psi"), "temperatures_c": vehicle.get("tyre_temperatures_c"),
            "wear": vehicle.get("tyre_wear"),
            "by_stint": [{"stint_id": st.stint_id, "compound": st.tyre_compound, "age_at_start_laps": st.tyre_age_start_laps} for st in stints]}


def missing_channels(vehicle: dict) -> list[dict]:
    return [{"channel": label, "field": key, "requires": needs} for key, label, needs in FUTURE_CHANNELS if vehicle.get(key) is None]
