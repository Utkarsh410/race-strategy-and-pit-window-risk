"""Fuel-load model.

* Fuel mass in kilograms, consumption in kg per lap, lap-time effect in
  seconds per kg of fuel on board.
* ``fuel_at_lap_start(lap)`` is the mass on board when ``lap`` begins.
* The fuel time effect is ``fuel_effect_s_per_kg * fuel_kg`` relative to an
  empty tank (the reference condition of the baseline lap time).
* Refuelling is never assumed: it is only modelled when the regulations
  explicitly set ``refuelling_allowed: true``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ArrayLike = np.ndarray | float | int


def calculate_fuel_remaining(fuel_start_kg: float, consumption_kg_per_lap: ArrayLike, laps_completed: ArrayLike) -> np.ndarray:
    """Fuel remaining after ``laps_completed`` laps (never negative)."""
    remaining = fuel_start_kg - np.asarray(consumption_kg_per_lap) * np.asarray(laps_completed, dtype=float)
    return np.maximum(remaining, 0.0)


def fuel_at_lap_start(lap: ArrayLike, fuel_start_kg: float, consumption_kg_per_lap: float, start_lap: int = 1) -> np.ndarray:
    """Fuel on board at the start of ``lap`` given ``fuel_start_kg`` at ``start_lap``."""
    return calculate_fuel_remaining(fuel_start_kg, consumption_kg_per_lap, np.asarray(lap) - start_lap)


def calculate_fuel_time_effect(fuel_kg: ArrayLike, fuel_effect_s_per_kg: float) -> np.ndarray:
    """Lap-time penalty (s) of carrying ``fuel_kg`` relative to an empty tank."""
    fuel = np.asarray(fuel_kg, dtype=float)
    if np.any(fuel < 0):
        raise ValueError("fuel mass cannot be negative")
    return fuel_effect_s_per_kg * fuel


@dataclass
class FuelFeasibility:
    feasible: bool
    required_kg: float
    available_kg: float
    margin_kg: float
    reason: str = ""
    per_stint_required_kg: list[float] = field(default_factory=list)


def validate_fuel_feasibility(
    fuel_start_kg: float,
    consumption_kg_per_lap: float,
    laps: int,
    *,
    reserve_kg: float = 0.0,
    refuelling_allowed: bool = False,
    fuel_capacity_kg: Optional[float] = None,
    stint_lengths: Optional[Sequence[int]] = None,
) -> FuelFeasibility:
    """Check that the planned laps can be completed with the available fuel.

    Without refuelling the starting load must cover all ``laps`` plus the
    reserve. With refuelling each stint must fit in the tank capacity.
    """
    required = consumption_kg_per_lap * laps + reserve_kg
    if not refuelling_allowed:
        margin = fuel_start_kg - required
        ok = margin >= -1e-9
        reason = "" if ok else (
            f"fuel shortfall: need {required:.2f} kg for {laps} laps (+reserve) but start with {fuel_start_kg:.2f} kg; "
            "refuelling not permitted"
        )
        return FuelFeasibility(ok, required, fuel_start_kg, margin, reason)
    if fuel_capacity_kg is None:
        return FuelFeasibility(False, required, fuel_start_kg, float("nan"), "refuelling allowed but fuel capacity unknown")
    lengths = list(stint_lengths) if stint_lengths else [laps]
    per_stint = [consumption_kg_per_lap * n + reserve_kg for n in lengths]
    worst = max(per_stint)
    margin = fuel_capacity_kg - worst
    ok = margin >= -1e-9 and fuel_start_kg + 1e-9 >= per_stint[0]
    reason = ""
    if not ok:
        reason = (
            f"stint fuel demand {worst:.2f} kg exceeds capacity {fuel_capacity_kg:.2f} kg"
            if margin < 0
            else f"starting fuel {fuel_start_kg:.2f} kg below first-stint demand {per_stint[0]:.2f} kg"
        )
    return FuelFeasibility(ok, required, fuel_capacity_kg, margin, reason, per_stint)


def calculate_refuel_amount(
    fuel_on_board_kg: float | np.ndarray,
    next_stint_laps: int | np.ndarray,
    consumption_kg_per_lap: float,
    fuel_capacity_kg: float,
    reserve_kg: float = 0.0,
) -> np.ndarray:
    """Fuel to add so the next stint can be completed (bounded by capacity)."""
    need = np.asarray(next_stint_laps) * consumption_kg_per_lap + reserve_kg - np.asarray(fuel_on_board_kg)
    room = fuel_capacity_kg - np.asarray(fuel_on_board_kg)
    return np.clip(need, 0.0, np.maximum(room, 0.0))


@dataclass
class FuelConsumptionEstimate:
    kg_per_lap: Optional[float]
    source: str  # "measured" | "assumption" | "unavailable"
    sample_count: int
    warnings: list[str] = field(default_factory=list)


def estimate_fuel_consumption(
    dataframe: Optional[pd.DataFrame] = None,
    *,
    assumed_kg_per_lap: Optional[float] = None,
) -> FuelConsumptionEstimate:
    """Estimate consumption from consecutive ``fuel_kg`` readings.

    Uses green-flag laps without pit activity. If no usable fuel data exist,
    returns the explicit ``assumed_kg_per_lap`` labelled as an assumption, or
    an ``unavailable`` result if no assumption was supplied.
    """
    if dataframe is not None and {"fuel_kg", "lap"} <= set(dataframe.columns) and dataframe["fuel_kg"].notna().sum() > 2:
        df = dataframe.sort_values(["driver", "lap"] if "driver" in dataframe.columns else ["lap"]).copy()
        group = df["driver"] if "driver" in df.columns else pd.Series(0, index=df.index)
        burn = -df.groupby(group)["fuel_kg"].diff()
        lap_gap = df.groupby(group)["lap"].diff()
        mask = (lap_gap == 1) & burn.notna() & (burn >= 0)
        if "track_status" in df.columns:
            mask &= df["track_status"].astype(str).str.upper().eq("GREEN")
        for col in ("pit_in", "pit_out"):
            if col in df.columns:
                mask &= ~df[col].fillna(False).astype(bool)
        values = burn[mask]
        if len(values) >= 3:
            est = float(values.median())
            logger.info("Measured fuel consumption %.3f kg/lap from %d laps", est, len(values))
            return FuelConsumptionEstimate(est, "measured", int(len(values)))
    if assumed_kg_per_lap is not None:
        return FuelConsumptionEstimate(
            assumed_kg_per_lap, "assumption", 0,
            [f"Fuel consumption not measurable from data; using configured ASSUMPTION {assumed_kg_per_lap:.3f} kg/lap."],
        )
    return FuelConsumptionEstimate(None, "unavailable", 0, ["No fuel data and no consumption assumption supplied."])
