"""Transparent additive lap-time model and lap-data pace analysis.

Model (seconds)::

    lap_time = reference_lap_time        compound reference (empty tank, fresh tyre,
                                         reference track temperature) + car pace offset
             + tyre_effect               degradation(wear) * temperature/uncertainty
                                         multipliers + out-lap warm-up
             + fuel_effect               fuel_effect_s_per_kg * fuel on board at lap start
             + traffic_effect            sampled / interaction traffic loss
             + weather_effect            wetness-vs-compound suitability penalty
             + track_condition_effect    -min(evolution * (lap - 1), cap) + standing start on lap 1
             + driver_effect             driver offset
             + random_error              0 in deterministic mode

Each effect appears exactly once. Track temperature only affects the
degradation *rate* (inside ``tyre_effect``), never the base pace, so it is
not double counted. Results are floored at
``minimum_lap_time_fraction * baseline_lap_time_s`` to prevent physically
impossible lap times.

:func:`predict_lap_time` is the single canonical implementation used by the
race simulator, the strategy optimiser and the Monte Carlo engine.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
from scipy import stats

from racesim.fuel_model import calculate_fuel_time_effect
from racesim.tyre_model import CompoundTable, tyre_time_effect
from racesim.weather_model import calculate_weather_pace_effect

if TYPE_CHECKING:
    from racesim.config import SimulatorSetup
    from racesim.schemas import WeatherConfig

logger = logging.getLogger(__name__)

ArrayLike = np.ndarray | float | int


@dataclass(frozen=True)
class LapModelContext:
    """Constants of the lap-time model derived from a :class:`SimulatorSetup`."""

    table: CompoundTable
    baseline_lap_time_s: float
    reference_temperature_c: float
    fuel_effect_s_per_kg: float
    track_evolution_s_per_lap: float
    track_evolution_cap_s: float
    standing_start_loss_s: float
    minimum_lap_time_s: float
    weather: "WeatherConfig"

    @classmethod
    def from_setup(cls, setup: "SimulatorSetup") -> "LapModelContext":
        r = setup.race
        return cls(
            table=CompoundTable.from_compounds(setup.compounds),
            baseline_lap_time_s=r.baseline_lap_time_s,
            reference_temperature_c=r.reference_track_temperature_c,
            fuel_effect_s_per_kg=r.fuel_effect_s_per_kg,
            track_evolution_s_per_lap=r.track_evolution_s_per_lap,
            track_evolution_cap_s=r.track_evolution_cap_s,
            standing_start_loss_s=r.standing_start_loss_s,
            minimum_lap_time_s=r.minimum_lap_time_fraction * r.baseline_lap_time_s,
            weather=setup.weather,
        )


def calculate_track_condition_effect(lap: ArrayLike, rate_s_per_lap: float, cap_s: float, standing_start_loss_s: float = 0.0) -> np.ndarray:
    """Track evolution (negative = faster) plus the lap-1 standing-start loss."""
    lap = np.asarray(lap)
    evo = -np.minimum(rate_s_per_lap * (lap - 1), cap_s)
    return evo + np.where(lap == 1, standing_start_loss_s, 0.0)


def calculate_lap_time(
    reference_lap_time_s: ArrayLike,
    tyre_effect_s: ArrayLike = 0.0,
    fuel_effect_s: ArrayLike = 0.0,
    traffic_effect_s: ArrayLike = 0.0,
    weather_effect_s: ArrayLike = 0.0,
    track_condition_effect_s: ArrayLike = 0.0,
    driver_effect_s: ArrayLike = 0.0,
    random_error_s: ArrayLike = 0.0,
    *,
    minimum_lap_time_s: Optional[float] = None,
) -> np.ndarray:
    """Sum the additive effects (s) and apply the physical floor."""
    total = (
        np.asarray(reference_lap_time_s, dtype=float)
        + tyre_effect_s
        + fuel_effect_s
        + traffic_effect_s
        + weather_effect_s
        + track_condition_effect_s
        + driver_effect_s
        + random_error_s
    )
    floor = 1e-3 if minimum_lap_time_s is None else minimum_lap_time_s
    if np.any(total < floor):
        logger.debug("Clipped %d lap time(s) at the physical floor %.3f s", int(np.sum(total < floor)), floor)
    return np.maximum(total, floor)


def predict_lap_time(
    ctx: LapModelContext,
    compound_idx: ArrayLike,
    lap: ArrayLike,
    tyre_age_laps: ArrayLike,
    wear_laps: ArrayLike,
    fuel_kg: ArrayLike,
    warmup_applies: ArrayLike,
    track_temperature_c: ArrayLike,
    wetness: ArrayLike = 0.0,
    *,
    pace_offset_s: ArrayLike = 0.0,
    degradation_multiplier: ArrayLike = 1.0,
    traffic_s: ArrayLike = 0.0,
    driver_s: ArrayLike = 0.0,
    random_error_s: ArrayLike = 0.0,
    return_components: bool = False,
):
    """Canonical green-flag lap-time prediction (vectorised).

    Returns the total lap time, or ``(total, components)`` where
    ``components`` holds each additive term (s).
    """
    c = np.asarray(compound_idx, dtype=int)
    reference = ctx.table.reference_lap_time_s[c] + pace_offset_s
    tyre = tyre_time_effect(
        ctx.table, c, tyre_age_laps, wear_laps, track_temperature_c, ctx.reference_temperature_c, warmup_applies, degradation_multiplier
    )
    fuel = calculate_fuel_time_effect(fuel_kg, ctx.fuel_effect_s_per_kg)
    weather = calculate_weather_pace_effect(wetness, ctx.table.type_code[c], ctx.weather)
    track = calculate_track_condition_effect(lap, ctx.track_evolution_s_per_lap, ctx.track_evolution_cap_s, ctx.standing_start_loss_s)
    total = calculate_lap_time(
        reference, tyre, fuel, traffic_s, weather, track, driver_s, random_error_s, minimum_lap_time_s=ctx.minimum_lap_time_s
    )
    if not return_components:
        return total
    return total, {
        "reference_s": reference,
        "tyre_s": tyre,
        "fuel_s": fuel,
        "traffic_s": np.asarray(traffic_s, dtype=float),
        "weather_s": weather,
        "track_s": track,
        "driver_s": np.asarray(driver_s, dtype=float),
        "random_s": np.asarray(random_error_s, dtype=float),
    }


# ---------------------------------------------------------------------------
# Lap-data analysis
# ---------------------------------------------------------------------------


def _representative(df: pd.DataFrame) -> pd.DataFrame:
    from racesim.data_validation import representative_lap_mask

    return df.loc[representative_lap_mask(df)]


def estimate_baseline_pace(
    dataframe: pd.DataFrame,
    *,
    fuel_effect_s_per_kg: Optional[float] = None,
    degradation_s_per_lap: Optional[dict[str, float]] = None,
) -> dict:
    """Estimate representative pace from lap data.

    Reference conditions of the returned ``baseline_lap_time_s``:
    representative (green, no pit, no flagged traffic) laps, corrected to an
    empty tank when ``fuel_kg`` and ``fuel_effect_s_per_kg`` are available, and
    to tyre age 0 when per-compound degradation slopes are supplied. Any
    correction that could not be applied is listed in ``warnings``.
    """
    warnings: list[str] = []
    df = _representative(dataframe)
    if df.empty:
        return {"baseline_lap_time_s": None, "sample_count": 0, "reference_conditions": "n/a",
                "warnings": ["No representative laps available."]}
    t = df["lap_time_s"].astype(float).copy()
    cond = []
    if fuel_effect_s_per_kg is not None and "fuel_kg" in df.columns and df["fuel_kg"].notna().all():
        t = t - fuel_effect_s_per_kg * df["fuel_kg"]
        cond.append("empty tank")
    else:
        warnings.append("Fuel correction not applied: pace is at the observed (unknown) fuel loads.")
        cond.append("observed fuel load")
    if degradation_s_per_lap and {"tyre_age_laps", "tyre_compound"} <= set(df.columns):
        slope = df["tyre_compound"].astype(str).str.upper().map(degradation_s_per_lap).fillna(0.0)
        t = t - slope * df["tyre_age_laps"].astype(float)
        cond.append("tyre age 0 (supplied linear slopes)")
    else:
        warnings.append("Tyre-age correction not applied.")
        cond.append("observed tyre ages")
    by_compound = None
    if "tyre_compound" in df.columns:
        by_compound = t.groupby(df["tyre_compound"].astype(str).str.upper()).median().to_dict()
    return {
        "baseline_lap_time_s": float(t.median()),
        "sample_count": int(len(t)),
        "reference_conditions": ", ".join(cond),
        "by_compound_s": by_compound,
        "warnings": warnings,
    }


SECTOR_COLUMNS = ("sector1_s", "sector2_s", "sector3_s")


def calculate_sector_statistics(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Per-driver sector statistics (mean, median, best, std) and theoretical best.

    Returns an empty frame when no sector columns are present.
    """
    sectors = [c for c in SECTOR_COLUMNS if c in dataframe.columns]
    if not sectors:
        return pd.DataFrame()
    df = _representative(dataframe)
    group = df["driver"] if "driver" in df.columns else pd.Series("ALL", index=df.index, name="driver")
    rows = []
    for drv, g in df.groupby(group):
        row = {"driver": drv}
        for s in sectors:
            v = g[s].dropna()
            row.update({f"{s}_mean": v.mean(), f"{s}_median": v.median(), f"{s}_best": v.min(), f"{s}_std": v.std()})
        if len(sectors) == 3:
            row["theoretical_best_s"] = sum(row[f"{s}_best"] for s in sectors)
            row["best_lap_s"] = g["lap_time_s"].min()
        rows.append(row)
    return pd.DataFrame(rows)


def calculate_stint_pace(dataframe: pd.DataFrame, *, fuel_effect_s_per_kg: Optional[float] = None) -> pd.DataFrame:
    """Per-stint pace summary using representative laps.

    Includes the raw lap-time slope (s/lap) and, when fuel data allow, a
    fuel-corrected slope, which is the better proxy for tyre degradation.
    """
    from racesim.data_validation import assign_stint_ids

    df = assign_stint_ids(_representative(dataframe))
    rows = []
    for sid, g in df.groupby("stint_id", sort=False):
        g = g.sort_values("lap")
        row = {
            "stint_id": sid,
            "driver": g["driver"].iloc[0] if "driver" in g.columns else "ALL",
            "compound": g["tyre_compound"].iloc[0] if "tyre_compound" in g.columns else None,
            "first_lap": int(g["lap"].min()),
            "last_lap": int(g["lap"].max()),
            "laps": int(len(g)),
            "mean_lap_time_s": g["lap_time_s"].mean(),
            "median_lap_time_s": g["lap_time_s"].median(),
            "best_lap_time_s": g["lap_time_s"].min(),
            "std_lap_time_s": g["lap_time_s"].std(),
            "raw_slope_s_per_lap": None,
            "fuel_corrected_slope_s_per_lap": None,
        }
        if len(g) >= 3:
            row["raw_slope_s_per_lap"] = float(np.polyfit(g["lap"], g["lap_time_s"], 1)[0])
            if fuel_effect_s_per_kg is not None and "fuel_kg" in g.columns and g["fuel_kg"].notna().all():
                y = g["lap_time_s"] - fuel_effect_s_per_kg * g["fuel_kg"]
                row["fuel_corrected_slope_s_per_lap"] = float(np.polyfit(g["lap"], y, 1)[0])
        rows.append(row)
    return pd.DataFrame(rows)


def detect_pace_trend(dataframe: pd.DataFrame, *, significance: float = 0.05) -> pd.DataFrame:
    """Linear trend of representative lap time against lap number per driver."""
    df = _representative(dataframe)
    group = df["driver"] if "driver" in df.columns else pd.Series("ALL", index=df.index)
    rows = []
    for drv, g in df.groupby(group):
        if len(g) < 5:
            rows.append({"driver": drv, "slope_s_per_lap": None, "p_value": None, "trend": "insufficient data", "laps": len(g)})
            continue
        res = stats.linregress(g["lap"].astype(float), g["lap_time_s"].astype(float))
        trend = "no significant trend"
        if res.pvalue < significance:
            trend = "slowing" if res.slope > 0 else "improving"
        rows.append({"driver": drv, "slope_s_per_lap": float(res.slope), "p_value": float(res.pvalue), "trend": trend, "laps": len(g)})
    return pd.DataFrame(rows)


def estimate_pace_uncertainty(dataframe: pd.DataFrame) -> dict:
    """Lap-to-lap variability after removing within-stint linear trends.

    Removing each stint's linear trend prevents tyre degradation and fuel burn
    from being mistaken for random variation. Returns the residual standard
    deviation and a robust (MAD-based) estimate in seconds.
    """
    from racesim.data_validation import assign_stint_ids

    df = assign_stint_ids(_representative(dataframe))
    resid = []
    for _, g in df.groupby("stint_id"):
        if len(g) >= 4:
            coef = np.polyfit(g["lap"].astype(float), g["lap_time_s"].astype(float), 1)
            resid.append(g["lap_time_s"].to_numpy() - np.polyval(coef, g["lap"].to_numpy()))
    if not resid:
        return {"lap_time_std_s": None, "robust_std_s": None, "sample_count": 0,
                "warning": "Insufficient stint data to estimate lap-to-lap variability."}
    r = np.concatenate(resid)
    mad = float(np.median(np.abs(r - np.median(r))) * 1.4826)
    return {"lap_time_std_s": float(r.std(ddof=1)), "robust_std_s": mad, "sample_count": int(r.size), "warning": None}
