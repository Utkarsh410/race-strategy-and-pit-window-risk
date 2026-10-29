"""Tyre degradation model.

Model (per compound, all times in seconds)::

    tyre_time(age) = reference_time
                     + m * [ linear * wear
                             + nonlinear * max(0, wear - threshold) ** 2 ]   (nonlinear model only)
                     + warmup_penalty * exp(-age / warmup_decay_laps)          (out-laps only)

* ``age`` is the number of laps already completed on the set at the start of
  the lap (0 on the first lap of a fresh set).
* ``wear`` is the *wear-equivalent* age. Under green flag it equals ``age``;
  under a safety car / VSC each lap adds only a fraction of a lap of wear.
* ``m`` is a degradation multiplier: track-temperature effect
  ``1 + temperature_sensitivity * (T - T_ref)`` times any parameter-uncertainty
  or car-specific multiplier.
* The warm-up penalty is a separate, documented term (see
  :func:`calculate_warmup_penalty`) so it is never folded into the pit loss.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Mapping, Optional

import numpy as np
import pandas as pd

from racesim.schemas import TyreModelConfig

logger = logging.getLogger(__name__)

COMPOUND_TYPE_CODES = {"dry": 0, "intermediate": 1, "wet": 2}

ArrayLike = np.ndarray | float | int


# ---------------------------------------------------------------------------
# Vectorised parameter table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompoundTable:
    """Array view of all compound parameters, indexed by compound index."""

    names: tuple[str, ...]
    reference_lap_time_s: np.ndarray
    linear_s_per_lap: np.ndarray
    nonlinear_coefficient: np.ndarray
    nonlinear_threshold_laps: np.ndarray
    use_nonlinear: np.ndarray
    warmup_penalty_s: np.ndarray
    warmup_decay_laps: np.ndarray
    degradation_uncertainty: np.ndarray
    temperature_sensitivity: np.ndarray
    maximum_recommended_age_laps: np.ndarray
    type_code: np.ndarray

    @classmethod
    def from_compounds(cls, compounds: Mapping[str, TyreModelConfig]) -> "CompoundTable":
        items = list(compounds.values())
        arr = lambda attr: np.array([getattr(c, attr) for c in items], dtype=float)  # noqa: E731
        return cls(
            names=tuple(c.compound_name for c in items),
            reference_lap_time_s=arr("reference_lap_time_s"),
            linear_s_per_lap=arr("linear_degradation_s_per_lap"),
            nonlinear_coefficient=arr("nonlinear_degradation_coefficient"),
            nonlinear_threshold_laps=arr("nonlinear_degradation_threshold_laps"),
            use_nonlinear=np.array([c.degradation_model == "nonlinear" for c in items]),
            warmup_penalty_s=arr("warmup_penalty_s"),
            warmup_decay_laps=arr("warmup_decay_laps"),
            degradation_uncertainty=arr("degradation_uncertainty"),
            temperature_sensitivity=arr("temperature_sensitivity"),
            maximum_recommended_age_laps=arr("maximum_recommended_age_laps"),
            type_code=np.array([COMPOUND_TYPE_CODES[c.compound_type] for c in items], dtype=int),
        )

    def index(self, name: str) -> int:
        try:
            return self.names.index(name.strip().upper())
        except ValueError as exc:
            raise KeyError(f"unknown tyre compound {name!r}; configured: {list(self.names)}") from exc

    def __len__(self) -> int:
        return len(self.names)


# ---------------------------------------------------------------------------
# Core model functions
# ---------------------------------------------------------------------------


def calculate_tyre_age(stint_start_lap: ArrayLike, lap: ArrayLike, starting_tyre_age_laps: ArrayLike = 0) -> ArrayLike:
    """Tyre age (laps completed on the set) at the start of ``lap``.

    For a stint starting on ``stint_start_lap`` with a set of age
    ``starting_tyre_age_laps``, the first lap has exactly that age and each
    subsequent lap adds one. A tyre change resets the age to the new set's
    starting age (0 for new tyres).
    """
    age = np.asarray(starting_tyre_age_laps) + (np.asarray(lap) - np.asarray(stint_start_lap))
    if np.any(age < 0):
        raise ValueError("lap precedes the stint start; tyre age would be negative")
    return age if isinstance(age, np.ndarray) and age.ndim else int(age)


def temperature_multiplier(track_temperature_c: ArrayLike, reference_temperature_c: float, sensitivity: ArrayLike) -> ArrayLike:
    """Degradation-rate multiplier from track temperature (floored at 0)."""
    return np.maximum(0.0, 1.0 + np.asarray(sensitivity) * (np.asarray(track_temperature_c) - reference_temperature_c))


def calculate_degradation(
    wear_laps: ArrayLike,
    linear_s_per_lap: ArrayLike,
    nonlinear_coefficient: ArrayLike = 0.0,
    nonlinear_threshold_laps: ArrayLike = 0.0,
    use_nonlinear: ArrayLike = True,
    multiplier: ArrayLike = 1.0,
) -> np.ndarray:
    """Age-related time loss (s) relative to a fresh tyre.

    Linear model: ``linear * wear``. Nonlinear model adds
    ``nonlinear * max(0, wear - threshold)**2``.
    """
    wear = np.asarray(wear_laps, dtype=float)
    lin = np.asarray(linear_s_per_lap) * wear
    over = np.maximum(0.0, wear - np.asarray(nonlinear_threshold_laps))
    nonlin = np.where(np.asarray(use_nonlinear), np.asarray(nonlinear_coefficient) * over**2, 0.0)
    return np.asarray(multiplier) * (lin + nonlin)


def calculate_warmup_penalty(
    tyre_age_laps: ArrayLike,
    warmup_penalty_s: ArrayLike,
    warmup_decay_laps: ArrayLike,
    applies: ArrayLike = True,
) -> np.ndarray:
    """Out-lap warm-up penalty (s): ``penalty * exp(-age / decay)``.

    The penalty is largest on the first lap of a new set (age 0) and decays
    exponentially. ``applies`` is ``False`` for the race-start stint, whose
    tyres are pre-conditioned (the standing start is modelled separately).
    """
    age = np.asarray(tyre_age_laps, dtype=float)
    pen = np.asarray(warmup_penalty_s) * np.exp(-age / np.asarray(warmup_decay_laps))
    return np.where(np.asarray(applies), pen, 0.0)


def tyre_time_effect(
    table: CompoundTable,
    compound_idx: ArrayLike,
    tyre_age_laps: ArrayLike,
    wear_laps: ArrayLike,
    track_temperature_c: ArrayLike,
    reference_temperature_c: float,
    warmup_applies: ArrayLike = True,
    degradation_multiplier: ArrayLike = 1.0,
) -> np.ndarray:
    """Tyre contribution (degradation + warm-up) excluding the compound reference."""
    c = np.asarray(compound_idx, dtype=int)
    mult = np.asarray(degradation_multiplier) * temperature_multiplier(
        track_temperature_c, reference_temperature_c, table.temperature_sensitivity[c]
    )
    deg = calculate_degradation(
        wear_laps,
        table.linear_s_per_lap[c],
        table.nonlinear_coefficient[c],
        table.nonlinear_threshold_laps[c],
        table.use_nonlinear[c],
        mult,
    )
    warm = calculate_warmup_penalty(tyre_age_laps, table.warmup_penalty_s[c], table.warmup_decay_laps[c], warmup_applies)
    return deg + warm


def predict_tyre_pace(
    compound: TyreModelConfig,
    tyre_age_laps: ArrayLike,
    *,
    wear_laps: Optional[ArrayLike] = None,
    track_temperature_c: Optional[float] = None,
    reference_temperature_c: float = 35.0,
    include_warmup: bool = False,
    degradation_multiplier: float = 1.0,
) -> np.ndarray:
    """Predicted lap time (s) of a compound under reference fuel conditions.

    Returns ``reference_lap_time_s + degradation (+ warm-up)``.
    """
    table = CompoundTable.from_compounds({compound.compound_name: compound})
    age = np.asarray(tyre_age_laps, dtype=float)
    temp = reference_temperature_c if track_temperature_c is None else track_temperature_c
    effect = tyre_time_effect(
        table,
        np.zeros(age.shape, dtype=int),
        age,
        age if wear_laps is None else wear_laps,
        temp,
        reference_temperature_c,
        include_warmup,
        degradation_multiplier,
    )
    return compound.reference_lap_time_s + effect


def calculate_tyre_life(
    compound: TyreModelConfig,
    *,
    loss_threshold_s: float = 1.5,
    track_temperature_c: Optional[float] = None,
    reference_temperature_c: float = 35.0,
    max_laps: int = 200,
) -> dict[str, float | int | str]:
    """Usable tyre life.

    ``performance_life_laps`` is the first age at which degradation relative to
    a fresh tyre exceeds ``loss_threshold_s`` (or ``max_laps``). The configured
    ``maximum_recommended_age_laps`` is reported separately; the recommended
    life is the minimum of the two.
    """
    ages = np.arange(max_laps + 1)
    pace = predict_tyre_pace(
        compound, ages, track_temperature_c=track_temperature_c, reference_temperature_c=reference_temperature_c
    )
    loss = pace - pace[0]
    over = np.nonzero(loss > loss_threshold_s)[0]
    perf_life = int(over[0]) if over.size else max_laps
    return {
        "compound": compound.compound_name,
        "performance_life_laps": perf_life,
        "maximum_recommended_age_laps": compound.maximum_recommended_age_laps,
        "recommended_life_laps": min(perf_life, compound.maximum_recommended_age_laps),
        "loss_threshold_s": loss_threshold_s,
    }


def compare_tyre_compounds(
    compounds: Mapping[str, TyreModelConfig],
    *,
    max_age_laps: int = 40,
    reference_temperature_c: float = 35.0,
    track_temperature_c: Optional[float] = None,
    include_warmup: bool = False,
    dry_only: bool = True,
    limit_to_recommended_age: bool = False,
) -> pd.DataFrame:
    """Long-format table of predicted pace by age for each compound.

    With ``limit_to_recommended_age`` each curve stops at the compound's
    ``maximum_recommended_age_laps``.
    """
    rows = []
    for name, comp in compounds.items():
        if dry_only and comp.compound_type != "dry":
            continue
        top = min(max_age_laps, comp.maximum_recommended_age_laps) if limit_to_recommended_age else max_age_laps
        ages = np.arange(top + 1)
        pace = predict_tyre_pace(
            comp,
            ages,
            track_temperature_c=track_temperature_c,
            reference_temperature_c=reference_temperature_c,
            include_warmup=include_warmup,
        )
        cum = np.cumsum(pace)
        for a, p, cs in zip(ages, pace, cum):
            rows.append(
                {
                    "compound": name,
                    "tyre_age_laps": int(a),
                    "predicted_lap_time_s": float(p),
                    "degradation_s": float(p - comp.reference_lap_time_s),
                    "mean_stint_lap_time_s": float(cs / (a + 1)),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Estimation from data
# ---------------------------------------------------------------------------


@dataclass
class DegradationEstimate:
    """Result of fitting a linear degradation slope from lap data."""

    compound: str
    linear_degradation_s_per_lap: Optional[float]
    intercept_s: Optional[float]
    standard_error_s_per_lap: Optional[float]
    sample_count: int
    stint_count: int
    r_squared: Optional[float]
    reliable: bool
    fuel_corrected: bool
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def _drop_warmup_laps(df: pd.DataFrame, exclude_warmup_laps: int) -> pd.DataFrame:
    """Drop early laps of stints that began after a pit stop (warm-up affected)."""
    if exclude_warmup_laps <= 0 or df.empty:
        return df
    later_stint = ~df["stint_id"].astype(str).str.endswith("_S1")
    return df[~(later_stint & (df["tyre_age_laps"] < exclude_warmup_laps))]


def _model_laps(df: pd.DataFrame) -> pd.DataFrame:
    """Rows suitable for pace modelling: green, not pit in/out, no traffic flag."""
    from racesim.data_validation import representative_lap_mask

    return df.loc[representative_lap_mask(df)].copy()


def fuel_corrected_lap_times(
    df: pd.DataFrame,
    fuel_effect_s_per_kg: Optional[float],
    fuel_consumption_kg_per_lap: Optional[float] = None,
) -> tuple[Optional[pd.Series], list[str]]:
    """Remove the fuel effect from lap times.

    Uses the ``fuel_kg`` column when available; otherwise the configured
    consumption assumption (only the lap-dependent part matters for slopes).
    Returns ``(None, warnings)`` if no correction is possible.
    """
    warnings: list[str] = []
    if fuel_effect_s_per_kg is None:
        return None, ["No fuel effect supplied: degradation estimate is confounded with fuel burn."]
    if "fuel_kg" in df.columns and df["fuel_kg"].notna().all():
        return df["lap_time_s"] - fuel_effect_s_per_kg * df["fuel_kg"], warnings
    if fuel_consumption_kg_per_lap is not None and "lap" in df.columns:
        warnings.append(
            "fuel_kg column unavailable: fuel correction uses the configured consumption ASSUMPTION "
            f"({fuel_consumption_kg_per_lap:.3f} kg/lap); absolute intercept is not zero-fuel referenced."
        )
        return df["lap_time_s"] + fuel_effect_s_per_kg * fuel_consumption_kg_per_lap * (df["lap"] - 1), warnings
    return None, ["Fuel load unknown and no consumption assumption supplied: estimate is confounded with fuel burn."]


def estimate_tyre_degradation(
    dataframe: pd.DataFrame,
    *,
    fuel_effect_s_per_kg: Optional[float] = None,
    fuel_consumption_kg_per_lap: Optional[float] = None,
    min_samples: int = 12,
    min_stints: int = 1,
    exclude_warmup_laps: int = 3,
) -> dict[str, DegradationEstimate]:
    """Estimate a linear degradation slope per compound.

    The fit is a *within-stint* regression (stint fixed effects) of
    fuel-corrected lap time on tyre age, using only representative laps
    (green flag, no pit in/out laps, no flagged traffic). This removes
    stint-to-stint offsets such as driver changes or track evolution between
    stints. Laps with tyre age below ``exclude_warmup_laps`` in stints that
    start after a pit stop are dropped, because the out-lap warm-up penalty
    would otherwise bias the slope downwards. Remaining confounders are reported as warnings; the estimate is
    flagged unreliable when data are insufficient or the slope is negative.
    """
    from racesim.data_validation import assign_stint_ids

    required = {"lap_time_s", "tyre_age_laps", "tyre_compound"}
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(f"degradation estimation needs columns {sorted(missing)}")

    df = _drop_warmup_laps(assign_stint_ids(_model_laps(dataframe)), exclude_warmup_laps)
    df = df[np.isfinite(df["lap_time_s"]) & df["tyre_age_laps"].notna()]
    results: dict[str, DegradationEstimate] = {}
    for compound, grp in df.groupby(df["tyre_compound"].astype(str).str.upper()):
        y, warnings = fuel_corrected_lap_times(grp, fuel_effect_s_per_kg, fuel_consumption_kg_per_lap)
        fuel_corrected = y is not None
        if y is None:
            y = grp["lap_time_s"]
        x = grp["tyre_age_laps"].astype(float)
        stints = grp["stint_id"]
        n, n_stints = len(grp), stints.nunique()
        x_d = x - x.groupby(stints).transform("mean")
        y_d = y - y.groupby(stints).transform("mean")
        sxx = float((x_d**2).sum())
        if n < min_samples or n_stints < min_stints or sxx <= 0:
            warnings.append(
                f"Insufficient data for {compound}: {n} representative laps in {n_stints} stint(s) "
                f"(need >= {min_samples} laps with tyre-age variation)."
            )
            results[compound] = DegradationEstimate(compound, None, None, None, n, n_stints, None, False, fuel_corrected, warnings)
            continue
        slope = float((x_d * y_d).sum() / sxx)
        resid = y_d - slope * x_d
        dof = max(n - n_stints - 1, 1)
        se = float(np.sqrt((resid**2).sum() / dof / sxx))
        sst = float((y_d**2).sum())
        r2 = 1 - float((resid**2).sum()) / sst if sst > 0 else None
        intercept = float((y - slope * x).mean())
        reliable = fuel_corrected
        if slope < 0:
            warnings.append(f"Negative slope for {compound} ({slope:.4f} s/lap): physically implausible, likely confounded.")
            reliable = False
        if se > max(abs(slope), 1e-9):
            warnings.append(f"Standard error for {compound} exceeds the slope magnitude: estimate is not significant.")
            reliable = False
        if "driver" in grp.columns and grp["driver"].nunique() > 1:
            warnings.append(f"{compound}: data pooled across {grp['driver'].nunique()} drivers (offsets absorbed per stint).")
        results[compound] = DegradationEstimate(
            compound, slope, intercept, se, n, n_stints, r2, reliable, fuel_corrected, warnings
        )
        logger.info("Degradation estimate %s: %.4f +/- %.4f s/lap (n=%d)", compound, slope, se, n)
    return results


def estimate_degradation_uncertainty(
    dataframe: pd.DataFrame,
    compound: str,
    *,
    fuel_effect_s_per_kg: Optional[float] = None,
    fuel_consumption_kg_per_lap: Optional[float] = None,
    n_bootstrap: int = 500,
    seed: int = 0,
) -> dict[str, Optional[float] | int | str]:
    """Stint-level bootstrap of the degradation slope for one compound.

    Stints (not individual laps) are resampled, so adjacent correlated laps
    are not treated as independent observations. Returns the bootstrap
    standard deviation and a 90 % percentile interval, or an
    ``insufficient-data`` status when fewer than two stints are available.
    """
    from racesim.data_validation import assign_stint_ids

    df = assign_stint_ids(_model_laps(dataframe))
    df = df[df["tyre_compound"].astype(str).str.upper() == compound.upper()]
    stint_ids = df["stint_id"].unique()
    if len(stint_ids) < 2:
        return {"compound": compound, "status": "insufficient-data", "stints": int(len(stint_ids)),
                "bootstrap_std_s_per_lap": None, "ci05": None, "ci95": None}
    rng = np.random.default_rng(seed)
    groups = {sid: g for sid, g in df.groupby("stint_id")}
    slopes = []
    for _ in range(n_bootstrap):
        pick = rng.choice(stint_ids, size=len(stint_ids), replace=True)
        sample = pd.concat([groups[s].assign(stint_id=f"{s}#{i}") for i, s in enumerate(pick)])
        est = estimate_tyre_degradation(
            sample,
            fuel_effect_s_per_kg=fuel_effect_s_per_kg,
            fuel_consumption_kg_per_lap=fuel_consumption_kg_per_lap,
            min_samples=3,
        ).get(compound.upper())
        if est is not None and est.linear_degradation_s_per_lap is not None:
            slopes.append(est.linear_degradation_s_per_lap)
    if len(slopes) < 10:
        return {"compound": compound, "status": "insufficient-data", "stints": int(len(stint_ids)),
                "bootstrap_std_s_per_lap": None, "ci05": None, "ci95": None}
    s = np.asarray(slopes)
    return {
        "compound": compound,
        "status": "ok",
        "stints": int(len(stint_ids)),
        "bootstrap_std_s_per_lap": float(s.std(ddof=1)),
        "ci05": float(np.percentile(s, 5)),
        "ci95": float(np.percentile(s, 95)),
    }
