"""Model calibration and held-out validation for the tyre / lap-time model.

* Data are split by **stint** (or by driver), never by random lap, so
  adjacent correlated laps from one stint cannot appear in both sets.
* The fuel effect is held at the configured value during fitting: within a
  stint tyre age and fuel load are almost perfectly collinear, so they cannot
  both be estimated from the same laps.
* Fits are bounded (non-negative degradation, threshold within the observed
  age range); unconstrained fits are not used.
* If there are too few held-out laps the report states that validation is
  inconclusive. No accuracy figure is ever fabricated.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from racesim.data_validation import assign_stint_ids, representative_lap_mask

logger = logging.getLogger(__name__)


def split_training_validation_data(
    dataframe: pd.DataFrame,
    *,
    method: str = "stint",
    validation_fraction: float = 0.3,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group-wise split of representative laps.

    ``method='stint'`` holds out whole stints (every compound keeps at least
    one training stint where possible); ``method='driver'`` holds out whole
    drivers.
    """
    df = assign_stint_ids(dataframe[representative_lap_mask(dataframe)])
    if df.empty:
        return df, df
    rng = np.random.default_rng(seed)
    if method == "driver":
        if "driver" not in df.columns or df["driver"].nunique() < 2:
            raise ValueError("driver split requires at least two drivers")
        drivers = np.array(sorted(df["driver"].unique()))
        n_val = max(1, int(round(validation_fraction * len(drivers))))
        val_groups = set(rng.choice(drivers, n_val, replace=False))
        mask = df["driver"].isin(val_groups)
    elif method == "stint":
        stints = df.groupby("stint_id")["tyre_compound"].first() if "tyre_compound" in df.columns else df.groupby("stint_id").size()
        val_groups = set()
        if "tyre_compound" in df.columns:
            for comp, ids in stints.groupby(stints).groups.items():
                ids = np.array(sorted(ids))
                if len(ids) >= 2:
                    k = max(1, int(round(validation_fraction * len(ids))))
                    val_groups |= set(rng.choice(ids, min(k, len(ids) - 1), replace=False))
        else:
            ids = np.array(sorted(stints.index))
            val_groups = set(rng.choice(ids, max(1, int(round(validation_fraction * len(ids)))), replace=False))
        mask = df["stint_id"].isin(val_groups)
    else:
        raise ValueError("method must be 'stint' or 'driver'")
    return df[~mask].copy(), df[mask].copy()


@dataclass
class FittedTyreModel:
    model: str
    fuel_effect_s_per_kg: float
    compound_params: dict[str, dict[str, float]]
    driver_offsets_s: dict[str, float]
    n_train: int
    fuel_source: str
    warnings: list[str] = field(default_factory=list)

    def predict(self, dataframe: pd.DataFrame) -> np.ndarray:
        comp = dataframe["tyre_compound"].astype(str).str.upper()
        age = dataframe["tyre_age_laps"].astype(float).to_numpy()
        pred = np.full(len(dataframe), np.nan)
        for name, p in self.compound_params.items():
            sel = (comp == name).to_numpy()
            over = np.maximum(0.0, age[sel] - p["threshold_laps"])
            pred[sel] = p["intercept_s"] + p["linear_s_per_lap"] * age[sel] + p["nonlinear_coefficient"] * over**2
        if "driver" in dataframe.columns:
            pred += dataframe["driver"].astype(str).map(self.driver_offsets_s).fillna(0.0).to_numpy()
        pred += self.fuel_effect_s_per_kg * _fuel(dataframe)
        return pred


def _fuel(df: pd.DataFrame) -> np.ndarray:
    if "fuel_kg" in df.columns and df["fuel_kg"].notna().all():
        return df["fuel_kg"].astype(float).to_numpy()
    return np.zeros(len(df))


def fit_tyre_model(
    train: pd.DataFrame,
    *,
    model: str = "linear",
    fuel_effect_s_per_kg: float = 0.0,
    min_laps_per_compound: int = 8,
) -> FittedTyreModel:
    """Bounded least-squares fit of the additive lap-time model.

    ``lap_time = intercept[c] + driver_offset[d] + fuel_effect * fuel
                 + linear[c] * age (+ nonlinear[c] * max(0, age - threshold[c])**2)``
    """
    if model not in ("linear", "nonlinear"):
        raise ValueError("model must be 'linear' or 'nonlinear'")
    needed = {"lap_time_s", "tyre_age_laps", "tyre_compound"}
    if not needed <= set(train.columns):
        raise ValueError(f"fit requires columns {sorted(needed)}")
    warnings = []
    fuel_source = "fuel_kg column"
    if not ("fuel_kg" in train.columns and train["fuel_kg"].notna().all()):
        fuel_source = "unavailable (fuel effect absorbed into intercept/slope)"
        warnings.append("fuel_kg missing: degradation estimate is confounded with fuel burn.")
    comp = train["tyre_compound"].astype(str).str.upper().to_numpy()
    counts = pd.Series(comp).value_counts()
    compounds = [c for c in sorted(counts.index) if counts[c] >= min_laps_per_compound]
    skipped = [c for c in counts.index if c not in compounds]
    if skipped:
        warnings.append(f"Compounds with too few training laps were not fitted: {skipped}")
    if not compounds:
        raise ValueError("insufficient training data for any compound")
    sel = np.isin(comp, compounds)
    df = train[sel]
    comp = comp[sel]
    y = df["lap_time_s"].astype(float).to_numpy() - fuel_effect_s_per_kg * _fuel(df)
    age = df["tyre_age_laps"].astype(float).to_numpy()
    drivers = sorted(df["driver"].astype(str).unique()) if "driver" in df.columns else []
    d_idx = df["driver"].astype(str).map({d: i for i, d in enumerate(drivers)}).to_numpy() if drivers else np.zeros(len(df), int)
    c_idx = np.array([compounds.index(c) for c in comp])
    nc, nd = len(compounds), max(len(drivers) - 1, 0)
    max_age = float(age.max()) if age.size else 1.0

    def unpack(x):
        intercept = x[:nc]
        lin = x[nc:2 * nc]
        drv = np.concatenate([[0.0], x[2 * nc:2 * nc + nd]]) if drivers else np.zeros(1)
        if model == "nonlinear":
            nl = x[2 * nc + nd:3 * nc + nd]
            thr = x[3 * nc + nd:4 * nc + nd]
        else:
            nl, thr = np.zeros(nc), np.zeros(nc)
        return intercept, lin, drv, nl, thr

    def resid(x):
        intercept, lin, drv, nl, thr = unpack(x)
        over = np.maximum(0.0, age - thr[c_idx])
        return intercept[c_idx] + drv[d_idx] + lin[c_idx] * age + nl[c_idx] * over**2 - y

    x0 = [np.median(y[c_idx == i]) for i in range(nc)] + [0.05] * nc + [0.0] * nd
    lb = [-np.inf] * nc + [0.0] * nc + [-np.inf] * nd
    ub = [np.inf] * nc + [5.0] * nc + [np.inf] * nd
    if model == "nonlinear":
        x0 += [0.001] * nc + [max_age / 2] * nc
        lb += [0.0] * nc + [0.0] * nc
        ub += [1.0] * nc + [max_age] * nc
    sol = least_squares(resid, np.array(x0, float), bounds=(np.array(lb, float), np.array(ub, float)))
    intercept, lin, drv, nl, thr = unpack(sol.x)
    params = {c: {"intercept_s": float(intercept[i]), "linear_s_per_lap": float(lin[i]),
                  "nonlinear_coefficient": float(nl[i]), "threshold_laps": float(thr[i])} for i, c in enumerate(compounds)}
    offsets = {d: float(drv[i]) for i, d in enumerate(drivers)} if drivers else {}
    logger.info("Fitted %s tyre model on %d laps", model, len(df))
    return FittedTyreModel(model, fuel_effect_s_per_kg, params, offsets, int(len(df)), fuel_source, warnings)


def calculate_mae(observed, predicted) -> float:
    """Mean absolute error (s)."""
    return float(np.mean(np.abs(np.asarray(predicted, float) - np.asarray(observed, float))))


def calculate_rmse(observed, predicted) -> float:
    """Root mean squared error (s)."""
    return float(np.sqrt(np.mean((np.asarray(predicted, float) - np.asarray(observed, float)) ** 2)))


def calculate_prediction_bias(observed, predicted) -> float:
    """Mean of (predicted - observed) in seconds; positive = model too slow."""
    return float(np.mean(np.asarray(predicted, float) - np.asarray(observed, float)))


def evaluate_lap_time_predictions(model: FittedTyreModel, validation: pd.DataFrame) -> pd.DataFrame:
    """Observed vs predicted lap times on held-out laps (unfitted compounds dropped)."""
    df = validation[validation["tyre_compound"].astype(str).str.upper().isin(model.compound_params)].copy()
    df["predicted_lap_time_s"] = model.predict(df)
    df["residual_s"] = df["predicted_lap_time_s"] - df["lap_time_s"]
    df["model"] = model.model
    return df


def generate_validation_report(
    dataframe: pd.DataFrame,
    *,
    fuel_effect_s_per_kg: float = 0.0,
    method: str = "stint",
    validation_fraction: float = 0.3,
    seed: int = 0,
    min_validation_samples: int = 30,
) -> dict:
    """Fit linear and nonlinear tyre models on training stints and score them on held-out stints."""
    report: dict = {"status": "inconclusive", "method": method, "models": {}, "warnings": [], "predictions": pd.DataFrame()}
    missing = {"lap_time_s", "tyre_age_laps", "tyre_compound"} - set(dataframe.columns)
    if missing:
        report["warnings"].append(f"Validation unavailable: missing columns {sorted(missing)}.")
        return report
    try:
        train, valid = split_training_validation_data(dataframe, method=method, validation_fraction=validation_fraction, seed=seed)
    except ValueError as exc:
        report["warnings"].append(str(exc))
        return report
    report["n_train"], report["n_validation"] = len(train), len(valid)
    if len(valid) < min_validation_samples or len(train) < min_validation_samples:
        report["warnings"].append(f"Insufficient data: {len(train)} training / {len(valid)} validation laps "
                                  f"(need >= {min_validation_samples} each). Validation is inconclusive.")
        return report
    preds = []
    for kind in ("linear", "nonlinear"):
        try:
            m = fit_tyre_model(train, model=kind, fuel_effect_s_per_kg=fuel_effect_s_per_kg)
        except ValueError as exc:
            report["warnings"].append(f"{kind} fit failed: {exc}")
            continue
        p = evaluate_lap_time_predictions(m, valid)
        if len(p) < min_validation_samples:
            report["warnings"].append(f"{kind}: only {len(p)} comparable validation laps - inconclusive.")
            continue
        report["models"][kind] = {
            "mae_s": calculate_mae(p["lap_time_s"], p["predicted_lap_time_s"]),
            "rmse_s": calculate_rmse(p["lap_time_s"], p["predicted_lap_time_s"]),
            "bias_s": calculate_prediction_bias(p["lap_time_s"], p["predicted_lap_time_s"]),
            "n_validation": int(len(p)),
            "n_train": m.n_train,
            "parameters": m.compound_params,
            "warnings": m.warnings,
        }
        preds.append(p)
    if report["models"]:
        report["status"] = "conclusive"
        report["predictions"] = pd.concat(preds, ignore_index=True)
        if len(report["models"]) == 2:
            lin, nl = report["models"]["linear"]["rmse_s"], report["models"]["nonlinear"]["rmse_s"]
            report["preferred_model"] = "nonlinear" if nl < lin * 0.98 else "linear"
            report["comparison"] = (f"Held-out RMSE linear {lin:.3f} s vs nonlinear {nl:.3f} s; "
                                    f"preferred: {report['preferred_model']} (nonlinear must improve RMSE by >2%).")
    if "data_source" in dataframe.columns and dataframe["data_source"].astype(str).str.contains("SYNTHETIC").any():
        report["warnings"].append("Validation performed on SYNTHETIC data generated by this model: it checks parameter "
                                  "recovery and code correctness, not real-world predictive accuracy.")
    return report
