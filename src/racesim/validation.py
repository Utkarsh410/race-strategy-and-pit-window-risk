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
    training_errors_s: np.ndarray = field(default_factory=lambda: np.zeros(0))  # observed - predicted on training laps

    def prediction_interval(self, dataframe: pd.DataFrame, level: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
        """Empirical prediction interval from training-residual quantiles.

        Distribution-free but optimistic: training residuals are in-sample,
        so held-out coverage is expected to fall short of ``level`` when the
        model does not generalise. Coverage is therefore *measured*, not assumed.
        """
        if self.training_errors_s.size < 10:
            raise ValueError("too few training residuals for a prediction interval")
        pred = self.predict(dataframe)
        lo, hi = np.quantile(self.training_errors_s, [(1 - level) / 2, 1 - (1 - level) / 2])
        return pred + lo, pred + hi

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
    training_errors = -resid(sol.x)
    intercept, lin, drv, nl, thr = unpack(sol.x)
    params = {c: {"intercept_s": float(intercept[i]), "linear_s_per_lap": float(lin[i]),
                  "nonlinear_coefficient": float(nl[i]), "threshold_laps": float(thr[i])} for i, c in enumerate(compounds)}
    offsets = {d: float(drv[i]) for i, d in enumerate(drivers)} if drivers else {}
    logger.info("Fitted %s tyre model on %d laps", model, len(df))
    return FittedTyreModel(model, fuel_effect_s_per_kg, params, offsets, int(len(df)), fuel_source, warnings, training_errors)


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
    interval_level: float = 0.9,
) -> dict:
    """Fit linear and nonlinear tyre models on training stints and score them on held-out stints.

    ``evidence_type`` states what the result can support: synthetic data only
    test self-consistency; ``historical_out_of_sample`` requires data that are
    labelled as non-synthetic. Mixed inputs are rejected rather than pooled.
    """
    report: dict = {"status": "inconclusive", "method": method, "models": {}, "warnings": [], "predictions": pd.DataFrame(),
                    "evidence_type": classify_evidence(dataframe), "interval_level": interval_level}
    if report["evidence_type"] == "mixed":
        report["warnings"].append("Dataset mixes SYNTHETIC and non-synthetic rows; validate them separately "
                                  "(use run_model_validation, which splits them). No metrics reported.")
        return report
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
    report["leakage_check"] = check_split_leakage(train, valid)
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
        coverage = None
        try:
            lo, hi = m.prediction_interval(p, interval_level)
            p["interval_low_s"], p["interval_high_s"] = lo, hi
            coverage = calculate_interval_coverage(p["lap_time_s"], lo, hi)
        except ValueError as exc:
            report["warnings"].append(f"{kind}: prediction interval unavailable ({exc}).")
        unseen = sorted(set(p["driver"].astype(str)) - set(m.driver_offsets_s)) if "driver" in p.columns and m.driver_offsets_s else []
        if unseen:
            report["warnings"].append(f"{kind}: drivers {unseen} not in training data; their offset is assumed 0 s.")
        report["models"][kind] = {
            "interval_coverage": coverage,
            "residual_std_s": float(np.std(m.training_errors_s, ddof=1)) if m.training_errors_s.size > 1 else None,
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
        report["residual_analysis"] = analyse_residuals(report["predictions"])
    if report["evidence_type"] == "synthetic_self_consistency":
        report["warnings"].append("Validation performed on SYNTHETIC data generated by this model: it checks parameter "
                                  "recovery and code correctness, not real-world predictive accuracy.")
    elif report["evidence_type"] == "unlabelled":
        report["warnings"].append("Data have no data_source column: provenance is unverified, so results are reported as "
                                  "'unlabelled' rather than historical evidence.")
    return report


# ---------------------------------------------------------------------------
# v2: evidence classification, leakage checks, coverage, residuals, calibration
# ---------------------------------------------------------------------------

EVIDENCE_TYPES = {
    "synthetic_self_consistency": "Synthetic data from RaceSim's own model: tests code and parameter recovery only.",
    "historical_out_of_sample": "Data labelled as recorded (non-synthetic) race data, scored on held-out stints.",
    "unlabelled": "No data_source label: provenance unverified; not presented as historical evidence.",
    "mixed": "Synthetic and non-synthetic rows mixed: must be validated separately.",
}


def classify_evidence(dataframe: pd.DataFrame) -> str:
    """Classify what kind of evidence a validation on ``dataframe`` provides."""
    if "data_source" not in dataframe.columns or dataframe["data_source"].isna().all():
        return "unlabelled"
    src = dataframe["data_source"].astype(str).str.upper()
    synthetic = src.str.contains("SYNTHETIC")
    if synthetic.all():
        return "synthetic_self_consistency"
    if synthetic.any():
        return "mixed"
    if src.isin(["UNLABELLED", "UNKNOWN", "NAN", ""]).all():
        return "unlabelled"
    return "historical_out_of_sample"


def check_split_leakage(train: pd.DataFrame, valid: pd.DataFrame) -> dict:
    """Verify that no stint appears in both sets and count neighbouring laps across the split.

    ``adjacent_cross_split_laps`` counts validation laps whose immediate
    neighbour (same driver, lap +/- 1) is in training. Under a stint split these
    can only be stint boundaries (pit laps are excluded), never laps inside a stint.
    """
    shared = sorted(set(train.get("stint_id", pd.Series(dtype=str))) & set(valid.get("stint_id", pd.Series(dtype=str))))
    adjacent = 0
    if {"driver", "lap"} <= set(train.columns) and {"driver", "lap"} <= set(valid.columns):
        tr = set(zip(train["driver"].astype(str), train["lap"].astype(int)))
        for d, lap in zip(valid["driver"].astype(str), valid["lap"].astype(int)):
            adjacent += int((d, lap - 1) in tr or (d, lap + 1) in tr)
    return {"shared_stints": shared, "leakage_free": not shared, "adjacent_cross_split_laps": adjacent}


def calculate_interval_coverage(observed, lower, upper) -> float:
    """Fraction of observations inside ``[lower, upper]``."""
    o = np.asarray(observed, float)
    return float(np.mean((o >= np.asarray(lower, float)) & (o <= np.asarray(upper, float))))


def analyse_residuals(predictions: pd.DataFrame) -> dict:
    """Residual diagnostics (residual = predicted - observed, seconds).

    Includes summaries by model/compound, by tyre-age band and by driver, the
    slope of residual against tyre age (non-zero means the degradation shape is
    wrong) and the within-stint lag-1 autocorrelation (positive means laps are
    not independent, so naive lap-level error bars would be too narrow).
    """
    if predictions is None or predictions.empty:
        return {}
    df = predictions.copy()
    df["age_band"] = pd.cut(df["tyre_age_laps"].astype(float), bins=[-0.5, 5.5, 10.5, 20.5, 30.5, 1e9],
                            labels=["0-5", "6-10", "11-20", "21-30", "31+"])
    agg = lambda g: g.agg(n=("residual_s", "size"), mean_s=("residual_s", "mean"), std_s=("residual_s", "std")).reset_index()  # noqa: E731
    out: dict = {
        "by_compound": agg(df.groupby(["model", "tyre_compound"])),
        "by_age_band": agg(df.groupby(["model", "age_band"], observed=True)),
    }
    if "driver" in df.columns:
        out["by_driver"] = agg(df.groupby(["model", "driver"]))
    rows = []
    for m, g in df.groupby("model"):
        slope = float(np.polyfit(g["tyre_age_laps"].astype(float), g["residual_s"], 1)[0]) if len(g) >= 3 else None
        ac = []
        if "stint_id" in g.columns:
            for _, st in g.sort_values("lap").groupby("stint_id"):
                r = st["residual_s"].to_numpy()
                if len(r) >= 4 and np.std(r) > 0:
                    ac.append(np.corrcoef(r[:-1], r[1:])[0, 1])
        rows.append({"model": m, "residual_vs_age_slope_s_per_lap": slope,
                     "mean_within_stint_lag1_autocorrelation": float(np.mean(ac)) if ac else None, "stints": len(ac)})
    out["diagnostics"] = pd.DataFrame(rows)
    return out


@dataclass
class CalibrationResult:
    """Fitted versus configured tyre parameters. Nothing is applied automatically."""

    table: pd.DataFrame
    recommended_overrides: dict[str, dict[str, float]]
    evidence_type: str
    warnings: list[str] = field(default_factory=list)


def calibrate_tyre_model(dataframe: pd.DataFrame, setup, *, min_samples: int = 12) -> CalibrationResult:
    """Estimate linear degradation per compound and compare with the configuration.

    Uses the within-stint, fuel-corrected estimator from :mod:`racesim.tyre_model`.
    Overrides are recommended only for compounds whose estimate is flagged
    reliable. Calibrating on SYNTHETIC data merely recovers the generating values.
    """
    from racesim.tyre_model import estimate_tyre_degradation

    evidence = classify_evidence(dataframe)
    warnings: list[str] = []
    if evidence == "synthetic_self_consistency":
        warnings.append("Calibration on SYNTHETIC data recovers the generating parameters; it is not real-world calibration.")
    est = estimate_tyre_degradation(dataframe, fuel_effect_s_per_kg=setup.race.fuel_effect_s_per_kg,
                                    fuel_consumption_kg_per_lap=setup.race.fuel_consumption_kg_per_lap, min_samples=min_samples)
    rows, overrides = [], {}
    for comp, e in est.items():
        cfg = setup.compounds.get(comp)
        rows.append({
            "compound": comp,
            "configured_linear_s_per_lap": cfg.linear_degradation_s_per_lap if cfg else None,
            "fitted_linear_s_per_lap": e.linear_degradation_s_per_lap,
            "standard_error_s_per_lap": e.standard_error_s_per_lap,
            "ci95_low": None if e.linear_degradation_s_per_lap is None else e.linear_degradation_s_per_lap - 1.96 * e.standard_error_s_per_lap,
            "ci95_high": None if e.linear_degradation_s_per_lap is None else e.linear_degradation_s_per_lap + 1.96 * e.standard_error_s_per_lap,
            "laps": e.sample_count, "stints": e.stint_count, "reliable": e.reliable,
            "configured_inside_ci95": (None if e.linear_degradation_s_per_lap is None or cfg is None else
                                       abs(cfg.linear_degradation_s_per_lap - e.linear_degradation_s_per_lap) <= 1.96 * e.standard_error_s_per_lap),
            "notes": " | ".join(e.warnings),
        })
        if e.reliable and cfg is not None:
            overrides[comp] = {"linear_degradation_s_per_lap": float(max(e.linear_degradation_s_per_lap, 0.0))}
    if not overrides:
        warnings.append("No compound has a reliable estimate: configuration left unchanged.")
    return CalibrationResult(pd.DataFrame(rows), overrides, evidence, warnings)


def apply_calibration(setup, calibration: CalibrationResult):
    """Return a new setup with the recommended linear-degradation overrides applied."""
    from racesim.config import with_overrides

    return with_overrides(setup, compounds=calibration.recommended_overrides) if calibration.recommended_overrides else setup


@dataclass
class ModelValidationResult:
    """Complete validation run: data quality, evidence type, metrics, calibration."""

    evidence_type: str
    status: str
    report: dict
    data_quality: dict
    calibration: Optional[CalibrationResult]
    provenance: dict
    separate_results: dict = field(default_factory=dict)

    @property
    def headline(self) -> str:
        if self.status != "conclusive":
            return "Validation inconclusive: insufficient or unsuitable data; no accuracy figures reported."
        label = {"synthetic_self_consistency": "SYNTHETIC self-consistency check (not real-world accuracy)",
                 "historical_out_of_sample": "Historical out-of-sample validation",
                 "unlabelled": "Validation on unlabelled data (provenance unverified)"}[self.evidence_type]
        pref = self.report.get("preferred_model", "linear")
        m = self.report["models"].get(pref) or next(iter(self.report["models"].values()))
        cov = "" if m.get("interval_coverage") is None else f", {self.report['interval_level']:.0%} interval coverage {m['interval_coverage']:.1%}"
        return f"{label}: {pref} model MAE {m['mae_s']:.3f} s, RMSE {m['rmse_s']:.3f} s, bias {m['bias_s']:+.3f} s, n = {m['n_validation']}{cov}."

    def metrics_table(self) -> pd.DataFrame:
        rows = [{"evidence_type": self.evidence_type, "model": k, **{x: v[x] for x in ("mae_s", "rmse_s", "bias_s", "interval_coverage",
                                                                                       "n_validation", "n_train")}}
                for k, v in self.report.get("models", {}).items()]
        return pd.DataFrame(rows)


def run_model_validation(
    dataframe: pd.DataFrame,
    setup,
    *,
    method: str = "stint",
    seed: int = 0,
    validation_fraction: float = 0.3,
    interval_level: float = 0.9,
    min_validation_samples: int = 30,
    provenance: Optional[dict] = None,
) -> ModelValidationResult:
    """Data-quality report, evidence classification, held-out validation and calibration.

    Mixed datasets are split by ``data_source`` and validated separately so
    synthetic results never contaminate historical metrics; the historical
    subset (if any) becomes the primary result.
    """
    from racesim.data_validation import calculate_data_quality_report, clean_lap_times

    quality = calculate_data_quality_report(dataframe)
    if not quality["validation"]["valid"]:
        return ModelValidationResult("unlabelled", "inconclusive", {"warnings": quality["validation"]["errors"], "models": {}},
                                     quality, None, provenance or {})
    annotated = clean_lap_times(dataframe).annotated
    evidence = classify_evidence(annotated)
    separate: dict = {}
    if evidence == "mixed":
        syn = annotated["data_source"].astype(str).str.upper().str.contains("SYNTHETIC")
        parts = {"synthetic_self_consistency": annotated[syn], "non_synthetic": annotated[~syn]}
        for name, part in parts.items():
            separate[name] = generate_validation_report(part, fuel_effect_s_per_kg=setup.race.fuel_effect_s_per_kg, method=method,
                                                        validation_fraction=validation_fraction, seed=seed,
                                                        min_validation_samples=min_validation_samples, interval_level=interval_level)
        annotated = parts["non_synthetic"]
        evidence = classify_evidence(annotated)
    kw = dict(fuel_effect_s_per_kg=setup.race.fuel_effect_s_per_kg, method=method, validation_fraction=validation_fraction,
              seed=seed, min_validation_samples=min_validation_samples, interval_level=interval_level)
    report = generate_validation_report(annotated, **kw)
    calibration = None
    if {"tyre_compound", "tyre_age_laps"} <= set(annotated.columns) and len(annotated):
        calibration = calibrate_tyre_model(annotated, setup)
    status = report["status"]
    if status == "conclusive" and not report.get("leakage_check", {}).get("leakage_free", True):
        status = "inconclusive"
        report["warnings"].append("Stint leakage detected between training and validation sets: metrics withheld.")
    return ModelValidationResult(evidence, status, report, quality, calibration, provenance or {}, separate)
