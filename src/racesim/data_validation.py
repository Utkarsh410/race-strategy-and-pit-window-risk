"""Lap-data validation and cleaning.

Principles:

* The original data are never modified; functions return copies or reports.
* Invalid values (non-numeric, non-positive, non-finite or grossly out of
  range lap times) are **flagged**, never replaced with plausible numbers.
* Slow laps are not deleted automatically. Outliers are flagged with a likely
  cause (pit in/out lap, safety car/VSC, traffic, lap 1) so the user can
  inspect them; only *unexplained* outliers are excluded from model fitting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MANDATORY_COLUMNS = ("lap", "lap_time_s")
GREEN_STATUSES = {"GREEN"}
KNOWN_TRACK_STATUSES = {"GREEN", "SC", "VSC", "YELLOW", "RED", "SAFETY_CAR"}


@dataclass
class ValidationReport:
    """Structured result of :func:`validate_race_data`."""

    valid: bool
    n_rows: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    missing_mandatory_columns: list[str] = field(default_factory=list)
    available_optional_columns: list[str] = field(default_factory=list)
    missing_optional_columns: list[str] = field(default_factory=list)
    invalid_lap_time_count: int = 0
    duplicate_lap_count: int = 0
    outlier_count: int = 0
    missing_values: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class CleaningResult:
    """``annotated``: every original row plus flag columns.
    ``cleaned``: invalid and duplicate rows removed; outliers kept but flagged."""

    annotated: pd.DataFrame
    cleaned: pd.DataFrame
    summary: dict


def _bool_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[col].fillna(False).astype(bool)


def _status(df: pd.DataFrame) -> Optional[pd.Series]:
    if "track_status" not in df.columns:
        return None
    return df["track_status"].astype(str).str.strip().str.upper()


def _driver_key(df: pd.DataFrame) -> pd.Series:
    return df["driver"].astype(str) if "driver" in df.columns else pd.Series("ALL", index=df.index)


def identify_missing_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Missing-value count and fraction for every column."""
    n = max(len(dataframe), 1)
    counts = dataframe.isna().sum()
    return pd.DataFrame({"column": counts.index, "missing_count": counts.values, "missing_fraction": counts.values / n})


def identify_duplicate_laps(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Rows sharing the same (driver, lap) key (all occurrences returned)."""
    keys = ["driver", "lap"] if "driver" in dataframe.columns else ["lap"]
    if not set(keys) <= set(dataframe.columns):
        return dataframe.iloc[0:0]
    return dataframe[dataframe.duplicated(subset=keys, keep=False)]


def identify_invalid_lap_times(
    dataframe: pd.DataFrame,
    *,
    min_lap_time_s: Optional[float] = None,
    max_lap_time_s: Optional[float] = None,
    relative_bounds: tuple[float, float] = (0.5, 3.0),
) -> pd.DataFrame:
    """Flag lap times that cannot be genuine measurements.

    A lap time is invalid when it is missing, non-numeric, non-finite,
    non-positive, outside the optional absolute bounds, or outside
    ``relative_bounds`` x the median valid lap time (gross recording errors).
    Returns the offending rows with an ``invalid_reason`` column.
    """
    if "lap_time_s" not in dataframe.columns:
        raise ValueError("lap_time_s column is required")
    raw = dataframe["lap_time_s"]
    t = pd.to_numeric(raw, errors="coerce")
    reason = pd.Series("", index=dataframe.index, dtype=object)
    reason[raw.isna()] = "missing"
    reason[(t.isna()) & raw.notna()] = "non-numeric"
    finite = np.isfinite(t.to_numpy(dtype=float, na_value=np.nan))
    reason[(~finite) & t.notna()] = "non-finite"
    reason[(t <= 0) & (reason == "")] = "non-positive"
    ok = (reason == "") & t.notna()
    if min_lap_time_s is not None:
        reason[ok & (t < min_lap_time_s)] = f"below absolute minimum {min_lap_time_s} s"
    if max_lap_time_s is not None:
        reason[ok & (t > max_lap_time_s)] = f"above absolute maximum {max_lap_time_s} s"
    ok = (reason == "") & t.notna()
    if ok.any():
        med = float(t[ok].median())
        lo, hi = relative_bounds
        reason[ok & (t < lo * med)] = f"below {lo:g}x median ({med:.2f} s)"
        reason[ok & (t > hi * med)] = f"above {hi:g}x median ({med:.2f} s)"
    out = dataframe[reason != ""].copy()
    out["invalid_reason"] = reason[reason != ""]
    return out


def _likely_cause(df: pd.DataFrame) -> pd.Series:
    cause = pd.Series("unexplained", index=df.index, dtype=object)
    if "traffic_loss_s" in df.columns:
        cause[pd.to_numeric(df["traffic_loss_s"], errors="coerce").fillna(0) > 0.3] = "traffic"
    status = _status(df)
    if status is not None:
        cause[~status.isin(GREEN_STATUSES)] = "race control (" + status[~status.isin(GREEN_STATUSES)] + ")"
    if "lap" in df.columns:
        cause[df["lap"] == 1] = "lap 1 / race start"
    cause[_bool_col(df, "pit_out")] = "pit out-lap"
    cause[_bool_col(df, "pit_in")] = "pit in-lap"
    return cause


def identify_lap_time_outliers(
    dataframe: pd.DataFrame,
    *,
    threshold: float = 3.5,
    method: str = "mad",
) -> pd.DataFrame:
    """Robust per-driver outlier detection on valid lap times.

    ``method='mad'``: modified z-score ``0.6745 * (x - median) / MAD``;
    ``method='iqr'``: outside ``[Q1 - k*IQR, Q3 + k*IQR]`` with ``k = threshold``.
    Returns flagged rows with ``outlier_score`` and ``likely_cause``. Rows are
    not removed from the data.
    """
    t = pd.to_numeric(dataframe["lap_time_s"], errors="coerce")
    invalid = identify_invalid_lap_times(dataframe).index
    valid = t.notna() & ~dataframe.index.isin(invalid)
    score = pd.Series(np.nan, index=dataframe.index)
    flagged = pd.Series(False, index=dataframe.index)
    for _, idx in dataframe[valid].groupby(_driver_key(dataframe[valid])).groups.items():
        x = t.loc[idx]
        if len(x) < 5:
            continue
        if method == "mad":
            med = x.median()
            mad = (x - med).abs().median()
            if mad == 0:
                continue
            z = 0.6745 * (x - med) / mad
            score.loc[idx] = z
            flagged.loc[idx] = z.abs() > threshold
        elif method == "iqr":
            q1, q3 = x.quantile([0.25, 0.75])
            iqr = q3 - q1
            score.loc[idx] = (x - x.median()) / (iqr if iqr > 0 else 1)
            flagged.loc[idx] = (x < q1 - threshold * iqr) | (x > q3 + threshold * iqr)
        else:
            raise ValueError("method must be 'mad' or 'iqr'")
    out = dataframe[flagged].copy()
    out["outlier_score"] = score[flagged]
    out["likely_cause"] = _likely_cause(dataframe)[flagged]
    return out


def validate_race_data(dataframe: pd.DataFrame) -> ValidationReport:
    """Structural and value validation of lap-level data."""
    from racesim.data_ingestion import OPTIONAL_COLUMNS

    cols = set(dataframe.columns)
    missing = [c for c in MANDATORY_COLUMNS if c not in cols]
    report = ValidationReport(
        valid=not missing,
        n_rows=len(dataframe),
        missing_mandatory_columns=missing,
        available_optional_columns=[c for c in OPTIONAL_COLUMNS if c in cols],
        missing_optional_columns=[c for c in OPTIONAL_COLUMNS if c not in cols and c not in MANDATORY_COLUMNS],
    )
    if missing:
        report.errors.append(
            f"Missing mandatory column(s): {missing}. Required: {list(MANDATORY_COLUMNS)} "
            "(lap = integer lap number, lap_time_s = lap time in seconds)."
        )
        return report
    if dataframe.empty:
        report.valid = False
        report.errors.append("The dataset contains no rows.")
        return report

    report.missing_values = {k: int(v) for k, v in dataframe.isna().sum().items() if v}
    laps = pd.to_numeric(dataframe["lap"], errors="coerce")
    if laps.isna().any() or (laps < 1).any() or (laps.dropna() % 1 != 0).any():
        report.errors.append("Column 'lap' must contain positive integers (1-indexed).")
        report.valid = False
    invalid = identify_invalid_lap_times(dataframe)
    report.invalid_lap_time_count = len(invalid)
    if len(invalid):
        report.warnings.append(f"{len(invalid)} invalid lap time(s) flagged (not replaced): "
                               f"{invalid['invalid_reason'].value_counts().to_dict()}")
    dup = identify_duplicate_laps(dataframe)
    report.duplicate_lap_count = len(dup)
    if len(dup):
        report.warnings.append(f"{len(dup)} rows share a duplicated (driver, lap) key.")
    if "driver" not in cols:
        report.warnings.append("No 'driver' column: all rows are treated as one driver.")
    for col in ("tyre_age_laps", "fuel_kg", "traffic_loss_s"):
        if col in cols and (pd.to_numeric(dataframe[col], errors="coerce") < 0).any():
            report.errors.append(f"Column '{col}' contains negative values.")
            report.valid = False
    status = _status(dataframe)
    if status is not None:
        unknown = sorted(set(status.dropna()) - KNOWN_TRACK_STATUSES - {"NAN"})
        if unknown:
            report.warnings.append(f"Unrecognised track_status values {unknown}; treated as non-green.")
    if len(invalid) < len(dataframe):
        report.outlier_count = len(identify_lap_time_outliers(dataframe))
    if report.outlier_count:
        report.warnings.append(f"{report.outlier_count} lap-time outlier(s) flagged for inspection (not deleted).")
    for col in ("tyre_compound", "tyre_age_laps"):
        if col not in cols:
            report.warnings.append(f"Optional column '{col}' missing: tyre-degradation analysis disabled.")
    if "fuel_kg" not in cols:
        report.warnings.append("Optional column 'fuel_kg' missing: fuel corrections rely on configured assumptions.")
    return report


def representative_lap_mask(dataframe: pd.DataFrame, *, traffic_threshold_s: float = 0.3) -> pd.Series:
    """Laps suitable for pace modelling.

    Excludes invalid lap times, lap 1, non-green laps, pit in/out laps,
    laps with recorded traffic loss above ``traffic_threshold_s`` and rows
    marked ``exclude_from_model``.
    """
    df = dataframe
    t = pd.to_numeric(df["lap_time_s"], errors="coerce")
    mask = t.notna() & (t > 0) & np.isfinite(t.astype(float).fillna(-1))
    if len(df):
        mask &= ~df.index.isin(identify_invalid_lap_times(df).index)
    if "lap" in df.columns:
        mask &= df["lap"] != 1
    status = _status(df)
    if status is not None:
        mask &= status.isin(GREEN_STATUSES)
    mask &= ~_bool_col(df, "pit_in") & ~_bool_col(df, "pit_out")
    if "traffic_loss_s" in df.columns:
        mask &= pd.to_numeric(df["traffic_loss_s"], errors="coerce").fillna(0) <= traffic_threshold_s
    if "exclude_from_model" in df.columns:
        mask &= ~df["exclude_from_model"].fillna(False).astype(bool)
    return mask


def assign_stint_ids(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a ``stint_id`` column.

    A new stint starts when (per driver, in lap order) a ``pit_out`` flag is
    set, the compound changes, the tyre age decreases, or a pit in-lap was
    the previous lap. Without any of these columns each driver is one stint.
    """
    df = dataframe.copy()
    if df.empty:
        df["stint_id"] = pd.Series(dtype=str)
        return df
    df["_order"] = np.arange(len(df))
    drv = _driver_key(df)
    df["_drv"] = drv
    df = df.sort_values(["_drv", "lap"])
    new = pd.Series(False, index=df.index)
    g = df.groupby("_drv")
    if "pit_out" in df.columns:
        new |= _bool_col(df, "pit_out")
    if "pit_in" in df.columns:
        new |= g["pit_in"].shift(1).fillna(False).astype(bool)
    if "tyre_compound" in df.columns:
        prev = g["tyre_compound"].shift(1)
        new |= prev.notna() & (prev != df["tyre_compound"])
    if "tyre_age_laps" in df.columns:
        new |= g["tyre_age_laps"].diff().fillna(0) < 0
    first = ~df["_drv"].duplicated()
    new |= first
    df["stint_id"] = df["_drv"] + "_S" + new.astype(int).groupby(df["_drv"]).cumsum().astype(str)
    return df.sort_values("_order").drop(columns=["_order", "_drv"])


def clean_lap_times(
    dataframe: pd.DataFrame,
    *,
    outlier_threshold: float = 3.5,
    outlier_method: str = "mad",
    exclude_unexplained_outliers: bool = True,
) -> CleaningResult:
    """Produce an annotated copy and a separate cleaned dataset.

    The cleaned dataset removes only invalid lap times and duplicate keys
    (first occurrence kept). Outliers remain, flagged; unexplained outliers are
    marked ``exclude_from_model`` when ``exclude_unexplained_outliers``.
    """
    ann = dataframe.copy()
    invalid = identify_invalid_lap_times(ann)
    ann["flag_invalid"] = ann.index.isin(invalid.index)
    ann["invalid_reason"] = invalid["invalid_reason"].reindex(ann.index).fillna("")
    keys = ["driver", "lap"] if "driver" in ann.columns else ["lap"]
    ann["flag_duplicate"] = ann.duplicated(subset=keys, keep="first")
    outliers = identify_lap_time_outliers(ann, threshold=outlier_threshold, method=outlier_method)
    ann["flag_outlier"] = ann.index.isin(outliers.index)
    ann["outlier_cause"] = outliers["likely_cause"].reindex(ann.index).fillna("")
    ann["exclude_from_model"] = ann["flag_invalid"] | ann["flag_duplicate"]
    if exclude_unexplained_outliers:
        ann["exclude_from_model"] |= ann["outlier_cause"].eq("unexplained")
    cleaned = ann[~ann["flag_invalid"] & ~ann["flag_duplicate"]].copy()
    cleaned["lap_time_s"] = pd.to_numeric(cleaned["lap_time_s"])
    summary = {
        "rows_original": int(len(dataframe)),
        "rows_cleaned": int(len(cleaned)),
        "invalid_removed": int(ann["flag_invalid"].sum()),
        "duplicates_removed": int(ann["flag_duplicate"].sum()),
        "outliers_flagged": int(ann["flag_outlier"].sum()),
        "outlier_causes": ann.loc[ann["flag_outlier"], "outlier_cause"].value_counts().to_dict(),
        "excluded_from_model": int(ann["exclude_from_model"].sum()),
    }
    logger.info("Cleaning summary: %s", summary)
    return CleaningResult(ann, cleaned, summary)


def calculate_data_quality_report(dataframe: pd.DataFrame) -> dict:
    """Counts describing data completeness and validity (no synthetic score)."""
    report = validate_race_data(dataframe)
    out = {"validation": report.as_dict(), "missing_values": identify_missing_values(dataframe).to_dict("records")}
    if report.valid:
        cleaning = clean_lap_times(dataframe)
        out["cleaning"] = cleaning.summary
        out["representative_laps"] = int(representative_lap_mask(cleaning.annotated).sum())
        if "driver" in dataframe.columns:
            out["drivers"] = sorted(dataframe["driver"].astype(str).unique().tolist())
        if "tyre_compound" in dataframe.columns:
            out["compounds"] = sorted(dataframe["tyre_compound"].dropna().astype(str).unique().tolist())
        if "data_source" in dataframe.columns:
            out["data_source"] = sorted(dataframe["data_source"].astype(str).unique().tolist())
    return out
