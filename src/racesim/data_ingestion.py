"""CSV / DataFrame ingestion and result export.

Lap-level CSV schema (column -> unit). Only ``lap`` and ``lap_time_s`` are
mandatory; every other column is optional.

=================  =====================================================
lap                integer, 1-indexed
driver             string identifier
lap_time_s         seconds
sector1_s..3_s     seconds
tyre_compound      string (e.g. SOFT, MEDIUM, HARD)
tyre_age_laps      integer laps completed on the set at lap start
fuel_kg            kilograms on board at lap start
track_status       GREEN | SC | VSC | YELLOW | RED
pit_in / pit_out   boolean (true/false, 1/0, yes/no)
traffic_loss_s     seconds lost to traffic (if known)
air_temp_c         degrees Celsius
track_temp_c       degrees Celsius
=================  =====================================================

Units are never inferred: a column such as ``lap_time`` or ``lap_time_ms``
is rejected with a message asking for an explicit ``lap_time_s`` column.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import pandas as pd

from racesim.data_validation import MANDATORY_COLUMNS, ValidationReport, validate_race_data

logger = logging.getLogger(__name__)

OPTIONAL_COLUMNS = (
    "lap", "driver", "lap_time_s", "sector1_s", "sector2_s", "sector3_s", "tyre_compound", "tyre_age_laps",
    "fuel_kg", "track_status", "pit_in", "pit_out", "traffic_loss_s", "air_temp_c", "track_temp_c",
)
NUMERIC_COLUMNS = ("lap_time_s", "sector1_s", "sector2_s", "sector3_s", "tyre_age_laps", "fuel_kg",
                   "traffic_loss_s", "air_temp_c", "track_temp_c")
BOOLEAN_COLUMNS = ("pit_in", "pit_out")
AMBIGUOUS_UNIT_COLUMNS = {
    "lap_time": "lap_time_s", "laptime": "lap_time_s", "lap_time_ms": "lap_time_s", "time": "lap_time_s",
    "fuel": "fuel_kg", "fuel_l": "fuel_kg", "tyre_age": "tyre_age_laps", "air_temp": "air_temp_c",
    "track_temp": "track_temp_c", "traffic_loss": "traffic_loss_s",
}
_TRUE = {"true", "1", "yes", "y", "t"}
_FALSE = {"false", "0", "no", "n", "f", "", "nan", "none"}


class DataValidationError(ValueError):
    """Raised when a dataset cannot be used (e.g. missing mandatory columns)."""


@dataclass
class IngestionResult:
    """``raw`` is the untouched input; ``data`` is the typed working copy."""

    raw: pd.DataFrame
    data: pd.DataFrame
    report: ValidationReport
    source: str
    coercion_warnings: list[str]


def _parse_bool(series: pd.Series, name: str) -> tuple[pd.Series, list[str]]:
    if series.dtype == bool:
        return series, []
    s = series.astype(str).str.strip().str.lower()
    unknown = ~(s.isin(_TRUE) | s.isin(_FALSE))
    warnings = [f"Column '{name}': {int(unknown.sum())} unrecognised boolean value(s) treated as missing."] if unknown.any() else []
    out = pd.Series(pd.NA, index=series.index, dtype="boolean")
    out[s.isin(_TRUE)] = True
    out[s.isin(_FALSE)] = False
    return out, warnings


def load_race_dataframe(dataframe: pd.DataFrame, *, source: str = "dataframe", strict: bool = True) -> IngestionResult:
    """Validate and type a lap-level DataFrame without modifying the original.

    Raises :class:`DataValidationError` if mandatory columns are missing and
    ``strict`` is true.
    """
    raw = dataframe.copy()
    df = dataframe.copy()
    df.columns = [str(c).strip() for c in df.columns]
    lower_map = {c: c.lower() for c in df.columns if c.lower() in OPTIONAL_COLUMNS and c != c.lower()}
    df = df.rename(columns=lower_map)
    ambiguous = {c: AMBIGUOUS_UNIT_COLUMNS[c.lower()] for c in df.columns
                 if c.lower() in AMBIGUOUS_UNIT_COLUMNS and AMBIGUOUS_UNIT_COLUMNS[c.lower()] not in df.columns}
    missing = [c for c in MANDATORY_COLUMNS if c not in df.columns]
    if missing and strict:
        hint = ""
        if ambiguous:
            hint = (" Columns with unknown units were found: "
                    + ", ".join(f"'{k}' (rename to '{v}' after converting to the documented unit)" for k, v in ambiguous.items())
                    + ". Units are not inferred automatically.")
        raise DataValidationError(f"Missing mandatory column(s) {missing}.{hint}")

    warnings: list[str] = [f"Column '{k}' ignored: unit unknown (expected '{v}')." for k, v in ambiguous.items()]
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            before = df[col].notna()
            df[col] = pd.to_numeric(df[col], errors="coerce")
            bad = int((before & df[col].isna()).sum())
            if bad:
                warnings.append(f"Column '{col}': {bad} non-numeric value(s) set to missing and flagged (not replaced).")
    if "lap" in df.columns:
        lap = pd.to_numeric(df["lap"], errors="coerce")
        df["lap"] = lap.astype("Int64") if lap.notna().all() and (lap % 1 == 0).all() else lap
    for col in BOOLEAN_COLUMNS:
        if col in df.columns:
            df[col], w = _parse_bool(df[col], col)
            warnings += w
    for col in ("driver", "tyre_compound", "track_status"):
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()
            if col != "driver":
                df[col] = df[col].str.upper()
    if "lap" in df.columns and df["lap"].dtype == "Int64":
        df["lap"] = df["lap"].astype(int)
    report = validate_race_data(df)
    report.warnings = warnings + report.warnings
    for w in report.warnings:
        logger.warning("[%s] %s", source, w)
    return IngestionResult(raw=raw, data=df, report=report, source=source, coercion_warnings=warnings)


def load_race_csv(path: Union[str, Path], **kwargs: Any) -> IngestionResult:
    """Load a lap-level CSV file (see module docstring for the schema)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    raw = pd.read_csv(path)
    return load_race_dataframe(raw, source=str(path), **kwargs)


def _to_frame(results: Any) -> pd.DataFrame:
    if isinstance(results, pd.DataFrame):
        return results
    if hasattr(results, "to_dataframe"):
        return results.to_dataframe()
    if isinstance(results, Iterable):
        frames = []
        for r in results:
            frame = _to_frame(r)
            if "strategy_id" not in frame.columns and hasattr(r, "strategy_id"):
                frame.insert(0, "strategy_id", r.strategy_id)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    raise TypeError(f"cannot export object of type {type(results).__name__}")


def export_simulation_results(results: Any, path: Union[str, Path]) -> Path:
    """Write simulation output (SimulationResult(s), MonteCarloResult or DataFrame) to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _to_frame(results).to_csv(path, index=False)
    logger.info("Exported simulation results to %s", path)
    return path


def export_strategy_comparison(results: Any, path: Union[str, Path]) -> Path:
    """Write a strategy comparison (OptimisationResult, evaluations or DataFrame) to CSV."""
    if hasattr(results, "ranking"):
        frame = results.ranking
    elif isinstance(results, list) and results and hasattr(results[0], "as_row"):
        frame = pd.DataFrame([r.as_row() for r in results])
    else:
        frame = _to_frame(results)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    logger.info("Exported strategy comparison to %s", path)
    return path


def load_historical_csv(path: Union[str, Path], *, source_label: Optional[str] = None, **kwargs: Any) -> tuple[IngestionResult, dict]:
    """Load recorded lap data for validation, returning the ingestion result and provenance.

    The ``data_source`` column is never invented: if the file lacks one, it is
    set to ``source_label`` when the user supplies it, otherwise to
    ``UNLABELLED`` (which validation treats as unverified provenance, not as
    historical evidence). A SHA-256 digest of the file is recorded so results
    can be traced to the exact input.
    """
    import hashlib

    path = Path(path)
    res = load_race_csv(path, **kwargs)
    if "data_source" not in res.data.columns:
        res.data["data_source"] = source_label or "UNLABELLED"
    elif source_label:
        logger.warning("data_source column present; ignoring source_label=%r", source_label)
    provenance = {
        "file": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": len(res.data),
        "data_source_labels": sorted(res.data["data_source"].astype(str).unique().tolist()),
        "drivers": int(res.data["driver"].nunique()) if "driver" in res.data.columns else None,
        "validation_errors": len(res.report.errors),
        "validation_warnings": len(res.report.warnings),
    }
    return res, provenance
