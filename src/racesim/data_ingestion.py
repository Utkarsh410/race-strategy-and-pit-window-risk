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

Units are never inferred: a numeric column such as ``lap_time`` or
``lap_time_ms`` is rejected with a message asking for an explicit
``lap_time_s`` column. Two things are accepted because their unit is not in
doubt: lap times written in race format (``1:50.485``, ``1:02:03.456``) in
any lap-time or sector column, and ``lap_number`` for ``lap``. Headers are
matched ignoring case, spaces and hyphens (``Lap Time`` -> ``lap_time``).

:func:`read_lap_file` reads an uploaded file of any supported type
(:data:`UPLOAD_TYPES`): lap tables as CSV / TSV / TXT (comma, semicolon or
tab separated, with or without a byte-order mark), Excel, JSON, JSON Lines or
Parquet; and laps derived from logger files - MoTeC i2 CSV exports and .ld
logs (:mod:`racesim.integrations.motec`), iRacing .ibt telemetry
(:mod:`racesim.integrations.iracing`) and RaceSim's own ACC session logs.
The file type is detected from the content where it can be, not only the name.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Optional, Union

import pandas as pd

from racesim.data_validation import MANDATORY_COLUMNS, ValidationReport, validate_race_data
from racesim.formatting import parse_lap_time

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
# same meaning and unit (RaceSim's own ACC lap export uses the first-listed spellings)
COLUMN_ALIASES = {"lap_number": "lap", "lap_no": "lap", "lap_num": "lap", "fuel_start_kg": "fuel_kg",
                  "sector_1_s": "sector1_s", "sector_2_s": "sector2_s", "sector_3_s": "sector3_s"}
RACE_FORMAT_COLUMNS = ("lap_time_s", "sector1_s", "sector2_s", "sector3_s")
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


def _header_key(name: str) -> str:
    return re.sub(r"[\s\-]+", "_", str(name).strip().lower())


def _race_format_seconds(series: pd.Series, *, only_race_format: bool = False) -> Optional[pd.Series]:
    """Convert m:ss(.mmm) / h:mm:ss(.mmm) text to seconds; None if the column holds no such text.

    Plain numbers pass through; text that is neither stays as it is, so the
    usual coercion sets it to missing and flags it. With ``only_race_format``
    (a column whose name leaves the unit open) any plain number means None.
    """
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return None
    text = series.astype("string").str.strip()
    timed = text.str.contains(":", regex=False).fillna(False)
    if not timed.any():
        return None
    if only_race_format and not (timed | text.isna() | (text == "")).all():
        return None
    values = []
    for v, is_timed in zip(series.tolist(), timed.tolist()):
        seconds = parse_lap_time(v) if is_timed else None
        values.append(seconds if seconds is not None else v)
    return pd.Series(values, index=series.index, dtype=object)


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
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    warnings: list[str] = []
    rename: dict[str, str] = {}
    for c in df.columns:
        key = COLUMN_ALIASES.get(_header_key(c), _header_key(c))
        known = key in OPTIONAL_COLUMNS or key in AMBIGUOUS_UNIT_COLUMNS
        if known and key != c and key not in df.columns and key not in rename.values():
            rename[c] = key
            if _header_key(c) in COLUMN_ALIASES:
                warnings.append(f"Column '{c}' used as '{key}'.")
    df = df.rename(columns=rename)
    # race-format text has an unambiguous unit, so e.g. a 'Lap Time' column of 1:50.485 values is accepted
    for c in list(df.columns):
        target = AMBIGUOUS_UNIT_COLUMNS.get(c)
        if target in RACE_FORMAT_COLUMNS and target not in df.columns:
            seconds = _race_format_seconds(df[c], only_race_format=True)
            if seconds is not None:
                df[target] = seconds
                df = df.drop(columns=c)
                warnings.append(f"Column '{c}' read as race-format times (m:ss.mmm) and converted to '{target}'.")
    for col in RACE_FORMAT_COLUMNS:
        if col in df.columns:
            seconds = _race_format_seconds(df[col])
            if seconds is not None:
                df[col] = seconds
                warnings.append(f"Column '{col}': race-format times (m:ss.mmm) converted to seconds.")
    ambiguous = {c: AMBIGUOUS_UNIT_COLUMNS[c] for c in df.columns
                 if c in AMBIGUOUS_UNIT_COLUMNS and AMBIGUOUS_UNIT_COLUMNS[c] not in df.columns}
    missing = [c for c in MANDATORY_COLUMNS if c not in df.columns]
    if missing and strict:
        hint = ""
        if ambiguous:
            hint = (" Columns with unknown units were found: "
                    + ", ".join(f"'{k}' (rename to '{v}' after converting to the documented unit)" for k, v in ambiguous.items())
                    + ". Units are not inferred automatically.")
        raise DataValidationError(f"Missing mandatory column(s) {missing}.{hint}")

    warnings += [f"Column '{k}' ignored: unit unknown (expected '{v}')." for k, v in ambiguous.items()]
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


def read_lap_csv(buffer: BinaryIO, name: str = "upload") -> tuple[pd.DataFrame, str, list[str]]:
    """Read an uploaded lap file: ``(frame, data label, notes)``.

    A MoTeC i2 CSV export is recognised from its first line and reduced to
    lap times from its beacon markers; any other file is read as a lap-level
    CSV with the separator detected (comma, semicolon or tab).
    """
    from racesim.integrations.motec import is_motec_csv, motec_laps

    if is_motec_csv(buffer):
        laps, header = motec_laps(buffer)
        where = " / ".join(x for x in (header.venue, header.vehicle) if x)
        return laps, f"Uploaded MoTeC file: {name}" + (f" ({where})" if where else ""), header.notes
    sample = buffer.read(64 * 1024)
    buffer.seek(0)
    text = sample.decode("utf-8-sig", errors="replace") if isinstance(sample, bytes) else sample
    first = text.splitlines()[0] if text else ""
    sep = max((",", ";", "\t"), key=first.count)
    df = pd.read_csv(buffer, sep=sep, encoding="utf-8-sig")
    notes = [] if sep == "," else [f"Separator detected: {'tab' if sep == chr(9) else repr(sep)}."]
    return df, f"Uploaded file: {name}", notes


UPLOAD_TYPES = ("csv", "tsv", "txt", "xlsx", "json", "jsonl", "parquet", "ld", "ibt")


def _records_frame(data: Any) -> pd.DataFrame:
    """Lap records from parsed JSON: a list of objects, or an object with a 'laps' list."""
    if isinstance(data, dict):
        data = data.get("laps")
    if not isinstance(data, list) or not all(isinstance(r, dict) for r in data):
        raise ValueError("JSON must be a list of lap objects, or an object with a 'laps' list of them.")
    return pd.DataFrame(data)


def _session_log_laps(records: list[dict], name: str) -> tuple[pd.DataFrame, str, list[str]]:
    from racesim.integrations.acc.live_analysis import replay_session_records

    analysis, last = replay_session_records(records)
    snap = analysis.snapshot(last)
    laps = pd.DataFrame(snap.get("laps", []))
    if laps.empty:
        raise ValueError("The ACC session log holds no completed lap.")
    kind = "MOCK ACC session log (synthetic, not ACC telemetry)" if snap.get("mock") else "Recorded ACC session log"
    notes = [f"{kind}: {len(laps)} lap(s) of session {snap.get('session_key')}, replayed through the live lap tracker. "
             "Tyre and fuel columns are filled only where the log carried them."]
    if len(snap.get("sessions") or []) > 1:
        notes.append(f"The log holds {len(snap['sessions'])} sessions; only the latest is loaded.")
    return laps, f"{'MOCK ' if snap.get('mock') else ''}Uploaded ACC session log: {name}", notes


def read_lap_file(buffer: BinaryIO, name: str = "upload") -> tuple[pd.DataFrame, str, list[str]]:
    """Read any supported upload: ``(lap frame, data label, notes)``. Raises ``ValueError`` with a reason if it cannot."""
    import json

    from racesim.integrations.iracing import ibt_laps, is_ibt
    from racesim.integrations.motec import is_motec_ld, ld_laps

    ext = Path(name).suffix.lower().lstrip(".")
    if is_motec_ld(buffer):
        laps, header = ld_laps(buffer)
        where = " / ".join(x for x in (header.venue, header.vehicle) if x)
        return laps, f"Uploaded MoTeC .ld log: {name}" + (f" ({where})" if where else ""), header.notes
    if is_ibt(buffer):
        laps, info = ibt_laps(buffer)
        where = " / ".join(x for x in (info.track, info.car) if x)
        return laps, f"Uploaded iRacing telemetry: {name}" + (f" ({where})" if where else ""), info.notes
    if ext in ("ld", "ibt"):
        raise ValueError(f"{name} does not look like a {'MoTeC .ld log' if ext == 'ld' else 'iRacing .ibt file'} (unrecognised header).")
    head = buffer.read(4)
    buffer.seek(0)
    if head[:2] == b"PK" or ext in ("xlsx", "xls"):
        try:
            import openpyxl  # noqa: F401
        except ImportError:
            raise ValueError("Excel files need the openpyxl package, which is not installed. "
                             "Install it with `.venv/bin/pip install openpyxl`, or save the sheet as CSV.") from None
        df = pd.read_excel(buffer, sheet_name=0)
        return df, f"Uploaded workbook: {name} (first sheet)", ["Read the first sheet of the workbook."]
    if head == b"PAR1" or ext == "parquet":
        return pd.read_parquet(buffer), f"Uploaded file: {name}", []
    if ext in ("json", "jsonl"):
        text = buffer.read().decode("utf-8-sig")
        buffer.seek(0)
        stripped = text.lstrip()
        if ext == "jsonl" or (stripped.startswith("{") and "\n{" in stripped):
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
            if records and records[0].get("record_type") == "session_start":
                return _session_log_laps(records, name)
            return pd.DataFrame(records), f"Uploaded file: {name}", []
        return _records_frame(json.loads(text)), f"Uploaded file: {name}", []
    return read_lap_csv(buffer, name)


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
