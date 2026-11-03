"""Calibrate RaceSim's model from recorded ACC session logs.

Recorded logs (``outputs/acc_sessions/*.jsonl``) are replayed through the live
lap tracker (:func:`replay_session_log`), so lap boundaries, out/in-laps and
invalid laps are decided by the same code the dashboard uses. Completed laps
are converted to RaceSim's lap CSV schema and passed to the existing
validation and calibration (:func:`run_model_validation`).

What the conversion does and does not assume
--------------------------------------------
* Only sessions whose source is ``acc_udp`` are used; mock sessions are
  dropped and counted.
* ``data_source = "ACC_RECORDED"``. RaceSim's evidence classifier therefore
  treats the laps as recorded (non-synthetic) data. They are recorded
  **simulator** sessions: suitable for fitting RaceSim to ACC, not evidence
  about real-world racing.
* ``driver`` is the ACC session key. RaceSim's lap schema needs (driver, lap)
  to be unique, and this keeps each session's offset (fuel, temperature,
  track evolution) separate in the fit.
* ``lap`` = ACC's laps-completed counter + 1. ACC does not count the out-lap
  from the pits, so RaceSim's lap 1 (always excluded as a start lap) is the
  out-lap, and ACC's first counted lap is kept.
* ``tyre_age_laps`` is laps since the stint began **only** for stints that
  start in the pits (a fresh set is assumed, and stated). For stints first
  seen on track the age is unknown and left empty.
* ``tyre_compound`` and ``fuel_kg`` are not in the UDP feed. The compound is
  ``UNKNOWN`` unless declared; fuel stays empty unless a consumption rate is
  declared, in which case only the *lap-dependent* fuel correction is applied.
* ``exclude_from_model`` is true for every lap the tracker did not classify as
  ``CLEAN_CANDIDATE`` (invalid, out/in-laps, uncertain, incomplete).

Nothing is extrapolated: every parameter has an explicit minimum amount of
data, and below it the report says what is missing instead of producing a value.
"""

from __future__ import annotations

import glob
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd

from racesim.integrations.acc.live_analysis import replay_session_log

logger = logging.getLogger(__name__)

DATA_SOURCE = "ACC_RECORDED"
UNKNOWN_COMPOUND = "UNKNOWN"
PIT_START_REASONS = {"session_start_in_pit", "pit_exit", "left_pits"}

# Minimum data per parameter (per track). Chosen to match the thresholds already
# used by RaceSim's estimators so the two never disagree.
MIN_BASELINE_CLEAN_LAPS = 5
MIN_VARIABILITY_CLEAN_LAPS = 8          # in stints with >= 4 clean laps (trend removed per stint)
MIN_VARIABILITY_STINT_LAPS = 4
MIN_DEGRADATION_LAPS = 12               # tyre_model.estimate_tyre_degradation default
MIN_VALIDATION_LAPS = 30                # validation.generate_validation_report default (each side)
MIN_PIT_STOPS = 1


@dataclass
class LapDataResult:
    laps: pd.DataFrame
    provenance: dict
    warnings: list[str] = field(default_factory=list)


def _stint_info(analysis) -> dict[tuple[str, int], dict]:
    out = {}
    for key, s in analysis.sessions.items():
        for st in s.tracker.stints:
            out[(key, st.stint_id)] = {"start_reason": st.start_reason, "pit_entry": st.pit_entry_session_time_s,
                                       "pit_exit": st.pit_exit_session_time_s}
    return out


def session_logs_to_lap_data(paths: Iterable[Union[str, Path]], *, compound: Optional[str] = None,
                             fuel_start_kg: Optional[float] = None, fuel_per_lap_kg: Optional[float] = None) -> LapDataResult:
    """Convert recorded ACC session logs into RaceSim's lap CSV schema (one row per completed lap)."""
    rows: list[dict] = []
    warnings: list[str] = []
    prov = {"files": [], "sessions_used": 0, "mock_sessions_skipped": 0, "empty_sessions": 0,
            "compound": f"declared: {compound}" if compound else "not in UDP feed: UNKNOWN",
            "fuel": (f"assumed: start {fuel_start_kg} kg, {fuel_per_lap_kg} kg/lap" if fuel_per_lap_kg is not None
                     else "not in UDP feed: unavailable"),
            "tyre_age": "laps since stint start, only for stints that start in the pits (fresh set assumed)"}
    for path in paths:
        path = Path(path)
        analysis, _ = replay_session_log(path)
        prov["files"].append(path.name)
        info = _stint_info(analysis)
        for key, s in analysis.sessions.items():
            if s.mock or s.source != "acc_udp":
                prov["mock_sessions_skipped"] += 1
                continue
            laps = s.tracker.laps
            if not laps:
                prov["empty_sessions"] += 1
                continue
            prov["sessions_used"] += 1
            session = (s.latest or {}).get("session") or {}
            track = session.get("track_name") or "UNKNOWN_TRACK"
            stype = session.get("session_type")
            for lap in laps:
                stint = info.get((key, lap.stint_id), {})
                from_pit = stint.get("start_reason") in PIT_START_REASONS
                age = (lap.stint_lap - 1) if (from_pit and lap.stint_lap is not None) else None
                if lap.tyre_age_laps is not None:  # a future data source may supply it directly
                    age = lap.tyre_age_laps
                sectors = lap.sector_times_s or []
                cls = lap.classification.value
                exclude = cls != "CLEAN_CANDIDATE"
                if stype == "RACE" and lap.lap_number == 1:
                    exclude = True  # race start lap
                fuel = None
                if lap.fuel_start_kg is not None:
                    fuel = lap.fuel_start_kg
                elif fuel_per_lap_kg is not None and fuel_start_kg is not None and lap.lap_number is not None:
                    fuel = max(fuel_start_kg - fuel_per_lap_kg * lap.lap_number, 0.0)
                rows.append({
                    "lap": int(lap.lap_number) + 1, "driver": key, "track_name": track, "session_type": stype,
                    "lap_time_s": lap.lap_time_s,
                    "sector1_s": sectors[0] if len(sectors) > 0 else None,
                    "sector2_s": sectors[1] if len(sectors) > 1 else None,
                    "sector3_s": sectors[2] if len(sectors) > 2 else None,
                    "tyre_compound": (lap.tyre_compound or compound or UNKNOWN_COMPOUND).upper(),
                    "tyre_age_laps": age, "fuel_kg": fuel,
                    "pit_in": cls == "IN_LAP", "pit_out": cls == "OUT_LAP",
                    "acc_lap_class": cls, "acc_flags": "|".join(lap.flags), "acc_stint_id": lap.stint_id,
                    "stint_start_reason": stint.get("start_reason"), "exclude_from_model": exclude,
                    "data_source": DATA_SOURCE, "source_file": path.name,
                })
    df = pd.DataFrame(rows)
    if prov["mock_sessions_skipped"]:
        warnings.append(f"{prov['mock_sessions_skipped']} mock session(s) skipped; mock data is never used for calibration.")
    if not df.empty and df["tyre_age_laps"].isna().any():
        warnings.append(f"{int(df['tyre_age_laps'].isna().sum())} lap(s) have unknown tyre age (stint first seen on track).")
    return LapDataResult(df, prov, warnings)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    return df[(~df["exclude_from_model"]) & df["lap_time_s"].notna() & (df["lap"] > 1)]


def _row(track, parameter, required, available, value=None, unit="", note="") -> dict:
    ok = available >= required
    return {"track": track, "parameter": parameter, "required": required, "available": available,
            "status": "estimated" if ok and value is not None else ("insufficient data" if not ok else "not estimable"),
            "still_needed": max(required - available, 0), "value": value if ok else None, "unit": unit, "note": note}


@dataclass
class AccCalibrationReport:
    lap_data: pd.DataFrame
    provenance: dict
    sufficiency: pd.DataFrame
    track_summaries: pd.DataFrame
    validation: dict = field(default_factory=dict)          # track -> ModelValidationResult
    calibration: dict = field(default_factory=dict)         # track -> {compound: DegradationEstimate}
    warnings: list[str] = field(default_factory=list)

    @property
    def any_estimate(self) -> bool:
        return bool((self.sufficiency["status"] == "estimated").any()) if not self.sufficiency.empty else False


def calibrate_from_acc_logs(paths: Sequence[Union[str, Path]], setup, *, track: Optional[str] = None,
                            compound: Optional[str] = None, fuel_start_kg: Optional[float] = None,
                            fuel_per_lap_kg: Optional[float] = None, fuel_effect_s_per_kg: Optional[float] = None) -> AccCalibrationReport:
    """Per-track sufficiency checks, pace/variability/pit-loss estimates, validation and degradation calibration."""
    res = session_logs_to_lap_data(paths, compound=compound, fuel_start_kg=fuel_start_kg, fuel_per_lap_kg=fuel_per_lap_kg)
    return calibrate_lap_data(res.laps, setup, provenance=res.provenance, warnings=res.warnings, track=track,
                              fuel_per_lap_kg=fuel_per_lap_kg, fuel_effect_s_per_kg=fuel_effect_s_per_kg)


def calibrate_lap_data(df: pd.DataFrame, setup, *, provenance: Optional[dict] = None, warnings: Optional[list[str]] = None,
                       track: Optional[str] = None, fuel_per_lap_kg: Optional[float] = None,
                       fuel_effect_s_per_kg: Optional[float] = None) -> AccCalibrationReport:
    """Estimates from converted lap data (the output of :func:`session_logs_to_lap_data`)."""
    from racesim.config import with_overrides
    from racesim.lap_time_model import estimate_pace_uncertainty
    from racesim.validation import run_model_validation

    warnings = list(warnings or [])
    provenance = provenance or {}
    suff, summaries, validations, calibrations = [], [], {}, {}
    if df.empty:
        warnings.append("No completed laps from real ACC sessions were found in the logs.")
        return AccCalibrationReport(df, provenance, pd.DataFrame(), pd.DataFrame(), warnings=warnings)
    tracks = [track] if track else sorted(df["track_name"].unique())
    # The fuel effect used for correction is an explicit input (or the configured value when fuel data exist).
    effect = fuel_effect_s_per_kg if fuel_effect_s_per_kg is not None else setup.race.fuel_effect_s_per_kg
    fit_setup = with_overrides(setup, race={"fuel_effect_s_per_kg": effect})
    for t in tracks:
        d = df[df["track_name"] == t]
        clean = _clean(d)
        n_clean = len(clean)
        summaries.append({"track": t, "sessions": d["driver"].nunique(), "laps_completed": len(d), "clean_laps": n_clean,
                          "invalid_laps": int((d["acc_lap_class"] == "INVALID").sum()),
                          "other_excluded": int(d["exclude_from_model"].sum() - (d["acc_lap_class"] == "INVALID").sum()),
                          "best_clean_lap_s": float(clean["lap_time_s"].min()) if n_clean else None,
                          "median_clean_lap_s": float(clean["lap_time_s"].median()) if n_clean else None,
                          "pit_stops_observed": int(d["pit_in"].sum())})
        med = float(clean["lap_time_s"].median()) if n_clean else None
        suff.append(_row(t, "observed clean pace (median lap)", MIN_BASELINE_CLEAN_LAPS, n_clean, med, "s",
                         "at the session's unknown fuel load; not the empty-tank baseline_lap_time_s until fuel is known"))
        stint_counts = clean.groupby(["driver", "acc_stint_id"]).size()
        usable = int(stint_counts[stint_counts >= MIN_VARIABILITY_STINT_LAPS].sum())
        var = None
        if usable >= MIN_VARIABILITY_CLEAN_LAPS:
            var = estimate_pace_uncertainty(d.assign(exclude_from_model=d["exclude_from_model"]))["robust_std_s"]
        suff.append(_row(t, "lap-to-lap variability (robust std)", MIN_VARIABILITY_CLEAN_LAPS, usable, var, "s",
                         f"clean laps in stints of >= {MIN_VARIABILITY_STINT_LAPS} clean laps; per-stint trend removed"))
        stops = int(d["pit_in"].sum())
        loss = None
        if stops >= MIN_PIT_STOPS and n_clean >= MIN_BASELINE_CLEAN_LAPS:  # same reference-pace minimum as the pace estimate
            ins = d[d["pit_in"] & d["lap_time_s"].notna()]["lap_time_s"]
            outs = d[d["pit_out"] & d["lap_time_s"].notna()]["lap_time_s"]
            if len(ins) and len(outs):
                loss = float(ins.median() + outs.median() - 2 * med)
        suff.append(_row(t, "pit loss (in-lap + out-lap - 2 x clean lap)", MIN_PIT_STOPS, stops, loss, "s",
                         "total time lost to a stop incl. stationary time and out-lap warm-up; RaceSim splits these, "
                         "so it is a cross-check, not a direct pit_lane_loss_s value"
                         + ("" if n_clean >= MIN_BASELINE_CLEAN_LAPS else f"; also needs >= {MIN_BASELINE_CLEAN_LAPS} clean laps as reference pace")))
        known_comp = (clean["tyre_compound"] != UNKNOWN_COMPOUND) & clean["tyre_age_laps"].notna()
        n_deg = int(known_comp.sum())
        fuel_known = clean["fuel_kg"].notna().all() and n_clean > 0
        note = "needs a declared compound (--compound) and known tyre age"
        if not fuel_known:
            note += "; fuel unknown, so the slope is confounded with fuel burn (declare --fuel-start/--fuel-per-lap)"
        slope = None
        if n_deg >= MIN_DEGRADATION_LAPS:
            from racesim.tyre_model import estimate_tyre_degradation

            # Fuel correction only from measured fuel or a declared rate - never the synthetic configured rate.
            corr = fuel_known or fuel_per_lap_kg is not None
            est = estimate_tyre_degradation(d[known_comp.reindex(d.index, fill_value=False) | d["exclude_from_model"]],
                                            fuel_effect_s_per_kg=effect if corr else None,
                                            fuel_consumption_kg_per_lap=fuel_per_lap_kg, min_samples=MIN_DEGRADATION_LAPS)
            calibrations[t] = est
            fitted = [e for e in est.values() if e.linear_degradation_s_per_lap is not None]
            if fitted:
                e = max(fitted, key=lambda x: x.sample_count)
                slope = e.linear_degradation_s_per_lap
                note += (f"; {e.compound}: {slope:.4f} +/- {e.standard_error_s_per_lap:.4f} s/lap, reliable: {e.reliable}"
                         + (f" ({'; '.join(e.warnings)})" if e.warnings else ""))
        suff.append(_row(t, "linear tyre degradation", MIN_DEGRADATION_LAPS, n_deg, slope, "s/lap", note))
        v = run_model_validation(d, fit_setup) if n_clean >= 2 * MIN_VALIDATION_LAPS else None
        if v is not None:
            validations[t] = v
        status_note = (v.headline if v is not None else f"needs >= {MIN_VALIDATION_LAPS} training and {MIN_VALIDATION_LAPS} held-out laps")
        status_note += "; whole stints are held out, so this also needs enough separate stints"
        suff.append(_row(t, "held-out validation (MAE / RMSE)", 2 * MIN_VALIDATION_LAPS, n_clean,
                         None if v is None or v.status != "conclusive" else v.metrics_table().iloc[0]["rmse_s"], "s", status_note))
    rep = AccCalibrationReport(df, provenance, pd.DataFrame(suff), pd.DataFrame(summaries), validations, calibrations, warnings)
    if not rep.any_estimate:
        rep.warnings.append("No parameter has enough data yet; the configuration is unchanged. See 'still_needed' per parameter.")
    return rep


def default_log_paths(directory: Union[str, Path]) -> list[Path]:
    return sorted(Path(p) for p in glob.glob(str(Path(directory) / "*.jsonl")))


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    fmt = lambda v: "" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{v:,.3f}" if isinstance(v, float) else str(v))  # noqa: E731
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(out)


def report_markdown(rep: AccCalibrationReport) -> str:
    """Plain-text report (Markdown) of the calibration run."""
    lines = ["# ACC calibration report", "",
             "Recorded ACC **simulator** sessions; real values, but they describe ACC, not real-world racing.", "",
             "## Data used", ""]
    for k, v in rep.provenance.items():
        lines.append(f"- **{k}**: {v}")
    if not rep.track_summaries.empty:
        lines += ["", "## Laps by track", "", _md_table(rep.track_summaries)]
    if not rep.sufficiency.empty:
        lines += ["", "## Parameters: data needed vs available", "",
                  _md_table(rep.sufficiency.drop(columns=["note"])), "", "Notes:", ""]
        for _, r in rep.sufficiency.iterrows():
            lines.append(f"- {r['track']} / {r['parameter']}: {r['note']}")
    if rep.warnings:
        lines += ["", "## Warnings", ""] + [f"- {w}" for w in rep.warnings]
    return "\n".join(lines) + "\n"
