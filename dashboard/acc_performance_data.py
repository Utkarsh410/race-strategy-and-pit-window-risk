"""Display tables for the ACC Performance page (no Streamlit here, so they can be tested directly)."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from racesim.formatting import format_lap_time
from racesim.integrations.acc.live_analysis import OVERVIEW_LAP_TIME_ROWS
from racesim.integrations.acc.race_state import Provenance

UNAVAILABLE = Provenance.UNAVAILABLE.value
RECENT_LAP_COLUMNS = ["lap_number", "lap_time_s", "classification", "sector_1_s", "sector_2_s", "sector_3_s", "position", "stint_id", "stint_lap",
                      "in_pit", "flags"]


def data_label(snap: dict) -> str:
    """Label for values derived from this snapshot."""
    if snap.get("mock"):
        return Provenance.MOCK.value
    if snap.get("replay"):
        return Provenance.REPLAY.value
    return Provenance.STALE.value if snap.get("stale") else Provenance.DERIVED.value


def overview_table(snap: dict) -> pd.DataFrame:
    rows = []
    for r in snap.get("overview", []):
        if r["value"] is not None and r["channel"] in OVERVIEW_LAP_TIME_ROWS:
            rows.append({**r, "value": format_lap_time(r["value"]), "unit": "m:ss.mmm"})
        else:
            rows.append({**r, "value": "" if r["value"] is None else str(r["value"])})
    return pd.DataFrame(rows, columns=["channel", "value", "unit", "label", "source"])


def with_lap_times(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """Display copy with each seconds column shown as m:ss.mmm and the ``_s`` suffix dropped (``lap_time_s`` -> ``lap_time``)."""
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out.insert(out.columns.get_loc(col), col.removesuffix("_s"), out.pop(col).map(format_lap_time))
    return out


def pace_display(table: pd.DataFrame) -> pd.DataFrame:
    """:func:`pace_table` for display: lap-time statistics in m:ss.mmm, the standard deviation in seconds."""
    out = table.copy()
    std = out["statistic"] == "Standard deviation"
    value = out.pop("value_s")
    out.insert(1, "value", [(None if pd.isna(v) else f"{v:.3f} s") if s else format_lap_time(v) for v, s in zip(value, std)])
    return out


def recent_laps_table(snap: dict, n: int = 15) -> pd.DataFrame:
    df = pd.DataFrame(snap.get("laps", []), columns=RECENT_LAP_COLUMNS)
    return df.tail(n).iloc[::-1].reset_index(drop=True)


def pace_table(snap: dict) -> pd.DataFrame:
    """Every statistic with its sample size. Too few laps gives an empty value and a stated requirement, never a number."""
    pace = snap.get("pace") or {}
    label = data_label(snap)
    rows = []
    entries = [(name.capitalize() + " lap time", pace.get(name)) for name in ("best", "mean", "median")]
    entries.append(("Standard deviation", pace.get("std")))
    entries += [(f"Rolling pace, last {w} valid laps", stat) for w, stat in (pace.get("rolling") or {}).items()]
    for name, stat in entries:
        if not stat:
            continue
        ok = stat["value"] is not None
        rows.append({"statistic": name, "value_s": stat["value"], "valid_laps_used": stat["n"], "laps_needed": stat["needed"],
                     "label": label if ok else f"{UNAVAILABLE}: {stat['n']} of {stat['needed']} valid laps"})
    return pd.DataFrame(rows, columns=["statistic", "value_s", "valid_laps_used", "laps_needed", "label"])


def sector_table(snap: dict) -> pd.DataFrame:
    return pd.DataFrame((snap.get("pace") or {}).get("sector_trends", []), columns=["sector", "n", "best_s", "median_s", "latest_s", "trend_s_per_lap"])


def stint_table(snap: dict) -> pd.DataFrame:
    pace = {r["stint_id"]: r for r in (snap.get("pace") or {}).get("stints", [])}
    rows = []
    for st in snap.get("stints", []):
        p = pace.get(st["stint_id"], {})
        rows.append({"stint_id": st["stint_id"], "laps": st["laps"], "first_lap": st["first_lap"], "last_lap": st["last_lap"],
                     "clean_laps": p.get("n", 0), "median_s": p.get("median_s"), "delta_to_previous_s": p.get("median_delta_to_previous_stint_s"),
                     "pace_trend_s_per_lap": p.get("pace_trend_s_per_lap"), "started": st["start_reason"], "ended": st["end_reason"] or "in progress",
                     "tyre_compound": st["tyre_compound"] or UNAVAILABLE})
    return pd.DataFrame(rows)


def fuel_table(snap: dict) -> pd.DataFrame:
    fuel = snap.get("fuel") or {}
    stale = bool(snap.get("stale"))
    def row(name, value, unit, derived=False, note=""):
        if value is None:
            label = UNAVAILABLE
        elif snap.get("mock") or snap.get("replay") or stale:
            label = data_label(snap)
        else:
            label = Provenance.DERIVED.value if derived else Provenance.LIVE.value
        return {"channel": name, "value": value, "unit": unit, "label": label, "note": note}
    used, needed = fuel.get("burn_laps_used", 0), fuel.get("burn_laps_needed", 3)
    return pd.DataFrame([
        row("Latest fuel", fuel.get("latest_fuel_kg"), "kg"),
        row("Consumption", fuel.get("burn_kg_per_lap"), "kg/lap", True, f"{used} reliable laps used, {needed} needed"),
        row("Estimated laps remaining", fuel.get("laps_remaining_estimate"), "laps", True, "Fuel / consumption"),
        row("Estimated time remaining", fuel.get("time_remaining_estimate_s"), "s", True, "Laps remaining x median clean lap"),
    ])


def tyre_table(snap: dict) -> pd.DataFrame:
    tyres = snap.get("tyres") or {}
    rows = []
    for name, key, unit in (("Compound", "compound", ""), ("Tyre age", "age_laps", "laps"), ("Pressures FL/FR/RL/RR", "pressures_psi", "psi"),
                            ("Temperatures FL/FR/RL/RR", "temperatures_c", "degC"), ("Wear", "wear", "")):
        value = tyres.get(key)
        label = UNAVAILABLE if value is None else (data_label(snap) if snap.get("mock") or snap.get("replay") or snap.get("stale") else Provenance.LIVE.value)
        rows.append({"channel": name, "value": "" if value is None else str(value), "unit": unit, "label": label})
    return pd.DataFrame(rows)


def strategy_inputs_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{**r, "value": "" if r["value"] is None else str(r["value"])} for r in rows],
                        columns=["input", "value", "category", "source", "uncertainty", "required"])


def candidates_table(result: Optional[dict]) -> pd.DataFrame:
    return pd.DataFrame((result or {}).get("candidates", []))
