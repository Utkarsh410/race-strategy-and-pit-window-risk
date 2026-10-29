"""Shared dashboard state, configuration overrides and cached computations.

Model-assumption widgets live in several sections but every section uses the
*same* :class:`SimulatorSetup`, built once per rerun from session state, so all
analyses share one race model. Cached functions are deterministic for their
inputs; the Monte Carlo cache key includes the seed and simulation count.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from racesim.config import SimulatorSetup, load_setup, regulation_warnings, with_baseline_lap_time, with_overrides
from racesim.data_ingestion import IngestionResult
from racesim.data_validation import CleaningResult
from racesim.schemas import RaceSnapshot


@st.cache_resource
def base_setup() -> SimulatorSetup:
    return load_setup()


def ss(key: str, default: Any) -> Any:
    """Initialise a widget key once and return its current value."""
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


def show(fig: go.Figure, key: Optional[str] = None) -> None:
    st.plotly_chart(fig, theme=None, key=key, config={"displaylogo": False})


def _scale_laps(laps: list[int], n0: int, n1: int, min_rem: int) -> list[int]:
    out, prev = [], 0
    for p in laps:
        q = min(max(round(p * n1 / n0), prev + 1), n1 - min_rem - (len(laps) - len(out) - 1))
        out.append(q)
        prev = q
    return out


def build_setup() -> SimulatorSetup:
    """Apply dashboard overrides (from session state) to the base configuration."""
    base = base_setup()
    r = base.race
    n = int(ss("ov_total_laps", r.total_laps))
    setup = with_baseline_lap_time(base, float(ss("ov_baseline", r.baseline_lap_time_s)))
    cons = float(ss("ov_fuel_cons", r.fuel_consumption_kg_per_lap))
    auto_fuel = bool(ss("ov_auto_fuel", True))
    fuel_start = round(cons * n + 5.0, 2) if auto_fuel else float(ss("ov_fuel_start", r.fuel_start_kg))
    pit_loss = float(ss("ov_pit_loss", r.pit_lane_loss_s))
    ps = base.pit_stop
    pit_updates: dict[str, Any] = {"stationary_std_s": float(ss("ov_stationary_std", ps.stationary_std_s))}
    if ps.has_components and r.pit_lane_loss_s > 0:
        f = pit_loss / r.pit_lane_loss_s
        pit_updates.update(pit_entry_loss_s=ps.pit_entry_loss_s * f, pit_transit_loss_s=ps.pit_transit_loss_s * f,
                           pit_exit_loss_s=ps.pit_exit_loss_s * f)
    pit_updates.update(sc_pit_lane_loss_s=float(ss("ov_sc_pit_loss", ps.sc_pit_lane_loss_s)),
                       vsc_pit_lane_loss_s=float(ss("ov_vsc_pit_loss", ps.vsc_pit_lane_loss_s)))
    race_updates = {
        "total_laps": n,
        "pit_lane_loss_s": pit_loss,
        "stationary_stop_time_s": float(ss("ov_stationary", r.stationary_stop_time_s)),
        "pit_loss_std_s": float(ss("ov_pit_std", r.pit_loss_std_s)),
        "baseline_lap_time_std_s": float(ss("ov_lap_std", r.baseline_lap_time_std_s)),
        "safety_car_probability_per_lap": float(ss("ov_sc_prob", r.safety_car_probability_per_lap)),
        "vsc_probability_per_lap": float(ss("ov_vsc_prob", r.vsc_probability_per_lap)),
        "fuel_start_kg": fuel_start,
        "fuel_consumption_kg_per_lap": cons,
        "fuel_effect_s_per_kg": float(ss("ov_fuel_effect", r.fuel_effect_s_per_kg)),
        "minimum_stint_laps": int(ss("ov_min_stint", r.minimum_stint_laps)),
        "maximum_stint_laps": int(ss("ov_max_stint", r.maximum_stint_laps)),
        "maximum_stops": int(ss("ov_max_stops", r.maximum_stops)),
        "track_temperature_c": float(ss("ov_track_temp", r.track_temperature_c)),
        "simulation_seed": int(ss("mc_seed", r.simulation_seed)),
    }
    unc_scale = float(ss("ov_deg_unc_scale", 1.0))
    comp_updates = {}
    for name, c in base.compounds.items():
        comp_updates[name] = {
            "linear_degradation_s_per_lap": float(ss(f"ov_lin_{name}", c.linear_degradation_s_per_lap)),
            "nonlinear_degradation_coefficient": float(ss(f"ov_nl_{name}", c.nonlinear_degradation_coefficient)),
            "nonlinear_degradation_threshold_laps": float(ss(f"ov_thr_{name}", c.nonlinear_degradation_threshold_laps)),
            "maximum_recommended_age_laps": int(ss(f"ov_age_{name}", c.maximum_recommended_age_laps)),
            "degradation_uncertainty": c.degradation_uncertainty * unc_scale,
        }
    regs = base.regulations
    allowed = ss("ov_compounds", list(base.dry_compounds))
    permitted = [c for c in base.compounds if c in allowed or base.compounds[c].compound_type != "dry"]
    reg_updates = {
        "mandatory_pit_stops": int(ss("ov_mandatory", regs.mandatory_pit_stops if regs.mandatory_pit_stops is not None else 0)),
        "minimum_distinct_dry_compounds": int(ss("ov_distinct", regs.minimum_distinct_dry_compounds or 1)),
        "permitted_compounds": permitted,
    }
    rivals = []
    for rv in base.rivals:
        d = rv.model_dump()
        if n != r.total_laps:
            d["pit_laps"] = _scale_laps(rv.pit_laps, r.total_laps, n, r.minimum_pit_laps_remaining)
        rivals.append(d)
    bl = base.baseline_strategy.model_dump() if base.baseline_strategy else None
    if bl and n != r.total_laps:
        bl["pit_laps"] = _scale_laps(bl["pit_laps"], r.total_laps, n, r.minimum_pit_laps_remaining)
    # The temperature rate is only used by the rising/cooling scenarios.
    weather = {"scenario": ss("ov_weather", base.weather.scenario),
               "temperature_change_c_per_lap": float(ss("ov_temp_rate", 0.2)),
               "rain_onset_lap": min(int(ss("ov_rain_lap", max(2, n // 2))), n),
               "wet_weather_enabled": bool(ss("ov_wet_enabled", base.weather.wet_weather_enabled))}
    return with_overrides(setup, race=race_updates, pit_stop=pit_updates, compounds=comp_updates, regulations=reg_updates,
                          rivals=rivals, baseline_strategy=bl, weather=weather)


def snapshot_from_state(setup: SimulatorSetup) -> RaceSnapshot:
    lap = int(ss("snap_lap", 0))
    if lap == 0:
        return RaceSnapshot(current_lap=0)
    comp = ss("snap_compound", setup.dry_compounds[0])
    return RaceSnapshot(current_lap=lap, tyre_compound=comp, tyre_age_laps=int(ss("snap_age", lap)),
                        stops_completed=int(ss("snap_stops", 0)), compounds_used=list(ss("snap_used", [])))


@dataclass
class DashboardContext:
    setup: SimulatorSetup
    setup_json: str
    snapshot: RaceSnapshot
    snapshot_json: str
    data: Optional[IngestionResult] = None
    cleaning: Optional[CleaningResult] = None
    data_label: str = "none"
    config_warnings: list[str] = field(default_factory=list)
    config_error: Optional[str] = None

    @property
    def lap_data(self) -> Optional[pd.DataFrame]:
        return None if self.cleaning is None else self.cleaning.annotated


def make_context() -> DashboardContext:
    try:
        setup = build_setup()
        err = None
    except Exception as exc:  # validation errors are shown to the user, base config used instead
        setup, err = base_setup(), f"Invalid configuration override, defaults used instead: {exc}"
    snap = snapshot_from_state(setup)
    return DashboardContext(setup, setup.model_dump_json(), snap, snap.model_dump_json(),
                            config_warnings=regulation_warnings(setup), config_error=err)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def setup_from_json(js: str) -> SimulatorSetup:
    return SimulatorSetup.model_validate_json(js)


def json_key(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str)


# ---------------------------------------------------------------------------
# Cached computations (deterministic for their inputs)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Optimising strategies...", max_entries=32)
def cached_optimisation(setup_json: str, snapshot_json: str = "", max_stops: Optional[int] = None):
    from racesim.strategy_optimizer import find_optimal_strategy

    setup = setup_from_json(setup_json)
    snap = RaceSnapshot.model_validate_json(snapshot_json) if snapshot_json else None
    return find_optimal_strategy(setup, snap, max_stops=max_stops)


@st.cache_data(show_spinner="Calculating pit window...", max_entries=32)
def cached_pit_window(setup_json: str, snapshot_json: str, n_stops: tuple[int, ...], events_json: str = "[]"):
    from racesim.pit_window import calculate_pit_window
    from racesim.schemas import RaceEvent

    events = [RaceEvent(**e) for e in json.loads(events_json)] or None
    snap = RaceSnapshot.model_validate_json(snapshot_json)
    return calculate_pit_window(setup_from_json(setup_json), snap, n_stops=list(n_stops), events=events)


@st.cache_data(show_spinner="Simulating safety-car scenarios...", max_entries=16)
def cached_sc_table(setup_json: str, strategy_json: str, sc_laps: tuple[int, ...], duration: int, kind: str):
    from racesim.schemas import EventType, Strategy
    from racesim.workflow import safety_car_scenario_table

    strategies = [Strategy.model_validate(s) for s in json.loads(strategy_json)]
    return safety_car_scenario_table(setup_from_json(setup_json), strategies, sc_start_laps=list(sc_laps),
                                     duration_laps=duration, kind=EventType(kind))


@st.cache_data(show_spinner="Generating synthetic demo data...", max_entries=8)
def cached_synthetic(setup_json: str, seed: int, anomalies: bool):
    from racesim.synthetic_data import generate_synthetic_race

    return generate_synthetic_race(setup_from_json(setup_json), seed=seed, inject_anomalies=anomalies)
