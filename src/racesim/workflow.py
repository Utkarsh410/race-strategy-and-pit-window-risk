"""End-to-end analysis workflow shared by the CLI, the HTML report and the dashboard."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from racesim.config import SimulatorSetup, regulation_warnings
from racesim.monte_carlo import MonteCarloResult, run_monte_carlo
from racesim.pit_window import PitWindowResult, calculate_pit_window
from racesim.race_simulator import simulate_race
from racesim.risk_metrics import compare_objectives, risk_reports_from_monte_carlo
from racesim.schemas import EventType, RaceEvent, RiskReport, SimulationResult, Strategy, default_strategy_name
from racesim.sensitivity import (
    SensitivityResult,
    rank_sensitive_parameters,
    vary_lap_time_variability,
    vary_pit_loss,
    vary_safety_car_probability,
    vary_tyre_degradation,
)
from racesim.strategy_optimizer import OptimisationResult, find_optimal_strategy

logger = logging.getLogger(__name__)


def baseline_strategy(setup: SimulatorSetup) -> Optional[Strategy]:
    b = setup.baseline_strategy
    if b is None:
        return None
    return Strategy.from_plan(setup.race.total_laps, b.pit_laps, b.compounds,
                              strategy_name="Baseline: " + default_strategy_name(b.pit_laps, b.compounds))


def shortlist_strategies(opt: OptimisationResult, setup: SimulatorSetup, max_count: int = 5) -> list[Strategy]:
    """Recommended strategy, the best of every other stop count / compound pattern, and the baseline."""
    out: list[Strategy] = []
    if opt.best is not None:
        out.append(opt.best.strategy)
    for _, e in sorted(opt.best_by_stops().items()):
        if all(e.strategy.strategy_id != s.strategy_id for s in out):
            out.append(e.strategy)
    seen_patterns = {tuple(s.planned_compounds) for s in out}
    for e in opt.evaluations:
        if len(out) >= max_count - 1:
            break
        if e.feasible and tuple(e.strategy.planned_compounds) not in seen_patterns:
            out.append(e.strategy)
            seen_patterns.add(tuple(e.strategy.planned_compounds))
    base = baseline_strategy(setup)
    if base is not None and all(base.strategy_id != s.strategy_id for s in out):
        out.append(base)
    return out[:max_count + 1]


def safety_car_scenario_table(setup: SimulatorSetup, strategies: Sequence[Strategy], *,
                              sc_start_laps: Optional[Sequence[int]] = None, duration_laps: int = 4,
                              kind: EventType = EventType.SAFETY_CAR, reactive: bool = True) -> pd.DataFrame:
    """Deterministic race time of each strategy with a single SC (or VSC) at different laps.

    With ``reactive`` the team may bring a stop forward under neutralisation
    when the model expects a gain; it is never forced.
    """
    n = setup.race.total_laps
    sc_start_laps = sc_start_laps or [max(2, round(n * f)) for f in (0.15, 0.3, 0.45, 0.6, 0.75)]
    rows = []
    for st in strategies:
        rows.append({"scenario": "No neutralisation", "strategy": st.strategy_name, "strategy_id": st.strategy_id,
                     "race_time_s": simulate_race(setup, st, reactive=reactive).predicted_race_time_s, "pit_laps": st.planned_pit_laps})
        for lap in sc_start_laps:
            ev = [RaceEvent(event_type=kind, start_lap=lap, duration_laps=min(duration_laps, n - lap + 1))]
            r = simulate_race(setup, st, events=ev, reactive=reactive)
            rows.append({"scenario": f"{'SC' if kind == EventType.SAFETY_CAR else 'VSC'} lap {lap}", "strategy": st.strategy_name,
                         "strategy_id": st.strategy_id, "race_time_s": r.predicted_race_time_s,
                         "pit_laps": [p.lap for p in r.pit_stops]})
    df = pd.DataFrame(rows)
    df["delta_to_best_in_scenario_s"] = df["race_time_s"] - df.groupby("scenario")["race_time_s"].transform("min")
    return df


def monte_carlo_sc_conditioning(mc: MonteCarloResult) -> pd.DataFrame:
    """Mean simulated race time per strategy split by whether a safety car occurred."""
    rows = []
    for sid, t in mc.race_times.items():
        for label, mask in (("Runs with safety car", mc.sc_occurred), ("Runs without safety car", ~mc.sc_occurred)):
            if mask.any():
                rows.append({"scenario": label, "strategy": mc.strategies[sid].strategy_name, "strategy_id": sid,
                             "mean_race_time_s": float(t[mask].mean()), "runs": int(mask.sum())})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["delta_to_best_in_scenario_s"] = df["mean_race_time_s"] - df.groupby("scenario")["mean_race_time_s"].transform("min")
    return df


@dataclass
class AnalysisBundle:
    setup: SimulatorSetup
    optimisation: OptimisationResult
    pit_window: PitWindowResult
    recommended_simulation: Optional[SimulationResult]
    shortlist: list[Strategy]
    sc_scenarios: pd.DataFrame
    monte_carlo: Optional[MonteCarloResult] = None
    risk_reports: dict[str, RiskReport] = field(default_factory=dict)
    objectives: Optional[pd.DataFrame] = None
    sc_conditioning: Optional[pd.DataFrame] = None
    sensitivity: list[SensitivityResult] = field(default_factory=list)
    sensitivity_ranking: Optional[pd.DataFrame] = None
    warnings: list[str] = field(default_factory=list)
    data_provenance: dict = field(default_factory=dict)


def run_full_analysis(
    setup: SimulatorSetup,
    *,
    n_sims: Optional[int] = None,
    seed: Optional[int] = None,
    include_rivals: bool = True,
    run_sensitivity: bool = True,
    sensitivity_sims: int = 300,
    data_provenance: Optional[dict] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> AnalysisBundle:
    """Optimise, compute the pit window, simulate scenarios and quantify risk."""
    warnings = regulation_warnings(setup)
    opt = find_optimal_strategy(setup)
    if opt.best is None:
        raise RuntimeError("no feasible strategy satisfies the configured constraints")
    k = max(opt.best.number_of_stops, 1)
    window = calculate_pit_window(setup, n_stops=[k])
    rec_sim = simulate_race(setup, opt.best.strategy, rivals=setup.rivals or None)
    short = shortlist_strategies(opt, setup)
    sc = safety_car_scenario_table(setup, short)
    bundle = AnalysisBundle(setup, opt, window, rec_sim, short, sc, warnings=warnings,
                            data_provenance=data_provenance or {"source": "configuration only (SYNTHETIC parameters)"})
    if n_sims != 0:
        mc = run_monte_carlo(setup, short, n_sims=n_sims, seed=seed, include_rivals=include_rivals and bool(setup.rivals),
                             progress_callback=progress_callback)
        base = baseline_strategy(setup)
        bundle.monte_carlo = mc
        bundle.risk_reports = risk_reports_from_monte_carlo(mc, baseline_id=base.strategy_id if base else None)
        bundle.objectives = compare_objectives(bundle.risk_reports, {sid: s.strategy_name for sid, s in mc.strategies.items()})
        bundle.sc_conditioning = monte_carlo_sc_conditioning(mc)
        for r in bundle.risk_reports.values():
            bundle.warnings += [w for w in r.warnings if w not in bundle.warnings]
    if run_sensitivity:
        bundle.sensitivity = [
            vary_tyre_degradation(setup),
            vary_pit_loss(setup),
            vary_safety_car_probability(setup, n_sims=sensitivity_sims, seed=seed),
            vary_lap_time_variability(setup, n_sims=sensitivity_sims, seed=seed),
        ]
        bundle.sensitivity_ranking = rank_sensitive_parameters(bundle.sensitivity)
    return bundle


def paired_deltas(mc: MonteCarloResult, baseline_id: str) -> dict[str, np.ndarray]:
    """Per-run race-time difference of each strategy versus the baseline (s)."""
    if baseline_id not in mc.race_times:
        return {}
    b = mc.race_times[baseline_id]
    return {sid: t - b for sid, t in mc.race_times.items() if sid != baseline_id}
