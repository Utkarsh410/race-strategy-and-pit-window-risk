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
    """Everything a report can show. ``analyses_run`` lists what was actually executed;
    sections for anything else are labelled as not run."""

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
    # v2 additions
    precision: Optional[pd.DataFrame] = None
    convergence: Optional[pd.DataFrame] = None
    convergence_summary: Optional[dict] = None
    tail_stability: dict = field(default_factory=dict)
    uncertainty_breakdown: Optional[pd.DataFrame] = None
    rolling_race: Optional[object] = None          # replanning.RollingRaceResult
    validation: Optional[object] = None            # validation.ModelValidationResult
    model_uncertainty: list[str] = field(default_factory=list)
    analyses_run: list[str] = field(default_factory=list)

    @property
    def decision_log(self):
        return None if self.rolling_race is None else self.rolling_race.decision_log


def run_sensitivity_suite(setup: SimulatorSetup, *, sensitivity_sims: int = 300, seed: Optional[int] = None) -> list[SensitivityResult]:
    """The standard set of sensitivity sweeps used by the CLI, report and dashboard."""
    return [
        vary_tyre_degradation(setup),
        vary_pit_loss(setup),
        vary_safety_car_probability(setup, n_sims=sensitivity_sims, seed=seed),
        vary_lap_time_variability(setup, n_sims=sensitivity_sims, seed=seed),
    ]


def default_replan_events(setup: SimulatorSetup, *, sc_lap: Optional[int] = None, sc_laps: int = 4,
                          rain_lap: Optional[int] = None, rain_laps: int = 12, rain_severity: float = 0.8) -> list[RaceEvent]:
    """Manual events for a rolling-horizon demonstration."""
    ev: list[RaceEvent] = []
    n = setup.race.total_laps
    if sc_lap:
        ev.append(RaceEvent(event_type=EventType.SAFETY_CAR, start_lap=sc_lap, duration_laps=min(sc_laps, n - sc_lap + 1)))
    if rain_lap:
        ev.append(RaceEvent(event_type=EventType.RAIN, start_lap=rain_lap, duration_laps=min(rain_laps, n - rain_lap + 1),
                            severity=rain_severity))
    return ev


def run_replanning_demo(setup: SimulatorSetup, *, strategy: Optional[Strategy] = None, events: Optional[Sequence[RaceEvent]] = None,
                        n_sims: Optional[int] = None, risk_aversion: Optional[float] = None, seed: Optional[int] = None,
                        truth_rivals=None):
    """Deterministic rolling-horizon race starting from ``strategy`` (default: the baseline plan)."""
    from racesim.replanning import ReplanningEngine, run_rolling_race

    strategy = strategy or baseline_strategy(setup) or find_optimal_strategy(setup).best.strategy
    engine = ReplanningEngine(setup, n_sims=n_sims, risk_aversion=risk_aversion, seed=seed)
    return run_rolling_race(setup, strategy, events=events, engine=engine, truth_rivals=truth_rivals)


def compute_uncertainty_diagnostics(bundle: "AnalysisBundle") -> None:
    """Simulation-precision diagnostics for the recommended strategy (in place)."""
    from racesim.uncertainty import (assess_tail_stability, convergence_diagnostics, convergence_summary,
                                     simulation_precision_table)

    mc = bundle.monte_carlo
    if mc is None:
        return
    base = baseline_strategy(bundle.setup)
    base_id = base.strategy_id if base is not None and base.strategy_id in mc.race_times else None
    rec = bundle.optimisation.best.strategy.strategy_id
    bundle.precision = simulation_precision_table(mc, base_id)
    if rec in mc.race_times:
        bundle.convergence = convergence_diagnostics(mc.race_times[rec], mc.race_times[base_id] if base_id and base_id != rec else None)
        bundle.convergence_summary = convergence_summary(bundle.convergence)
    bundle.tail_stability = {sid: assess_tail_stability(t, n_boot=200) for sid, t in mc.race_times.items()}
    if "uncertainty diagnostics" not in bundle.analyses_run:
        bundle.analyses_run.append("uncertainty diagnostics")


def assemble_bundle(
    setup: SimulatorSetup,
    *,
    optimisation: Optional[OptimisationResult] = None,
    monte_carlo: Optional[MonteCarloResult] = None,
    sensitivity: Optional[list[SensitivityResult]] = None,
    rolling_race=None,
    validation=None,
    uncertainty_breakdown: Optional[pd.DataFrame] = None,
    data_provenance: Optional[dict] = None,
) -> AnalysisBundle:
    """Build a report bundle from analyses that have already been run (no hidden recomputation
    of Monte Carlo, sensitivity, replanning or validation)."""
    opt = optimisation or find_optimal_strategy(setup)
    if opt.best is None:
        raise RuntimeError("no feasible strategy satisfies the configured constraints")
    k = max(opt.best.number_of_stops, 1)
    window = calculate_pit_window(setup, n_stops=[k])
    rec_sim = simulate_race(setup, opt.best.strategy, rivals=setup.rivals or None)
    short = shortlist_strategies(opt, setup)
    sc = safety_car_scenario_table(setup, short)
    bundle = AnalysisBundle(setup, opt, window, rec_sim, short, sc, warnings=list(regulation_warnings(setup)),
                            data_provenance=data_provenance or {"source": "configuration only (SYNTHETIC parameters)"},
                            analyses_run=["deterministic optimisation", "pit window", "safety-car scenarios"])
    if monte_carlo is not None:
        base = baseline_strategy(setup)
        bundle.monte_carlo = monte_carlo
        bundle.risk_reports = risk_reports_from_monte_carlo(monte_carlo, baseline_id=base.strategy_id if base else None)
        names = {sid: st.strategy_name for sid, st in monte_carlo.strategies.items()}
        bundle.objectives = compare_objectives(bundle.risk_reports, names)
        bundle.sc_conditioning = monte_carlo_sc_conditioning(monte_carlo)
        for r in bundle.risk_reports.values():
            bundle.warnings += [w for w in r.warnings if w not in bundle.warnings]
        bundle.analyses_run.append("Monte Carlo")
        compute_uncertainty_diagnostics(bundle)
    if sensitivity:
        bundle.sensitivity = list(sensitivity)
        bundle.sensitivity_ranking = rank_sensitive_parameters(bundle.sensitivity)
        bundle.analyses_run.append("sensitivity")
    if uncertainty_breakdown is not None:
        bundle.uncertainty_breakdown = uncertainty_breakdown
        bundle.analyses_run.append("uncertainty breakdown")
    if rolling_race is not None:
        bundle.rolling_race = rolling_race
        bundle.analyses_run.append("dynamic replanning")
    if validation is not None:
        bundle.validation = validation
        bundle.analyses_run.append("model validation")
    from racesim.uncertainty import model_uncertainty_summary

    vstat = None
    if validation is not None:
        vstat = f"{validation.status} ({validation.evidence_type})"
    bundle.model_uncertainty = model_uncertainty_summary(setup, sensitivity_ranking=bundle.sensitivity_ranking,
                                                         validation_status=vstat, breakdown=uncertainty_breakdown)
    return bundle


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
    replan_events: Optional[Sequence[RaceEvent]] = None,
    run_replanning: bool = False,
    replan_sims: Optional[int] = None,
    risk_aversion: Optional[float] = None,
    validation_data: Optional[pd.DataFrame] = None,
    run_breakdown: bool = False,
    breakdown_sims: int = 1000,
) -> AnalysisBundle:
    """Optimise, compute the pit window, simulate scenarios and quantify risk.

    Optional v2 analyses: rolling-horizon replanning (``run_replanning``),
    model validation on ``validation_data`` and an uncertainty-source
    breakdown (``run_breakdown``).
    """
    opt = find_optimal_strategy(setup)
    if opt.best is None:
        raise RuntimeError("no feasible strategy satisfies the configured constraints")
    mc = None
    if n_sims != 0:
        short = shortlist_strategies(opt, setup)
        mc = run_monte_carlo(setup, short, n_sims=n_sims, seed=seed, include_rivals=include_rivals and bool(setup.rivals),
                             progress_callback=progress_callback)
    sens = run_sensitivity_suite(setup, sensitivity_sims=sensitivity_sims, seed=seed) if run_sensitivity else None
    rolling = None
    if run_replanning:
        rolling = run_replanning_demo(setup, events=replan_events, n_sims=replan_sims, risk_aversion=risk_aversion, seed=seed)
    val = None
    if validation_data is not None:
        from racesim.validation import run_model_validation

        val = run_model_validation(validation_data, setup, provenance=data_provenance)
    breakdown = None
    if run_breakdown:
        from racesim.uncertainty import uncertainty_breakdown as _breakdown

        base = baseline_strategy(setup)
        breakdown = _breakdown(setup, opt.best.strategy, base, n_sims=breakdown_sims, seed=seed,
                               include_rivals=include_rivals and bool(setup.rivals))
    return assemble_bundle(setup, optimisation=opt, monte_carlo=mc, sensitivity=sens, rolling_race=rolling, validation=val,
                           uncertainty_breakdown=breakdown, data_provenance=data_provenance)


def paired_deltas(mc: MonteCarloResult, baseline_id: str) -> dict[str, np.ndarray]:
    """Per-run race-time difference of each strategy versus the baseline (s)."""
    if baseline_id not in mc.race_times:
        return {}
    b = mc.race_times[baseline_id]
    return {sid: t - b for sid, t in mc.race_times.items() if sid != baseline_id}
