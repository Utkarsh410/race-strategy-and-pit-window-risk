"""Deterministic strategy evaluation and optimisation.

All candidates are evaluated with the *same* deterministic race model
(the canonical lap-time model with zero random error, green flag, the
configured deterministic weather profile, mean pit costs). The objective is
minimum predicted total race time subject to the hard constraints checked by
:func:`racesim.strategy_generator.validate_strategy`.

Evaluation uses cached per-stint cumulative lap-time tables, so thousands of
candidates can be compared quickly; :func:`simulate_strategy` is verified in
the tests to match :func:`racesim.race_simulator.simulate_race` exactly.

The objective is pluggable (``objective`` argument) to support later
position-based or championship-points objectives.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from racesim.config import SimulatorSetup
from racesim.fuel_model import calculate_refuel_amount
from racesim.lap_time_model import LapModelContext, predict_lap_time
from racesim.pit_stop_model import calculate_total_pit_cost
from racesim.schemas import RaceSnapshot, Stint, Strategy, TrackStatus
from racesim.strategy_generator import StrategyValidation, generate_candidate_strategies, validate_strategy
from racesim.weather_model import generate_weather_scenario

logger = logging.getLogger(__name__)

MODEL_ASSUMPTIONS = (
    "Deterministic evaluation: zero random lap error, mean pit costs, no safety car or traffic.",
    "Pit cost = pit-lane loss (excl. stationary) + stationary time + rejoin loss, booked once on the in-lap.",
    "Out-lap warm-up penalty applied by the tyre model on stints that start after a pit stop.",
    "Fuel effect relative to an empty tank; fuel burn at the configured consumption assumption.",
    "Same race model and conditions for every candidate strategy.",
)


@dataclass
class StrategyEvaluation:
    """Deterministic evaluation of one strategy (times in seconds)."""

    strategy: Strategy
    predicted_race_time_s: float
    total_pit_cost_s: float
    tyre_time_s: float
    fuel_time_s: float
    racing_time_s: float
    feasible: bool
    reasons: list[str] = field(default_factory=list)
    assumptions: tuple[str, ...] = MODEL_ASSUMPTIONS

    @property
    def number_of_stops(self) -> int:
        return self.strategy.number_of_stops

    def as_row(self, baseline_time_s: Optional[float] = None) -> dict:
        s = self.strategy
        row = {
            "strategy_id": s.strategy_id,
            "strategy_name": s.strategy_name,
            "stops": s.number_of_stops,
            "pit_laps": ", ".join(map(str, s.planned_pit_laps)) or "-",
            "compounds": " > ".join(s.planned_compounds),
            "stint_lengths": ", ".join(map(str, s.stint_lengths)),
            "race_time_s": self.predicted_race_time_s,
            "total_pit_cost_s": self.total_pit_cost_s,
            "tyre_time_s": self.tyre_time_s,
            "fuel_time_s": self.fuel_time_s,
            "feasible": self.feasible,
            "infeasibility_reasons": "; ".join(self.reasons),
        }
        if baseline_time_s is not None:
            row["delta_to_baseline_s"] = self.predicted_race_time_s - baseline_time_s
        return row


class StrategyEvaluator:
    """Fast deterministic evaluator with cached stint tables."""

    def __init__(self, setup: SimulatorSetup, snapshot: Optional[RaceSnapshot] = None, ctx: Optional[LapModelContext] = None):
        self.setup = setup
        self.ctx = ctx or LapModelContext.from_setup(setup)
        self.snapshot = snapshot
        race = setup.race
        weather = generate_weather_scenario(setup.weather, race.total_laps, race.track_temperature_c)
        self.track_temp = weather.track_temperature_c
        self.wetness = weather.wetness
        self.start_lap = 1 if snapshot is None or snapshot.current_lap == 0 else snapshot.current_lap + 1
        default_fuel = race.fuel_start_kg - race.fuel_consumption_kg_per_lap * (self.start_lap - 1)
        self.start_fuel = snapshot.fuel_kg if snapshot is not None and snapshot.fuel_kg is not None else default_fuel
        if setup.refuelling_allowed and setup.regulations.fuel_capacity_kg is not None:
            self.start_fuel = min(self.start_fuel, setup.regulations.fuel_capacity_kg)
        self.green_pit = calculate_total_pit_cost(race, setup.pit_stop, TrackStatus.GREEN)
        self._tables: dict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def fuel_at(self, lap: np.ndarray | int) -> np.ndarray:
        """Fuel at lap start without refuelling (never negative)."""
        return np.maximum(self.start_fuel - self.setup.race.fuel_consumption_kg_per_lap * (np.asarray(lap) - self.start_lap), 0.0)

    def stint_arrays(self, stint: Stint, fuel_start_kg: Optional[float] = None) -> dict[str, np.ndarray]:
        """Per-lap total, tyre and fuel components for one stint."""
        laps = np.arange(stint.start_lap, stint.end_lap + 1)
        k = laps - stint.start_lap
        age = stint.starting_tyre_age_laps + k
        fuel = self.fuel_at(laps) if fuel_start_kg is None else np.maximum(fuel_start_kg - self.setup.race.fuel_consumption_kg_per_lap * k, 0.0)
        total, comp = predict_lap_time(
            self.ctx, self.ctx.table.index(stint.tyre_compound), laps, age, age, fuel, stint.start_lap > 1,
            self.track_temp[laps - 1], self.wetness[laps - 1],
            driver_s=self.setup.driver_offsets_s.get(stint.driver_id, 0.0), return_components=True,
        )
        return {"lap": laps, "lap_time_s": total, "tyre_s": np.broadcast_to(comp["tyre_s"], laps.shape),
                "fuel_s": np.broadcast_to(comp["fuel_s"], laps.shape)}

    def _table(self, stint: Stint) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = (stint.tyre_compound, stint.start_lap, stint.starting_tyre_age_laps, stint.driver_id)
        if key not in self._tables:
            full = Stint(start_lap=stint.start_lap, end_lap=self.setup.race.total_laps, tyre_compound=stint.tyre_compound,
                         starting_tyre_age_laps=stint.starting_tyre_age_laps, driver_id=stint.driver_id)
            a = self.stint_arrays(full)
            self._tables[key] = (np.cumsum(a["lap_time_s"]), np.cumsum(a["tyre_s"]), np.cumsum(a["fuel_s"]))
        return self._tables[key]

    def stint_totals(self, stint: Stint) -> tuple[float, float, float]:
        """(racing time, tyre time, fuel time) for a stint without refuelling."""
        tot, tyre, fuel = self._table(stint)
        i = stint.length_laps - 1
        return float(tot[i]), float(tyre[i]), float(fuel[i])


def simulate_stint(setup: SimulatorSetup, stint: Stint, *, fuel_start_kg: Optional[float] = None,
                   evaluator: Optional[StrategyEvaluator] = None) -> pd.DataFrame:
    """Deterministic per-lap prediction for a stint (s)."""
    ev = evaluator or StrategyEvaluator(setup)
    return pd.DataFrame(ev.stint_arrays(stint, fuel_start_kg))


def simulate_strategy(
    setup: SimulatorSetup,
    strategy: Strategy,
    *,
    evaluator: Optional[StrategyEvaluator] = None,
    check_feasibility: bool = True,
) -> StrategyEvaluation:
    """Deterministic evaluation of a complete strategy."""
    ev = evaluator or StrategyEvaluator(setup)
    validation = check_strategy_feasibility(setup, strategy, ev.snapshot) if check_feasibility else StrategyValidation(True)
    race = setup.race
    racing = tyre = fuel_t = pit_total = 0.0
    if setup.refuelling_allowed:
        fuel = ev.start_fuel
        cap = setup.regulations.fuel_capacity_kg or race.fuel_start_kg
        for k, st in enumerate(strategy.stints):
            a = ev.stint_arrays(st, fuel)
            racing += float(a["lap_time_s"].sum())
            tyre += float(a["tyre_s"].sum())
            fuel_t += float(a["fuel_s"].sum())
            fuel = max(fuel - race.fuel_consumption_kg_per_lap * st.length_laps, 0.0)
            if k < len(strategy.stints) - 1:
                nxt = strategy.stints[k + 1].length_laps
                added = float(calculate_refuel_amount(fuel, nxt, race.fuel_consumption_kg_per_lap, cap))
                pit_total += calculate_total_pit_cost(race, setup.pit_stop, TrackStatus.GREEN, fuel_added_kg=added,
                                                      regulations=setup.regulations).total_s
                fuel += added
    else:
        for st in strategy.stints:
            r, t, f = ev.stint_totals(st)
            racing, tyre, fuel_t = racing + r, tyre + t, fuel_t + f
        pit_total = ev.green_pit.total_s * strategy.number_of_stops
    return StrategyEvaluation(strategy, racing + pit_total, pit_total, tyre, fuel_t, racing,
                              validation.valid, validation.reasons)


def calculate_total_race_time(setup: SimulatorSetup, strategy: Strategy, evaluator: Optional[StrategyEvaluator] = None) -> float:
    """Predicted deterministic race time (s)."""
    return simulate_strategy(setup, strategy, evaluator=evaluator, check_feasibility=False).predicted_race_time_s


def check_strategy_feasibility(setup: SimulatorSetup, strategy: Strategy, snapshot: Optional[RaceSnapshot] = None) -> StrategyValidation:
    """Hard-constraint check (delegates to :func:`validate_strategy`)."""
    return validate_strategy(strategy, setup, snapshot)


def compare_strategies(
    setup: SimulatorSetup,
    strategies: Sequence[Strategy],
    *,
    baseline: Optional[Strategy] = None,
    evaluator: Optional[StrategyEvaluator] = None,
    include_infeasible: bool = False,
) -> pd.DataFrame:
    """Rank strategies by predicted race time under identical assumptions."""
    ev = evaluator or StrategyEvaluator(setup)
    evals = [simulate_strategy(setup, s, evaluator=ev) for s in strategies]
    base_t = simulate_strategy(setup, baseline, evaluator=ev).predicted_race_time_s if baseline is not None else None
    rows = [e.as_row(base_t) for e in evals if include_infeasible or e.feasible]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(["feasible", "race_time_s"], ascending=[False, True]).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    df["delta_to_best_s"] = df["race_time_s"] - df.loc[df["feasible"], "race_time_s"].min()
    return df


@dataclass
class OptimisationResult:
    best: Optional[StrategyEvaluation]
    evaluations: list[StrategyEvaluation]
    ranking: pd.DataFrame
    candidates_evaluated: int
    rejected: list[tuple[str, str]]
    explanation: str
    baseline: Optional[StrategyEvaluation] = None
    lap_steps: dict[int, int] = field(default_factory=dict)

    def best_by_stops(self) -> dict[int, StrategyEvaluation]:
        out: dict[int, StrategyEvaluation] = {}
        for e in self.evaluations:
            if e.feasible and (e.number_of_stops not in out or e.predicted_race_time_s < out[e.number_of_stops].predicted_race_time_s):
                out[e.number_of_stops] = e
        return out


def _neighbours(strategy: Strategy, setup: SimulatorSetup, radius: int) -> list[Strategy]:
    out = []
    for i in range(strategy.number_of_stops):
        for d in range(-radius, radius + 1):
            if d == 0:
                continue
            laps = list(strategy.planned_pit_laps)
            laps[i] += d
            try:
                out.append(Strategy.from_plan(setup.race.total_laps, laps, list(strategy.planned_compounds),
                                              start_lap=strategy.start_lap,
                                              starting_tyre_age_laps=strategy.stints[0].starting_tyre_age_laps))
            except ValueError:
                continue
    return out


def _refine(setup: SimulatorSetup, start: StrategyEvaluation, ev: StrategyEvaluator, radius: int,
            objective: Callable[[StrategyEvaluation], float], seen: dict[str, StrategyEvaluation]) -> StrategyEvaluation:
    """Coordinate-descent refinement of pit laps at single-lap resolution."""
    best = start
    improved = True
    while improved:
        improved = False
        for cand in _neighbours(best.strategy, setup, radius):
            if cand.strategy_id not in seen:
                seen[cand.strategy_id] = simulate_strategy(setup, cand, evaluator=ev)
            e = seen[cand.strategy_id]
            if e.feasible and objective(e) < objective(best) - 1e-9:
                best, improved = e, True
    return best


def explain_recommendation(best: StrategyEvaluation, evaluations: Sequence[StrategyEvaluation]) -> str:
    """Text explaining why the best strategy ranks above key alternatives."""
    feas = sorted([e for e in evaluations if e.feasible], key=lambda e: e.predicted_race_time_s)
    if not feas:
        return "No feasible strategy found."
    lines = [
        f"Recommended: {best.strategy.strategy_name} - predicted {best.predicted_race_time_s:.2f} s "
        f"({best.number_of_stops} stop(s), pit cost {best.total_pit_cost_s:.1f} s, tyre time {best.tyre_time_s:.1f} s)."
    ]
    best_pattern = tuple(best.strategy.planned_compounds)
    by_pattern: dict[tuple, StrategyEvaluation] = {}
    for e in feas:
        by_pattern.setdefault(tuple(e.strategy.planned_compounds), e)
    by_stops: dict[int, StrategyEvaluation] = {}
    for pattern, e in by_pattern.items():
        if pattern != best_pattern:
            by_stops.setdefault(e.number_of_stops, e)
    # Closest different compound pattern first, then the best of every other stop count.
    alternatives = [e for p, e in by_pattern.items() if p != best_pattern][:1]
    alternatives += [e for _, e in sorted(by_stops.items()) if e not in alternatives]
    same_pattern = [e for e in feas if tuple(e.strategy.planned_compounds) == best_pattern and e is not best]
    if same_pattern:
        lines.append(f"{len(same_pattern)} timing variant(s) of the same compound sequence lie within "
                     f"{same_pattern[-1].predicted_race_time_s - best.predicted_race_time_s:.2f} s; "
                     f"the nearest is +{same_pattern[0].predicted_race_time_s - best.predicted_race_time_s:.2f} s.")
    for alt in alternatives[:4]:
        d = alt.predicted_race_time_s - best.predicted_race_time_s
        d_pit = alt.total_pit_cost_s - best.total_pit_cost_s
        d_tyre = alt.tyre_time_s - best.tyre_time_s
        d_fuel = alt.fuel_time_s - best.fuel_time_s
        d_base = d - d_pit - d_tyre - d_fuel
        lines.append(
            f"vs {alt.strategy.strategy_name}: +{d:.2f} s (pit {d_pit:+.2f}, tyre {d_tyre:+.2f}, fuel {d_fuel:+.2f}, "
            f"compound base pace/other {d_base:+.2f})."
        )
    lines.append("Deterministic ranking only: it does not account for safety cars, traffic or variability; "
                 "see the Monte Carlo risk analysis before committing. No strategy guarantees a result.")
    return "\n".join(lines)


def find_optimal_strategy(
    setup: SimulatorSetup,
    snapshot: Optional[RaceSnapshot] = None,
    *,
    max_stops: Optional[int] = None,
    min_stops: int = 0,
    compounds: Optional[Sequence[str]] = None,
    refine_top: int = 12,
    objective: Optional[Callable[[StrategyEvaluation], float]] = None,
    candidates: Optional[Sequence[Strategy]] = None,
) -> OptimisationResult:
    """Find the minimum-race-time feasible strategy.

    1. Generate candidates (coarse grid within the candidate limit).
    2. Evaluate all with the same deterministic model.
    3. Refine the best ``refine_top`` candidates (distinct compound/stop
       patterns) by single-lap coordinate descent.
    """
    objective = objective or (lambda e: e.predicted_race_time_s)
    ev = StrategyEvaluator(setup, snapshot)
    if candidates is None:
        cset = generate_candidate_strategies(setup, snapshot, max_stops=max_stops, min_stops=min_stops, compounds=compounds)
        cands, rejected, steps = cset.strategies, cset.rejected, cset.lap_step_by_stops
    else:
        cands, rejected, steps = list(candidates), [], {}
    seen: dict[str, StrategyEvaluation] = {}
    for s in cands:
        seen[s.strategy_id] = simulate_strategy(setup, s, evaluator=ev)
    feas = sorted([e for e in seen.values() if e.feasible], key=objective)
    radius = max(max(steps.values(), default=1), 1)
    patterns: dict[tuple, StrategyEvaluation] = {}
    for e in feas:
        patterns.setdefault((tuple(e.strategy.planned_compounds),), e)
    starts = list(patterns.values())[:refine_top]
    for e in starts:
        _refine(setup, e, ev, radius, objective, seen)
    evaluations = sorted(seen.values(), key=lambda e: (not e.feasible, objective(e)))
    feasible = [e for e in evaluations if e.feasible]
    best = feasible[0] if feasible else None
    baseline_eval = None
    if setup.baseline_strategy is not None and (snapshot is None or snapshot.current_lap == 0):
        try:
            base = Strategy.from_plan(setup.race.total_laps, setup.baseline_strategy.pit_laps, setup.baseline_strategy.compounds,
                                      strategy_name="Baseline: " + "-".join(c[0] for c in setup.baseline_strategy.compounds))
            baseline_eval = simulate_strategy(setup, base, evaluator=ev)
        except ValueError as exc:
            logger.warning("Baseline strategy invalid: %s", exc)
    base_t = baseline_eval.predicted_race_time_s if baseline_eval else None
    ranking = pd.DataFrame([e.as_row(base_t) for e in feasible])
    if not ranking.empty:
        ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
        ranking["delta_to_best_s"] = ranking["race_time_s"] - ranking["race_time_s"].iloc[0]
    explanation = explain_recommendation(best, evaluations) if best else "No feasible strategy satisfies the constraints."
    if best is None:
        logger.warning("No feasible strategy found (%d candidates, %d rejections)", len(cands), len(rejected))
    else:
        logger.info("Optimal strategy %s: %.3f s", best.strategy.strategy_id, best.predicted_race_time_s)
    return OptimisationResult(best, evaluations, ranking, len(seen), rejected, explanation, baseline_eval, steps)


# ---------------------------------------------------------------------------
# Undercut / overcut (same-car, deterministic, green flag)
# ---------------------------------------------------------------------------


def _window_time(ev: StrategyEvaluator, compound: str, age_at_first_lap: int, first_lap: int, pit_lap: int,
                 new_compound: str, last_lap: int, pace_offset_s: float = 0.0, warmup_first: bool = False) -> float:
    ctx = ev.ctx
    laps_old = np.arange(first_lap, pit_lap + 1)
    laps_new = np.arange(pit_lap + 1, last_lap + 1)
    t = 0.0
    if laps_old.size:
        age = age_at_first_lap + (laps_old - first_lap)
        t += float(np.sum(predict_lap_time(ctx, ctx.table.index(compound), laps_old, age, age, ev.fuel_at(laps_old), warmup_first,
                                           ev.track_temp[laps_old - 1], ev.wetness[laps_old - 1], pace_offset_s=pace_offset_s)))
    t += ev.green_pit.total_s
    if laps_new.size:
        age = laps_new - (pit_lap + 1)
        t += float(np.sum(predict_lap_time(ctx, ctx.table.index(new_compound), laps_new, age, age, ev.fuel_at(laps_new), True,
                                           ev.track_temp[laps_new - 1], ev.wetness[laps_new - 1], pace_offset_s=pace_offset_s)))
    return t


def calculate_undercut_gain(
    setup: SimulatorSetup,
    *,
    tyre_compound: str,
    tyre_age_laps_at_pit: int,
    new_compound: str,
    own_pit_lap: int,
    rival_pit_lap: int,
    rival_tyre_compound: Optional[str] = None,
    rival_tyre_age_laps_at_pit: Optional[int] = None,
    rival_new_compound: Optional[str] = None,
    rival_pace_offset_s: float = 0.0,
) -> dict[str, float]:
    """Time gained (s) relative to a rival by pitting ``rival_pit_lap - own_pit_lap`` laps earlier.

    Both cars are timed from lap ``own_pit_lap`` to lap ``rival_pit_lap + 1``
    (the rival's out-lap). ``tyre_age_laps_at_pit`` is the tyre age at the
    start of ``own_pit_lap``. Positive gain means the undercut gains time;
    it is computed from tyre-age pace difference, out-lap warm-up and pit
    costs, not a fixed bonus.
    """
    if rival_pit_lap <= own_pit_lap:
        raise ValueError("undercut requires the rival to pit after the own car")
    ev = StrategyEvaluator(setup)
    r_comp = rival_tyre_compound or tyre_compound
    r_age = tyre_age_laps_at_pit if rival_tyre_age_laps_at_pit is None else rival_tyre_age_laps_at_pit
    r_new = rival_new_compound or new_compound
    end = rival_pit_lap + 1
    own = _window_time(ev, tyre_compound, tyre_age_laps_at_pit, own_pit_lap, own_pit_lap, new_compound, end)
    rival = _window_time(ev, r_comp, r_age, own_pit_lap, rival_pit_lap, r_new, end, rival_pace_offset_s)
    return {"gain_s": rival - own, "own_time_s": own, "rival_time_s": rival, "window_start_lap": own_pit_lap, "window_end_lap": end}


def calculate_overcut_gain(
    setup: SimulatorSetup,
    *,
    tyre_compound: str,
    tyre_age_laps_at_rival_pit: int,
    new_compound: str,
    own_pit_lap: int,
    rival_pit_lap: int,
    rival_tyre_compound: Optional[str] = None,
    rival_tyre_age_laps_at_pit: Optional[int] = None,
    rival_new_compound: Optional[str] = None,
    rival_pace_offset_s: float = 0.0,
) -> dict[str, float]:
    """Time gained (s) relative to a rival by staying out until ``own_pit_lap > rival_pit_lap``.

    Both cars are timed from ``rival_pit_lap`` to ``own_pit_lap + 1``.
    """
    if own_pit_lap <= rival_pit_lap:
        raise ValueError("overcut requires the own car to pit after the rival")
    ev = StrategyEvaluator(setup)
    r_comp = rival_tyre_compound or tyre_compound
    r_age = tyre_age_laps_at_rival_pit if rival_tyre_age_laps_at_pit is None else rival_tyre_age_laps_at_pit
    r_new = rival_new_compound or new_compound
    end = own_pit_lap + 1
    own = _window_time(ev, tyre_compound, tyre_age_laps_at_rival_pit, rival_pit_lap, own_pit_lap, new_compound, end)
    rival = _window_time(ev, r_comp, r_age, rival_pit_lap, rival_pit_lap, r_new, end, rival_pace_offset_s)
    return {"gain_s": rival - own, "own_time_s": own, "rival_time_s": rival, "window_start_lap": rival_pit_lap, "window_end_lap": end}
