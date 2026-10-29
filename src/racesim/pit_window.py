"""Pit-window calculator.

For every candidate in-lap ``L`` of the *next* stop, the calculator finds the
best feasible continuation (remaining stops, compounds and pit laps) with the
deterministic race model and reports its predicted race time. The
recommended lap is the minimum-cost feasible lap - never the midpoint of the
window. Infeasible laps carry explicit rejection reasons.

Constraints considered: current lap, remaining distance, current tyre age,
minimum / maximum stint length, recommended tyre life, minimum laps
remaining after a stop, fuel feasibility, mandatory stops, compound rules,
driver-time rules (via :func:`validate_strategy`), pit cost and expected
degradation. Optional manual race-control events are evaluated by full
simulation of each lap's best continuation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from racesim.config import SimulatorSetup
from racesim.race_simulator import simulate_race
from racesim.schemas import RaceEvent, RaceSnapshot, Strategy
from racesim.strategy_generator import generate_compound_sequences, generate_stop_sequences, validate_strategy
from racesim.strategy_optimizer import StrategyEvaluator, find_optimal_strategy, simulate_strategy

logger = logging.getLogger(__name__)


@dataclass
class PitWindowResult:
    """Pit-window analysis for the next stop."""

    table: pd.DataFrame
    earliest_feasible_lap: Optional[int]
    latest_feasible_lap: Optional[int]
    recommended_lap: Optional[int]
    recommended_strategy: Optional[Strategy]
    recommended_race_time_s: Optional[float]
    stops_considered: list[int]
    warnings: list[str] = field(default_factory=list)


def _context(setup: SimulatorSetup, snapshot: Optional[RaceSnapshot]) -> tuple[int, int, int, list[str], int]:
    """(first lap of the remaining race, laps already on set, start age, compounds used, stops done)."""
    if snapshot is None or snapshot.current_lap == 0:
        return 1, 0, snapshot.tyre_age_laps if snapshot else 0, [], 0
    used = list(dict.fromkeys(snapshot.compounds_used + ([snapshot.tyre_compound] if snapshot.tyre_compound else [])))
    return snapshot.current_lap + 1, snapshot.tyre_age_laps, snapshot.tyre_age_laps, used, snapshot.stops_completed


def calculate_earliest_feasible_pit_lap(setup: SimulatorSetup, snapshot: Optional[RaceSnapshot] = None) -> int:
    """Earliest in-lap allowed by the minimum stint length (before other checks)."""
    start, on_set, _, _, _ = _context(setup, snapshot)
    return max(start, start - 1 + setup.effective_min_stint_laps - on_set)


def calculate_latest_feasible_pit_lap(setup: SimulatorSetup, snapshot: Optional[RaceSnapshot] = None,
                                      compound: Optional[str] = None) -> int:
    """Latest in-lap allowed by max stint length, recommended tyre life and laps remaining."""
    start, on_set, age0, _, _ = _context(setup, snapshot)
    latest = min(start - 1 + setup.effective_max_stint_laps - on_set,
                 setup.race.total_laps - setup.race.minimum_pit_laps_remaining)
    comp = compound or (snapshot.tyre_compound if snapshot else None)
    if comp is not None:
        latest = min(latest, start - 1 + setup.compounds[comp].maximum_recommended_age_laps - age0)
    return latest


def _continuations(setup: SimulatorSetup, snapshot: Optional[RaceSnapshot], pit_lap: int, n_stops: int,
                   budget: int) -> list[Strategy]:
    start, on_set, age0, used, _ = _context(setup, snapshot)
    first = snapshot.tyre_compound if snapshot else None
    seqs = generate_compound_sequences(setup, n_stops, first_compound=first, compounds_used=used)
    kwargs = dict(min_stint_laps=setup.effective_min_stint_laps, max_stint_laps=setup.effective_max_stint_laps,
                  min_laps_remaining=setup.race.minimum_pit_laps_remaining)
    step = 1
    tails = generate_stop_sequences(pit_lap + 1, setup.race.total_laps, n_stops - 1, **kwargs, lap_step=step)
    while len(tails) * max(len(seqs), 1) > budget and step < setup.race.total_laps:
        step += 1
        tails = generate_stop_sequences(pit_lap + 1, setup.race.total_laps, n_stops - 1, **kwargs, lap_step=step)
    out = []
    for seq in seqs:
        for tail in tails:
            try:
                out.append(Strategy.from_plan(setup.race.total_laps, [pit_lap, *tail], list(seq), start_lap=start,
                                              starting_tyre_age_laps=age0))
            except ValueError:
                continue
    return out


def calculate_pit_window_costs(
    setup: SimulatorSetup,
    snapshot: Optional[RaceSnapshot] = None,
    *,
    n_stops: Optional[Sequence[int]] = None,
    events: Optional[Sequence[RaceEvent]] = None,
    budget_per_lap: int = 600,
) -> pd.DataFrame:
    """Predicted race time of the best continuation for each candidate in-lap.

    ``n_stops`` lists the remaining stop counts to consider (default: 1 to the
    planning maximum). Returns one row per lap with feasibility flags,
    rejection reasons and, when ``events`` are given, the cost under that
    race-control scenario.
    """
    start, on_set, age0, _, stops_done = _context(setup, snapshot)
    race = setup.race
    stop_counts = list(n_stops) if n_stops is not None else list(range(1, race.maximum_stops + 1))
    ev = StrategyEvaluator(setup, snapshot)
    rows = []
    last_lap = race.total_laps - 1
    for lap in range(start, last_lap + 1):
        stint_len = lap - start + 1 + on_set
        reasons: list[str] = []
        if stint_len < setup.effective_min_stint_laps:
            reasons.append(f"current stint {stint_len} laps < minimum {setup.effective_min_stint_laps}")
        if stint_len > setup.effective_max_stint_laps:
            reasons.append(f"current stint {stint_len} laps > maximum {setup.effective_max_stint_laps}")
        if race.total_laps - lap < race.minimum_pit_laps_remaining:
            reasons.append(f"fewer than {race.minimum_pit_laps_remaining} laps remain after the stop")
        best = None
        first_failure = None
        if not reasons:
            for k in stop_counts:
                for strat in _continuations(setup, snapshot, lap, k, budget_per_lap):
                    check = validate_strategy(strat, setup, snapshot)
                    if not check.valid:
                        first_failure = first_failure or check.reasons[0]
                        continue
                    e = simulate_strategy(setup, strat, evaluator=ev, check_feasibility=False)
                    if best is None or e.predicted_race_time_s < best.predicted_race_time_s:
                        best = e
            if best is None:
                reasons.append(first_failure or "no feasible continuation")
        row = {
            "pit_lap": lap,
            "feasible": best is not None,
            "tyre_age_at_stop_laps": age0 + (lap - start + 1),
            "predicted_race_time_s": best.predicted_race_time_s if best else np.nan,
            "best_continuation": best.strategy.strategy_name if best else "",
            "strategy_id": best.strategy.strategy_id if best else "",
            "stops_remaining": best.strategy.number_of_stops if best else np.nan,
            "rejection_reasons": "; ".join(reasons),
        }
        if events and best is not None:
            row["scenario_race_time_s"] = simulate_race(setup, best.strategy, events=events, reactive=False).predicted_race_time_s
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty and df["feasible"].any():
        df["delta_to_best_s"] = df["predicted_race_time_s"] - df.loc[df["feasible"], "predicted_race_time_s"].min()
        if "scenario_race_time_s" in df.columns:
            df["scenario_delta_to_best_s"] = df["scenario_race_time_s"] - df["scenario_race_time_s"].min()
    return df


def rank_pit_laps(window: pd.DataFrame, column: str = "predicted_race_time_s") -> pd.DataFrame:
    """Feasible pit laps ordered by predicted cost (lowest first)."""
    feas = window[window["feasible"]].sort_values([column, "pit_lap"]).reset_index(drop=True)
    feas.insert(0, "rank", np.arange(1, len(feas) + 1))
    return feas


def validate_pit_window(result: PitWindowResult) -> list[str]:
    """Consistency checks on a pit-window result; returns problems found."""
    problems = []
    if result.recommended_lap is None:
        problems.append("No feasible pit lap: the window is empty under the configured constraints.")
        return problems
    if not (result.earliest_feasible_lap <= result.recommended_lap <= result.latest_feasible_lap):
        problems.append("Recommended lap lies outside the feasible window.")
    row = result.table.set_index("pit_lap").loc[result.recommended_lap]
    if not bool(row["feasible"]):
        problems.append("Recommended lap is not feasible.")
    feas = result.table[result.table["feasible"]]
    if float(row["predicted_race_time_s"]) > float(feas["predicted_race_time_s"].min()) + 1e-9:
        problems.append("Recommended lap is not the minimum-cost feasible lap.")
    gaps = np.diff(feas["pit_lap"].to_numpy())
    if np.any(gaps > 1):
        problems.append("Feasible laps are not contiguous: the window contains infeasible gaps.")
    return problems


def calculate_pit_window(
    setup: SimulatorSetup,
    snapshot: Optional[RaceSnapshot] = None,
    *,
    n_stops: Optional[Sequence[int]] = None,
    events: Optional[Sequence[RaceEvent]] = None,
    budget_per_lap: int = 600,
) -> PitWindowResult:
    """Full pit-window analysis for the next stop.

    If ``n_stops`` is not given, the remaining stop count of the overall
    optimal strategy is used (at least one stop).
    """
    warnings: list[str] = []
    if n_stops is None:
        opt = find_optimal_strategy(setup, snapshot)
        k = max(opt.best.number_of_stops, 1) if opt.best else 1
        n_stops = [k]
    table = calculate_pit_window_costs(setup, snapshot, n_stops=n_stops, events=events, budget_per_lap=budget_per_lap)
    feas = table[table["feasible"]] if not table.empty else table
    if feas.empty:
        warnings.append("No feasible pit lap under the configured constraints.")
        return PitWindowResult(table, None, None, None, None, None, list(n_stops), warnings)
    best_row = rank_pit_laps(table).iloc[0]
    start, _, age0, _, _ = _context(setup, snapshot)
    # Rebuild the recommended strategy from its identifier components.
    parts = best_row["strategy_id"].split(":", 1)[1].split(">")
    comps, laps = parts[0::2], [int(x) for x in parts[1::2]]
    rec_strategy = Strategy.from_plan(setup.race.total_laps, laps, comps, start_lap=start, starting_tyre_age_laps=age0)
    result = PitWindowResult(
        table=table,
        earliest_feasible_lap=int(feas["pit_lap"].min()),
        latest_feasible_lap=int(feas["pit_lap"].max()),
        recommended_lap=int(best_row["pit_lap"]),
        recommended_strategy=rec_strategy,
        recommended_race_time_s=float(best_row["predicted_race_time_s"]),
        stops_considered=list(n_stops),
        warnings=warnings,
    )
    result.warnings += validate_pit_window(result)
    mid = (result.earliest_feasible_lap + result.latest_feasible_lap) / 2
    logger.info("Pit window %d-%d, recommended %d (midpoint %.1f)", result.earliest_feasible_lap,
                result.latest_feasible_lap, result.recommended_lap, mid)
    return result
