"""Candidate strategy generation with constraint checks and pruning.

The search space (stop count x compound sequence x pit laps) is enumerated
deterministically. When it would exceed ``maximum_strategy_candidates`` the
pit-lap grid is coarsened (lap step 1, 2, 3, ...) per stop count so the total
stays within budget; the optimiser then refines the best candidates locally
at single-lap resolution. Every rejected or pruned candidate class is
recorded with a reason.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from racesim.config import SimulatorSetup
from racesim.fuel_model import validate_fuel_feasibility
from racesim.schemas import RaceSnapshot, Strategy

logger = logging.getLogger(__name__)


@dataclass
class StrategyValidation:
    valid: bool
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class CandidateSet:
    strategies: list[Strategy]
    rejected: list[tuple[str, str]] = field(default_factory=list)
    lap_step_by_stops: dict[int, int] = field(default_factory=dict)
    truncated: bool = False

    def __len__(self) -> int:
        return len(self.strategies)


def _first_stint_context(setup: SimulatorSetup, snapshot: Optional[RaceSnapshot]) -> tuple[int, Optional[str], int, int, list[str]]:
    """(start_lap, fixed first compound, starting age, laps already on set, compounds used)."""
    if snapshot is None or snapshot.current_lap == 0:
        comp = snapshot.tyre_compound if snapshot else None
        age = snapshot.tyre_age_laps if snapshot else 0
        return 1, comp, age, 0, []
    used = list(dict.fromkeys(snapshot.compounds_used + ([snapshot.tyre_compound] if snapshot.tyre_compound else [])))
    return snapshot.current_lap + 1, snapshot.tyre_compound, snapshot.tyre_age_laps, snapshot.tyre_age_laps, used


def validate_strategy(
    strategy: Strategy,
    setup: SimulatorSetup,
    snapshot: Optional[RaceSnapshot] = None,
    *,
    enforce_recommended_tyre_age: bool = True,
) -> StrategyValidation:
    """Check a strategy against planning limits and championship rules.

    For a mid-race ``snapshot`` the first stint continues the current tyre
    set; laps already completed on it count toward the stint-length limits.
    """
    reasons: list[str] = []
    warnings: list[str] = []
    race, regs = setup.race, setup.regulations
    start_lap, _, _, laps_on_set, used = _first_stint_context(setup, snapshot)
    stops_done = snapshot.stops_completed if snapshot else 0
    if strategy.start_lap != start_lap:
        reasons.append(f"strategy starts on lap {strategy.start_lap}, expected {start_lap}")
    if strategy.end_lap != race.total_laps:
        reasons.append(f"strategy ends on lap {strategy.end_lap}, race distance is {race.total_laps}")
    min_stint, max_stint = setup.effective_min_stint_laps, setup.effective_max_stint_laps
    for k, st in enumerate(strategy.stints):
        if st.tyre_compound not in setup.compounds:
            reasons.append(f"stint {k + 1}: unknown compound {st.tyre_compound}")
            continue
        if regs.permitted_compounds is not None and st.tyre_compound not in regs.permitted_compounds:
            reasons.append(f"stint {k + 1}: compound {st.tyre_compound} not permitted")
        length = st.length_laps + (laps_on_set if k == 0 else 0)
        if length < min_stint:
            reasons.append(f"stint {k + 1}: {length} laps < minimum {min_stint}")
        if length > max_stint:
            reasons.append(f"stint {k + 1}: {length} laps > maximum {max_stint}")
        end_age = st.starting_tyre_age_laps + st.length_laps
        rec = setup.compounds[st.tyre_compound].maximum_recommended_age_laps
        if end_age > rec:
            msg = f"stint {k + 1}: tyre age reaches {end_age} > recommended {rec} for {st.tyre_compound}"
            (reasons if enforce_recommended_tyre_age else warnings).append(msg)
    for p in strategy.planned_pit_laps:
        if race.total_laps - p < race.minimum_pit_laps_remaining:
            reasons.append(f"pit on lap {p} leaves fewer than {race.minimum_pit_laps_remaining} laps")
    total_stops = stops_done + strategy.number_of_stops
    if regs.mandatory_pit_stops is not None and total_stops < regs.mandatory_pit_stops:
        reasons.append(f"{total_stops} stop(s) < mandatory {regs.mandatory_pit_stops}")
    if regs.maximum_pit_stops is not None and total_stops > regs.maximum_pit_stops:
        reasons.append(f"{total_stops} stop(s) > maximum {regs.maximum_pit_stops}")
    if total_stops > race.maximum_stops + stops_done:
        reasons.append(f"{strategy.number_of_stops} planned stop(s) > planning maximum {race.maximum_stops}")
    compounds_all = used + list(strategy.planned_compounds)
    wet_used = any(setup.compounds[c].compound_type != "dry" for c in compounds_all if c in setup.compounds)
    if regs.minimum_distinct_dry_compounds is not None and not (wet_used and regs.wet_running_waives_compound_rule):
        distinct = {c for c in compounds_all if c in setup.compounds and setup.compounds[c].compound_type == "dry"}
        if len(distinct) < regs.minimum_distinct_dry_compounds:
            reasons.append(f"uses {len(distinct)} distinct dry compound(s) < required {regs.minimum_distinct_dry_compounds}")
    fuel_start = snapshot.fuel_kg if snapshot and snapshot.fuel_kg is not None else (
        race.fuel_start_kg - race.fuel_consumption_kg_per_lap * (start_lap - 1))
    fuel = validate_fuel_feasibility(
        fuel_start, race.fuel_consumption_kg_per_lap, race.total_laps - start_lap + 1,
        refuelling_allowed=setup.refuelling_allowed, fuel_capacity_kg=regs.fuel_capacity_kg,
        stint_lengths=strategy.stint_lengths,
    )
    if not fuel.feasible:
        reasons.append(fuel.reason)
    if regs.driver_rules:
        laps_by_driver: dict[str, int] = {}
        for st in strategy.stints:
            laps_by_driver[st.driver_id] = laps_by_driver.get(st.driver_id, 0) + st.length_laps
        for rule in regs.driver_rules:
            got = laps_by_driver.get(rule.driver_id, 0)
            if got < rule.minimum_laps:
                reasons.append(f"driver {rule.driver_id}: {got} laps < minimum {rule.minimum_laps}")
            if rule.maximum_laps is not None and got > rule.maximum_laps:
                reasons.append(f"driver {rule.driver_id}: {got} laps > maximum {rule.maximum_laps}")
    return StrategyValidation(not reasons, reasons, warnings)


def generate_compound_sequences(
    setup: SimulatorSetup,
    n_stops: int,
    *,
    first_compound: Optional[str] = None,
    compounds_used: Sequence[str] = (),
    compounds: Optional[Sequence[str]] = None,
) -> list[tuple[str, ...]]:
    """Dry compound sequences of length ``n_stops + 1`` satisfying the compound rule."""
    pool = list(compounds) if compounds is not None else setup.dry_compounds
    rule = setup.regulations.minimum_distinct_dry_compounds
    seqs = []
    for seq in itertools.product(pool, repeat=n_stops + 1):
        if first_compound is not None and seq[0] != first_compound:
            continue
        if rule is not None and len(set(seq) | set(compounds_used)) < rule:
            continue
        seqs.append(seq)
    return seqs


def generate_stop_sequences(
    start_lap: int,
    total_laps: int,
    n_stops: int,
    *,
    min_stint_laps: int,
    max_stint_laps: int,
    min_laps_remaining: int,
    lap_step: int = 1,
    first_stint_offset_laps: int = 0,
    max_first_stint_laps: Optional[int] = None,
) -> list[tuple[int, ...]]:
    """All in-lap tuples for ``n_stops`` stops meeting stint-length limits.

    Uses recursive generation with feasibility pruning: a partial sequence is
    abandoned as soon as the remaining laps cannot be split into valid stints.
    ``first_stint_offset_laps`` counts laps already driven on the current set.
    Pit laps are taken on a grid of ``lap_step`` laps (anchored at the
    earliest feasible lap) to bound the search size.
    """
    last_allowed = total_laps - min_laps_remaining
    out: list[tuple[int, ...]] = []
    first_max = max_stint_laps if max_first_stint_laps is None else min(max_stint_laps, max_first_stint_laps)

    def rec(prev_end: int, k: int, acc: tuple[int, ...]) -> None:
        remaining_stints = n_stops - k + 1
        laps_left = total_laps - prev_end
        if remaining_stints == 1:
            if min_stint_laps <= laps_left + (first_stint_offset_laps if k == 0 else 0) <= (first_max if k == 0 else max_stint_laps):
                out.append(acc)
            return
        offset = first_stint_offset_laps if k == 0 else 0
        lo = prev_end + max(min_stint_laps - offset, 1)
        hi = min(prev_end + (first_max if k == 0 else max_stint_laps) - offset, last_allowed)
        for p in range(lo, hi + 1, lap_step):
            rest = total_laps - p
            if rest < (remaining_stints - 1) * min_stint_laps or rest > (remaining_stints - 1) * max_stint_laps:
                continue
            rec(p, k + 1, acc + (p,))

    if n_stops == 0:
        laps = total_laps - start_lap + 1 + first_stint_offset_laps
        return [()] if min_stint_laps <= laps <= first_max else []
    rec(start_lap - 1, 0, ())
    return out


def remove_duplicate_strategies(strategies: Iterable[Strategy]) -> list[Strategy]:
    """Drop strategies with an identical identifier, keeping the first."""
    seen: set[str] = set()
    unique = []
    for s in strategies:
        if s.strategy_id not in seen:
            seen.add(s.strategy_id)
            unique.append(s)
    return unique


def generate_candidate_strategies(
    setup: SimulatorSetup,
    snapshot: Optional[RaceSnapshot] = None,
    *,
    max_stops: Optional[int] = None,
    min_stops: int = 0,
    compounds: Optional[Sequence[str]] = None,
    lap_step: Optional[int] = None,
    max_candidates: Optional[int] = None,
    enforce_recommended_tyre_age: bool = True,
) -> CandidateSet:
    """Generate valid candidate strategies deterministically.

    The candidate budget (``maximum_strategy_candidates``) is split evenly
    across stop counts; the pit-lap grid step for each stop count is the
    smallest that keeps it within its share.
    """
    race = setup.race
    budget = max_candidates or race.maximum_strategy_candidates
    max_stops = race.maximum_stops if max_stops is None else max_stops
    start_lap, first_comp, age0, laps_on_set, used = _first_stint_context(setup, snapshot)
    stops_done = snapshot.stops_completed if snapshot else 0
    regs = setup.regulations
    lo_stops = max(min_stops, (regs.mandatory_pit_stops or 0) - stops_done, 0)
    if regs.maximum_pit_stops is not None:
        max_stops = min(max_stops, regs.maximum_pit_stops - stops_done)
    stop_counts = list(range(lo_stops, max_stops + 1))
    rejected: list[tuple[str, str]] = []
    if (regs.mandatory_pit_stops or 0) - stops_done > min_stops:
        rejected += [(f"{k}-stop", "below mandatory stop count") for k in range(min_stops, lo_stops)]
    if not stop_counts:
        return CandidateSet([], rejected or [("all", "no stop count satisfies the regulations")])
    share = max(budget // len(stop_counts), 1)
    strategies: list[Strategy] = []
    steps: dict[int, int] = {}
    truncated = False

    for n_stops in stop_counts:
        seqs = generate_compound_sequences(setup, n_stops, first_compound=first_comp, compounds_used=used, compounds=compounds)
        if not seqs:
            rejected.append((f"{n_stops}-stop", "no compound sequence satisfies the compound rules"))
            continue
        kwargs = dict(min_stint_laps=setup.effective_min_stint_laps, max_stint_laps=setup.effective_max_stint_laps,
                      min_laps_remaining=race.minimum_pit_laps_remaining, first_stint_offset_laps=laps_on_set)
        step = lap_step or 1
        laps_grid = generate_stop_sequences(start_lap, race.total_laps, n_stops, lap_step=step, **kwargs)
        while lap_step is None and len(laps_grid) * len(seqs) > share and step < race.total_laps:
            step += 1
            laps_grid = generate_stop_sequences(start_lap, race.total_laps, n_stops, lap_step=step, **kwargs)
        steps[n_stops] = step
        if not laps_grid:
            rejected.append((f"{n_stops}-stop", "no pit-lap sequence satisfies stint-length limits"))
            continue
        count = 0
        for seq in seqs:
            for laps in laps_grid:
                if count >= share:
                    truncated = True
                    break
                strat = Strategy.from_plan(race.total_laps, list(laps), list(seq), start_lap=start_lap, starting_tyre_age_laps=age0)
                check = validate_strategy(strat, setup, snapshot, enforce_recommended_tyre_age=enforce_recommended_tyre_age)
                if check.valid:
                    strategies.append(strat)
                    count += 1
                elif len(rejected) < 5000:
                    rejected.append((strat.strategy_id, "; ".join(check.reasons)))
        if step > 1:
            rejected.append((f"{n_stops}-stop", f"pit-lap grid coarsened to every {step} laps to respect the candidate limit"))
    strategies = remove_duplicate_strategies(strategies)
    logger.info("Generated %d candidate strategies (steps %s, %d rejections)", len(strategies), steps, len(rejected))
    return CandidateSet(strategies, rejected, steps, truncated)
