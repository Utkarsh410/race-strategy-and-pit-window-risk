"""Rolling-horizon strategy replanning.

At the end of each lap the engine observes the race state and decides whether
to re-optimise. When it does, it:

1. builds a :class:`RaceSnapshot` from the *observed* state (lap, tyre,
   fuel, completed stops) so only the remaining race is optimised;
2. generates feasible future strategies with the existing optimiser
   (:func:`find_optimal_strategy`) and keeps a shortlist: the deterministic
   optimum, the best plan for each stop count, the best plan for each next-pit
   lap in the horizon, the current plan and a no-further-stop option;
3. evaluates every shortlisted plan with the shared engine
   (:func:`run_race`) starting from the exact observed state (fuel, tyre
   wear, last track status, elapsed time), with known race control projected
   forward, the observed weather projected forward and updated rival
   estimates;
4. optionally runs a conditioned Monte Carlo over the remaining laps with
   common random numbers and compares expected time and downside risk;
5. records a :class:`DecisionRecord` in the decision log.

No second lap-time model exists: every evaluation goes through
:func:`racesim.lap_time_model.predict_lap_time` via the simulator. Completed
laps, elapsed time and completed stops are inputs, never re-simulated.

Simplifications (documented, and shown in the dashboard)
--------------------------------------------------------
* The remaining SC/VSC length is unknown to the team; the engine assumes the
  configured mean duration minus laps already run (at least one lap).
* Ongoing rain is assumed to last the configured rain duration minus laps
  already run; future random rain in the Monte Carlo is kept only for runs
  consistent with the observed dry track.
* Rival plans come from the configured priors, shifted when an observed stop
  differs from the prior; overdue stops are assumed
  ``rival_overdue_stop_laps`` laps away. Rival pace is re-estimated from
  observed lap times using the shared model.
* While being evaluated, a plan is followed as written (``reactive=False``) in
  the deterministic comparison; the Monte Carlo uses the configured reactive
  policy as a proxy for future re-planning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from racesim import __version__
from racesim.config import SimulatorSetup, setup_to_dict
from racesim.decision_log import AlternativeRecord, DecisionLog, DecisionRecord, configuration_hash
from racesim.lap_time_model import LapModelContext, predict_lap_time
from racesim.monte_carlo import build_scenario, resolve_flags
from racesim.pit_stop_model import calculate_total_pit_cost
from racesim.race_simulator import (
    HISTORY_KEYS,
    SENTINEL_LAP,
    CarSpec,
    Scenario,
    advance_lap,
    calculate_final_result,
    deterministic_scenario,
    initial_race_state,
    run_race,
)
from racesim.risk_metrics import calculate_cvar
from racesim.safety_car_model import STATUS_LABELS
from racesim.schemas import RaceEvent, RaceSnapshot, RivalConfig, Strategy, TrackStatus
from racesim.strategy_generator import validate_strategy
from racesim.strategy_optimizer import find_optimal_strategy
from racesim.weather_model import generate_weather_scenario, project_conditions

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Observed state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RivalObservation:
    """What the team observes about a rival at the end of a lap."""

    rival_id: str
    gap_s: float                      # own elapsed - rival elapsed (positive: own car behind)
    tyre_compound: str
    tyre_age_laps: int
    stops_completed: int
    completed_pit_laps: tuple[int, ...] = ()
    observed_pace_offset_s: Optional[float] = None  # from recent laps, None if unavailable


@dataclass(frozen=True)
class ObservedRaceState:
    """Own-car and race state at the end of ``current_lap`` (all times in seconds)."""

    current_lap: int
    elapsed_s: float
    tyre_compound: str
    tyre_age_laps: int
    tyre_wear_laps: float
    fuel_kg: float
    compound_sequence: tuple[str, ...]          # compounds run so far, in order, ending with the current set
    completed_pit_laps: tuple[int, ...]
    plan_pit_laps: tuple[int, ...]              # remaining planned in-laps (> current_lap)
    plan_compounds: tuple[str, ...]             # compound fitted at each remaining planned stop
    track_status: int = 0
    neutralisation_laps_elapsed: int = 0
    wetness: float = 0.0
    raining: bool = False
    rain_laps_elapsed: int = 0
    track_temperature_c: Optional[float] = None
    rivals: tuple[RivalObservation, ...] = ()

    @property
    def stops_completed(self) -> int:
        return len(self.completed_pit_laps)

    def snapshot(self) -> RaceSnapshot:
        return RaceSnapshot(current_lap=self.current_lap, tyre_compound=self.tyre_compound, tyre_age_laps=self.tyre_age_laps,
                            fuel_kg=self.fuel_kg, stops_completed=self.stops_completed,
                            compounds_used=list(self.compound_sequence[:-1]))

    def as_dict(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__ if k != "rivals"}
        d["track_status"] = STATUS_LABELS[int(self.track_status)]
        d["rivals"] = [r.__dict__ for r in self.rivals]
        return d

    def fingerprint(self) -> tuple:
        return tuple(round(v, 6) if isinstance(v, float) else v for v in self.__dict__.values() if not isinstance(v, tuple)) + (
            self.compound_sequence, self.completed_pit_laps, self.plan_pit_laps, self.plan_compounds,
            tuple((r.rival_id, round(r.gap_s, 6), r.tyre_compound, r.tyre_age_laps, r.stops_completed) for r in self.rivals))


@dataclass
class StrategyDecision:
    """Engine output: the record plus the recommended future strategy and its predictions."""

    record: DecisionRecord
    strategy: Optional[Strategy]
    predicted_elapsed_s: Optional[np.ndarray] = None   # own absolute elapsed at end of laps decision_lap+1 .. N
    predicted_fuel_kg: Optional[np.ndarray] = None
    predicted_gaps_s: dict[str, np.ndarray] = field(default_factory=dict)
    decision_lap: int = 0

    @property
    def action(self) -> str:
        return self.record.action

    @property
    def recommended_pit_lap(self) -> Optional[int]:
        return self.record.recommended_pit_lap

    @property
    def replanned(self) -> bool:
        return self.record.replanned

    def predicted_at(self, arr: Optional[np.ndarray], lap: int) -> Optional[float]:
        i = lap - (self.decision_lap + 1)
        return None if arr is None or not 0 <= i < len(arr) else float(arr[i])


# ---------------------------------------------------------------------------
# Rival estimates
# ---------------------------------------------------------------------------


def update_rival_estimate(setup: SimulatorSetup, prior: Optional[RivalConfig], obs: RivalObservation, current_lap: int,
                          own_elapsed_s: float) -> CarSpec:
    """Rival car for the remaining race, combining the prior plan with observations.

    * Completed stops are taken from the observation. If the last observed stop
      differs from the prior plan, the remaining planned stops shift by the same
      number of laps.
    * Planned stops that are already overdue are assumed to happen
      ``rival_overdue_stop_laps`` laps from now.
    * Without a prior plan the rival is assumed to stop once when its tyres
      reach the compound's recommended maximum age (if that is before the end).
    * Pace offset blends the prior and the observed estimate with
      ``rival_pace_update_weight``.
    """
    cfg = setup.replanning
    n = setup.race.total_laps
    last_allowed = n - setup.race.minimum_pit_laps_remaining
    k = obs.stops_completed
    if prior is not None:
        future = list(prior.pit_laps[k:])
        comps_after = [c.upper() for c in prior.compounds[k + 1:]]
        if k > 0 and k <= len(prior.pit_laps) and obs.completed_pit_laps:
            delta = obs.completed_pit_laps[-1] - prior.pit_laps[k - 1]
            future = [p + delta for p in future]
        pace = prior.pace_offset_s
        deg = prior.degradation_multiplier
        std = prior.lap_time_std_s
        name = prior.name
    else:
        life = setup.compounds[obs.tyre_compound].maximum_recommended_age_laps - obs.tyre_age_laps
        future = [current_lap + max(life, 1)] if current_lap + life < n else []
        comps_after = [max(setup.dry_compounds, key=lambda c: setup.compounds[c].maximum_recommended_age_laps)]
        pace, deg, std, name = 0.0, 1.0, None, obs.rival_id
    fixed, nxt = [], current_lap + cfg.rival_overdue_stop_laps
    for p in future:
        p = max(p, nxt if not fixed else fixed[-1] + setup.effective_min_stint_laps)
        if p > last_allowed:
            break
        fixed.append(p)
    comps = [obs.tyre_compound] + (comps_after + [comps_after[-1] if comps_after else obs.tyre_compound] * len(fixed))[:len(fixed)]
    if obs.observed_pace_offset_s is not None:
        pace = cfg.rival_pace_update_weight * obs.observed_pace_offset_s + (1 - cfg.rival_pace_update_weight) * pace
    strategy = Strategy.from_plan(n, fixed, comps, start_lap=current_lap + 1, starting_tyre_age_laps=obs.tyre_age_laps,
                                  strategy_name=f"{name} (estimated)")
    return CarSpec(obs.rival_id, strategy, pace_offset_s=pace, degradation_multiplier=deg,
                   start_offset_s=own_elapsed_s - obs.gap_s, lap_time_std_s=std, label=name)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def _expected_neutral_laps_remaining(setup: SimulatorSetup, state: ObservedRaceState) -> int:
    if state.track_status == TrackStatus.GREEN:
        return 0
    rc = setup.race_control
    lo, hi = ((rc.sc_duration_min_laps, rc.sc_duration_max_laps) if state.track_status == TrackStatus.SAFETY_CAR
              else (rc.vsc_duration_min_laps, rc.vsc_duration_max_laps))
    return max(1, int(round((lo + hi) / 2)) - state.neutralisation_laps_elapsed)


def _conditioned_arrays(setup: SimulatorSetup, state: ObservedRaceState) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Known future: track status, temperature and wetness from the observation."""
    n, lap = setup.race.total_laps, state.current_lap
    status = np.zeros(n, dtype=np.int8)
    r = min(_expected_neutral_laps_remaining(setup, state), n - lap)
    status[lap:lap + r] = state.track_status
    rain_left = max(1, setup.weather.rain_duration_laps - state.rain_laps_elapsed) if state.raining else 0
    temp0 = setup.race.track_temperature_c if state.track_temperature_c is None else state.track_temperature_c
    temp, wet = project_conditions(setup.weather, n, lap + 1, wetness=state.wetness, track_temperature_c=temp0,
                                   rain_laps_remaining=rain_left)
    return status, temp, wet, r, rain_left


class ReplanningEngine:
    """Rolling-horizon decision engine with a persistent decision log."""

    def __init__(self, setup: SimulatorSetup, *, seed: Optional[int] = None, rival_priors: Optional[Sequence[RivalConfig]] = None,
                 n_sims: Optional[int] = None, risk_aversion: Optional[float] = None):
        self.setup = setup
        self.ctx = LapModelContext.from_setup(setup)
        self.seed = setup.race.simulation_seed if seed is None else seed
        self.n_sims = setup.replanning.monte_carlo_simulations if n_sims is None else n_sims
        self.risk_aversion = setup.replanning.risk_aversion if risk_aversion is None else risk_aversion
        self.priors = {r.rival_id: r for r in (setup.rivals if rival_priors is None else rival_priors)}
        cfg = setup_to_dict(setup)
        self.config_hash = configuration_hash(cfg)
        self.log = DecisionLog(configuration=cfg)
        self.last_state: Optional[ObservedRaceState] = None
        self.last_decision: Optional[StrategyDecision] = None
        self.optimisations = 0

    # -- triggers -----------------------------------------------------------
    def detect_triggers(self, state: ObservedRaceState) -> list[str]:
        """Reasons to re-optimise; an empty list means the plan is carried forward."""
        prev, dec, cfg = self.last_state, self.last_decision, self.setup.replanning
        if prev is None or dec is None:
            return ["initial evaluation"]
        if state.fingerprint() == prev.fingerprint():
            return []
        if state.current_lap < prev.current_lap:
            return ["new race (lap counter reset)"]
        t: list[str] = []
        if state.track_status != prev.track_status:
            t.append(f"race control changed {STATUS_LABELS[int(prev.track_status)]} -> {STATUS_LABELS[int(state.track_status)]}")
        if state.raining != prev.raining:
            t.append("rain started" if state.raining else "rain stopped")
        elif abs(state.wetness - prev.wetness) > cfg.wetness_change_threshold:
            t.append(f"track wetness changed {prev.wetness:.2f} -> {state.wetness:.2f}")
        prev_r = {r.rival_id: r for r in prev.rivals}
        for r in state.rivals:
            p = prev_r.get(r.rival_id)
            if p is not None and r.stops_completed != p.stops_completed:
                t.append(f"rival {r.rival_id} pitted (lap {r.completed_pit_laps[-1] if r.completed_pit_laps else '?'})")
            pred = dec.predicted_at(dec.predicted_gaps_s.get(r.rival_id), state.current_lap)
            if pred is not None and abs(r.gap_s - pred) > cfg.gap_deviation_threshold_s:
                t.append(f"gap to {r.rival_id} {r.gap_s:+.2f} s vs predicted {pred:+.2f} s")
        pe = dec.predicted_at(dec.predicted_elapsed_s, state.current_lap)
        if pe is not None and abs(state.elapsed_s - pe) > cfg.elapsed_deviation_threshold_s:
            t.append(f"elapsed time {state.elapsed_s - pe:+.2f} s vs prediction")
        pf = dec.predicted_at(dec.predicted_fuel_kg, state.current_lap + 1)
        if pf is not None and abs(state.fuel_kg - pf) > cfg.fuel_deviation_threshold_kg:
            t.append(f"fuel {state.fuel_kg - pf:+.2f} kg vs prediction")
        planned = dec.strategy.planned_pit_laps if dec.strategy else []
        expected_stops = list(prev.completed_pit_laps) + [p for p in planned if prev.current_lap < p <= state.current_lap]
        if list(state.completed_pit_laps) != expected_stops:
            t.append(f"stops completed {list(state.completed_pit_laps)} differ from plan {expected_stops}")
        nxt = dec.recommended_pit_lap
        if nxt is not None and nxt > state.current_lap and nxt - (state.current_lap + 1) <= cfg.replan_within_laps_of_stop \
                and prev.current_lap < state.current_lap and dec.record.lap < nxt - 1 - cfg.replan_within_laps_of_stop:
            t.append(f"planned stop on lap {nxt} is within {cfg.replan_within_laps_of_stop} lap(s): confirm")
        if state.current_lap - dec.decision_lap >= cfg.replan_every_laps:
            t.append(f"periodic re-evaluation ({cfg.replan_every_laps} laps)")
        return t

    # -- evaluation ---------------------------------------------------------
    def _own_car(self, strategy: Strategy, state: ObservedRaceState) -> CarSpec:
        return CarSpec("OWN", strategy, start_offset_s=state.elapsed_s, initial_fuel_kg=state.fuel_kg,
                       initial_wear_laps=state.tyre_wear_laps, initial_track_status=state.track_status, label="Own car")

    def _rivals(self, state: ObservedRaceState) -> list[CarSpec]:
        if not self.setup.replanning.include_rivals:
            return []
        return [update_rival_estimate(self.setup, self.priors.get(o.rival_id), o, state.current_lap, state.elapsed_s)
                for o in state.rivals]

    def _candidates(self, state: ObservedRaceState) -> tuple[dict[str, Strategy], list[str], object]:
        setup, snap = self.setup, state.snapshot()
        warnings: list[str] = []
        opt = find_optimal_strategy(setup, snap)
        self.optimisations += 1
        cands: dict[str, Strategy] = {}

        def add(label: str, s: Strategy) -> None:
            for k, v in cands.items():
                if v.strategy_id == s.strategy_id:
                    return
            cands[label] = s

        if opt.best is not None:
            add("deterministic optimum", opt.best.strategy)
            for k, e in sorted(opt.best_by_stops().items()):
                add(f"best {k}-stop", e.strategy)
            first_pit: dict[int, object] = {}
            for e in opt.evaluations:
                if e.feasible and e.strategy.planned_pit_laps and e.strategy.planned_pit_laps[0] not in first_pit:
                    first_pit[e.strategy.planned_pit_laps[0]] = e
            for p in range(state.current_lap + 1, state.current_lap + 1 + setup.replanning.pit_lap_horizon_laps):
                if p in first_pit:
                    add(f"pit on lap {p}", first_pit[p].strategy)
        for label, s in self._weather_candidates(state, snap).items():
            add(label, s)
        n = setup.race.total_laps
        try:
            plan = Strategy.from_plan(n, list(state.plan_pit_laps), [state.tyre_compound, *state.plan_compounds],
                                      start_lap=state.current_lap + 1, starting_tyre_age_laps=state.tyre_age_laps)
            v = validate_strategy(plan, setup, snap)
            if v.valid:
                cands = {"current plan": plan, **{k: s for k, s in cands.items() if s.strategy_id != plan.strategy_id}}
            else:
                warnings.append("Current plan no longer feasible: " + "; ".join(v.reasons[:3]))
        except ValueError as exc:
            warnings.append(f"Current plan cannot be represented: {exc}")
        if not cands and opt.best is None:
            warnings += [f"{c}: {r}" for c, r in opt.rejected[:3]]
        return cands, warnings, opt

    def _weather_candidates(self, state: ObservedRaceState, snap: RaceSnapshot) -> dict[str, Strategy]:
        """Wet-tyre plans when the track is wet, and plans back to slicks when on wet tyres.

        The dry-compound optimiser cannot produce these, so they are built here
        and checked with the same constraint validator. Pit-lap grids are coarse
        (3 laps) to bound the evaluation cost.
        """
        setup, ctx, lap = self.setup, self.ctx, state.current_lap
        if not setup.weather.wet_weather_enabled:
            return {}
        n, last = setup.race.total_laps, setup.race.total_laps - setup.race.minimum_pit_laps_remaining
        permitted = setup.regulations.permitted_compounds
        allowed = [c for c in setup.compounds if permitted is None or c in permitted]
        wet_types = [c for c in allowed if setup.compounds[c].compound_type != "dry"]
        dry = [c for c in allowed if setup.compounds[c].compound_type == "dry"]
        on_wet = setup.compounds[state.tyre_compound].compound_type != "dry"
        out: dict[str, Strategy] = {}

        def add(label: str, pits: list[int], comps: list[str]) -> None:
            try:
                st = Strategy.from_plan(n, pits, comps, start_lap=lap + 1, starting_tyre_age_laps=state.tyre_age_laps)
            except ValueError:
                return
            if validate_strategy(st, setup, snap).valid:
                out[label] = st

        grid = lambda a: list(range(a, last + 1, 3)) + ([last] if (last - a) % 3 else [])  # noqa: E731
        if on_wet:
            add(f"stay on {state.tyre_compound} to the finish", [], [state.tyre_compound])
            for d in dry:
                for p in grid(lap + 1):
                    add(f"back to {d} on lap {p}", [p], [state.tyre_compound, d])
        elif state.raining or state.wetness >= 0.05:
            peak = max(state.wetness, setup.weather.peak_wetness)
            from racesim.weather_model import calculate_weather_pace_effect

            best_wet = min(wet_types, key=lambda c: setup.compounds[c].reference_lap_time_s
                           + float(calculate_weather_pace_effect(peak, ctx.table.type_code[ctx.table.index(c)], setup.weather)),
                           default=None)
            if best_wet is not None:
                p1 = lap + 1
                add(f"{best_wet} on lap {p1} to the finish", [p1], [state.tyre_compound, best_wet])
                for d in dry:
                    for p2 in grid(p1 + setup.effective_min_stint_laps):
                        add(f"{best_wet} on lap {p1}, {d} on lap {p2}", [p1, p2], [state.tyre_compound, best_wet, d])
        return out

    def _deterministic(self, strategy: Strategy, state: ObservedRaceState, rivals: list[CarSpec]):
        status, temp, wet, _, _ = _conditioned_arrays(self.setup, state)
        n = self.setup.race.total_laps
        scen = Scenario(1, n, status[None, :], temp[None, :], wet[None, :], np.ones((1, len(self.ctx.table))), True, {}, None,
                        "replanning: known conditions")
        out = run_race(self.setup, [self._own_car(strategy, state)] + rivals, scen, record_history=True, reactive=False,
                       interactions=bool(rivals), ctx=self.ctx)
        return out

    def _monte_carlo_scenario(self, state: ObservedRaceState) -> Optional[Scenario]:
        if self.n_sims <= 0:
            return None
        setup, lap = self.setup, state.current_lap
        flags = resolve_flags(setup)
        scen = build_scenario(setup, self.n_sims, self.seed, batch_index=10_000 + lap, flags=flags)
        status, temp, wet, r, rain_left = _conditioned_arrays(setup, state)
        scen.track_status[:, :lap] = 0
        if r:
            # The team does not know when the neutralisation ends: sample the remaining
            # length per run from the configured duration range (at least one lap).
            rc = setup.race_control
            lo, hi = ((rc.sc_duration_min_laps, rc.sc_duration_max_laps) if state.track_status == TrackStatus.SAFETY_CAR
                      else (rc.vsc_duration_min_laps, rc.vsc_duration_max_laps))
            e = state.neutralisation_laps_elapsed
            from racesim.monte_carlo import rng_for

            rem = rng_for(self.seed, "replan_neutral_length", lap).integers(max(lo - e, 1), max(hi - e, 1) + 1, size=self.n_sims)
            idx = np.arange(setup.race.total_laps)[None, :]
            known = (idx >= lap) & (idx < lap + rem[:, None])
            scen.track_status[:, lap:lap + max(hi, 1) + 1] = 0
            scen.track_status[known] = state.track_status
        scen.track_temperature_c[:] = temp[None, :]
        if state.wetness > 0 or state.raining or setup.weather.rain_probability_per_race == 0:
            scen.wetness[:] = wet[None, :]
        else:  # keep random future rain only in runs consistent with a dry track now
            inconsistent = scen.wetness[:, lap - 1] > 0 if lap > 0 else np.zeros(self.n_sims, bool)
            scen.wetness[inconsistent] = wet
            scen.wetness[:, :lap] = 0.0
        scen.description = f"replanning Monte Carlo from lap {lap} (seed {self.seed})"
        return scen

    def _factors(self, state: ObservedRaceState, best: AlternativeRecord, alts: list[AlternativeRecord],
                 plan_alt: Optional[AlternativeRecord], triggers: list[str], neutral_laps: int, rain_left: int) -> list[str]:
        s, setup = self.setup, self.setup
        f = [f"Trigger: {t}" for t in triggers]
        if state.track_status != TrackStatus.GREEN:
            g = calculate_total_pit_cost(s.race, s.pit_stop, TrackStatus.GREEN).total_s
            nc = calculate_total_pit_cost(s.race, s.pit_stop, state.track_status).total_s
            f.append(f"{STATUS_LABELS[int(state.track_status)]} active ({state.neutralisation_laps_elapsed} lap(s) run, "
                     f"{neutral_laps} more assumed): pit cost {nc:.1f} s vs {g:.1f} s under green (configured inputs).")
        if state.raining or state.wetness > 0:
            f.append(f"Track wetness {state.wetness:.2f}" + (f", rain assumed for {rain_left} more lap(s)." if state.raining else "."))
        mx = setup.compounds[state.tyre_compound].maximum_recommended_age_laps
        f.append(f"Current {state.tyre_compound} tyres: age {state.tyre_age_laps} laps (wear-equivalent {state.tyre_wear_laps:.1f}), "
                 f"recommended maximum {mx}.")
        remaining = s.race.total_laps - state.current_lap
        f.append(f"Fuel {state.fuel_kg:.2f} kg for {remaining} laps (need {remaining * s.race.fuel_consumption_kg_per_lap:.2f} kg at the assumed rate).")
        key = "expected_remaining_s" if best.expected_remaining_s is not None else "deterministic_remaining_s"
        others = sorted((a for a in alts if a is not best), key=lambda a: getattr(a, key))
        if others:
            o = others[0]
            d = getattr(o, key) - getattr(best, key)
            extra = "" if o.delta_cvar_vs_recommended_s is None else f", CVaR95 {o.delta_cvar_vs_recommended_s:+.2f} s"
            f.append(f"Next-best option '{o.label}' ({o.strategy_name}): {d:+.2f} s {'expected' if key.startswith('exp') else 'deterministic'}{extra}.")
        if plan_alt is not None and plan_alt is not best:
            d = getattr(plan_alt, key) - getattr(best, key)
            f.append(f"Continuing the current plan is {d:+.2f} s versus the recommendation.")
        mean_best = min(alts, key=lambda a: getattr(a, key))
        if mean_best is not best and self.risk_aversion > 0 and mean_best.cvar95_remaining_s is not None:
            f.append(f"Chosen over the lowest-mean option '{mean_best.label}' because risk aversion {self.risk_aversion} "
                     f"penalises its CVaR95 ({mean_best.cvar95_remaining_s:.2f} s vs {best.cvar95_remaining_s:.2f} s).")
        if best.probability_ahead_of_all_rivals is not None:
            f.append(f"P(finish ahead of all rivals) {best.probability_ahead_of_all_rivals:.0%} with the recommendation"
                     + ("" if plan_alt is None or plan_alt.probability_ahead_of_all_rivals is None else
                        f" vs {plan_alt.probability_ahead_of_all_rivals:.0%} on the current plan") + ".")
        elif best.deterministic_position is not None:
            f.append(f"Deterministic predicted position P{best.deterministic_position}.")
        return f

    def _record(self, state: ObservedRaceState, **kw) -> DecisionRecord:
        return DecisionRecord(decision_id=len(self.log) + 1, lap=state.current_lap, simulated_time_s=state.elapsed_s,
                              race_state=state.as_dict(), model_version=__version__, random_seed=self.seed,
                              configuration_hash=self.config_hash, monte_carlo_simulations=self.n_sims,
                              risk_aversion=self.risk_aversion, **kw)

    def _full_plan(self, state: ObservedRaceState, strategy: Optional[Strategy]) -> dict:
        future_p = list(strategy.planned_pit_laps) if strategy else []
        future_c = list(strategy.planned_compounds[1:]) if strategy else []
        return {"completed_pit_laps": list(state.completed_pit_laps), "future_pit_laps": future_p,
                "compounds": list(state.compound_sequence) + future_c,
                "all_pit_laps": list(state.completed_pit_laps) + future_p}

    @staticmethod
    def _action(strategy: Optional[Strategy], lap: int) -> tuple[str, Optional[int]]:
        if strategy is None:
            return "NO_FEASIBLE_STRATEGY", None
        future = [p for p in strategy.planned_pit_laps if p > lap]
        if not future:
            return "NO_FURTHER_STOP", None
        return ("BOX_THIS_LAP" if future[0] == lap + 1 else "STAY_OUT"), future[0]

    def decide(self, state: ObservedRaceState, *, force: bool = False) -> StrategyDecision:
        """Return a recommendation for the remaining race (re-optimising only when triggered)."""
        if state.current_lap >= self.setup.race.total_laps:
            raise ValueError("the race is finished; nothing to plan")
        triggers = self.detect_triggers(state)
        if force and not triggers:
            triggers = ["forced re-evaluation"]
        if not triggers and self.last_decision is not None:
            return self._carry_forward(state)
        decision = self._optimise(state, triggers)
        self.last_state, self.last_decision = state, decision
        self.log.append(decision.record)
        return decision

    def _carry_forward(self, state: ObservedRaceState) -> StrategyDecision:
        prev = self.last_decision
        action, pit = self._action(prev.strategy, state.current_lap)
        rec = self._record(
            state, action=action, recommended_pit_lap=pit,
            recommended_strategy_id=prev.record.recommended_strategy_id,
            recommended_strategy_name=prev.record.recommended_strategy_name,
            full_race_plan=self._full_plan(state, prev.strategy), replanned=False, triggers=[], alternatives=[],
            expected_difference_vs_current_plan_s=0.0, risk_comparison={},
            influencing_factors=[f"No relevant change since the lap {prev.decision_lap} decision; plan carried forward."],
            warnings=[])
        decision = replace(prev, record=rec)
        self.last_state = state
        self.log.append(rec)
        return decision

    def _optimise(self, state: ObservedRaceState, triggers: list[str]) -> StrategyDecision:
        setup = self.setup
        cands, warnings, opt = self._candidates(state)
        _, _, _, neutral_laps, rain_left = _conditioned_arrays(setup, state)
        if not cands:
            rec = self._record(state, action="NO_FEASIBLE_STRATEGY", recommended_pit_lap=None, recommended_strategy_id=None,
                               recommended_strategy_name=None, full_race_plan=self._full_plan(state, None), replanned=True,
                               triggers=triggers, alternatives=[], expected_difference_vs_current_plan_s=None, risk_comparison={},
                               influencing_factors=[f"Trigger: {t}" for t in triggers] + ["No remaining strategy satisfies the constraints."],
                               warnings=warnings or ["No feasible remaining strategy."])
            logger.warning("Lap %d: no feasible remaining strategy", state.current_lap)
            return StrategyDecision(rec, None, decision_lap=state.current_lap)
        rivals = self._rivals(state)
        alts: list[AlternativeRecord] = []
        det: dict[str, object] = {}
        for label, s in cands.items():
            out = self._deterministic(s, state, rivals)
            det[label] = out
            own = out.cars["OWN"]
            alts.append(AlternativeRecord(label, s.strategy_id, s.strategy_name, s.planned_pit_laps[0] if s.planned_pit_laps else None,
                                          float(own.race_time_s[0]),
                                          deterministic_position=int(own.finishing_position[0]) if own.finishing_position is not None else None))
        scen = self._monte_carlo_scenario(state)
        samples: dict[str, np.ndarray] = {}
        ahead: dict[str, np.ndarray] = {}
        if scen is not None:
            react = scen.flags.get("reactive", False)
            for a, (label, s) in zip(alts, cands.items()):
                out = run_race(setup, [self._own_car(s, state)] + rivals, scen, reactive=react, interactions=bool(rivals), ctx=self.ctx)
                t = out.cars["OWN"].race_time_s
                samples[label] = t
                a.expected_remaining_s, a.std_remaining_s = float(t.mean()), float(t.std(ddof=1)) if t.size > 1 else 0.0
                a.cvar95_remaining_s = calculate_cvar(t)
                if rivals:
                    own_f = out.cars["OWN"].finish_elapsed_s
                    ah = np.all([own_f < out.cars[r.car_id].finish_elapsed_s for r in rivals], axis=0)
                    ahead[label] = ah
                    a.probability_ahead_of_all_rivals = float(ah.mean())
        lam = self.risk_aversion
        for a in alts:
            a.score_s = (a.expected_remaining_s + lam * (a.cvar95_remaining_s - a.expected_remaining_s)
                         if a.expected_remaining_s is not None else a.deterministic_remaining_s)
        best = min(alts, key=lambda a: (a.score_s, a.deterministic_remaining_s))
        plan_alt = next((a for a in alts if a.label == "current plan"), None)
        hysteresis_note = None
        thr = setup.replanning.plan_change_threshold_s
        if plan_alt is not None and best is not plan_alt and plan_alt.score_s - best.score_s <= thr:
            hysteresis_note = (f"'{best.label}' scores only {plan_alt.score_s - best.score_s:.2f} s better than the current plan "
                               f"(threshold {thr:.2f} s): current plan kept to avoid switching between near-equal options.")
            best = plan_alt
        for a in alts:
            if a.expected_remaining_s is not None:
                a.delta_expected_vs_recommended_s = a.expected_remaining_s - best.expected_remaining_s
                a.delta_cvar_vs_recommended_s = a.cvar95_remaining_s - best.cvar95_remaining_s
                d = samples[a.label] - samples[best.label]
                a.probability_faster_than_recommended = float(np.mean((d < -1e-6) + 0.5 * (np.abs(d) <= 1e-6)))
            else:
                a.delta_expected_vs_recommended_s = a.deterministic_remaining_s - best.deterministic_remaining_s
        strat = cands[best.label]
        key = "expected_remaining_s" if best.expected_remaining_s is not None else "deterministic_remaining_s"
        diff = None if plan_alt is None else getattr(best, key) - getattr(plan_alt, key)
        risk = {"basis": "monte_carlo" if scen is not None else "deterministic",
                "recommended": {"label": best.label, "expected_s": best.expected_remaining_s, "cvar95_s": best.cvar95_remaining_s,
                                "std_s": best.std_remaining_s, "deterministic_s": best.deterministic_remaining_s}}
        if plan_alt is not None:
            risk["current_plan"] = {"expected_s": plan_alt.expected_remaining_s, "cvar95_s": plan_alt.cvar95_remaining_s,
                                    "deterministic_s": plan_alt.deterministic_remaining_s}
        if scen is not None:
            risk["lowest_expected_option"] = min(alts, key=lambda a: a.expected_remaining_s).label
            risk["lowest_cvar95_option"] = min(alts, key=lambda a: a.cvar95_remaining_s).label
            if self.n_sims < 1000:
                warnings.append(f"{self.n_sims} Monte Carlo runs per option: tail (CVaR95) comparisons rest on "
                                f"{max(1, self.n_sims // 20)} runs and are imprecise.")
        action, pit = self._action(strat, state.current_lap)
        rec = self._record(state, action=action, recommended_pit_lap=pit, recommended_strategy_id=strat.strategy_id,
                           recommended_strategy_name=strat.strategy_name, full_race_plan=self._full_plan(state, strat),
                           replanned=True, triggers=triggers, alternatives=alts, expected_difference_vs_current_plan_s=diff,
                           risk_comparison=risk,
                           influencing_factors=self._factors(state, best, alts, plan_alt, triggers, neutral_laps, rain_left)
                           + ([hysteresis_note] if hysteresis_note else []),
                           warnings=warnings)
        out = det[best.label]
        h = out.history["OWN"]
        elapsed = h["elapsed_s"][0] + state.elapsed_s
        fuel = h["fuel_kg"][0]
        gaps = {}
        for r in rivals:
            gaps[r.car_id] = elapsed - (out.history[r.car_id]["elapsed_s"][0] + r.start_offset_s)
        logger.info("Lap %d decision: %s (pit %s) via %s", state.current_lap, action, pit, best.label)
        return StrategyDecision(rec, strat, elapsed, fuel, gaps, decision_lap=state.current_lap)


# ---------------------------------------------------------------------------
# Rolling-horizon race (truth simulation + engine in the loop)
# ---------------------------------------------------------------------------


def replace_future_plan(state, ctx: LapModelContext, strategy: Strategy) -> None:
    """Overwrite the not-yet-executed part of a single-row race plan (row 0)."""
    k = int(state.stops_done[0])
    pits = list(strategy.planned_pit_laps)
    comps = [ctx.table.index(c) for c in strategy.planned_compounds[1:]]
    need = k + len(pits) + 2
    width = state.plan_pit_laps.shape[1]
    if need > width:
        state.plan_pit_laps = np.pad(state.plan_pit_laps, ((0, 0), (0, need - width)), constant_values=SENTINEL_LAP)
        state.plan_compounds = np.pad(state.plan_compounds, ((0, 0), (0, need - width)), mode="edge")
        state.driver_offsets = np.pad(state.driver_offsets, (0, need - width), mode="edge")
    state.plan_pit_laps[0, k:] = SENTINEL_LAP
    state.plan_pit_laps[0, k:k + len(pits)] = pits
    last = comps[-1] if comps else int(state.compound_idx[0])
    state.plan_compounds[0, k + 1:] = last
    state.plan_compounds[0, k + 1:k + 1 + len(comps)] = comps
    dry = [c for c in [int(state.compound_idx[0]), *comps] if ctx.table.type_code[c] == 0]
    if dry:
        state.final_dry_compound[0] = dry[-1]


def _run_length(values: np.ndarray, upto: int, predicate) -> int:
    n = 0
    for i in range(upto - 1, -1, -1):
        if predicate(values[i]):
            n += 1
        else:
            break
    return n


def _observed_pace_offset(setup, ctx, h: dict, temp: np.ndarray, wet: np.ndarray, lap: int, window: int = 3) -> Optional[float]:
    """Mean of (observed - model) on recent clean green laps; the car's pace offset estimate."""
    res = []
    for i in range(lap - 1, max(lap - 1 - 3 * window, 0), -1):
        if len(res) >= window:
            break
        if h["status"][i][0] != TrackStatus.GREEN or h["pit_time_s"][i][0] > 0 or h["interaction_s"][i][0] != 0 or i == 0:
            continue
        if i > 0 and h["pit_time_s"][i - 1][0] > 0:
            continue  # out-lap
        age = float(h["tyre_age_laps"][i][0])
        pred = predict_lap_time(ctx, int(h["compound_idx"][i][0]), i + 1, age, age, float(h["fuel_kg"][i][0]), True,
                                float(temp[i]), float(wet[i]))
        res.append(float(h["lap_time_s"][i][0]) - float(pred))
    return float(np.mean(res)) if res else None


@dataclass
class RollingRaceResult:
    total_race_time_s: float
    static_plan_race_time_s: float
    reactive_policy_race_time_s: float
    decision_log: DecisionLog
    laps: pd.DataFrame
    own_pit_laps: list[int]
    own_compounds: list[str]
    finishing_position: Optional[int]
    static_finishing_position: Optional[int]
    rival_results: dict
    events: list[RaceEvent]

    @property
    def gain_vs_static_s(self) -> float:
        return self.static_plan_race_time_s - self.total_race_time_s


def observe_state(setup, ctx, states, hist, scenario: Scenario, raining: np.ndarray, lap: int, compounds: list[str],
                  pit_laps: dict[str, list[int]]) -> ObservedRaceState:
    """Build the team's observation from a single-row truth simulation after ``lap``."""
    own = states[0]
    h = hist["OWN"]
    status = scenario.track_status[0]
    k = int(own.stops_done[0])
    plan = [int(p) for p in own.plan_pit_laps[0, k:] if p < SENTINEL_LAP and p > lap]
    pcomps = [ctx.table.names[int(c)] for c in own.plan_compounds[0, k + 1:k + 1 + len(plan)]]
    cur_status = int(status[lap - 1])
    rivals = []
    for s in states[1:]:
        cid = s.car.car_id
        rivals.append(RivalObservation(
            cid, float(own.elapsed_s[0] - s.elapsed_s[0]), ctx.table.names[int(s.compound_idx[0])], int(s.tyre_age_laps[0]),
            int(s.stops_done[0]), tuple(pit_laps[cid]),
            _observed_pace_offset(setup, ctx, hist[cid], scenario.track_temperature_c[0], scenario.wetness[0], lap)))
    return ObservedRaceState(
        current_lap=lap, elapsed_s=float(own.elapsed_s[0]), tyre_compound=ctx.table.names[int(own.compound_idx[0])],
        tyre_age_laps=int(own.tyre_age_laps[0]), tyre_wear_laps=float(own.tyre_wear_laps[0]), fuel_kg=float(own.fuel_kg[0]),
        compound_sequence=tuple(compounds), completed_pit_laps=tuple(pit_laps["OWN"]), plan_pit_laps=tuple(plan),
        plan_compounds=tuple(pcomps), track_status=cur_status,
        neutralisation_laps_elapsed=_run_length(status, lap, lambda v: v == cur_status) if cur_status else 0,
        wetness=float(scenario.wetness[0, lap - 1]), raining=bool(raining[lap - 1]),
        rain_laps_elapsed=_run_length(raining, lap, bool), track_temperature_c=float(scenario.track_temperature_c[0, lap - 1]),
        rivals=tuple(rivals))


def run_rolling_race(
    setup: SimulatorSetup,
    strategy: Strategy,
    *,
    events: Optional[Sequence[RaceEvent]] = None,
    truth_rivals: Optional[Sequence[RivalConfig]] = None,
    engine: Optional[ReplanningEngine] = None,
    replan: bool = True,
) -> RollingRaceResult:
    """Deterministic rolling-horizon race: the engine decides after every lap.

    The "truth" race uses the shared engine with the given events (deterministic
    scenario). ``truth_rivals`` may differ from the engine's priors
    (``setup.rivals``) to represent unexpected rival behaviour. The same race
    is also run with the static plan, and with the v1 reactive heuristic, for
    comparison.
    """
    engine = engine or ReplanningEngine(setup)
    ctx = engine.ctx
    events = list(events or [])
    truth_rivals = list(setup.rivals if truth_rivals is None else truth_rivals)
    scenario = deterministic_scenario(setup, events)
    raining = generate_weather_scenario(setup.weather, setup.race.total_laps, setup.race.track_temperature_c,
                                        rain_events=events).raining
    cars = [CarSpec("OWN", strategy, label="Own car")] + [CarSpec.from_rival(r, setup.race.total_laps) for r in truth_rivals]
    multi = len(cars) > 1
    states = [initial_race_state(setup, ctx, c, 1) for c in cars]
    rand = [scenario.car_randomness(c.car_id) for c in cars]
    hist = {c.car_id: {k: [] for k in HISTORY_KEYS} for c in cars}
    pit_laps: dict[str, list[int]] = {c.car_id: [] for c in cars}
    compounds = [strategy.planned_compounds[0]]
    n = setup.race.total_laps
    for lap in range(1, n + 1):
        if replan and lap > 1:
            obs = observe_state(setup, ctx, states, hist, scenario, raining, lap - 1, compounds, pit_laps)
            dec = engine.decide(obs)
            if dec.replanned and dec.strategy is not None:
                replace_future_plan(states[0], ctx, dec.strategy)
        outs = advance_lap(setup, ctx, scenario, states, rand, lap, react_to_neutralisation=False, react_to_rain=False,
                           interactions=multi, history=hist)
        for c, o in zip(cars, outs):
            if o.pitting[0]:
                pit_laps[c.car_id].append(lap)
                if c.car_id == "OWN":
                    compounds.append(ctx.table.names[int(states[0].compound_idx[0])])
    final = calculate_final_result(states)
    static = run_race(setup, cars, scenario, reactive=False, interactions=multi, ctx=ctx).cars
    reactive = run_race(setup, cars, scenario, reactive=True, interactions=multi, ctx=ctx).cars
    h = hist["OWN"]
    laps = pd.DataFrame({
        "lap": np.arange(1, n + 1), "lap_time_s": np.array(h["lap_time_s"])[:, 0], "pit_time_s": np.array(h["pit_time_s"])[:, 0],
        "safety_car_bunching_s": np.array(h["bunching_s"])[:, 0],
        "elapsed_s": np.array(h["elapsed_s"])[:, 0], "tyre_compound": [ctx.table.names[int(c[0])] for c in h["compound_idx"]],
        "tyre_age_laps": np.array(h["tyre_age_laps"])[:, 0].astype(int), "fuel_kg": np.array(h["fuel_kg"])[:, 0],
        "track_status": [STATUS_LABELS[int(s[0])] for s in h["status"]], "wetness": scenario.wetness[0],
    })
    own = final["OWN"]
    return RollingRaceResult(
        total_race_time_s=float(own.race_time_s[0]), static_plan_race_time_s=float(static["OWN"].race_time_s[0]),
        reactive_policy_race_time_s=float(reactive["OWN"].race_time_s[0]), decision_log=engine.log, laps=laps,
        own_pit_laps=pit_laps["OWN"], own_compounds=compounds,
        finishing_position=int(own.finishing_position[0]) if own.finishing_position is not None else None,
        static_finishing_position=int(static["OWN"].finishing_position[0]) if static["OWN"].finishing_position is not None else None,
        rival_results={cid: {"race_time_s": float(o.race_time_s[0]), "pit_laps": pit_laps[cid],
                             "finishing_position": None if o.finishing_position is None else int(o.finishing_position[0])}
                       for cid, o in final.items() if cid != "OWN"},
        events=events,
    )
