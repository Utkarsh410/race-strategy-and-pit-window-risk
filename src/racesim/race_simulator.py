"""Lap-by-lap race simulator with an explicit, vectorised race state.

One engine serves every use case:

* a single deterministic race (``n_sims = 1``, all random inputs zero),
* a single stochastic race (one row of a sampled :class:`Scenario`),
* a Monte Carlo batch (``n_sims`` rows simulated simultaneously),
* several cars (own car + rivals) sharing the same race-control and weather
  scenario, with simplified car-to-car interactions.

Lap and time accounting
-----------------------
For every lap ``L`` (1-indexed):

1. read the race-control status and weather for lap ``L``;
2. apply reactive race events (SC/VSC pit calls, rain tyre calls);
3. compute the lap time from the canonical lap-time model
   (:func:`racesim.lap_time_model.predict_lap_time`), the neutralised pace and
   the restart effect;
4. if ``L`` is a pit lap, add the pit cost (booked once, on the in-lap) and fit
   the next tyre set (age 0 on lap ``L + 1``); otherwise age the tyres;
5. ``elapsed += lap_time + pit_cost``.

The race terminates after lap ``total_laps``. Without car interactions the
race time equals ``sum(lap times) + sum(pit costs)`` exactly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from racesim.config import SimulatorSetup
from racesim.fuel_model import calculate_refuel_amount, fuel_at_lap_start
from racesim.lap_time_model import LapModelContext, predict_lap_time
from racesim.pit_stop_model import (
    calculate_stationary_time,
    lane_loss_from_normals,
    stationary_from_normals,
)
from racesim.safety_car_model import (
    STATUS_LABELS,
    calculate_neutralisation_pit_gain,
    calculate_restart_effect,
    generate_race_control_events,
    neutralised_lap_time,
)
from racesim.schemas import (
    PitStopRecord,
    RaceEvent,
    RivalConfig,
    SimulationResult,
    Stint,
    StintRecord,
    Strategy,
    TrackStatus,
)
from racesim.weather_model import calculate_weather_pace_effect, generate_weather_scenario

logger = logging.getLogger(__name__)

SENTINEL_LAP = 1_000_000
EXTRA_STOP_SLOTS = 3  # room for reactive (weather) stops beyond the plan
MIN_FOLLOW_GAP_S = 0.2
RAIN_SWITCH_MARGIN_S = 1.0


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CarSpec:
    """A car to simulate: strategy plus car-specific pace assumptions."""

    car_id: str
    strategy: Strategy
    pace_offset_s: float = 0.0
    degradation_multiplier: float = 1.0
    start_offset_s: float = 0.0
    lap_time_std_s: Optional[float] = None
    label: str = ""

    @classmethod
    def from_rival(cls, rival: RivalConfig, total_laps: int) -> "CarSpec":
        strategy = Strategy.from_plan(
            total_laps, list(rival.pit_laps), list(rival.compounds),
            starting_tyre_age_laps=rival.starting_tyre_age_laps, strategy_name=f"{rival.name} plan",
        )
        return cls(rival.rival_id, strategy, rival.pace_offset_s, rival.degradation_multiplier,
                   rival.start_gap_s, rival.lap_time_std_s, rival.name)


@dataclass
class CarRandomness:
    """Car-specific random inputs (zeros in deterministic mode).

    * ``lap_noise_z`` - standard normals for lap-to-lap variability
    * ``pace_bias_s`` - race-long pace bias (parameter uncertainty)
    * ``fuel_multiplier`` - consumption multiplier (parameter uncertainty)
    * ``pit_*`` - per-stop draws for pit-lane loss and stationary time (event uncertainty)
    * ``traffic_s`` - random traffic loss per lap
    """

    lap_noise_z: np.ndarray
    pace_bias_s: np.ndarray
    fuel_multiplier: np.ndarray
    pit_lane_z: np.ndarray
    pit_stationary_z: np.ndarray
    pit_slow_u: np.ndarray
    traffic_s: np.ndarray

    @classmethod
    def zeros(cls, n_sims: int, n_laps: int, n_stop_slots: int = 12) -> "CarRandomness":
        return cls(
            np.zeros((n_sims, n_laps)), np.zeros(n_sims), np.ones(n_sims),
            np.zeros((n_sims, n_stop_slots)), np.zeros((n_sims, n_stop_slots)), np.ones((n_sims, n_stop_slots)),
            np.zeros((n_sims, n_laps)),
        )


@dataclass
class Scenario:
    """Common random inputs shared by every strategy/car in a comparison.

    Using one scenario for all strategies gives *common random numbers*: the
    same safety cars, weather, tyre-parameter draws and lap noise, so the
    difference between strategies is not polluted by independent noise.
    """

    n_sims: int
    n_laps: int
    track_status: np.ndarray            # (n, L) int8
    track_temperature_c: np.ndarray     # (n, L)
    wetness: np.ndarray                 # (n, L)
    degradation_multiplier: np.ndarray  # (n, C) parameter uncertainty per compound
    deterministic: bool = True
    flags: dict = field(default_factory=dict)
    car_factory: Optional[Callable[[str], CarRandomness]] = None
    description: str = "deterministic"
    _cache: dict = field(default_factory=dict, repr=False)

    def car_randomness(self, car_id: str) -> CarRandomness:
        if car_id not in self._cache:
            self._cache[car_id] = (
                self.car_factory(car_id) if self.car_factory is not None else CarRandomness.zeros(self.n_sims, self.n_laps)
            )
        return self._cache[car_id]

    def row(self, i: int) -> "Scenario":
        """Single-simulation view (for detailed replay of one Monte Carlo run)."""
        sl = slice(i, i + 1)
        parent = self

        def factory(car_id: str) -> CarRandomness:
            r = parent.car_randomness(car_id)
            return CarRandomness(*(getattr(r, f)[sl] for f in CarRandomness.__dataclass_fields__))

        return Scenario(1, self.n_laps, self.track_status[sl], self.track_temperature_c[sl], self.wetness[sl],
                        self.degradation_multiplier[sl], self.deterministic, dict(self.flags), factory,
                        f"{self.description} [run {i}]")


def deterministic_scenario(setup: SimulatorSetup, events: Optional[Sequence[RaceEvent]] = None, n_sims: int = 1) -> Scenario:
    """Scenario with no random variation; optional manual SC/VSC/rain events."""
    n_laps = setup.race.total_laps
    status = generate_race_control_events(setup, n_laps, n_sims=n_sims, manual_events=list(events or []))
    weather = generate_weather_scenario(setup.weather, n_laps, setup.race.track_temperature_c,
                                        rain_events=list(events or []), n_sims=n_sims)
    desc = "deterministic" + (f" with {len(events)} manual event(s)" if events else "")
    return Scenario(n_sims, n_laps, status, weather.track_temperature_c, weather.wetness,
                    np.ones((n_sims, len(setup.compounds))), True, {}, None, desc)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class RaceState:
    """Explicit vectorised state of one car across ``n`` simulations.

    Values describe the car at the *start* of lap ``lap_completed + 1``.
    """

    car: CarSpec
    lap_completed: int
    elapsed_s: np.ndarray
    interaction_s: np.ndarray
    tyre_age_laps: np.ndarray
    tyre_wear_laps: np.ndarray
    compound_idx: np.ndarray
    fuel_kg: np.ndarray
    stint_index: np.ndarray
    stint_start_lap: np.ndarray
    stops_done: np.ndarray
    plan_pit_laps: np.ndarray
    plan_compounds: np.ndarray
    driver_offsets: np.ndarray
    final_dry_compound: np.ndarray
    prev_status: np.ndarray
    pit_cost_total_s: np.ndarray
    tyre_time_total_s: np.ndarray
    fuel_time_total_s: np.ndarray
    reactive_stops: np.ndarray
    fuel_shortfall: np.ndarray
    used_wet_tyres: np.ndarray
    pit_records: list[list[PitStopRecord]] = field(default_factory=list)
    decisions: list[dict] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.elapsed_s)


def initial_race_state(setup: SimulatorSetup, ctx: LapModelContext, car: CarSpec, n_sims: int,
                       fuel_start_kg: Optional[float] = None, record: bool = False) -> RaceState:
    """Initial state for a car whose strategy starts on ``strategy.start_lap``."""
    st = car.strategy
    width = st.number_of_stops + EXTRA_STOP_SLOTS
    pit = np.full((n_sims, width), SENTINEL_LAP, dtype=int)
    pit[:, : st.number_of_stops] = st.planned_pit_laps
    comps_idx = [ctx.table.index(c) for c in st.planned_compounds]
    comps = np.tile(np.array(comps_idx + [comps_idx[-1]] * (width + 1 - len(comps_idx))), (n_sims, 1))
    offs = [setup.driver_offsets_s.get(s.driver_id, 0.0) for s in st.stints]
    offs = np.array(offs + [offs[-1]] * (width + 1 - len(offs)))
    dry = [i for i in comps_idx if ctx.table.type_code[i] == 0]
    if fuel_start_kg is None:
        fuel_start_kg = float(fuel_at_lap_start(st.start_lap, setup.race.fuel_start_kg, setup.race.fuel_consumption_kg_per_lap))
        if setup.refuelling_allowed and setup.regulations.fuel_capacity_kg is not None:
            fuel_start_kg = min(fuel_start_kg, setup.regulations.fuel_capacity_kg)
    zeros = np.zeros(n_sims)
    age0 = st.stints[0].starting_tyre_age_laps
    return RaceState(
        car=car,
        lap_completed=st.start_lap - 1,
        elapsed_s=np.full(n_sims, car.start_offset_s, dtype=float),
        interaction_s=zeros.copy(),
        tyre_age_laps=np.full(n_sims, age0, dtype=float),
        tyre_wear_laps=np.full(n_sims, age0, dtype=float),
        compound_idx=comps[:, 0].copy(),
        fuel_kg=np.full(n_sims, fuel_start_kg, dtype=float),
        stint_index=np.zeros(n_sims, dtype=int),
        stint_start_lap=np.full(n_sims, st.start_lap, dtype=int),
        stops_done=np.zeros(n_sims, dtype=int),
        plan_pit_laps=pit,
        plan_compounds=comps,
        driver_offsets=offs,
        final_dry_compound=np.full(n_sims, dry[-1] if dry else comps_idx[-1]),
        prev_status=np.zeros(n_sims, dtype=np.int8),
        pit_cost_total_s=zeros.copy(),
        tyre_time_total_s=zeros.copy(),
        fuel_time_total_s=zeros.copy(),
        reactive_stops=np.zeros(n_sims, dtype=int),
        fuel_shortfall=np.zeros(n_sims, dtype=bool),
        used_wet_tyres=ctx.table.type_code[comps[:, 0]] != 0,
        pit_records=[[] for _ in range(n_sims)] if record else [],
    )


@dataclass
class LapOutcome:
    """Result of simulating one lap for one car (arrays over simulations)."""

    lap: int
    status: np.ndarray
    lap_time_s: np.ndarray
    tyre_s: np.ndarray
    fuel_s: np.ndarray
    pitting: np.ndarray
    lane_loss_s: np.ndarray
    stationary_s: np.ndarray
    rejoin_s: np.ndarray
    fuel_added_kg: np.ndarray
    next_compound_idx: np.ndarray
    fuel_burn_kg: np.ndarray
    wear_increment: np.ndarray
    fuel_before_kg: np.ndarray
    tyre_age_before: np.ndarray
    compound_before: np.ndarray

    @property
    def pit_cost_s(self) -> np.ndarray:
        return np.where(self.pitting, self.lane_loss_s + self.stationary_s + self.rejoin_s, 0.0)


def _gather(arr2d: np.ndarray, idx: np.ndarray) -> np.ndarray:
    idx = np.minimum(idx, arr2d.shape[1] - 1)
    return arr2d[np.arange(arr2d.shape[0]), idx]


# ---------------------------------------------------------------------------
# Lap-level functions
# ---------------------------------------------------------------------------


def apply_race_event(state: RaceState, setup: SimulatorSetup, ctx: LapModelContext, scenario: Scenario,
                     lap: int, *, react_to_neutralisation: bool, react_to_rain: bool,
                     gain_cache: Optional[dict] = None) -> None:
    """Apply reactive strategy changes at the start of ``lap`` (mutates the plan).

    * **SC/VSC**: bring the next planned stop forward to this lap if feasible
      and expected to gain time (:func:`calculate_neutralisation_pit_gain`).
    * **Rain**: switch to the best-suited compound when the expected pace gain
      exceeds a margin; switch back to the plan's final dry compound when the
      track dries.
    """
    status = scenario.track_status[:, lap - 1]
    n_total = setup.race.total_laps
    rc = setup.race_control
    rows = np.arange(state.n)
    next_pit = _gather(state.plan_pit_laps, state.stops_done)
    laps_left = n_total - lap

    if react_to_neutralisation:
        neutral = status != TrackStatus.GREEN
        cand = (
            neutral
            & (next_pit > lap)
            & (next_pit < SENTINEL_LAP)
            & (next_pit - lap <= rc.neutralisation_pit_advance_max_laps)
            & (lap - state.stint_start_lap + 1 >= setup.effective_min_stint_laps)
            & (laps_left >= setup.race.minimum_pit_laps_remaining)
        )
        following = _gather(state.plan_pit_laps, state.stops_done + 1)
        following = np.where(following >= SENTINEL_LAP, n_total, following)
        cand &= (following - lap) <= setup.effective_max_stint_laps
        cache = gain_cache if gain_cache is not None else {}
        for i in np.nonzero(cand)[0]:
            s = int(status[i])
            wear_factor = rc.sc_tyre_wear_factor if s == TrackStatus.SAFETY_CAR else rc.vsc_tyre_wear_factor
            fuel_factor = rc.sc_fuel_factor if s == TrackStatus.SAFETY_CAR else rc.vsc_fuel_factor
            age_end = state.tyre_age_laps[i] + 1
            wear_end = round(float(state.tyre_wear_laps[i] + wear_factor), 1)
            fuel_end = round(float(max(state.fuel_kg[i] - setup.race.fuel_consumption_kg_per_lap * fuel_factor, 0.0)) * 2) / 2
            nxt = int(_gather(state.plan_compounds, state.stops_done + 1)[i])
            key = (lap, s, int(state.compound_idx[i]), int(age_end), wear_end, fuel_end, int(state.stint_start_lap[i] > 1),
                   nxt, int(next_pit[i]), int(following[i]), round(float(scenario.track_temperature_c[i, lap - 1]), 1))
            if key not in cache:
                cache[key] = calculate_neutralisation_pit_gain(
                    setup, ctx, status=s, current_lap=lap, compound_idx=int(state.compound_idx[i]),
                    tyre_age_laps=age_end, wear_laps=wear_end, fuel_kg=fuel_end,
                    warmup_applies=bool(state.stint_start_lap[i] > 1), next_compound_idx=nxt,
                    planned_pit_lap=int(next_pit[i]), horizon_end_lap=int(following[i]),
                    track_temperature_c=float(scenario.track_temperature_c[i, lap - 1]),
                )["gain_s"]
            if cache[key] > 0:
                state.plan_pit_laps[i, state.stops_done[i]] = lap
                state.reactive_stops[i] += 1
                if state.pit_records:
                    state.decisions.append({"sim": int(i), "lap": lap, "status": STATUS_LABELS[s],
                                            "action": "pit_now", "expected_gain_s": float(cache[key])})

    if react_to_rain and setup.weather.wet_weather_enabled:
        wet = scenario.wetness[:, lap - 1]
        if np.any(wet > 0) or np.any(state.used_wet_tyres):
            permitted = setup.regulations.permitted_compounds
            allowed = np.array([permitted is None or name in permitted for name in ctx.table.names])
            pace = ctx.table.reference_lap_time_s[None, :] + calculate_weather_pace_effect(
                wet[:, None], ctx.table.type_code[None, :], setup.weather)
            pace = np.where(allowed[None, :], pace, np.inf)
            cur = state.compound_idx
            cur_pace = pace[rows, cur]
            best = np.argmin(pace, axis=1)
            cur_dry = ctx.table.type_code[cur] == 0
            best_dry = ctx.table.type_code[best] == 0
            final_dry = state.final_dry_compound
            to_wet = cur_dry & ~best_dry & (cur_pace - pace[rows, best] > RAIN_SWITCH_MARGIN_S)
            to_dry = ~cur_dry & (cur_pace - pace[rows, final_dry] > RAIN_SWITCH_MARGIN_S)
            switch = (to_wet | to_dry) & (laps_left >= setup.race.minimum_pit_laps_remaining)
            switch &= state.stops_done < state.plan_pit_laps.shape[1] - 1
            for i in np.nonzero(switch)[0]:
                k = state.stops_done[i]
                state.plan_pit_laps[i, k:] = SENTINEL_LAP
                state.plan_pit_laps[i, k] = lap
                state.plan_compounds[i, k + 1:] = best[i] if to_wet[i] else final_dry[i]
                state.reactive_stops[i] += 1


def simulate_lap(state: RaceState, setup: SimulatorSetup, ctx: LapModelContext, scenario: Scenario,
                 randomness: CarRandomness, lap: int) -> LapOutcome:
    """Compute one lap for every simulation without mutating the state."""
    car = state.car
    rc = setup.race_control
    flags = scenario.flags
    status = scenario.track_status[:, lap - 1]
    temp = scenario.track_temperature_c[:, lap - 1]
    wet = scenario.wetness[:, lap - 1]
    std = setup.race.baseline_lap_time_std_s if car.lap_time_std_s is None else car.lap_time_std_s
    noise = std * randomness.lap_noise_z[:, lap - 1] if flags.get("lap_variability", False) else 0.0
    traffic = randomness.traffic_s[:, lap - 1] if flags.get("traffic", False) else 0.0
    deg_mult = car.degradation_multiplier * scenario.degradation_multiplier[np.arange(state.n), state.compound_idx]
    green, comp = predict_lap_time(
        ctx, state.compound_idx, lap, state.tyre_age_laps, state.tyre_wear_laps, state.fuel_kg,
        state.stint_start_lap > 1, temp, wet,
        pace_offset_s=car.pace_offset_s + randomness.pace_bias_s,
        degradation_multiplier=deg_mult,
        traffic_s=np.where(status == TrackStatus.GREEN, traffic, 0.0),
        driver_s=state.driver_offsets[np.minimum(state.stint_index, len(state.driver_offsets) - 1)],
        random_error_s=noise,
        return_components=True,
    )
    lap_time = np.where(status != TrackStatus.GREEN, np.maximum(green, neutralised_lap_time(setup, status)), green)
    lap_time = lap_time + calculate_restart_effect(state.prev_status, status, setup)

    sc = status == TrackStatus.SAFETY_CAR
    vsc = status == TrackStatus.VSC
    wear_inc = np.select([sc, vsc], [rc.sc_tyre_wear_factor, rc.vsc_tyre_wear_factor], 1.0)
    fuel_factor = np.select([sc, vsc], [rc.sc_fuel_factor, rc.vsc_fuel_factor], 1.0)
    burn = setup.race.fuel_consumption_kg_per_lap * fuel_factor * randomness.fuel_multiplier

    next_pit = _gather(state.plan_pit_laps, state.stops_done)
    pitting = next_pit == lap
    k = np.minimum(state.stops_done, randomness.pit_lane_z.shape[1] - 1)
    rows = np.arange(state.n)
    pit_var = flags.get("pit_variability", False)
    lane = lane_loss_from_normals(setup.race, setup.pit_stop, status, randomness.pit_lane_z[rows, k] if pit_var else 0.0)
    fuel_added = np.zeros(state.n)
    if setup.refuelling_allowed and np.any(pitting):
        nxt2 = _gather(state.plan_pit_laps, state.stops_done + 1)
        stint_after = np.where(nxt2 >= SENTINEL_LAP, setup.race.total_laps, nxt2) - lap
        cap = setup.regulations.fuel_capacity_kg or setup.race.fuel_start_kg
        fuel_after_lap = np.maximum(state.fuel_kg - burn, 0.0)
        fuel_added = np.where(pitting, calculate_refuel_amount(fuel_after_lap, stint_after, setup.race.fuel_consumption_kg_per_lap, cap), 0.0)
    mean_stat = calculate_stationary_time(setup.race, setup.pit_stop, fuel_added, setup.regulations)
    stationary = stationary_from_normals(
        setup.race, setup.pit_stop,
        randomness.pit_stationary_z[rows, k] if pit_var else 0.0,
        randomness.pit_slow_u[rows, k] if pit_var else 1.0,
        mean_stat,
    )
    return LapOutcome(
        lap=lap, status=status, lap_time_s=lap_time, tyre_s=comp["tyre_s"], fuel_s=comp["fuel_s"],
        pitting=pitting, lane_loss_s=lane, stationary_s=stationary,
        rejoin_s=np.full(state.n, setup.pit_stop.rejoin_loss_s), fuel_added_kg=fuel_added,
        next_compound_idx=_gather(state.plan_compounds, state.stops_done + 1), fuel_burn_kg=burn,
        wear_increment=wear_inc, fuel_before_kg=state.fuel_kg.copy(), tyre_age_before=state.tyre_age_laps.copy(),
        compound_before=state.compound_idx.copy(),
    )


def update_race_state(state: RaceState, outcome: LapOutcome, ctx: LapModelContext,
                      extra_time_s: Optional[np.ndarray] = None) -> None:
    """Advance the state by one lap (fuel, tyres, stops, elapsed time).

    ``extra_time_s`` carries car-interaction adjustments (dirty air), which are
    tracked separately so the isolated race time remains available.
    """
    lap = outcome.lap
    pit_cost = outcome.pit_cost_s
    extra = np.zeros(state.n) if extra_time_s is None else extra_time_s
    state.elapsed_s += outcome.lap_time_s + pit_cost + extra
    state.interaction_s += extra
    state.pit_cost_total_s += pit_cost
    state.tyre_time_total_s += outcome.tyre_s
    state.fuel_time_total_s += outcome.fuel_s
    need = outcome.fuel_burn_kg
    state.fuel_shortfall |= state.fuel_kg + 1e-9 < need
    state.fuel_kg = np.maximum(state.fuel_kg - need, 0.0) + outcome.fuel_added_kg

    p = outcome.pitting
    if state.pit_records:
        for i in np.nonzero(p)[0]:
            state.pit_records[i].append(PitStopRecord(
                lap=lap, compound_from=ctx.table.names[outcome.compound_before[i]],
                compound_to=ctx.table.names[outcome.next_compound_idx[i]],
                lane_loss_s=float(outcome.lane_loss_s[i]), stationary_s=float(outcome.stationary_s[i]),
                rejoin_loss_s=float(outcome.rejoin_s[i]), total_cost_s=float(pit_cost[i]),
                track_status=STATUS_LABELS[int(outcome.status[i])],
                reason="planned" if lap in state.car.strategy.planned_pit_laps else "reactive",
                fuel_added_kg=float(outcome.fuel_added_kg[i]),
            ))
    state.tyre_age_laps = np.where(p, 0.0, state.tyre_age_laps + 1)
    state.tyre_wear_laps = np.where(p, 0.0, state.tyre_wear_laps + outcome.wear_increment)
    state.compound_idx = np.where(p, outcome.next_compound_idx, state.compound_idx)
    state.used_wet_tyres |= ctx.table.type_code[state.compound_idx] != 0
    state.stint_index = state.stint_index + p
    state.stint_start_lap = np.where(p, lap + 1, state.stint_start_lap)
    state.stops_done = state.stops_done + p
    state.prev_status = outcome.status.copy()
    state.lap_completed = lap


def _dirty_air_adjustment(states: list[RaceState], outcomes: list[LapOutcome], setup: SimulatorSetup) -> list[np.ndarray]:
    """Simplified wheel-to-wheel model from aggregate times only.

    A car starting the lap within ``dirty_air_threshold_s`` behind another
    car, both on track under green, passes only if it is quicker by at least
    ``overtake_pace_delta_s`` this lap; otherwise it loses
    ``dirty_air_loss_s_per_lap`` and cannot finish the lap ahead of that car.
    """
    tr = setup.traffic
    m, n = len(states), states[0].n
    e0 = np.stack([s.elapsed_s for s in states])
    lt = np.stack([o.lap_time_s for o in outcomes])
    pit = np.stack([o.pitting for o in outcomes])
    pitc = np.stack([o.pit_cost_s for o in outcomes])
    green = outcomes[0].status == TrackStatus.GREEN
    new_e = e0 + lt + pitc
    extra = np.zeros((m, n))
    order = np.argsort(e0, axis=0, kind="stable")
    cols = np.arange(n)
    for r in range(1, m):
        j, a = order[r], order[r - 1]
        gap = e0[j, cols] - e0[a, cols]
        close = green & (gap < tr.dirty_air_threshold_s) & ~pit[j, cols] & ~pit[a, cols]
        cannot_pass = lt[j, cols] > lt[a, cols] - tr.overtake_pace_delta_s
        stuck = close & cannot_pass
        target = np.maximum(new_e[j, cols] + tr.dirty_air_loss_s_per_lap, new_e[a, cols] + MIN_FOLLOW_GAP_S)
        add = np.where(stuck, target - new_e[j, cols], 0.0)
        extra[j, cols] += add
        new_e[j, cols] += add
    return [extra[i] for i in range(m)]


def _apply_safety_car_bunching(states: list[RaceState], setup: SimulatorSetup, ending: np.ndarray) -> None:
    """Compress gaps to ``sc_bunching_gap_s`` at the end of a safety-car period."""
    m = len(states)
    e = np.stack([s.elapsed_s for s in states])
    order = np.argsort(e, axis=0, kind="stable")
    cols = np.arange(e.shape[1])
    gap_cap = setup.race_control.sc_bunching_gap_s
    for r in range(1, m):
        j, a = order[r], order[r - 1]
        new = np.minimum(e[j, cols], e[a, cols] + gap_cap)
        new = np.where(ending, new, e[j, cols])
        delta = new - e[j, cols]
        e[j, cols] = new
        for car_i in range(m):
            sel = j == car_i
            states[car_i].elapsed_s[sel] += delta[sel]
            states[car_i].interaction_s[sel] += delta[sel]


# ---------------------------------------------------------------------------
# Race-level functions
# ---------------------------------------------------------------------------


@dataclass
class CarOutcome:
    """Batch results for one car (arrays over simulations)."""

    car_id: str
    strategy_id: str
    race_time_s: np.ndarray           # own race duration incl. interactions
    isolated_race_time_s: np.ndarray  # without car interactions
    finish_elapsed_s: np.ndarray      # including start offset (for classification)
    pit_cost_total_s: np.ndarray
    tyre_time_total_s: np.ndarray
    fuel_time_total_s: np.ndarray
    n_stops: np.ndarray
    reactive_stops: np.ndarray
    fuel_shortfall: np.ndarray
    finishing_position: Optional[np.ndarray] = None


@dataclass
class RaceOutput:
    cars: dict[str, CarOutcome]
    history: Optional[dict[str, dict[str, np.ndarray]]] = None
    pit_records: Optional[dict[str, list[list[PitStopRecord]]]] = None
    decisions: Optional[dict[str, list[dict]]] = None
    scenario: Optional[Scenario] = None


def calculate_final_result(states: list[RaceState]) -> dict[str, CarOutcome]:
    """Build per-car outcomes and finishing positions (by finishing elapsed time)."""
    out: dict[str, CarOutcome] = {}
    for s in states:
        out[s.car.car_id] = CarOutcome(
            car_id=s.car.car_id, strategy_id=s.car.strategy.strategy_id,
            race_time_s=s.elapsed_s - s.car.start_offset_s,
            isolated_race_time_s=s.elapsed_s - s.car.start_offset_s - s.interaction_s,
            finish_elapsed_s=s.elapsed_s.copy(), pit_cost_total_s=s.pit_cost_total_s.copy(),
            tyre_time_total_s=s.tyre_time_total_s.copy(), fuel_time_total_s=s.fuel_time_total_s.copy(),
            n_stops=s.stops_done.copy(), reactive_stops=s.reactive_stops.copy(), fuel_shortfall=s.fuel_shortfall.copy(),
        )
    if len(states) > 1:
        e = np.stack([s.elapsed_s for s in states])
        ranks = np.argsort(np.argsort(e, axis=0, kind="stable"), axis=0, kind="stable") + 1
        for i, s in enumerate(states):
            out[s.car.car_id].finishing_position = ranks[i]
    return out


def run_race(
    setup: SimulatorSetup,
    cars: Sequence[CarSpec],
    scenario: Scenario,
    *,
    record_history: bool = False,
    reactive: Optional[bool] = None,
    interactions: bool = True,
    ctx: Optional[LapModelContext] = None,
) -> RaceOutput:
    """Simulate all ``cars`` over all scenario rows (the core engine)."""
    ctx = ctx or LapModelContext.from_setup(setup)
    n, n_laps = scenario.n_sims, setup.race.total_laps
    if scenario.n_laps != n_laps:
        raise ValueError("scenario lap count does not match the race distance")
    starts = {c.strategy.start_lap for c in cars}
    if len(starts) != 1:
        raise ValueError("all cars must start on the same lap")
    for c in cars:
        if c.strategy.end_lap != n_laps:
            raise ValueError(f"strategy {c.strategy.strategy_id} does not finish on lap {n_laps}")
    react_sc = setup.race_control.react_to_neutralisation if reactive is None else (reactive and setup.race_control.react_to_neutralisation)
    react_rain = setup.weather.react_to_rain if reactive is None else (reactive and setup.weather.react_to_rain)
    record = record_history and n <= 50
    states = [initial_race_state(setup, ctx, c, n, record=record) for c in cars]
    rand = [scenario.car_randomness(c.car_id) for c in cars]
    multi = interactions and len(cars) > 1
    hist_keys = ("lap_time_s", "pit_time_s", "elapsed_s", "fuel_kg", "tyre_age_laps", "compound_idx", "status")
    hist = {c.car_id: {k: [] for k in hist_keys} for c in cars} if record else None
    caches = [dict() for _ in cars]
    start_lap = next(iter(starts))
    for lap in range(start_lap, n_laps + 1):
        for s, cache in zip(states, caches):
            apply_race_event(s, setup, ctx, scenario, lap, react_to_neutralisation=react_sc,
                             react_to_rain=react_rain, gain_cache=cache)
        outcomes = [simulate_lap(s, setup, ctx, scenario, r, lap) for s, r in zip(states, rand)]
        extras = _dirty_air_adjustment(states, outcomes, setup) if multi else [None] * len(states)
        for s, o, ex in zip(states, outcomes, extras):
            if hist is not None:
                h = hist[s.car.car_id]
                h["fuel_kg"].append(o.fuel_before_kg.copy())
                h["tyre_age_laps"].append(o.tyre_age_before.copy())
                h["compound_idx"].append(o.compound_before.copy())
                h["status"].append(o.status.copy())
                h["lap_time_s"].append(o.lap_time_s + (0 if ex is None else ex))
                h["pit_time_s"].append(o.pit_cost_s)
            update_race_state(s, o, ctx, ex)
        if multi:
            st = scenario.track_status[:, lap - 1]
            nxt = scenario.track_status[:, lap] if lap < n_laps else np.zeros(n, dtype=np.int8)
            ending = (st == TrackStatus.SAFETY_CAR) & (nxt != TrackStatus.SAFETY_CAR)
            if np.any(ending):
                _apply_safety_car_bunching(states, setup, ending)
        if hist is not None:
            for s in states:
                hist[s.car.car_id]["elapsed_s"].append(s.elapsed_s - s.car.start_offset_s)
    results = calculate_final_result(states)
    history = None
    if hist is not None:
        history = {cid: {k: np.stack(v, axis=1) for k, v in h.items()} for cid, h in hist.items()}
    return RaceOutput(
        cars=results, history=history,
        pit_records={s.car.car_id: s.pit_records for s in states} if record else None,
        decisions={s.car.car_id: s.decisions for s in states} if record else None,
        scenario=scenario,
    )


def _stint_records(strategy_laps: np.ndarray, compounds: list[str], ages: np.ndarray, lap_times: np.ndarray,
                   pit_laps: list[int]) -> list[StintRecord]:
    records = []
    bounds = [int(strategy_laps[0]) - 1, *pit_laps, int(strategy_laps[-1])]
    for k in range(len(bounds) - 1):
        lo, hi = bounds[k] + 1, bounds[k + 1]
        sel = (strategy_laps >= lo) & (strategy_laps <= hi)
        lt = lap_times[sel]
        records.append(StintRecord(k, lo, hi, compounds[np.argmax(sel)], int(ages[np.argmax(sel)]), int(sel.sum()),
                                   float(lt.sum()), float(lt.mean()) if lt.size else float("nan")))
    return records


def simulate_race(
    setup: SimulatorSetup,
    strategy: Strategy,
    *,
    events: Optional[Sequence[RaceEvent]] = None,
    scenario: Optional[Scenario] = None,
    rivals: Optional[Sequence[RivalConfig | CarSpec]] = None,
    reactive: Optional[bool] = None,
    ctx: Optional[LapModelContext] = None,
    car: Optional[CarSpec] = None,
) -> SimulationResult:
    """Simulate one race in detail and return a :class:`SimulationResult`.

    Deterministic by default (no random variation). Pass a one-row
    ``scenario`` (e.g. ``mc_scenario.row(i)``) for a stochastic replay, and
    ``rivals`` to obtain a finishing position. ``car`` overrides the default
    own-car specification (e.g. to simulate a rival on its own).
    """
    ctx = ctx or LapModelContext.from_setup(setup)
    if scenario is None:
        scenario = deterministic_scenario(setup, events)
    if scenario.n_sims != 1:
        raise ValueError("simulate_race expects a single-row scenario; use run_race for batches")
    own = car or CarSpec("OWN", strategy, label="Own car")
    oid, strategy = own.car_id, own.strategy
    cars = [own] + [r if isinstance(r, CarSpec) else CarSpec.from_rival(r, setup.race.total_laps) for r in (rivals or [])]
    out = run_race(setup, cars, scenario, record_history=True, reactive=reactive, ctx=ctx)
    h = out.history[oid]
    res = out.cars[oid]
    laps = np.arange(strategy.start_lap, setup.race.total_laps + 1)
    compounds = [ctx.table.names[i] for i in h["compound_idx"][0]]
    pit_records = out.pit_records[oid][0]
    stints = _stint_records(laps, compounds, h["tyre_age_laps"][0], h["lap_time_s"][0], [p.lap for p in pit_records])
    total = float(res.race_time_s[0])
    meta = {
        "start_lap": strategy.start_lap,
        "scenario": scenario.description,
        "deterministic": scenario.deterministic,
        "isolated_race_time_s": float(res.isolated_race_time_s[0]),
        "interaction_time_s": total - float(res.isolated_race_time_s[0]),
        "tyre_time_s": float(res.tyre_time_total_s[0]),
        "fuel_time_s": float(res.fuel_time_total_s[0]),
        "fuel_shortfall": bool(res.fuel_shortfall[0]),
        "reactive_decisions": out.decisions[oid],
        "planned_pit_laps": list(strategy.planned_pit_laps),
        "strategy_name": strategy.strategy_name,
    }
    if len(cars) > 1:
        meta["rivals"] = {
            cid: {"race_time_s": float(o.race_time_s[0]), "finish_elapsed_s": float(o.finish_elapsed_s[0]),
                  "finishing_position": int(o.finishing_position[0])}
            for cid, o in out.cars.items() if cid != oid
        }
        meta["gap_history_s"] = {
            cid: (out.history[oid]["elapsed_s"][0] + own.start_offset_s)
            - (out.history[cid]["elapsed_s"][0] + next(c.start_offset_s for c in cars if c.car_id == cid))
            for cid in out.cars if cid != oid
        }
    logger.info("Simulated %s: %.3f s (%d stop(s))", strategy.strategy_id, total, len(pit_records))
    return SimulationResult(
        strategy_id=own.strategy.strategy_id,
        predicted_race_time_s=total,
        pit_stops=pit_records,
        stint_records=stints,
        lap_time_history=h["lap_time_s"][0],
        fuel_history=h["fuel_kg"][0],
        tyre_age_history=h["tyre_age_laps"][0].astype(int),
        event_history=[STATUS_LABELS[int(s)] for s in h["status"][0]],
        compound_history=compounds,
        pit_time_history=h["pit_time_s"][0],
        elapsed_history=h["elapsed_s"][0],
        finishing_position=int(res.finishing_position[0]) if res.finishing_position is not None else None,
        simulation_metadata=meta,
    )


def simulate_stint(setup: SimulatorSetup, stint: Stint, *, fuel_start_kg: Optional[float] = None,
                   ctx: Optional[LapModelContext] = None) -> pd.DataFrame:
    """Deterministic lap-by-lap prediction for a single stint (green flag).

    Returns a table with lap, tyre age, fuel and lap-time components.
    """
    ctx = ctx or LapModelContext.from_setup(setup)
    laps = np.arange(stint.start_lap, stint.end_lap + 1)
    k = laps - stint.start_lap
    if fuel_start_kg is None:
        fuel_start_kg = float(fuel_at_lap_start(stint.start_lap, setup.race.fuel_start_kg, setup.race.fuel_consumption_kg_per_lap))
    fuel = np.maximum(fuel_start_kg - setup.race.fuel_consumption_kg_per_lap * k, 0.0)
    age = stint.starting_tyre_age_laps + k
    weather = generate_weather_scenario(setup.weather, setup.race.total_laps, setup.race.track_temperature_c)
    total, comp = predict_lap_time(
        ctx, ctx.table.index(stint.tyre_compound), laps, age, age, fuel, stint.start_lap > 1,
        weather.track_temperature_c[laps - 1], weather.wetness[laps - 1],
        driver_s=setup.driver_offsets_s.get(stint.driver_id, 0.0), return_components=True,
    )
    df = pd.DataFrame({"lap": laps, "tyre_age_laps": age, "fuel_kg": fuel, "lap_time_s": total})
    for key, val in comp.items():
        df[key.replace("_s", "_effect_s") if key != "reference_s" else "reference_s"] = np.broadcast_to(val, laps.shape)
    return df
