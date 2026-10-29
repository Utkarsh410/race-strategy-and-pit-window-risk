"""Safety-car (SC) and virtual-safety-car (VSC) model.

Event model
-----------
* Per lap, when no neutralisation is active and the cooldown has elapsed, an
  SC starts with probability ``safety_car_probability_per_lap`` and otherwise
  a VSC with ``vsc_probability_per_lap``. Durations are uniform integers in the
  configured ranges. SC and VSC are mutually exclusive: contradictory
  simultaneous states are never generated, and overlapping manual events are
  rejected.
* Neutralised laps run at ``baseline_lap_time_s * {sc,vsc}_lap_time_factor``
  (or the car's green pace if that is slower), consume a fraction of tyre wear
  and fuel, and the first green lap after the period carries a restart loss.
* Pit-lane loss under SC / VSC uses the distinct circuit-specific inputs from
  :class:`PitStopModelConfig`.

Limitations (documented)
------------------------
* Event hazard is constant per lap; real incident rates depend on circuit,
  lap 1 chaos, weather and field spread.
* No red flags, no lapped-car procedures, no pit-lane closure.
* The reactive pit decision uses the team's deterministic expected model, so
  it can be wrong ex post (this is intentional "strategy uncertainty").
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

from racesim.lap_time_model import LapModelContext, predict_lap_time
from racesim.pit_stop_model import calculate_total_pit_cost
from racesim.schemas import EventType, RaceEvent, TrackStatus

if TYPE_CHECKING:
    from racesim.config import SimulatorSetup

logger = logging.getLogger(__name__)

STATUS_LABELS = {int(TrackStatus.GREEN): "GREEN", int(TrackStatus.SAFETY_CAR): "SC", int(TrackStatus.VSC): "VSC"}
_EVENT_TO_STATUS = {EventType.SAFETY_CAR: TrackStatus.SAFETY_CAR, EventType.VSC: TrackStatus.VSC}


def events_to_track_status(events: Sequence[RaceEvent], n_laps: int) -> np.ndarray:
    """Convert manual SC/VSC events to a per-lap status array (index = lap - 1).

    Raises ``ValueError`` if two neutralisation events overlap.
    """
    status = np.zeros(n_laps, dtype=np.int8)
    owner = np.full(n_laps, -1)
    for k, ev in enumerate(events):
        if ev.event_type not in _EVENT_TO_STATUS:
            continue
        if ev.start_lap > n_laps:
            raise ValueError(f"event starts on lap {ev.start_lap} beyond race distance {n_laps}")
        sl = slice(ev.start_lap - 1, min(ev.end_lap, n_laps))
        if np.any(owner[sl] >= 0):
            raise ValueError(f"contradictory race-control events overlap on laps {ev.start_lap}-{ev.end_lap}")
        owner[sl] = k
        status[sl] = int(_EVENT_TO_STATUS[ev.event_type])
    return status


def track_status_to_events(status: np.ndarray) -> list[RaceEvent]:
    """Convert a per-lap status array back into a list of events."""
    events: list[RaceEvent] = []
    s = np.asarray(status)
    lap = 0
    while lap < len(s):
        if s[lap] != TrackStatus.GREEN:
            start, kind = lap, s[lap]
            while lap < len(s) and s[lap] == kind:
                lap += 1
            etype = EventType.SAFETY_CAR if kind == TrackStatus.SAFETY_CAR else EventType.VSC
            events.append(RaceEvent(event_type=etype, start_lap=start + 1, duration_laps=lap - start))
        else:
            lap += 1
    return events


def generate_race_control_events(
    setup: "SimulatorSetup",
    n_laps: int,
    *,
    rng: Optional[np.random.Generator] = None,
    n_sims: int = 1,
    manual_events: Optional[Sequence[RaceEvent]] = None,
    sc_probability: Optional[float] = None,
    vsc_probability: Optional[float] = None,
) -> np.ndarray:
    """Per-lap track status of shape ``(n_sims, n_laps)``.

    Manual events (when given, even an empty list) override random generation.
    Without manual events and without ``rng`` the race is fully green.
    """
    if manual_events is not None:
        return np.tile(events_to_track_status(manual_events, n_laps), (n_sims, 1))
    status = np.zeros((n_sims, n_laps), dtype=np.int8)
    if rng is None:
        return status
    rc = setup.race_control
    p_sc = setup.race.safety_car_probability_per_lap if sc_probability is None else sc_probability
    p_vsc = setup.race.vsc_probability_per_lap if vsc_probability is None else vsc_probability
    remaining = np.zeros(n_sims, dtype=int)
    kind = np.zeros(n_sims, dtype=np.int8)
    cooldown = np.zeros(n_sims, dtype=int)
    last_start = n_laps - rc.no_new_event_final_laps
    for lap in range(1, n_laps + 1):
        u = rng.random(n_sims)
        d_sc = rng.integers(rc.sc_duration_min_laps, rc.sc_duration_max_laps + 1, size=n_sims)
        d_vsc = rng.integers(rc.vsc_duration_min_laps, rc.vsc_duration_max_laps + 1, size=n_sims)
        eligible = (remaining == 0) & (cooldown == 0) & (lap >= rc.earliest_event_lap) & (lap <= last_start)
        start_sc = eligible & (u < p_sc)
        start_vsc = eligible & ~start_sc & (u < p_sc + p_vsc)
        remaining = np.where(start_sc, d_sc, np.where(start_vsc, d_vsc, remaining))
        kind = np.where(start_sc, TrackStatus.SAFETY_CAR, np.where(start_vsc, TrackStatus.VSC, kind)).astype(np.int8)
        active = remaining > 0
        status[:, lap - 1] = np.where(active, kind, TrackStatus.GREEN)
        cooldown = np.where(active, 0, np.maximum(cooldown - 1, 0))
        ended = active & (remaining == 1)
        remaining = np.maximum(remaining - 1, 0)
        cooldown = np.where(ended, rc.cooldown_laps, cooldown)
    return status


@dataclass(frozen=True)
class NeutralisationEffect:
    """Effects of one SC or VSC period (seconds unless stated)."""

    status: TrackStatus
    start_lap: int
    duration_laps: int
    neutralised_lap_time_s: float
    pit_lane_loss_s: float
    green_pit_lane_loss_s: float
    pit_cost_saving_s: float
    tyre_wear_factor: float
    fuel_factor: float
    restart_loss_s: float

    @property
    def event(self) -> RaceEvent:
        etype = EventType.SAFETY_CAR if self.status == TrackStatus.SAFETY_CAR else EventType.VSC
        return RaceEvent(event_type=etype, start_lap=self.start_lap, duration_laps=self.duration_laps)


def neutralised_lap_time(setup: "SimulatorSetup", status: np.ndarray | int) -> np.ndarray:
    """Lap time (s) imposed by an SC/VSC; 0 for green laps."""
    s = np.asarray(status)
    rc, base = setup.race_control, setup.race.baseline_lap_time_s
    return np.select([s == TrackStatus.SAFETY_CAR, s == TrackStatus.VSC], [base * rc.sc_lap_time_factor, base * rc.vsc_lap_time_factor], 0.0)


def _neutralisation(setup: "SimulatorSetup", status: TrackStatus, start_lap: int, duration_laps: int) -> NeutralisationEffect:
    rc = setup.race_control
    green = calculate_total_pit_cost(setup.race, setup.pit_stop, TrackStatus.GREEN)
    neutral = calculate_total_pit_cost(setup.race, setup.pit_stop, status)
    sc = status == TrackStatus.SAFETY_CAR
    return NeutralisationEffect(
        status=status,
        start_lap=start_lap,
        duration_laps=duration_laps,
        neutralised_lap_time_s=float(neutralised_lap_time(setup, int(status))),
        pit_lane_loss_s=neutral.pit_lane_loss_s,
        green_pit_lane_loss_s=green.pit_lane_loss_s,
        pit_cost_saving_s=green.total_s - neutral.total_s,
        tyre_wear_factor=rc.sc_tyre_wear_factor if sc else rc.vsc_tyre_wear_factor,
        fuel_factor=rc.sc_fuel_factor if sc else rc.vsc_fuel_factor,
        restart_loss_s=rc.sc_restart_loss_s if sc else rc.vsc_restart_loss_s,
    )


def simulate_safety_car(setup: "SimulatorSetup", start_lap: int, duration_laps: int) -> NeutralisationEffect:
    """Effects of a safety-car period starting on ``start_lap``."""
    return _neutralisation(setup, TrackStatus.SAFETY_CAR, start_lap, duration_laps)


def simulate_vsc(setup: "SimulatorSetup", start_lap: int, duration_laps: int) -> NeutralisationEffect:
    """Effects of a VSC period starting on ``start_lap``."""
    return _neutralisation(setup, TrackStatus.VSC, start_lap, duration_laps)


def calculate_restart_effect(previous_status: np.ndarray | int, status: np.ndarray | int, setup: "SimulatorSetup") -> np.ndarray:
    """Restart loss (s) on the first green lap after an SC or VSC."""
    prev, cur = np.asarray(previous_status), np.asarray(status)
    rc = setup.race_control
    green = cur == TrackStatus.GREEN
    return np.where(green & (prev == TrackStatus.SAFETY_CAR), rc.sc_restart_loss_s,
                    np.where(green & (prev == TrackStatus.VSC), rc.vsc_restart_loss_s, 0.0))


def _stint_time(ctx: LapModelContext, setup: "SimulatorSetup", compound_idx: int, first_lap: int, last_lap: int,
                age0: float, wear0: float, fuel0: float, warmup: bool, temp: float) -> float:
    if last_lap < first_lap:
        return 0.0
    laps = np.arange(first_lap, last_lap + 1)
    k = laps - first_lap
    fuel = np.maximum(fuel0 - setup.race.fuel_consumption_kg_per_lap * k, 0.0)
    return float(np.sum(predict_lap_time(ctx, compound_idx, laps, age0 + k, wear0 + k, fuel, warmup, temp)))


def calculate_neutralisation_pit_gain(
    setup: "SimulatorSetup",
    ctx: LapModelContext,
    *,
    status: TrackStatus | int,
    current_lap: int,
    compound_idx: int,
    tyre_age_laps: float,
    wear_laps: float,
    fuel_kg: float,
    warmup_applies: bool,
    next_compound_idx: int,
    planned_pit_lap: int,
    horizon_end_lap: int,
    track_temperature_c: Optional[float] = None,
) -> dict[str, float]:
    """Expected time gained (s) by pitting now under SC/VSC instead of on the plan.

    Compares, over laps ``current_lap + 1 .. horizon_end_lap`` (end of the
    next stint):

    * **Plan A** - stay out until ``planned_pit_lap`` and pit under green;
    * **Plan B** - pit at the end of ``current_lap`` under neutralisation.

    State arguments describe the car at the *end* of ``current_lap``.
    Post-decision laps are evaluated at green pace in both plans (laps still
    neutralised affect both plans similarly). A positive gain means stopping
    now is expected to be faster; it is not assumed to be beneficial.
    """
    status = TrackStatus(int(status))
    if status == TrackStatus.GREEN:
        raise ValueError("neutralisation pit gain requested on a green lap")
    temp = setup.race.track_temperature_c if track_temperature_c is None else track_temperature_c
    cons = setup.race.fuel_consumption_kg_per_lap
    green_cost = calculate_total_pit_cost(setup.race, setup.pit_stop, TrackStatus.GREEN).total_s
    neutral_cost = calculate_total_pit_cost(setup.race, setup.pit_stop, status).total_s
    p = planned_pit_lap
    # Plan A: old tyres to the planned lap, then new tyres to the horizon.
    a_old = _stint_time(ctx, setup, compound_idx, current_lap + 1, p, tyre_age_laps, wear_laps, fuel_kg, warmup_applies, temp)
    fuel_at_p = max(fuel_kg - cons * (p - current_lap), 0.0)
    a_new = _stint_time(ctx, setup, next_compound_idx, p + 1, horizon_end_lap, 0, 0, fuel_at_p, True, temp)
    # Plan B: new tyres from the next lap to the horizon.
    b_new = _stint_time(ctx, setup, next_compound_idx, current_lap + 1, horizon_end_lap, 0, 0, fuel_kg, True, temp)
    plan_a = a_old + a_new + green_cost
    plan_b = b_new + neutral_cost
    return {
        "gain_s": plan_a - plan_b,
        "plan_stay_out_s": plan_a,
        "plan_pit_now_s": plan_b,
        "green_pit_cost_s": green_cost,
        "neutralised_pit_cost_s": neutral_cost,
    }


def update_strategy_after_event(
    setup: "SimulatorSetup",
    ctx: LapModelContext,
    *,
    planned_pit_laps: Sequence[int],
    planned_compounds: Sequence[str],
    stops_completed: int,
    status: TrackStatus | int,
    current_lap: int,
    stint_start_lap: int,
    tyre_age_laps: float,
    wear_laps: float,
    fuel_kg: float,
) -> tuple[list[int], dict]:
    """Decide whether to bring the next planned stop forward to ``current_lap``.

    Checks feasibility (minimum current stint, maximum next stint, minimum laps
    remaining, advance window) before evaluating
    :func:`calculate_neutralisation_pit_gain`. Returns the (possibly unchanged)
    pit-lap list and a decision record.
    """
    pit_laps = list(planned_pit_laps)
    decision: dict = {"lap": current_lap, "status": STATUS_LABELS[int(status)], "action": "stay_out", "reason": ""}
    if TrackStatus(int(status)) == TrackStatus.GREEN:
        decision["reason"] = "green flag"
        return pit_laps, decision
    if stops_completed >= len(pit_laps):
        decision["reason"] = "no further planned stop"
        return pit_laps, decision
    p = pit_laps[stops_completed]
    n = setup.race.total_laps
    following = pit_laps[stops_completed + 1] if stops_completed + 1 < len(pit_laps) else n
    checks = [
        (p <= current_lap, "planned stop is this lap or earlier"),
        (p - current_lap > setup.race_control.neutralisation_pit_advance_max_laps, "planned stop outside advance window"),
        (current_lap - stint_start_lap + 1 < setup.effective_min_stint_laps, "current stint below minimum length"),
        (n - current_lap < setup.race.minimum_pit_laps_remaining, "too few laps remaining"),
        (following - current_lap > setup.effective_max_stint_laps, "next stint would exceed maximum length"),
    ]
    for failed, reason in checks:
        if failed:
            decision["reason"] = reason
            return pit_laps, decision
    comp_now = ctx.table.index(planned_compounds[stops_completed])
    comp_next = ctx.table.index(planned_compounds[stops_completed + 1])
    gain = calculate_neutralisation_pit_gain(
        setup, ctx, status=status, current_lap=current_lap, compound_idx=comp_now, tyre_age_laps=tyre_age_laps,
        wear_laps=wear_laps, fuel_kg=fuel_kg, warmup_applies=stint_start_lap > 1, next_compound_idx=comp_next,
        planned_pit_lap=p, horizon_end_lap=following,
    )
    decision.update(gain)
    if gain["gain_s"] > 0:
        pit_laps[stops_completed] = current_lap
        decision["action"] = "pit_now"
        decision["reason"] = f"expected gain {gain['gain_s']:.2f} s"
    else:
        decision["reason"] = f"expected loss {-gain['gain_s']:.2f} s"
    logger.debug("Neutralisation decision: %s", decision)
    return pit_laps, decision
