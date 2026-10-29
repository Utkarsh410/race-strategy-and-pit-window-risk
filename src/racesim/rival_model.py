"""Rival strategy model (simplified, aggregate-time based).

Assumptions (always shown to the user):

* Cars are represented by elapsed race time only; there is no track-position
  or GPS data. Track order is inferred from elapsed time.
* Overtaking is modelled by a pace threshold (``overtake_pace_delta_s``); a car
  within ``dirty_air_threshold_s`` that is not quick enough loses
  ``dirty_air_loss_s_per_lap`` and stays behind. This is a documented
  simplification, not a wheel-to-wheel prediction.
* Gaps use consistent accounting: ``gap = own_elapsed - rival_elapsed``
  including start offsets and every pit cost. Positive gap = own car behind.
* The undercut/overcut is evaluated by simulating both cars with the same race
  model; it is never a fixed time bonus.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from racesim.config import SimulatorSetup, with_overrides
from racesim.data_validation import representative_lap_mask
from racesim.pit_window import PitWindowResult, calculate_pit_window
from racesim.race_simulator import CarSpec, deterministic_scenario, run_race, simulate_race
from racesim.schemas import RaceEvent, RaceSnapshot, RivalConfig, SimulationResult, Strategy
from racesim.strategy_optimizer import calculate_undercut_gain

logger = logging.getLogger(__name__)

RIVAL_ASSUMPTIONS = (
    "Aggregate elapsed-time model: no track-position data; order inferred from elapsed time.",
    "Overtaking requires a pace advantage >= overtake_pace_delta_s; otherwise dirty-air loss applies.",
    "Both cars share the same race-control scenario, weather and tyre model.",
    "Rival pace is a constant offset plus a degradation multiplier; driver behaviour is not modelled.",
)


def estimate_rival_pace(dataframe: pd.DataFrame, rival_driver: str, reference_driver: str,
                        *, fuel_effect_s_per_kg: Optional[float] = None) -> dict:
    """Median per-lap pace difference (rival - reference, s/lap) on laps where both are representative.

    Comparing like-for-like laps removes common track and fuel effects. If
    tyre ages differ systematically the estimate is confounded; that is
    reported as a warning.
    """
    if "driver" not in dataframe.columns:
        return {"pace_offset_s": None, "sample_count": 0, "warnings": ["No driver column: rival pace cannot be estimated."]}
    df = dataframe[representative_lap_mask(dataframe)].copy()
    t = df["lap_time_s"].astype(float)
    if fuel_effect_s_per_kg is not None and "fuel_kg" in df.columns:
        t = t - fuel_effect_s_per_kg * df["fuel_kg"].fillna(0)
    df["t"] = t
    a = df[df["driver"] == rival_driver].set_index("lap")
    b = df[df["driver"] == reference_driver].set_index("lap")
    common = a.index.intersection(b.index)
    warnings = []
    if len(common) < 5:
        return {"pace_offset_s": None, "sample_count": int(len(common)),
                "warnings": ["Fewer than 5 comparable laps: rival pace estimate unavailable."]}
    diff = a.loc[common, "t"] - b.loc[common, "t"]
    if "tyre_age_laps" in df.columns:
        age_diff = (a.loc[common, "tyre_age_laps"] - b.loc[common, "tyre_age_laps"]).abs().mean()
        if age_diff > 3:
            warnings.append(f"Mean tyre-age difference {age_diff:.1f} laps: pace offset is confounded with degradation.")
    return {"pace_offset_s": float(diff.median()), "sample_count": int(len(common)),
            "std_s": float(diff.std()), "warnings": warnings}


def _rival_setup(setup: SimulatorSetup, rival: RivalConfig) -> SimulatorSetup:
    comp = {name: {"linear_degradation_s_per_lap": c.linear_degradation_s_per_lap * rival.degradation_multiplier,
                   "nonlinear_degradation_coefficient": c.nonlinear_degradation_coefficient * rival.degradation_multiplier}
            for name, c in setup.compounds.items()}
    return with_overrides(setup, compounds=comp)


def predict_rival_pit_window(setup: SimulatorSetup, rival: RivalConfig, snapshot: Optional[RaceSnapshot] = None,
                             n_stops: Optional[Sequence[int]] = None) -> PitWindowResult:
    """Pit window the rival would compute if optimising with the same model.

    Uses the rival's degradation multiplier (its pace offset does not change
    the optimal timing in this model).
    """
    if snapshot is None:
        snapshot = RaceSnapshot(current_lap=0, tyre_compound=rival.compounds[0], tyre_age_laps=rival.starting_tyre_age_laps)
    return calculate_pit_window(_rival_setup(setup, rival), snapshot,
                                n_stops=n_stops or [max(len(rival.pit_laps), 1)])


def simulate_rival_strategy(setup: SimulatorSetup, rival: RivalConfig, *, events: Optional[Sequence[RaceEvent]] = None) -> SimulationResult:
    """Deterministic race of the rival alone on its configured strategy."""
    car = CarSpec.from_rival(rival, setup.race.total_laps)
    return simulate_race(setup, car.strategy, events=events, car=car)


def calculate_time_gap(own_elapsed_s: np.ndarray, rival_elapsed_s: np.ndarray,
                       own_offset_s: float = 0.0, rival_offset_s: float = 0.0) -> np.ndarray:
    """Per-lap gap (s) = own elapsed - rival elapsed (positive: own behind)."""
    return (np.asarray(own_elapsed_s) + own_offset_s) - (np.asarray(rival_elapsed_s) + rival_offset_s)


def predict_rejoin_position(own_elapsed_s: float, others_elapsed_s: Sequence[float]) -> dict:
    """Position after a stop given elapsed times at the same timing line.

    Returns the position (1 = ahead of all others) and the gaps to the cars
    immediately ahead and behind.
    """
    others = np.sort(np.asarray(others_elapsed_s, dtype=float))
    ahead = others[others < own_elapsed_s]
    behind = others[others >= own_elapsed_s]
    return {
        "position": int(len(ahead) + 1),
        "gap_to_car_ahead_s": float(own_elapsed_s - ahead.max()) if ahead.size else None,
        "gap_to_car_behind_s": float(behind.min() - own_elapsed_s) if behind.size else None,
    }


def estimate_traffic_loss(gap_to_car_ahead_s: Optional[float], pace_advantage_s_per_lap: float, laps: int,
                          setup: SimulatorSetup) -> float:
    """Expected time lost behind a car after rejoining (s), simplified.

    No loss if there is no car within the dirty-air threshold or the pace
    advantage allows a pass; otherwise ``dirty_air_loss_s_per_lap`` per lap.
    """
    tr = setup.traffic
    if gap_to_car_ahead_s is None or gap_to_car_ahead_s > tr.dirty_air_threshold_s:
        return 0.0
    if pace_advantage_s_per_lap >= tr.overtake_pace_delta_s:
        return 0.0
    return tr.dirty_air_loss_s_per_lap * max(laps, 0)


@dataclass
class CutAnalysis:
    """Outcome of an undercut / overcut evaluation (times in seconds)."""

    kind: str
    own_pit_lap: int
    rival_pit_lap: int
    initial_gap_s: float
    gap_after_both_stops_s: float
    final_gap_s: float
    gain_s: float
    successful: bool
    own_rejoin: dict
    pace_only_gain_s: float
    gap_history: pd.DataFrame
    assumptions: tuple[str, ...] = RIVAL_ASSUMPTIONS
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("kind", "own_pit_lap", "rival_pit_lap", "initial_gap_s",
                                              "gap_after_both_stops_s", "final_gap_s", "gain_s", "successful",
                                              "pace_only_gain_s")} | {"own_rejoin_position": self.own_rejoin.get("position")}


def _evaluate_cut(setup: SimulatorSetup, kind: str, *, current_lap: int, gap_s: float, own_compound: str,
                  own_tyre_age_laps: int, own_new_compound: str, own_pit_lap: int, rival_compound: str,
                  rival_tyre_age_laps: int, rival_new_compound: str, rival_pit_lap: int,
                  rival_pace_offset_s: float, rival_degradation_multiplier: float,
                  events: Optional[Sequence[RaceEvent]]) -> CutAnalysis:
    n = setup.race.total_laps
    start = current_lap + 1
    own = CarSpec("OWN", Strategy.from_plan(n, [own_pit_lap], [own_compound, own_new_compound], start_lap=start,
                                            starting_tyre_age_laps=own_tyre_age_laps), start_offset_s=gap_s)
    rival = CarSpec("RIVAL", Strategy.from_plan(n, [rival_pit_lap], [rival_compound, rival_new_compound], start_lap=start,
                                                starting_tyre_age_laps=rival_tyre_age_laps),
                    pace_offset_s=rival_pace_offset_s, degradation_multiplier=rival_degradation_multiplier)
    out = run_race(setup, [own, rival], deterministic_scenario(setup, events), record_history=True, reactive=False)
    e_own = out.history["OWN"]["elapsed_s"][0] + gap_s
    e_riv = out.history["RIVAL"]["elapsed_s"][0]
    gap = calculate_time_gap(e_own, e_riv)
    laps = np.arange(start, n + 1)
    after = max(own_pit_lap, rival_pit_lap) + 1
    gap_after = float(gap[after - start])
    rejoin = predict_rejoin_position(float(e_own[own_pit_lap - start]), [float(e_riv[own_pit_lap - start])])
    age_at_pit = own_tyre_age_laps + (own_pit_lap - start)
    if kind == "undercut":
        pace = calculate_undercut_gain(setup, tyre_compound=own_compound, tyre_age_laps_at_pit=age_at_pit,
                                       new_compound=own_new_compound, own_pit_lap=own_pit_lap, rival_pit_lap=rival_pit_lap,
                                       rival_tyre_compound=rival_compound,
                                       rival_tyre_age_laps_at_pit=rival_tyre_age_laps + (own_pit_lap - start),
                                       rival_new_compound=rival_new_compound, rival_pace_offset_s=rival_pace_offset_s)["gain_s"]
    else:
        from racesim.strategy_optimizer import calculate_overcut_gain

        pace = calculate_overcut_gain(setup, tyre_compound=own_compound,
                                      tyre_age_laps_at_rival_pit=own_tyre_age_laps + (rival_pit_lap - start),
                                      new_compound=own_new_compound, own_pit_lap=own_pit_lap, rival_pit_lap=rival_pit_lap,
                                      rival_tyre_compound=rival_compound,
                                      rival_tyre_age_laps_at_pit=rival_tyre_age_laps + (rival_pit_lap - start),
                                      rival_new_compound=rival_new_compound, rival_pace_offset_s=rival_pace_offset_s)["gain_s"]
    notes = []
    if out.cars["OWN"].race_time_s[0] != out.cars["OWN"].isolated_race_time_s[0]:
        notes.append(f"Car interaction (dirty air / SC bunching) cost the own car "
                     f"{out.cars['OWN'].race_time_s[0] - out.cars['OWN'].isolated_race_time_s[0]:.2f} s.")
    history = pd.DataFrame({"lap": laps, "own_elapsed_s": e_own, "rival_elapsed_s": e_riv, "gap_s": gap})
    return CutAnalysis(kind, own_pit_lap, rival_pit_lap, gap_s, gap_after, float(gap[-1]), gap_s - gap_after,
                       gap_after < 0, rejoin, float(pace), history, notes=notes)


def evaluate_undercut(setup: SimulatorSetup, *, current_lap: int, gap_s: float, own_compound: str, own_tyre_age_laps: int,
                      own_new_compound: str, own_pit_lap: int, rival_pit_lap: int, rival_compound: Optional[str] = None,
                      rival_tyre_age_laps: Optional[int] = None, rival_new_compound: Optional[str] = None,
                      rival_pace_offset_s: float = 0.0, rival_degradation_multiplier: float = 1.0,
                      events: Optional[Sequence[RaceEvent]] = None) -> CutAnalysis:
    """Undercut: own car (``gap_s`` behind) pits before the rival.

    Accounts for the current gap, own out-lap warm-up, rival pace and tyre
    degradation, both pit costs, rival pit timing and dirty-air traffic.
    Success means the own car is ahead once both cars have completed their
    out-laps.
    """
    if own_pit_lap >= rival_pit_lap:
        raise ValueError("undercut requires own_pit_lap < rival_pit_lap")
    return _evaluate_cut(setup, "undercut", current_lap=current_lap, gap_s=gap_s, own_compound=own_compound,
                         own_tyre_age_laps=own_tyre_age_laps, own_new_compound=own_new_compound, own_pit_lap=own_pit_lap,
                         rival_compound=rival_compound or own_compound,
                         rival_tyre_age_laps=own_tyre_age_laps if rival_tyre_age_laps is None else rival_tyre_age_laps,
                         rival_new_compound=rival_new_compound or own_new_compound, rival_pit_lap=rival_pit_lap,
                         rival_pace_offset_s=rival_pace_offset_s, rival_degradation_multiplier=rival_degradation_multiplier,
                         events=events)


def evaluate_overcut(setup: SimulatorSetup, *, current_lap: int, gap_s: float, own_compound: str, own_tyre_age_laps: int,
                     own_new_compound: str, own_pit_lap: int, rival_pit_lap: int, rival_compound: Optional[str] = None,
                     rival_tyre_age_laps: Optional[int] = None, rival_new_compound: Optional[str] = None,
                     rival_pace_offset_s: float = 0.0, rival_degradation_multiplier: float = 1.0,
                     events: Optional[Sequence[RaceEvent]] = None) -> CutAnalysis:
    """Overcut: own car stays out after the rival pits.

    Depends on the remaining pace of the old tyres versus the rival's out-lap
    and new-tyre pace, degradation, traffic on the rival's rejoin and the
    expected rejoin position.
    """
    if own_pit_lap <= rival_pit_lap:
        raise ValueError("overcut requires own_pit_lap > rival_pit_lap")
    return _evaluate_cut(setup, "overcut", current_lap=current_lap, gap_s=gap_s, own_compound=own_compound,
                         own_tyre_age_laps=own_tyre_age_laps, own_new_compound=own_new_compound, own_pit_lap=own_pit_lap,
                         rival_compound=rival_compound or own_compound,
                         rival_tyre_age_laps=own_tyre_age_laps if rival_tyre_age_laps is None else rival_tyre_age_laps,
                         rival_new_compound=rival_new_compound or own_new_compound, rival_pit_lap=rival_pit_lap,
                         rival_pace_offset_s=rival_pace_offset_s, rival_degradation_multiplier=rival_degradation_multiplier,
                         events=events)


def evaluate_rival_response(setup: SimulatorSetup, *, current_lap: int, gap_s: float, own_compound: str,
                            own_tyre_age_laps: int, own_new_compound: str, own_pit_lap: int,
                            rival_pit_laps: Sequence[int], **kwargs) -> pd.DataFrame:
    """Sensitivity of an own pit call to the rival's response lap."""
    rows = []
    for rp in rival_pit_laps:
        if rp == own_pit_lap:
            continue
        fn = evaluate_undercut if own_pit_lap < rp else evaluate_overcut
        try:
            res = fn(setup, current_lap=current_lap, gap_s=gap_s, own_compound=own_compound, own_tyre_age_laps=own_tyre_age_laps,
                     own_new_compound=own_new_compound, own_pit_lap=own_pit_lap, rival_pit_lap=rp, **kwargs)
        except ValueError as exc:
            logger.debug("Skipping rival response lap %d: %s", rp, exc)
            continue
        rows.append(res.as_dict())
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# v2: gap evolution, rejoin estimates, traffic exposure, response scenarios,
# safety-car position changes. All use the shared engine (run_race) with the
# simplified aggregate-time interaction model described in the module docstring.
# ---------------------------------------------------------------------------


def _multi_car_run(setup: SimulatorSetup, own_strategy: Strategy, rivals: Sequence[RivalConfig | CarSpec],
                   events: Optional[Sequence[RaceEvent]] = None, interactions: bool = True):
    n = setup.race.total_laps
    cars = [CarSpec("OWN", own_strategy, label="Own car")] + [
        r if isinstance(r, CarSpec) else CarSpec.from_rival(r, n) for r in rivals]
    out = run_race(setup, cars, deterministic_scenario(setup, events), record_history=True, reactive=False,
                   interactions=interactions)
    return cars, out


def gap_evolution(setup: SimulatorSetup, own_strategy: Strategy, rivals: Sequence[RivalConfig | CarSpec], *,
                  events: Optional[Sequence[RaceEvent]] = None, interactions: bool = True) -> pd.DataFrame:
    """Lap-by-lap gap to each rival with the events that moved it.

    ``gap_s = own elapsed - rival elapsed`` (start offsets included; positive =
    own car behind). ``gap_change_s`` is the change over the lap, and the pit,
    dirty-air and safety-car-bunching columns explain it.
    """
    from racesim.safety_car_model import STATUS_LABELS

    cars, out = _multi_car_run(setup, own_strategy, rivals, events, interactions)
    laps = np.arange(own_strategy.start_lap, setup.race.total_laps + 1)
    own_h = out.history["OWN"]
    own_e = own_h["elapsed_s"][0] + cars[0].start_offset_s
    rows = []
    for c in cars[1:]:
        h = out.history[c.car_id]
        gap = calculate_time_gap(own_e, h["elapsed_s"][0] + c.start_offset_s)
        start_gap = cars[0].start_offset_s - c.start_offset_s
        rows.append(pd.DataFrame({
            "lap": laps, "rival_id": c.car_id, "rival": c.label or c.car_id, "gap_s": gap,
            "gap_change_s": np.diff(np.concatenate([[start_gap], gap])),
            "own_pit": own_h["pit_time_s"][0] > 0, "rival_pit": h["pit_time_s"][0] > 0,
            "own_dirty_air_s": own_h["interaction_s"][0], "own_sc_bunching_s": own_h["bunching_s"][0],
            "rival_sc_bunching_s": h["bunching_s"][0],
            "track_status": [STATUS_LABELS[int(s)] for s in own_h["status"][0]],
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def predict_pit_rejoin(setup: SimulatorSetup, own_strategy: Strategy, pit_lap: int, rivals: Sequence[RivalConfig | CarSpec], *,
                       events: Optional[Sequence[RaceEvent]] = None) -> dict:
    """Where the own car rejoins after stopping on ``pit_lap`` and how exposed it is to traffic.

    Rejoin position compares the own elapsed time at the end of the in-lap
    (including the pit cost) with every other car's elapsed time at the end of
    the same lap. Traffic exposure counts cars within ``rejoin_traffic_window_s``
    ahead at the rejoin and the dirty-air time actually lost over the next
    ``traffic_exposure_laps`` laps in the simulation.
    """
    if pit_lap not in own_strategy.planned_pit_laps:
        raise ValueError(f"lap {pit_lap} is not a planned stop of {own_strategy.strategy_name}")
    cars, out = _multi_car_run(setup, own_strategy, rivals, events)
    i = pit_lap - own_strategy.start_lap
    own_e = float(out.history["OWN"]["elapsed_s"][0][i] + cars[0].start_offset_s)
    others = {c.car_id: float(out.history[c.car_id]["elapsed_s"][0][i] + c.start_offset_s) for c in cars[1:]}
    rejoin = predict_rejoin_position(own_e, list(others.values()))
    tr = setup.traffic
    ahead_close = [cid for cid, e in others.items() if 0 < own_e - e <= tr.rejoin_traffic_window_s]
    window = slice(i + 1, min(i + 1 + tr.traffic_exposure_laps, len(out.history["OWN"]["interaction_s"][0])))
    lost = float(out.history["OWN"]["interaction_s"][0][window].sum())
    return {
        "pit_lap": pit_lap, "pit_cost_s": float(out.history["OWN"]["pit_time_s"][0][i]),
        "rejoin_position": rejoin["position"], "gap_to_car_ahead_s": rejoin["gap_to_car_ahead_s"],
        "gap_to_car_behind_s": rejoin["gap_to_car_behind_s"], "cars_within_traffic_window": ahead_close,
        "dirty_air_loss_next_laps_s": lost, "laps_checked": tr.traffic_exposure_laps,
        "elapsed_by_car_s": {"OWN": own_e, **others},
    }


def compare_pit_lap_rejoins(setup: SimulatorSetup, own_strategy: Strategy, rivals: Sequence[RivalConfig | CarSpec],
                            pit_laps: Sequence[int], *, stop_index: int = 0,
                            events: Optional[Sequence[RaceEvent]] = None) -> pd.DataFrame:
    """Rejoin position, traffic exposure and race time for alternative laps of one stop."""
    rows = []
    for p in pit_laps:
        laps = list(own_strategy.planned_pit_laps)
        laps[stop_index] = p
        try:
            st = Strategy.from_plan(setup.race.total_laps, laps, list(own_strategy.planned_compounds),
                                    start_lap=own_strategy.start_lap,
                                    starting_tyre_age_laps=own_strategy.stints[0].starting_tyre_age_laps)
        except ValueError:
            continue
        r = predict_pit_rejoin(setup, st, p, rivals, events=events)
        res = simulate_race(setup, st, events=events, rivals=list(rivals), reactive=False)
        rows.append({"pit_lap": p, "rejoin_position": r["rejoin_position"], "gap_to_car_ahead_s": r["gap_to_car_ahead_s"],
                     "cars_in_traffic_window": len(r["cars_within_traffic_window"]),
                     "dirty_air_loss_next_laps_s": r["dirty_air_loss_next_laps_s"],
                     "race_time_s": res.predicted_race_time_s, "finishing_position": res.finishing_position})
    return pd.DataFrame(rows)


RESPONSES = ("cover next lap", "planned lap", "extend 3 laps", "extend 6 laps")


def rival_response_scenarios(setup: SimulatorSetup, *, current_lap: int, gap_s: float, own_compound: str, own_tyre_age_laps: int,
                             own_new_compound: str, own_pit_lap: int, rival_planned_pit_lap: int,
                             rival_compound: Optional[str] = None, rival_tyre_age_laps: Optional[int] = None,
                             rival_new_compound: Optional[str] = None, rival_pace_offset_s: float = 0.0,
                             rival_degradation_multiplier: float = 1.0, events: Optional[Sequence[RaceEvent]] = None) -> pd.DataFrame:
    """Outcome of an own pit call against named rival responses.

    Responses: cover (pit the lap after the own stop), the rival's planned
    lap, and extending 3 or 6 laps beyond the plan. Each is simulated with the
    same model; rows where the response lap is infeasible are omitted.
    """
    last = setup.race.total_laps - setup.race.minimum_pit_laps_remaining
    laps = {"cover next lap": own_pit_lap + 1, "planned lap": rival_planned_pit_lap,
            "extend 3 laps": rival_planned_pit_lap + 3, "extend 6 laps": rival_planned_pit_lap + 6}
    rows = []
    for name, rp in laps.items():
        if rp <= current_lap or rp > last or rp == own_pit_lap:
            continue
        fn = evaluate_undercut if own_pit_lap < rp else evaluate_overcut
        res = fn(setup, current_lap=current_lap, gap_s=gap_s, own_compound=own_compound, own_tyre_age_laps=own_tyre_age_laps,
                 own_new_compound=own_new_compound, own_pit_lap=own_pit_lap, rival_pit_lap=rp, rival_compound=rival_compound,
                 rival_tyre_age_laps=rival_tyre_age_laps, rival_new_compound=rival_new_compound,
                 rival_pace_offset_s=rival_pace_offset_s, rival_degradation_multiplier=rival_degradation_multiplier, events=events)
        rows.append({"response": name, **res.as_dict()})
    return pd.DataFrame(rows)


def safety_car_position_changes(setup: SimulatorSetup, own_strategy: Strategy, rivals: Sequence[RivalConfig | CarSpec], *,
                                sc_start_lap: int, duration_laps: int = 4) -> pd.DataFrame:
    """Order and gaps before the SC, after bunching, and at the finish, versus a green race.

    Order is inferred from elapsed time (simplified model). Plans are followed
    as written (no reactive stops) so the effect of the neutralisation alone is
    isolated.
    """
    n = setup.race.total_laps
    ev = [RaceEvent(event_type="safety_car", start_lap=sc_start_lap, duration_laps=min(duration_laps, n - sc_start_lap + 1))]
    end = min(sc_start_lap + duration_laps - 1, n)
    cars, out = _multi_car_run(setup, own_strategy, rivals, ev)
    _, green = _multi_car_run(setup, own_strategy, rivals, None)
    rows = []
    for c in cars:
        def at(o, lap):
            return float(o.history[c.car_id]["elapsed_s"][0][lap - own_strategy.start_lap] + c.start_offset_s)
        rows.append({"car_id": c.car_id, "car": c.label or c.car_id,
                     "elapsed_before_sc_s": at(out, sc_start_lap - 1) if sc_start_lap > 1 else c.start_offset_s,
                     "elapsed_after_sc_s": at(out, end), "elapsed_finish_s": at(out, n), "green_finish_s": at(green, n),
                     "sc_bunching_s": float(out.history[c.car_id]["bunching_s"][0].sum())})
    df = pd.DataFrame(rows)
    for col, pos in (("elapsed_before_sc_s", "position_before_sc"), ("elapsed_after_sc_s", "position_after_sc"),
                     ("elapsed_finish_s", "position_finish"), ("green_finish_s", "position_finish_green")):
        df[pos] = df[col].rank(method="first").astype(int)
    leader_before, leader_after = df["elapsed_before_sc_s"].min(), df["elapsed_after_sc_s"].min()
    df["gap_to_leader_before_s"] = df["elapsed_before_sc_s"] - leader_before
    df["gap_to_leader_after_s"] = df["elapsed_after_sc_s"] - leader_after
    df["positions_gained_vs_green"] = df["position_finish_green"] - df["position_finish"]
    return df
