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
