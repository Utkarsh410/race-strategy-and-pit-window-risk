"""Phase 4: gap accounting, rejoin estimates, traffic exposure, responses, SC position changes."""

import numpy as np
import pytest

from racesim.config import with_overrides
from racesim.race_simulator import CarSpec, simulate_race
from racesim.rival_model import (
    compare_pit_lap_rejoins, gap_evolution, predict_pit_rejoin, predict_rejoin_position, rival_response_scenarios,
    safety_car_position_changes,
)
from racesim.schemas import RaceEvent, Strategy

OWN = Strategy.from_plan(60, [25, 43], ["MEDIUM", "SOFT", "SOFT"])


def test_gap_accounting_matches_independent_runs(setup):
    """Without interactions the gap equals the difference of two single-car simulations incl. start offset."""
    g = gap_evolution(setup, OWN, setup.rivals[:1], interactions=False)
    own = simulate_race(setup, OWN)
    riv_car = CarSpec.from_rival(setup.rivals[0], 60)
    riv = simulate_race(setup, riv_car.strategy, car=riv_car)
    expected = own.elapsed_history - (riv.elapsed_history + setup.rivals[0].start_gap_s)
    assert np.allclose(g["gap_s"].to_numpy(), expected)
    # gap changes add up: start gap + sum(changes) = final gap
    start = 0.0 - setup.rivals[0].start_gap_s
    assert start + g["gap_change_s"].sum() == pytest.approx(g["gap_s"].iloc[-1])
    # a pit stop moves the gap by the pit cost plus the lap-time difference
    lap = 25
    row = g.set_index("lap").loc[lap]
    own_lt = own.lap_time_history[lap - 1] + own.pit_time_history[lap - 1]
    riv_lt = riv.lap_time_history[lap - 1] + riv.pit_time_history[lap - 1]
    assert row["gap_change_s"] == pytest.approx(own_lt - riv_lt) and row["own_pit"]


def test_gap_with_interactions_reconciles(setup):
    g = gap_evolution(setup, OWN, setup.rivals, events=[RaceEvent(event_type="safety_car", start_lap=30, duration_laps=4)])
    for rid, d in g.groupby("rival_id"):
        start = -next(r.start_gap_s for r in setup.rivals if r.rival_id == rid)
        assert start + d["gap_change_s"].sum() == pytest.approx(d["gap_s"].iloc[-1])
    assert (g["own_sc_bunching_s"] <= 0).all()


def test_rejoin_hand_calculation():
    r = predict_rejoin_position(1000.0 + 22.5, [1010.0, 1020.0, 1030.0])
    assert r["position"] == 3 and r["gap_to_car_ahead_s"] == pytest.approx(2.5) and r["gap_to_car_behind_s"] == pytest.approx(7.5)


def test_pit_rejoin_uses_same_lap_elapsed(setup):
    r = predict_pit_rejoin(setup, OWN, 25, setup.rivals)
    own_e = r["elapsed_by_car_s"]["OWN"]
    others = [v for k, v in r["elapsed_by_car_s"].items() if k != "OWN"]
    assert r["rejoin_position"] == 1 + sum(o < own_e for o in others)
    assert r["pit_cost_s"] == pytest.approx(setup.race.total_pit_cost_s)
    with pytest.raises(ValueError):
        predict_pit_rejoin(setup, OWN, 24, setup.rivals)


def test_traffic_exposure_configurable(setup):
    tbl = compare_pit_lap_rejoins(setup, OWN, setup.rivals, range(20, 30))
    assert set(tbl["pit_lap"]) == set(range(20, 30))
    assert (tbl["dirty_air_loss_next_laps_s"] >= 0).all()
    no_air = with_overrides(setup, traffic={"dirty_air_loss_s_per_lap": 0.0, "dirty_air_threshold_s": 0.0})
    tbl0 = compare_pit_lap_rejoins(no_air, OWN, no_air.rivals, range(20, 30))
    assert (tbl0["dirty_air_loss_next_laps_s"] == 0).all()


def test_rival_response_scenarios(setup):
    df = rival_response_scenarios(setup, current_lap=18, gap_s=1.5, own_compound="MEDIUM", own_tyre_age_laps=18,
                                  own_new_compound="HARD", own_pit_lap=21, rival_planned_pit_lap=24)
    assert list(df["response"]) == ["cover next lap", "planned lap", "extend 3 laps", "extend 6 laps"]
    assert (df["gain_s"] == df["initial_gap_s"] - df["gap_after_both_stops_s"]).all()


def test_safety_car_position_changes(setup):
    df = safety_car_position_changes(setup, OWN, setup.rivals, sc_start_lap=30)
    assert sorted(df["position_after_sc"]) == [1, 2, 3]
    after = df.sort_values("position_after_sc")["gap_to_leader_after_s"].to_numpy()
    assert np.all(np.diff(after) <= setup.race_control.sc_bunching_gap_s + 1e-9)  # gaps compressed by bunching
