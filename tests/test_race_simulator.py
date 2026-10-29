import numpy as np
import pytest

from racesim.race_simulator import CarSpec, deterministic_scenario, run_race, simulate_race, simulate_stint
from racesim.schemas import Stint, Strategy


def test_time_accounting(setup):
    r = simulate_race(setup, Strategy.from_plan(60, [20, 40], ["MEDIUM", "HARD", "SOFT"]))
    assert r.predicted_race_time_s == pytest.approx(r.lap_time_history.sum() + sum(p.total_cost_s for p in r.pit_stops))
    assert r.predicted_race_time_s == pytest.approx(r.elapsed_history[-1])
    assert np.allclose(np.diff(r.elapsed_history), r.lap_time_history[1:] + r.pit_time_history[1:])


def test_distance_and_termination(setup):
    r = simulate_race(setup, Strategy.from_plan(60, [30], ["MEDIUM", "HARD"]))
    assert len(r.lap_time_history) == 60
    assert r.to_dataframe()["lap"].iloc[-1] == 60
    with pytest.raises(ValueError):
        run_race(setup, [CarSpec("X", Strategy.from_plan(50, [25], ["MEDIUM", "HARD"]))], deterministic_scenario(setup))


def test_pit_laps_and_state_updates(setup):
    st = Strategy.from_plan(60, [15, 38], ["SOFT", "MEDIUM", "HARD"])
    r = simulate_race(setup, st)
    assert [p.lap for p in r.pit_stops] == [15, 38]
    age = r.tyre_age_history
    assert age[0] == 0 and age[14] == 14 and age[15] == 0 and age[37] == 22 and age[38] == 0
    assert r.compound_history[14] == "SOFT" and r.compound_history[15] == "MEDIUM" and r.compound_history[38] == "HARD"
    assert np.allclose(np.diff(r.fuel_history), -setup.race.fuel_consumption_kg_per_lap)
    assert [s.laps for s in r.stint_records] == [15, 23, 22]


def test_stint_simulation(setup):
    df = simulate_stint(setup, Stint(start_lap=21, end_lap=30, tyre_compound="HARD"))
    assert len(df) == 10 and df["tyre_age_laps"].iloc[0] == 0
    assert df["tyre_effect_s"].iloc[0] > df["tyre_effect_s"].iloc[3]  # warm-up on out-lap


def test_rivals_and_positions(setup):
    r = simulate_race(setup, Strategy.from_plan(60, [26], ["MEDIUM", "HARD"]), rivals=setup.rivals)
    assert r.finishing_position in (1, 2, 3)
    assert set(r.simulation_metadata["rivals"]) == {"RIVAL_1", "RIVAL_2"}
