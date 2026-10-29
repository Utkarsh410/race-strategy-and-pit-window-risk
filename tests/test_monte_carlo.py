import numpy as np
import pytest

from racesim.monte_carlo import compare_strategy_distributions, decompose_uncertainty, rng_for, run_monte_carlo
from racesim.race_simulator import simulate_race
from racesim.risk_metrics import calculate_strategy_risk_report
from racesim.schemas import Strategy

S1 = Strategy.from_plan(60, [26], ["MEDIUM", "HARD"])
S2 = Strategy.from_plan(60, [20, 40], ["MEDIUM", "SOFT", "SOFT"])


def test_seed_reproducible(setup):
    a = run_monte_carlo(setup, [S1], n_sims=200, seed=11)
    b = run_monte_carlo(setup, [S1], n_sims=200, seed=11)
    c = run_monte_carlo(setup, [S1], n_sims=200, seed=12)
    assert np.array_equal(a.race_times[S1.strategy_id], b.race_times[S1.strategy_id])
    assert not np.array_equal(a.race_times[S1.strategy_id], c.race_times[S1.strategy_id])


def test_batches_return_expected_count(setup):
    r = run_monte_carlo(setup, [S1, S2], n_sims=1234, seed=1, batch_size=500)
    assert all(len(v) == 1234 for v in r.race_times.values())
    assert len(r.to_dataframe()) == 2468


def test_deterministic_flags_match_simulator(setup):
    off = dict(parameter=False, lap_variability=False, events=False, pit_variability=False, traffic=False, reactive=False)
    r = run_monte_carlo(setup, [S1], n_sims=5, seed=3, flags=off)
    assert np.allclose(r.race_times[S1.strategy_id], simulate_race(setup, S1).predicted_race_time_s)


def test_common_random_numbers_reduce_noise(setup):
    r = run_monte_carlo(setup, [S1, S2], n_sims=600, seed=4)
    cmp = compare_strategy_distributions(r, S2.strategy_id, S1.strategy_id)
    assert cmp["std_difference_s"] < cmp["independent_std_s"]
    assert 0 <= cmp["probability_a_faster"] <= 1


def test_missing_rival_no_fabricated_probability(setup):
    r = run_monte_carlo(setup, [S1], n_sims=100, seed=2, include_rivals=False)
    rep = calculate_strategy_risk_report(r.race_times[S1.strategy_id])
    assert rep.probability_of_beating_rival is None and r.positions is None
    rr = run_monte_carlo(setup, [S1], n_sims=100, seed=2, include_rivals=True)
    assert set(rr.rival_finish_elapsed[S1.strategy_id]) == {"RIVAL_1", "RIVAL_2"}


def test_independent_streams():
    a = rng_for(1, "x").random(5)
    b = rng_for(1, "y").random(5)
    assert not np.allclose(a, b)
    assert np.array_equal(a, rng_for(1, "x").random(5))


def test_uncertainty_decomposition(setup):
    df = decompose_uncertainty(setup, S1, n_sims=100, seed=1)
    assert "all sources combined" in set(df["source"])
    assert (df["std_s"] >= 0).all()
