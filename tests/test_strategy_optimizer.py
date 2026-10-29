import pytest

from racesim.race_simulator import simulate_race
from racesim.schemas import Strategy
from racesim.strategy_optimizer import (
    calculate_overcut_gain, calculate_total_race_time, calculate_undercut_gain, check_strategy_feasibility, compare_strategies,
    find_optimal_strategy, simulate_strategy,
)


def test_known_optimum(simple_setup):
    res = find_optimal_strategy(simple_setup)
    best = res.best
    assert best.strategy.planned_compounds == ["FAST", "SLOW"]
    assert best.strategy.planned_pit_laps == [17]
    assert best.predicted_race_time_s == pytest.approx(17 * 99 + 3 * 100 + 22.5)
    assert best.total_pit_cost_s == pytest.approx(22.5)
    assert "Recommended" in res.explanation


def test_identical_strategies_equal(simple_setup):
    s = Strategy.from_plan(20, [12], ["FAST", "SLOW"])
    assert calculate_total_race_time(simple_setup, s) == calculate_total_race_time(simple_setup, Strategy.from_plan(20, [12], ["FAST", "SLOW"]))


def test_infeasible_excluded(simple_setup):
    bad = Strategy.from_plan(20, [], ["FAST"])
    assert not check_strategy_feasibility(simple_setup, bad).valid
    df = compare_strategies(simple_setup, [bad, Strategy.from_plan(20, [12], ["FAST", "SLOW"])])
    assert len(df) == 1 and bad.strategy_id not in set(df["strategy_id"])
    res = find_optimal_strategy(simple_setup, candidates=[bad])
    assert res.best is None


def test_optimizer_matches_simulator(setup):
    res = find_optimal_strategy(setup)
    for e in res.evaluations[:5]:
        assert simulate_race(setup, e.strategy).predicted_race_time_s == pytest.approx(e.predicted_race_time_s, abs=1e-6)


def test_candidate_limit_respected(make_simple_setup):
    res = find_optimal_strategy(make_simple_setup(maximum_strategy_candidates=40))
    assert res.best is not None


def test_undercut_overcut_not_fixed_bonus(setup):
    u1 = calculate_undercut_gain(setup, tyre_compound="MEDIUM", tyre_age_laps_at_pit=10, new_compound="HARD", own_pit_lap=11, rival_pit_lap=12)
    u2 = calculate_undercut_gain(setup, tyre_compound="MEDIUM", tyre_age_laps_at_pit=30, new_compound="HARD", own_pit_lap=31, rival_pit_lap=32)
    assert u2["gain_s"] > u1["gain_s"]  # older tyres -> bigger undercut
    o = calculate_overcut_gain(setup, tyre_compound="MEDIUM", tyre_age_laps_at_rival_pit=10, new_compound="HARD", own_pit_lap=13, rival_pit_lap=11)
    assert "gain_s" in o


def test_breakdown_fields(setup):
    e = simulate_strategy(setup, Strategy.from_plan(60, [26], ["MEDIUM", "HARD"]))
    row = e.as_row()
    for k in ("race_time_s", "total_pit_cost_s", "tyre_time_s", "fuel_time_s", "stint_lengths", "feasible"):
        assert k in row
    assert e.assumptions
