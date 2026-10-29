import numpy as np
import pytest

from racesim.risk_metrics import (
    calculate_cvar, calculate_expected_race_time, calculate_probability_of_beating_baseline,
    calculate_probability_of_beating_rival, calculate_race_time_percentiles, calculate_race_time_std,
    calculate_strategy_risk_report, compare_objectives,
)


def test_hand_calculated():
    x = np.arange(1, 101, dtype=float)
    assert calculate_expected_race_time(x) == 50.5
    assert calculate_race_time_std([1, 2, 3, 4]) == pytest.approx(np.std([1, 2, 3, 4], ddof=1))
    p = calculate_race_time_percentiles([10, 20, 30, 40, 50])
    assert p[50] == 30 and p[5] == pytest.approx(12.0)


def test_cvar_convention():
    x = np.arange(1, 101, dtype=float)
    assert calculate_cvar(x) == pytest.approx(np.mean([96, 97, 98, 99, 100]))
    assert calculate_cvar(np.arange(1, 21, dtype=float)) == 20.0   # ceil(0.05*20)=1 worst
    assert calculate_cvar(np.arange(1, 22, dtype=float)) == pytest.approx(20.5)  # ceil(1.05)=2 worst


def test_ties_half():
    assert calculate_probability_of_beating_baseline([1, 2, 3, 4], [2, 2, 2, 2]) == pytest.approx((1 + 0.5) / 4)
    a, b = np.array([1.0, 3, 5]), np.array([2.0, 3, 4])
    assert calculate_probability_of_beating_baseline(a, b) + calculate_probability_of_beating_baseline(b, a) == pytest.approx(1)


def test_rival_probability_none_without_data():
    assert calculate_probability_of_beating_rival([1, 2], None) is None
    assert calculate_probability_of_beating_rival([1, 2], [3, 0]) == 0.5


def test_report_and_tail_warning():
    small = calculate_strategy_risk_report(np.arange(100.0))
    assert small.warnings and small.probability_of_beating_baseline is None and small.probability_of_beating_rival is None
    big = calculate_strategy_risk_report(np.random.default_rng(0).normal(100, 1, 2000))
    assert not big.warnings and big.number_of_simulations == 2000
    multi = calculate_strategy_risk_report([1.0, 2.0], own_finish_s=[1.0, 2.0], rival_finish_s={"A": [2.0, 1.0], "B": [3.0, 3.0]})
    assert multi.probability_of_beating_rival == 0.5


def test_objectives_can_disagree():
    a = calculate_strategy_risk_report(np.array([10.0] * 50 + [30.0] * 50))   # mean 20, high spread
    b = calculate_strategy_risk_report(np.array([20.5] * 100))                 # mean 20.5, no spread
    df = compare_objectives({"A": a, "B": b}).set_index("objective")
    assert df.loc["Best expected race time", "strategy_id"] == "A"
    assert df.loc["Lowest downside risk (CVaR 95)", "strategy_id"] == "B"
    assert df.loc["Highest P(beat all rivals)", "best_strategy"] == "unavailable"
