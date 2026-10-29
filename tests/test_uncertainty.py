"""Phase 5: confidence intervals, convergence, tail stability, source separation."""

import numpy as np
import pytest

from racesim.config import with_overrides
from racesim.monte_carlo import resolve_flags, run_monte_carlo
from racesim.risk_metrics import calculate_strategy_risk_report
from racesim.schemas import Strategy
from racesim.sensitivity import vary_uncertain_parameters
from racesim.uncertainty import (
    MODEL_UNCERTAINTY_STATEMENT, assess_tail_stability, bootstrap_interval, convergence_diagnostics, convergence_summary,
    model_uncertainty_summary, simulation_precision_table, uncertainty_breakdown, wilson_interval,
)

A = Strategy.from_plan(60, [25, 43], ["MEDIUM", "SOFT", "SOFT"])
B = Strategy.from_plan(60, [26], ["MEDIUM", "HARD"])


def test_wilson_hand_values():
    lo, hi = wilson_interval(0.5, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)
    assert wilson_interval(0.0, 50)[0] == pytest.approx(0.0, abs=1e-12) and wilson_interval(1.0, 50)[1] == pytest.approx(1.0)
    # wider for fewer runs
    assert np.diff(wilson_interval(0.9, 100)) > np.diff(wilson_interval(0.9, 10_000))


def test_risk_report_intervals_contain_estimates():
    rng = np.random.default_rng(1)
    a = rng.normal(100, 2, 2000)
    b = a + rng.normal(0.5, 1, 2000)
    r = calculate_strategy_risk_report(a, baseline_times_s=b)
    lo, hi = r.probability_of_beating_baseline_ci95
    assert lo <= r.probability_of_beating_baseline <= hi
    assert r.mean_race_time_ci95_s[0] < r.expected_race_time_s < r.mean_race_time_ci95_s[1]
    assert r.cvar_95_ci95_s[0] <= r.cvar_95_s <= r.cvar_95_ci95_s[1]
    assert r.monte_carlo_standard_error_s == pytest.approx(a.std(ddof=1) / np.sqrt(a.size))
    assert r.probability_of_beating_rival_ci95 is None  # no rival data -> no interval fabricated


def test_convergence_shrinks_intervals(setup):
    mc = run_monte_carlo(setup, [A, B], n_sims=2000, seed=3)
    d = convergence_diagnostics(mc.race_times[A.strategy_id], mc.race_times[B.strategy_id])
    width = d["p_ci_high"] - d["p_ci_low"]
    assert width.iloc[-1] < width.iloc[0]
    s = convergence_summary(d)
    assert set(s["checks"]) == {"mean", "cvar", "probability", "paired_delta"} and isinstance(s["converged"], bool)


def test_tail_stability_flags_small_samples():
    rng = np.random.default_rng(0)
    small = assess_tail_stability(rng.normal(100, 5, 200))
    assert not small.stable and any("tail" in r for r in small.reasons)
    big = assess_tail_stability(rng.normal(100, 5, 20000), n_boot=100)
    assert big.stable and big.ci95_s[0] <= big.cvar95_s <= big.ci95_s[1]


def test_bootstrap_reproducible():
    x = np.random.default_rng(2).normal(0, 1, 500)
    assert bootstrap_interval(x, seed=4) == bootstrap_interval(x, seed=4)


def test_precision_table_and_crn(setup):
    mc = run_monte_carlo(setup, [A, B], n_sims=800, seed=5, include_rivals=True)
    t = simulation_precision_table(mc, B.strategy_id).set_index("strategy_id")
    assert {"mean_se_s", "cvar95_ci95_s", "p_beat_baseline_ci95", "paired_delta_ci95_s", "p_beat_all_rivals_ci95"} <= set(t.columns)
    # common random numbers: paired difference is far less noisy than the absolute race time
    d = mc.race_times[A.strategy_id] - mc.race_times[B.strategy_id]
    assert d.std() < 0.2 * mc.race_times[A.strategy_id].std()


def test_sources_separated(setup):
    df = uncertainty_breakdown(setup, A, B, n_sims=300, seed=1, include_rivals=True).set_index("source")
    assert {"lap-to-lap variability", "tyre/pace/fuel parameter uncertainty", "pit-stop variability",
            "race events (SC/VSC/rain) + traffic", "rival behaviour", "all sources combined"} <= set(df.index)
    # lap noise is common to both strategies lap by lap, so it cancels in the paired difference
    assert df.loc["lap-to-lap variability", "paired_delta_std_s"] < 1e-6
    # rival behaviour does not change the own car's isolated race time, only rival outcomes
    assert df.loc["rival behaviour", "race_time_std_s"] < 1e-6
    assert df.loc["race events (SC/VSC/rain) + traffic", "race_time_std_s"] > df.loc["race events (SC/VSC/rain) + traffic", "paired_delta_std_s"]


def test_rival_behaviour_off_by_default_and_reproducible(setup):
    assert resolve_flags(setup)["rival_behaviour"] is False
    on = with_overrides(setup, monte_carlo={"include_rival_uncertainty": True})
    a = run_monte_carlo(on, [A], n_sims=300, seed=9, include_rivals=True)
    b = run_monte_carlo(on, [A], n_sims=300, seed=9, include_rivals=True)
    off = run_monte_carlo(setup, [A], n_sims=300, seed=9, include_rivals=True)
    rid = "RIVAL_1"
    assert np.array_equal(a.rival_finish_elapsed[A.strategy_id][rid], b.rival_finish_elapsed[A.strategy_id][rid])
    assert not np.array_equal(a.rival_finish_elapsed[A.strategy_id][rid], off.rival_finish_elapsed[A.strategy_id][rid])
    assert np.allclose(a.race_times[A.strategy_id], off.race_times[A.strategy_id], atol=1e-9)  # own isolated time unaffected


def test_uncertain_parameter_sensitivity(setup):
    df = vary_uncertain_parameters(setup, A, B, n_sims=200, seed=1, ranges={"pit_loss_std_s": (0.0, 2.0)})
    assert len(df) == 2 and df["p_beat_baseline"].between(0, 1).all()


def test_model_uncertainty_statement(setup):
    lines = model_uncertainty_summary(setup)
    assert lines[0] == MODEL_UNCERTAINTY_STATEMENT
    assert any("not run" in l for l in lines)
