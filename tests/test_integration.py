import pandas as pd
import pytest

from racesim.data_ingestion import export_strategy_comparison, load_race_csv
from racesim.monte_carlo import run_monte_carlo
from racesim.race_simulator import simulate_race
from racesim.reporting import export_race_summary, export_strategy_summary, generate_strategy_report
from racesim.risk_metrics import calculate_strategy_risk_report
from racesim.sensitivity import rank_sensitive_parameters, vary_pit_loss, vary_tyre_degradation
from racesim.strategy_optimizer import find_optimal_strategy
from racesim.synthetic_data import generate_synthetic_race, write_synthetic_dataset
from racesim.validation import generate_validation_report, split_training_validation_data
from racesim.workflow import run_full_analysis


def test_end_to_end(setup, tmp_path):
    df, meta = generate_synthetic_race(setup, seed=7)
    csv, meta_path = write_synthetic_dataset(df, meta, tmp_path / "demo.csv")
    assert meta["data_source"] == "SYNTHETIC" and meta_path.exists()
    ing = load_race_csv(csv)
    assert ing.report.valid and (ing.data["data_source"] == "SYNTHETIC").all()

    opt = find_optimal_strategy(setup)
    assert opt.best is not None and opt.best.feasible
    sim = simulate_race(setup, opt.best.strategy)
    assert sim.predicted_race_time_s == pytest.approx(opt.best.predicted_race_time_s)

    mc = run_monte_carlo(setup, [opt.best.strategy, opt.baseline.strategy], n_sims=200, seed=1)
    rep = calculate_strategy_risk_report(mc.race_times[opt.best.strategy.strategy_id],
                                         baseline_times_s=mc.race_times[opt.baseline.strategy.strategy_id])
    assert rep.number_of_simulations == 200 and rep.probability_of_beating_baseline is not None

    p1 = export_strategy_comparison(opt, tmp_path / "cmp.csv")
    p2 = export_race_summary(sim, tmp_path / "race.csv")
    p3 = export_strategy_summary(opt, tmp_path / "summary.csv")
    assert len(pd.read_csv(p1)) == len(opt.ranking) and len(pd.read_csv(p2)) == 60 and p3.exists()


def test_full_report(setup, tmp_path):
    bundle = run_full_analysis(setup, n_sims=200, seed=3, run_sensitivity=False)
    html = generate_strategy_report(bundle, tmp_path / "report.html")
    assert (tmp_path / "report.html").exists()
    for section in ("1. Race configuration", "5. Pit-window analysis", "6. Risk metrics", "7. Safety-car",
                    "9. Model limitations", "10. Data provenance", "SYNTHETIC"):
        assert section in html
    assert "Sensitivity analysis was not run" in html


def test_validation_split_has_no_stint_leakage(setup):
    df, _ = generate_synthetic_race(setup, seed=5)
    train, valid = split_training_validation_data(df, method="stint")
    assert set(train["stint_id"]).isdisjoint(set(valid["stint_id"]))
    rep = generate_validation_report(df, fuel_effect_s_per_kg=setup.race.fuel_effect_s_per_kg, min_validation_samples=10)
    assert rep["status"] in ("conclusive", "inconclusive")
    if rep["status"] == "conclusive":
        assert {"linear", "nonlinear"} <= set(rep["models"])
        assert any("SYNTHETIC" in w for w in rep["warnings"])


def test_validation_inconclusive_small(setup):
    df, _ = generate_synthetic_race(setup, seed=5)
    rep = generate_validation_report(df.head(15))
    assert rep["status"] == "inconclusive" and rep["models"] == {}


def test_sensitivity(setup):
    r1 = vary_pit_loss(setup, values_s=[10.0, 20.0, 35.0])
    r2 = vary_tyre_degradation(setup, multipliers=[0.8, 1.0, 1.4])
    assert len(r1.table) == 3 and r1.table["stops"].iloc[0] >= r1.table["stops"].iloc[-1]
    ranking = rank_sensitive_parameters([r1, r2])
    assert set(ranking["parameter"]) == {"pit_lane_loss_s", "tyre_degradation_multiplier"}
