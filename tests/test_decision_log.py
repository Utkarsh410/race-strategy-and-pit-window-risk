"""Phase 7: decision log content and export."""

import json

import pandas as pd

from racesim import __version__
from racesim.replanning import ReplanningEngine, run_rolling_race
from racesim.schemas import RaceEvent, Strategy

REQUIRED = {"decision_id", "lap", "simulated_time_s", "action", "recommended_pit_lap", "recommended_strategy", "triggers",
            "expected_difference_vs_current_plan_s", "risk_comparison", "influencing_factors", "warnings", "race_state",
            "full_race_plan", "model_version", "random_seed", "configuration_hash", "monte_carlo_simulations"}


def test_log_export_and_fields(setup, tmp_path):
    eng = ReplanningEngine(setup, n_sims=0)
    res = run_rolling_race(setup, Strategy.from_plan(60, [26], ["MEDIUM", "HARD"]),
                           events=[RaceEvent(event_type="safety_car", start_lap=18, duration_laps=4)], engine=eng)
    log = res.decision_log
    assert len(log) == 59  # one record per decision point (laps 1..59)
    p, alt = log.export_csv(tmp_path / "decisions.csv")
    df, adf = pd.read_csv(p), pd.read_csv(alt)
    assert REQUIRED <= set(df.columns)
    assert (df["model_version"] == __version__).all() and (df["random_seed"] == setup.race.simulation_seed).all()
    assert df["configuration_hash"].nunique() == 1
    assert set(df["action"]) <= {"BOX_THIS_LAP", "STAY_OUT", "NO_FURTHER_STOP", "NO_FEASIBLE_STRATEGY"}
    assert set(adf["decision_id"]) <= set(df["decision_id"])
    json.loads(df["race_state"].iloc[0])
    # every replanned record lists the computed alternatives; the recommended one is among them
    for r in log.records:
        if r.replanned:
            assert r.recommended_strategy_id in {a.strategy_id for a in r.alternatives}
    html = log.to_html()
    assert "<table>" in html and "Model version" in html
    j = log.export_json(tmp_path / "log.json")
    assert json.loads(j.read_text())["records"][0]["lap"] == 1


def test_factors_reference_computed_numbers(setup):
    eng = ReplanningEngine(setup, n_sims=0)
    res = run_rolling_race(setup, Strategy.from_plan(60, [26], ["MEDIUM", "HARD"]),
                           events=[RaceEvent(event_type="safety_car", start_lap=18, duration_laps=4)], engine=eng)
    rec = next(r for r in res.decision_log.records if r.lap == 18)
    alt_names = {a.strategy_name for a in rec.alternatives}
    nxt = [f for f in rec.influencing_factors if f.startswith("Next-best option")]
    assert nxt and any(name in nxt[0] for name in alt_names)
