"""Phase 2: historical-data validation workflow.

The 'historical' fixture below is a deterministic formula-based table built
for unit tests only (labelled UNIT_TEST_FIXTURE); it is not race data.
"""

import numpy as np
import pandas as pd
import pytest

from racesim.data_ingestion import load_historical_csv
from racesim.synthetic_data import generate_synthetic_race
from racesim.validation import (
    analyse_residuals, apply_calibration, calculate_interval_coverage, calibrate_tyre_model, check_split_leakage,
    classify_evidence, generate_validation_report, run_model_validation, split_training_validation_data,
)


def fixture_laps(n_drivers=4, stints=(18, 22, 20), label="UNIT_TEST_FIXTURE", seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_drivers):
        lap = 1
        for k, length in enumerate(stints):
            comp = ["MEDIUM", "HARD", "MEDIUM"][k]
            for age in range(length):
                rows.append({"lap": lap, "driver": f"T{d}", "lap_time_s": 100 + 0.2 * d + 0.04 * age + 0.03 * (110 - 1.75 * (lap - 1))
                             + rng.normal(0, 0.2), "tyre_compound": comp, "tyre_age_laps": age, "fuel_kg": 110 - 1.75 * (lap - 1),
                             "track_status": "GREEN", "pit_in": age == length - 1 and k < len(stints) - 1,
                             "pit_out": age == 0 and k > 0, "data_source": label})
                lap += 1
    return pd.DataFrame(rows)


def test_evidence_classification(setup):
    syn, _ = generate_synthetic_race(setup, seed=1)
    assert classify_evidence(syn) == "synthetic_self_consistency"
    assert classify_evidence(fixture_laps()) == "historical_out_of_sample"
    assert classify_evidence(fixture_laps().drop(columns=["data_source"])) == "unlabelled"
    assert classify_evidence(pd.concat([syn, fixture_laps()])) == "mixed"


def test_stint_split_has_no_leakage():
    df = fixture_laps()
    train, valid = split_training_validation_data(df, method="stint", seed=3)
    chk = check_split_leakage(train, valid)
    assert chk["leakage_free"] and chk["shared_stints"] == []
    # leakage is detected when a stint is in both sets
    bad = check_split_leakage(train, pd.concat([valid, train.head(3)]))
    assert not bad["leakage_free"]


def test_metrics_and_coverage_reported():
    r = generate_validation_report(fixture_laps(), fuel_effect_s_per_kg=0.03, min_validation_samples=20)
    assert r["status"] == "conclusive" and r["evidence_type"] == "historical_out_of_sample"
    m = r["models"]["linear"]
    for k in ("mae_s", "rmse_s", "bias_s", "n_validation", "interval_coverage"):
        assert m[k] is not None
    assert 0 <= m["interval_coverage"] <= 1 and m["rmse_s"] >= m["mae_s"]
    p = r["predictions"]
    assert m["interval_coverage"] == pytest.approx(calculate_interval_coverage(
        p.loc[p.model == "linear", "lap_time_s"], p.loc[p.model == "linear", "interval_low_s"], p.loc[p.model == "linear", "interval_high_s"]))
    assert "diagnostics" in r["residual_analysis"]


def test_interval_coverage_hand_example():
    assert calculate_interval_coverage([1, 2, 3, 4], [0, 2.5, 2, 5], [2, 3, 4, 6]) == 0.5


def test_insufficient_data_is_inconclusive():
    r = run_model_validation(fixture_laps(n_drivers=1, stints=(6, 6)), _setup())
    assert r.status == "inconclusive" and "inconclusive" in r.headline.lower()
    assert r.metrics_table().empty


def test_incomplete_fields_supported():
    df = fixture_laps().drop(columns=["fuel_kg", "track_status", "pit_in", "pit_out"])
    r = generate_validation_report(df, min_validation_samples=20)
    assert r["status"] == "conclusive"
    assert any("fuel_kg missing" in w for m in r["models"].values() for w in m["warnings"])
    assert generate_validation_report(df.drop(columns=["tyre_age_laps"]))["status"] == "inconclusive"


def test_synthetic_kept_separate_from_historical(setup):
    syn, _ = generate_synthetic_race(setup, seed=2)
    mixed = pd.concat([syn, fixture_laps()], ignore_index=True)
    res = run_model_validation(mixed, setup, min_validation_samples=20)
    assert set(res.separate_results) == {"synthetic_self_consistency", "non_synthetic"}
    assert res.evidence_type == "historical_out_of_sample"
    assert (res.report["predictions"]["data_source"] == "UNIT_TEST_FIXTURE").all()
    syn_only = run_model_validation(syn, setup)
    assert "SYNTHETIC self-consistency" in syn_only.headline or syn_only.status == "inconclusive"
    assert generate_validation_report(mixed)["models"] == {}  # mixed data never pooled


def test_calibration(setup):
    cal = calibrate_tyre_model(fixture_laps(), setup)
    med = cal.table.set_index("compound").loc["MEDIUM"]
    assert med["fitted_linear_s_per_lap"] == pytest.approx(0.04, abs=0.02)
    assert cal.evidence_type == "historical_out_of_sample"
    new = apply_calibration(setup, cal)
    if "MEDIUM" in cal.recommended_overrides:
        assert new.compounds["MEDIUM"].linear_degradation_s_per_lap == pytest.approx(med["fitted_linear_s_per_lap"])
    assert setup.compounds["MEDIUM"].linear_degradation_s_per_lap == 0.04  # original untouched


def test_historical_csv_provenance(tmp_path):
    p = tmp_path / "laps.csv"
    fixture_laps().drop(columns=["data_source"]).to_csv(p, index=False)
    res, prov = load_historical_csv(p)
    assert prov["data_source_labels"] == ["UNLABELLED"] and len(prov["sha256"]) == 64
    res2, prov2 = load_historical_csv(p, source_label="TEAM_LOGS_2026")
    assert classify_evidence(res2.data) == "historical_out_of_sample"


def test_residual_analysis_empty():
    assert analyse_residuals(pd.DataFrame()) == {}


def _setup():
    from racesim.config import load_setup
    return load_setup()
