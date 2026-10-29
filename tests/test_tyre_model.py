import numpy as np
import pytest

from racesim.schemas import TyreModelConfig
from racesim.tyre_model import (
    calculate_degradation, calculate_tyre_age, calculate_tyre_life, calculate_warmup_penalty, compare_tyre_compounds,
    estimate_degradation_uncertainty, estimate_tyre_degradation, predict_tyre_pace,
)


def comp(**kw):
    base = dict(compound_name="T", reference_lap_time_s=100.0, linear_degradation_s_per_lap=0.0, maximum_recommended_age_laps=40)
    base.update(kw)
    return TyreModelConfig(**base)


def test_zero_degradation():
    p = predict_tyre_pace(comp(), np.arange(30))
    assert np.allclose(p, 100.0)


def test_linear_degradation_increases():
    p = predict_tyre_pace(comp(linear_degradation_s_per_lap=0.05, degradation_model="linear"), np.arange(30))
    assert np.all(np.diff(p) > 0)
    assert p[10] - p[0] == pytest.approx(0.5)


def test_nonlinear_threshold():
    d = calculate_degradation(np.arange(20), 0.0, 0.01, 10)
    assert np.allclose(d[:11], 0.0)
    assert d[15] == pytest.approx(0.01 * 25)
    assert np.allclose(calculate_degradation(np.arange(20), 0.0, 0.01, 10, use_nonlinear=False), 0.0)


def test_tyre_age_and_reset():
    assert calculate_tyre_age(1, 1) == 0
    assert calculate_tyre_age(11, 15, starting_tyre_age_laps=3) == 7
    with pytest.raises(ValueError):
        calculate_tyre_age(10, 5)


def test_warmup_decay():
    w = calculate_warmup_penalty(np.arange(5), 1.5, 1.0)
    assert w[0] == pytest.approx(1.5)
    assert w[2] == pytest.approx(1.5 * np.exp(-2))
    assert np.all(np.diff(w) < 0)
    assert calculate_warmup_penalty(0, 1.5, 1.0, applies=False) == 0


def test_life_and_comparison(setup):
    life = calculate_tyre_life(setup.compounds["SOFT"])
    assert life["recommended_life_laps"] <= life["maximum_recommended_age_laps"]
    df = compare_tyre_compounds(setup.compounds, max_age_laps=10)
    assert set(df["compound"]) == {"SOFT", "MEDIUM", "HARD"}


def test_estimation_recovers_known_slope(lap_df):
    est = estimate_tyre_degradation(lap_df, fuel_effect_s_per_kg=0.0, min_samples=5, exclude_warmup_laps=0)
    assert est["MEDIUM"].linear_degradation_s_per_lap == pytest.approx(0.05, abs=0.03)


def test_insufficient_data_warning(lap_df):
    est = estimate_tyre_degradation(lap_df.head(6), fuel_effect_s_per_kg=0.03)
    e = est["MEDIUM"]
    assert not e.reliable and e.linear_degradation_s_per_lap is None and e.warnings
    assert estimate_degradation_uncertainty(lap_df.head(6), "MEDIUM")["status"] == "insufficient-data"


def test_unknown_fuel_flags_confounding(lap_df):
    est = estimate_tyre_degradation(lap_df.drop(columns=["fuel_kg"]), min_samples=5)
    assert not est["MEDIUM"].reliable
    assert any("confounded" in w for w in est["MEDIUM"].warnings)
