import numpy as np
import pytest

from racesim.lap_time_model import (
    LapModelContext, calculate_lap_time, calculate_sector_statistics, calculate_stint_pace, detect_pace_trend,
    estimate_baseline_pace, estimate_pace_uncertainty, predict_lap_time,
)


def test_additive_sum_and_floor():
    assert calculate_lap_time(100, 1, 2, 0.5, 0, -0.2, 0.1, 0.3) == pytest.approx(103.7)
    assert calculate_lap_time(100, random_error_s=-500, minimum_lap_time_s=85.0) == 85.0
    assert calculate_lap_time(10, random_error_s=-50) > 0


def test_components_each_applied_once(setup):
    ctx = LapModelContext.from_setup(setup)
    i = ctx.table.index("MEDIUM")
    total, comp = predict_lap_time(ctx, i, 10, 5, 5, 80.0, False, 35.0, 0.0, return_components=True)
    assert total == pytest.approx(sum(float(np.asarray(v)) for v in comp.values()))
    assert comp["fuel_s"] == pytest.approx(80 * setup.race.fuel_effect_s_per_kg)
    assert comp["reference_s"] == pytest.approx(setup.compounds["MEDIUM"].reference_lap_time_s)


def test_deterministic_vs_stochastic(setup):
    ctx = LapModelContext.from_setup(setup)
    a = predict_lap_time(ctx, 0, 5, 3, 3, 50, False, 35)
    b = predict_lap_time(ctx, 0, 5, 3, 3, 50, False, 35)
    c = predict_lap_time(ctx, 0, 5, 3, 3, 50, False, 35, random_error_s=0.4)
    assert a == b and c == pytest.approx(a + 0.4)


def test_analysis_functions(lap_df):
    assert estimate_baseline_pace(lap_df, fuel_effect_s_per_kg=0.03)["baseline_lap_time_s"] is not None
    assert len(calculate_stint_pace(lap_df)) == 4
    assert set(detect_pace_trend(lap_df)["driver"]) == {"D1", "D2"}
    assert estimate_pace_uncertainty(lap_df)["lap_time_std_s"] > 0
    assert calculate_sector_statistics(lap_df).empty  # no sector columns: graceful
