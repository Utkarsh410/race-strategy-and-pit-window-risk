import numpy as np
import pandas as pd
import pytest

from racesim.fuel_model import (
    calculate_fuel_remaining, calculate_fuel_time_effect, estimate_fuel_consumption, validate_fuel_feasibility,
)
from racesim.schemas import Strategy
from racesim.strategy_generator import validate_strategy


def test_fuel_decreases_linearly():
    f = calculate_fuel_remaining(100.0, 2.0, np.arange(5))
    assert np.allclose(f, [100, 98, 96, 94, 92])


def test_fuel_never_negative():
    assert calculate_fuel_remaining(10.0, 2.0, 50) == 0.0
    with pytest.raises(ValueError):
        calculate_fuel_time_effect(-1.0, 0.03)


def test_feasibility():
    assert validate_fuel_feasibility(110, 1.75, 60).feasible
    bad = validate_fuel_feasibility(100, 1.75, 60)
    assert not bad.feasible and "refuelling not permitted" in bad.reason


def test_infeasible_strategy_rejected(make_simple_setup):
    s = make_simple_setup(fuel_start_kg=30.0)  # needs 40 kg for 20 laps
    v = validate_strategy(Strategy.from_plan(20, [10], ["FAST", "SLOW"]), s)
    assert not v.valid and any("fuel shortfall" in r for r in v.reasons)


def test_consumption_estimate_and_assumption(lap_df):
    est = estimate_fuel_consumption(lap_df)
    assert est.source == "measured" and est.kg_per_lap == pytest.approx(2.0)
    a = estimate_fuel_consumption(pd.DataFrame({"lap": [1, 2]}), assumed_kg_per_lap=1.5)
    assert a.source == "assumption" and a.warnings
    assert estimate_fuel_consumption(None).kg_per_lap is None
