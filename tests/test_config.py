import pytest
from pydantic import ValidationError

from racesim.config import (
    build_setup, load_race_config, load_regulations, regulation_warnings, with_baseline_lap_time, with_overrides,
)
from racesim.schemas import RaceConfig, Strategy

from conftest import SIMPLE_REGS, simple_race_dict


def test_default_config_loads(setup):
    assert setup.race.total_laps == 60
    assert setup.race.simulation_seed == 42
    assert "SYNTHETIC" in setup.race.data_label
    assert set(setup.dry_compounds) == {"SOFT", "MEDIUM", "HARD"}
    assert load_race_config().baseline_lap_time_s == 100.0
    assert load_regulations().refuelling_allowed is False


def test_pit_convention_total_cost(setup):
    # pit_lane_loss excludes stationary time; total = lane + stationary
    assert setup.race.total_pit_cost_s == pytest.approx(setup.race.pit_lane_loss_s + setup.race.stationary_stop_time_s)


def test_invalid_values_rejected():
    d = simple_race_dict()["race"]
    for bad in ({"total_laps": 0}, {"safety_car_probability_per_lap": 1.5}, {"pit_lane_loss_s": -1},
                {"minimum_stint_laps": 10, "maximum_stint_laps": 5}, {"fuel_start_kg": -5}):
        with pytest.raises(ValidationError):
            RaceConfig(**{**d, **bad})


def test_pit_components_must_sum():
    d = simple_race_dict()
    d["pit_stop"].update(pit_entry_loss_s=5, pit_transit_loss_s=5, pit_exit_loss_s=5)  # 15 != 20
    with pytest.raises(ValidationError):
        build_setup(d, SIMPLE_REGS)


def test_unconfigured_rules_warn(make_simple_setup):
    s = make_simple_setup(regs={"mandatory_pit_stops": None, "refuelling_allowed": None})
    w = regulation_warnings(s)
    assert any("mandatory" in x for x in w)
    assert any("Refuelling" in x for x in w)
    assert s.refuelling_allowed is False


def test_overrides_revalidate(simple_setup):
    s2 = with_overrides(simple_setup, race={"total_laps": 30})
    assert s2.race.total_laps == 30 and simple_setup.race.total_laps == 20
    with pytest.raises(ValidationError):
        with_overrides(simple_setup, race={"total_laps": -3})


def test_baseline_shift_keeps_compound_offsets(simple_setup):
    s2 = with_baseline_lap_time(simple_setup, 90.0)
    assert s2.compounds["FAST"].reference_lap_time_s == pytest.approx(89.0)
    assert s2.compounds["SLOW"].reference_lap_time_s == pytest.approx(90.0)


def test_strategy_schema_inclusive_stints():
    s = Strategy.from_plan(20, [8], ["fast", "slow"])
    assert [(x.start_lap, x.end_lap) for x in s.stints] == [(1, 8), (9, 20)]
    assert s.stint_lengths == [8, 12] and sum(s.stint_lengths) == 20
    with pytest.raises(ValueError):
        Strategy.from_plan(20, [8, 5], ["A", "B", "C"])
