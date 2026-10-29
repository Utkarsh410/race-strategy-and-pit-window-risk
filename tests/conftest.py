"""Shared fixtures."""

from __future__ import annotations

import pandas as pd
import pytest

from racesim.config import build_setup, load_setup


@pytest.fixture(scope="session")
def setup():
    """The shipped default (synthetic) configuration."""
    return load_setup()


def simple_race_dict(**race_overrides) -> dict:
    """A small analytic configuration with a hand-computable optimum.

    FAST is 1 s/lap quicker than SLOW, neither degrades, no fuel effect, no
    standing-start loss. FAST has a 1 s warm-up penalty, SLOW none, so the
    optimum is FAST first for as long as possible, then the minimum SLOW stint.
    """
    race = dict(
        race_name="Unit test race", circuit_name="Test circuit", total_laps=20, baseline_lap_time_s=100.0,
        pit_lane_loss_s=20.0, stationary_stop_time_s=2.5, pit_loss_std_s=0.5, baseline_lap_time_std_s=0.3,
        fuel_start_kg=50.0, fuel_consumption_kg_per_lap=2.0, fuel_effect_s_per_kg=0.0,
        safety_car_probability_per_lap=0.0, vsc_probability_per_lap=0.0, track_temperature_c=35.0,
        ambient_temperature_c=25.0, reference_track_temperature_c=35.0, simulation_seed=1, simulation_count=100,
        minimum_stint_laps=3, maximum_stint_laps=50, minimum_pit_laps_remaining=1, maximum_strategy_candidates=5000,
        maximum_stops=2, standing_start_loss_s=0.0,
    )
    race.update(race_overrides)
    return {
        "race": race,
        "compounds": {
            "FAST": dict(reference_lap_time_s=99.0, linear_degradation_s_per_lap=0.0, warmup_penalty_s=1.0, warmup_decay_laps=1.0,
                         maximum_recommended_age_laps=100, degradation_model="linear"),
            "SLOW": dict(reference_lap_time_s=100.0, linear_degradation_s_per_lap=0.0, warmup_penalty_s=0.0, warmup_decay_laps=1.0,
                         maximum_recommended_age_laps=100, degradation_model="linear"),
        },
        "pit_stop": {"sc_pit_lane_loss_s": 8.0, "vsc_pit_lane_loss_s": 12.0, "stationary_std_s": 0.2},
        "weather": {"wet_weather_enabled": False},
        "traffic": {"traffic_probability_per_lap": 0.0},
        "baseline_strategy": {"pit_laps": [10], "compounds": ["FAST", "SLOW"]},
    }


SIMPLE_REGS = {
    "championship_name": "Test series", "minimum_stint_laps": 3, "maximum_stint_laps": 50, "mandatory_pit_stops": 1,
    "maximum_pit_stops": None, "minimum_distinct_dry_compounds": 2, "permitted_compounds": ["FAST", "SLOW"],
    "refuelling_allowed": False, "driver_rules": [],
}


@pytest.fixture
def simple_setup():
    return build_setup(simple_race_dict(), SIMPLE_REGS)


@pytest.fixture
def make_simple_setup():
    def _make(regs: dict | None = None, **race):
        return build_setup(simple_race_dict(**race), {**SIMPLE_REGS, **(regs or {})})
    return _make


@pytest.fixture
def lap_df():
    """Small valid lap dataset for two drivers."""
    rows = []
    for drv, off in (("D1", 0.0), ("D2", 0.3)):
        for lap in range(1, 21):
            stint2 = lap > 10
            age = lap - 11 if stint2 else lap - 1
            rows.append({
                "lap": lap, "driver": drv, "lap_time_s": 100 + off + 0.05 * age + (0.2 if lap % 3 == 0 else -0.1),
                "tyre_compound": "HARD" if stint2 else "MEDIUM", "tyre_age_laps": age,
                "fuel_kg": 60 - 2.0 * (lap - 1), "track_status": "GREEN",
                "pit_in": lap == 10, "pit_out": lap == 11,
            })
    return pd.DataFrame(rows)
