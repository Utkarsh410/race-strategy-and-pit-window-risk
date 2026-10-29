import numpy as np
import pytest

from racesim.config import with_overrides
from racesim.race_simulator import simulate_race
from racesim.schemas import RaceEvent, Strategy
from racesim.weather_model import calculate_weather_pace_effect, estimate_wet_tyre_suitability, generate_weather_scenario


def test_stable_and_trends(setup):
    w = generate_weather_scenario(setup.weather, 20, 30.0)
    assert np.allclose(w.track_temperature_c, 30.0) and np.allclose(w.wetness, 0.0)
    rising = with_overrides(setup, weather={"scenario": "rising_temperature", "temperature_change_c_per_lap": 0.5}).weather
    t = generate_weather_scenario(rising, 5, 30.0).track_temperature_c
    assert np.allclose(np.diff(t), 0.5)
    cooling = with_overrides(setup, weather={"scenario": "cooling_temperature", "temperature_change_c_per_lap": 0.5}).weather
    assert np.all(np.diff(generate_weather_scenario(cooling, 5, 30.0).track_temperature_c) < 0)


def test_rain_onset_and_drying(setup):
    cfg = with_overrides(setup, weather={"scenario": "rain_onset", "rain_onset_lap": 5, "rain_duration_laps": 4}).weather
    w = generate_weather_scenario(cfg, 20, 30.0)
    assert w.wetness[3] == 0 and w.wetness[4] > 0 and w.wetness[7] >= w.wetness[4]
    assert w.wetness[15] < w.wetness[8]


def test_suitability_and_disable(setup):
    cfg = setup.weather
    assert calculate_weather_pace_effect(0.0, 0, cfg) == 0
    assert calculate_weather_pace_effect(0.8, 0, cfg) > calculate_weather_pace_effect(0.8, 2, cfg)
    df = estimate_wet_tyre_suitability(cfg, {"MEDIUM": (100.0, 0), "INTERMEDIATE": (106.0, 1), "WET": (109.0, 2)})
    assert df[df.wetness == 0.0]["best_compound"].iloc[0] == "MEDIUM"
    assert df[df.wetness == 1.0]["best_compound"].iloc[0] in ("WET", "INTERMEDIATE")
    off = with_overrides(setup, weather={"wet_weather_enabled": False}).weather
    w = generate_weather_scenario(off, 10, 30.0, rain_events=[RaceEvent(event_type="rain", start_lap=2, duration_laps=5)])
    assert np.allclose(w.wetness, 0) and w.warnings


def test_rain_triggers_tyre_change(setup):
    st = Strategy.from_plan(60, [26], ["MEDIUM", "HARD"])
    r = simulate_race(setup, st, events=[RaceEvent(event_type="rain", start_lap=30, duration_laps=15, severity=0.8)])
    assert any(c in ("INTERMEDIATE", "WET") for c in r.compound_history)
    assert r.predicted_race_time_s == pytest.approx(r.lap_time_history.sum() + sum(p.total_cost_s for p in r.pit_stops))
    assert [p.reason for p in r.pit_stops][1:] == ["reactive", "reactive"]
