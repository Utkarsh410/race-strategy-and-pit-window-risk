import numpy as np
import pytest

from racesim.pit_stop_model import (
    calculate_green_flag_pit_loss, calculate_neutralised_pit_loss, calculate_pit_loss_uncertainty, calculate_total_pit_cost,
    sample_pit_stop_time,
)
from racesim.race_simulator import simulate_race
from racesim.schemas import RaceEvent, Strategy, TrackStatus


def test_total_counts_each_component_once(setup):
    c = calculate_total_pit_cost(setup.race, setup.pit_stop)
    assert c.pit_entry_s + c.pit_transit_s + c.pit_exit_s == pytest.approx(c.pit_lane_loss_s)
    assert c.total_s == pytest.approx(setup.race.pit_lane_loss_s + setup.race.stationary_stop_time_s + setup.pit_stop.rejoin_loss_s)
    assert calculate_green_flag_pit_loss(setup.race, setup.pit_stop) == setup.race.pit_lane_loss_s


def test_stationary_not_double_counted_in_race(simple_setup):
    s0 = Strategy.from_plan(20, [10], ["FAST", "SLOW"])
    r = simulate_race(simple_setup, s0)
    assert len(r.pit_stops) == 1
    assert r.pit_stops[0].total_cost_s == pytest.approx(22.5)
    assert r.predicted_race_time_s == pytest.approx(r.lap_time_history.sum() + 22.5)


def test_neutralised_loss_applied(simple_setup):
    assert calculate_neutralised_pit_loss(simple_setup.pit_stop, TrackStatus.SAFETY_CAR) == 8.0
    assert calculate_total_pit_cost(simple_setup.race, simple_setup.pit_stop, TrackStatus.VSC).total_s == pytest.approx(14.5)
    with pytest.raises(ValueError):
        calculate_neutralised_pit_loss(simple_setup.pit_stop, TrackStatus.GREEN)
    ev = [RaceEvent(event_type="safety_car", start_lap=9, duration_laps=3)]
    r = simulate_race(simple_setup, Strategy.from_plan(20, [10], ["FAST", "SLOW"]), events=ev, reactive=False)
    assert r.pit_stops[0].track_status == "SC" and r.pit_stops[0].lane_loss_s == pytest.approx(8.0)


def test_sampling_reproducible(setup):
    a = sample_pit_stop_time(setup.race, setup.pit_stop, np.random.default_rng(5), 100)
    b = sample_pit_stop_time(setup.race, setup.pit_stop, np.random.default_rng(5), 100)
    assert np.array_equal(a["total_s"], b["total_s"])
    det = sample_pit_stop_time(setup.race, setup.pit_stop, np.random.default_rng(5), 3, stochastic=False)
    assert np.allclose(det["total_s"], setup.race.total_pit_cost_s)
    u = calculate_pit_loss_uncertainty(setup.race, setup.pit_stop)
    assert u["std_s"] > 0 and u["p05_s"] < u["p95_s"]
