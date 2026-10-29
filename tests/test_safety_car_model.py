import numpy as np
import pytest

from racesim.lap_time_model import LapModelContext
from racesim.race_simulator import simulate_race
from racesim.safety_car_model import (
    calculate_neutralisation_pit_gain, calculate_restart_effect, events_to_track_status, generate_race_control_events,
    simulate_safety_car, simulate_vsc, track_status_to_events, update_strategy_after_event,
)
from racesim.schemas import RaceEvent, Strategy, TrackStatus


def test_manual_events_and_round_trip():
    ev = [RaceEvent(event_type="safety_car", start_lap=5, duration_laps=3), RaceEvent(event_type="vsc", start_lap=20, duration_laps=2)]
    st = events_to_track_status(ev, 30)
    assert list(np.nonzero(st == 1)[0] + 1) == [5, 6, 7] and list(np.nonzero(st == 2)[0] + 1) == [20, 21]
    back = track_status_to_events(st)
    assert [(e.event_type.value, e.start_lap, e.duration_laps) for e in back] == [("safety_car", 5, 3), ("vsc", 20, 2)]


def test_overlapping_events_rejected():
    with pytest.raises(ValueError):
        events_to_track_status([RaceEvent(event_type="safety_car", start_lap=5, duration_laps=3),
                                RaceEvent(event_type="vsc", start_lap=6, duration_laps=2)], 30)


def test_random_generation_reproducible_and_exclusive(setup):
    a = generate_race_control_events(setup, 60, rng=np.random.default_rng(3), n_sims=500)
    b = generate_race_control_events(setup, 60, rng=np.random.default_rng(3), n_sims=500)
    assert np.array_equal(a, b)
    assert set(np.unique(a)) <= {0, 1, 2}
    assert (a[:, 0] == 0).all()  # earliest_event_lap = 2
    assert 0.3 < (a == 1).any(axis=1).mean() < 0.7


def test_sc_and_vsc_differ(setup):
    sc, vsc = simulate_safety_car(setup, 10, 4), simulate_vsc(setup, 10, 2)
    assert sc.pit_lane_loss_s != vsc.pit_lane_loss_s
    assert sc.pit_cost_saving_s > vsc.pit_cost_saving_s > 0
    assert calculate_restart_effect(TrackStatus.SAFETY_CAR, TrackStatus.GREEN, setup) == setup.race_control.sc_restart_loss_s


def test_pit_under_sc_not_always_beneficial(setup):
    ctx = LapModelContext.from_setup(setup)
    i_m, i_h = ctx.table.index("MEDIUM"), ctx.table.index("HARD")
    # Planned stop next lap anyway: small gain. Planned stop far away with a long stint after: may lose.
    near = calculate_neutralisation_pit_gain(setup, ctx, status=1, current_lap=25, compound_idx=i_m, tyre_age_laps=25,
                                             wear_laps=25, fuel_kg=66, warmup_applies=False, next_compound_idx=i_h,
                                             planned_pit_lap=26, horizon_end_lap=60)
    assert near["gain_s"] > 0
    pits, d = update_strategy_after_event(setup, ctx, planned_pit_laps=[50], planned_compounds=["HARD", "MEDIUM"],
                                          stops_completed=0, status=1, current_lap=5, stint_start_lap=1,
                                          tyre_age_laps=5, wear_laps=5, fuel_kg=100)
    assert pits == [50] and d["action"] == "stay_out"


def test_sc_slows_race(setup):
    st = Strategy.from_plan(60, [26], ["MEDIUM", "HARD"])
    green = simulate_race(setup, st).predicted_race_time_s
    sc = simulate_race(setup, st, events=[RaceEvent(event_type="safety_car", start_lap=40, duration_laps=4)], reactive=False)
    assert sc.predicted_race_time_s > green
    assert sc.event_history[39:43] == ["SC"] * 4
