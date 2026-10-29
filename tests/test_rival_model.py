import numpy as np
import pandas as pd
import pytest

from racesim.rival_model import (
    calculate_time_gap, estimate_rival_pace, estimate_traffic_loss, evaluate_overcut, evaluate_rival_response,
    evaluate_undercut, predict_rejoin_position, predict_rival_pit_window, simulate_rival_strategy,
)
from racesim.synthetic_data import generate_synthetic_race

KW = dict(current_lap=18, gap_s=1.0, own_compound="MEDIUM", own_tyre_age_laps=18, own_new_compound="HARD")


def test_gap_and_rejoin():
    assert np.allclose(calculate_time_gap([10, 20], [9, 21], own_offset_s=1.0), [2.0, 0.0])
    r = predict_rejoin_position(100.0, [95.0, 101.0, 120.0])
    assert r["position"] == 2 and r["gap_to_car_ahead_s"] == 5.0 and r["gap_to_car_behind_s"] == 1.0


def test_traffic_loss(setup):
    assert estimate_traffic_loss(None, 0.0, 5, setup) == 0.0
    assert estimate_traffic_loss(0.5, 2.0, 5, setup) == 0.0          # fast enough to pass
    assert estimate_traffic_loss(0.5, 0.1, 5, setup) == pytest.approx(5 * setup.traffic.dirty_air_loss_s_per_lap)


def test_undercut_depends_on_inputs(setup):
    a = evaluate_undercut(setup, own_pit_lap=20, rival_pit_lap=23, **KW)
    b = evaluate_undercut(setup, own_pit_lap=20, rival_pit_lap=23, **{**KW, "own_new_compound": "SOFT"}, rival_new_compound="HARD")
    assert a.kind == "undercut" and a.gain_s != b.gain_s
    assert a.gain_s == pytest.approx(KW["gap_s"] - a.gap_after_both_stops_s)
    assert a.successful == (a.gap_after_both_stops_s < 0)
    with pytest.raises(ValueError):
        evaluate_undercut(setup, own_pit_lap=25, rival_pit_lap=23, **KW)


def test_overcut_and_response(setup):
    o = evaluate_overcut(setup, own_pit_lap=25, rival_pit_lap=21, **KW)
    assert o.kind == "overcut" and len(o.gap_history) == 60 - 18
    resp = evaluate_rival_response(setup, own_pit_lap=22, rival_pit_laps=range(19, 27), **KW)
    assert set(resp["kind"]) == {"undercut", "overcut"} and 22 not in set(resp["rival_pit_lap"])


def test_rival_window_and_sim(setup):
    w = predict_rival_pit_window(setup, setup.rivals[0])
    assert w.recommended_lap is not None and w.earliest_feasible_lap <= w.recommended_lap <= w.latest_feasible_lap
    r = simulate_rival_strategy(setup, setup.rivals[0])
    assert [p.lap for p in r.pit_stops] == setup.rivals[0].pit_laps


def test_rival_pace_estimate(setup):
    df, _ = generate_synthetic_race(setup, seed=3, random_events=False)
    est = estimate_rival_pace(df, "SYN_DRIVER_B", "SYN_DRIVER_A", fuel_effect_s_per_kg=0.03)
    assert est["pace_offset_s"] is not None and est["sample_count"] >= 5
    assert estimate_rival_pace(df.drop(columns=["driver"]), "A", "B")["pace_offset_s"] is None
