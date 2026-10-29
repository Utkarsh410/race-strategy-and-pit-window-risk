import pytest

from racesim.pit_window import (
    calculate_earliest_feasible_pit_lap, calculate_latest_feasible_pit_lap, calculate_pit_window, rank_pit_laps,
    validate_pit_window,
)
from racesim.schemas import RaceEvent, RaceSnapshot


def test_window_respects_constraints(simple_setup):
    w = calculate_pit_window(simple_setup, n_stops=[1])
    assert w.earliest_feasible_lap == 3   # minimum stint 3
    assert w.latest_feasible_lap == 17    # final stint must be >= 3 laps
    t = w.table.set_index("pit_lap")
    assert not t.loc[2, "feasible"] and "minimum" in t.loc[2, "rejection_reasons"]
    assert not t.loc[18, "feasible"]
    assert validate_pit_window(w) == []


def test_recommended_by_objective_not_midpoint(simple_setup):
    w = calculate_pit_window(simple_setup, n_stops=[1])
    mid = (w.earliest_feasible_lap + w.latest_feasible_lap) / 2
    assert w.recommended_lap == 17 != mid
    assert w.recommended_race_time_s == pytest.approx(17 * 99 + 3 * 100 + 22.5)
    ranked = rank_pit_laps(w.table)
    assert ranked.iloc[0]["pit_lap"] == 17
    assert ranked["predicted_race_time_s"].is_monotonic_increasing


def test_infeasible_window_identified(make_simple_setup):
    s = make_simple_setup(minimum_stint_laps=11, regs={"minimum_stint_laps": 11})
    w = calculate_pit_window(s, n_stops=[1])
    assert w.recommended_lap is None and w.warnings
    assert any("empty" in p for p in validate_pit_window(w))


def test_snapshot_window(simple_setup):
    snap = RaceSnapshot(current_lap=5, tyre_compound="FAST", tyre_age_laps=5)
    assert calculate_earliest_feasible_pit_lap(simple_setup, snap) == 6
    assert calculate_latest_feasible_pit_lap(simple_setup, snap) <= 19
    w = calculate_pit_window(simple_setup, snap, n_stops=[1])
    assert w.table["pit_lap"].min() == 6 and w.recommended_lap == 17


def test_scenario_costs(simple_setup):
    ev = [RaceEvent(event_type="safety_car", start_lap=8, duration_laps=3)]
    w = calculate_pit_window(simple_setup, n_stops=[1], events=ev)
    assert "scenario_race_time_s" in w.table.columns
