"""Strategy inputs, readiness gating and the bridge to the existing optimiser."""

import pytest
from live_fixtures import Script, drive, feed

from racesim.integrations.acc.live_analysis import LiveAnalysis
from racesim.integrations.acc.live_strategy import BLOCKED, ELIGIBLE, SCENARIO_ONLY, LiveAssumptions, build_inputs, run_live_strategy
from racesim.integrations.acc.race_state import Provenance as P

FULL = dict(total_laps=40, tyre_compound="MEDIUM", tyre_age_laps=6, fuel_kg=60.0, fuel_kg_per_lap=1.75)


def snapshot(n_laps=6, session_type="RACE", lap_time=100.0, source="acc_udp", now=None, **vehicle):
    s, an = Script(session_type=session_type, source=source), LiveAnalysis()
    frames = [(0.5, s.tel(0.5, **vehicle))] + drive(s, 0.5, 0, [lap_time] * n_laps, **vehicle)
    feed(an, frames)
    return an.snapshot(frames[-1][0] if now is None else now)


def test_nothing_received_is_blocked():
    inp = build_inputs(LiveAnalysis().snapshot(0.0))
    assert inp.status == BLOCKED and any("No telemetry" in r for r in inp.reasons)


def test_udp_only_with_no_assumptions_is_blocked_and_nothing_is_zero():
    inp = build_inputs(snapshot())
    assert inp.status == BLOCKED
    for name in ("tyre_compound", "tyre_age_laps", "fuel_kg", "fuel_kg_per_lap"):
        assert inp.value(name) is None and inp.inputs[name].category == P.UNAVAILABLE
    assert any("fuel_kg" in r for r in inp.reasons)


def test_blocked_result_runs_no_model(setup):
    result = run_live_strategy(setup, build_inputs(snapshot()))
    assert result.status == BLOCKED and result.candidates == [] and result.recommended_pit_lap is None


def test_every_input_category_is_labelled():
    inp = build_inputs(snapshot(), LiveAssumptions(**FULL, pit_lane_loss_s=25.0))
    assert inp.inputs["laps_completed"].category == P.LIVE
    assert inp.inputs["observed_pace_s"].category == P.DERIVED
    assert inp.inputs["tyre_compound"].category == P.CONFIGURED
    assert inp.inputs["remaining_laps"].category == P.CONFIGURED
    assert inp.inputs["pit_lane_loss_s"].category == P.CONFIGURED
    assert inp.inputs["track_temperature_c"].category == P.LIVE
    assert inp.inputs["stationary_stop_time_s"].category == P.UNAVAILABLE


def test_measured_channel_wins_over_assumption():
    inp = build_inputs(snapshot(fuel_kg=33.0), LiveAssumptions(fuel_kg=99.0))
    assert inp.value("fuel_kg") == 33.0 and inp.inputs["fuel_kg"].category == P.LIVE


def test_too_few_clean_laps_blocks_on_pace():
    inp = build_inputs(snapshot(n_laps=2), LiveAssumptions(**FULL))
    assert inp.status == BLOCKED and inp.value("observed_pace_s") is None


def test_remaining_laps_derived_from_time_and_pace():
    snap = snapshot()
    inp = build_inputs(snap, LiveAssumptions(tyre_compound="MEDIUM", tyre_age_laps=1, fuel_kg=50.0, fuel_kg_per_lap=1.0))
    assert inp.inputs["remaining_laps"].category == P.DERIVED
    assert inp.value("remaining_laps") == 30  # (3600 - 600.5) s left at 100 s/lap, rounded up
    assert inp.inputs["remaining_laps"].uncertainty


def test_stale_data_blocks():
    snap = snapshot(now=10_000.0)
    inp = build_inputs(snap, LiveAssumptions(**FULL))
    assert inp.status == BLOCKED and any("DISCONNECTED" in r for r in inp.reasons)


def test_race_distance_already_reached_blocks():
    inp = build_inputs(snapshot(), LiveAssumptions(**{**FULL, "total_laps": 6}))
    assert inp.status == BLOCKED and any("already been reached" in r for r in inp.reasons)


def test_assumed_tyres_and_fuel_give_scenarios_without_a_recommendation(setup):
    inp = build_inputs(snapshot(), LiveAssumptions(**FULL))
    assert inp.status == SCENARIO_ONLY
    result = run_live_strategy(setup, inp)
    assert result.candidates and result.recommended_pit_lap is None
    assert result.label == P.MODEL.value
    assert "predicted_remaining_time_s" in result.candidates[0] and result.candidates[0]["delta_to_best_s"] == 0


def test_practice_cannot_be_recommendation_eligible():
    inp = build_inputs(snapshot(session_type="PRACTICE", fuel_kg=60.0, tyre_compound="MEDIUM", tyre_age_laps=6),
                       LiveAssumptions(total_laps=40, fuel_kg_per_lap=1.75, tyre_model_calibrated=True))
    assert inp.status == SCENARIO_ONLY and any("not RACE" in r for r in inp.reasons)


def test_mock_cannot_be_recommendation_eligible():
    s, an = Script(session_type="RACE", source="mock"), LiveAnalysis()
    measured = dict(tyre_compound="MEDIUM", tyre_age_laps=6, fuel_kg=60.0)
    for t, frame in [(0.5, s.tel(0.5, **measured))] + drive(s, 0.5, 0, [100.0] * 6, **measured):
        an.on_message("telemetry", frame, "new", t, mock=True)
    inp = build_inputs(an.snapshot(600.5), LiveAssumptions(total_laps=40, fuel_kg_per_lap=1.75, tyre_model_calibrated=True))
    assert inp.status == SCENARIO_ONLY and any("mock" in r for r in inp.reasons)


def test_uncalibrated_model_prevents_recommendation():
    measured = dict(tyre_compound="MEDIUM", tyre_age_laps=6)
    s, an = Script(session_type="RACE"), LiveAnalysis()
    frames = [(float(t), s.tel(float(t), fuel_kg=60.0 - t * 0.02, **measured)) for t in range(0, 3)]
    t, laps, fuel = 2.0, 0, 60.0
    for _ in range(6):
        for k in range(1, 100):
            frames.append((t + k, s.tel(t + k, laps=laps, last=100.0 if laps else None, fuel_kg=fuel - k * 0.02, **measured)))
        t, laps, fuel = t + 100.0, laps + 1, fuel - 2.0
        frames.append((t, s.tel(t, laps=laps, last=100.0, fuel_kg=fuel, **measured)))
    feed(an, frames)
    snap = an.snapshot(t)
    assert snap["fuel"]["burn_laps_used"] >= 3
    inp = build_inputs(snap, LiveAssumptions(total_laps=40))
    assert inp.status == SCENARIO_ONLY and any("calibrated" in r for r in inp.reasons)
    assert build_inputs(snap, LiveAssumptions(total_laps=40, tyre_model_calibrated=True)).status == ELIGIBLE


def test_eligible_inputs_reach_the_optimiser_and_give_a_pit_lap(setup):
    s, an = Script(session_type="RACE"), LiveAnalysis()
    measured = dict(tyre_compound="SOFT", tyre_age_laps=6)
    frames, t, laps, fuel = [(0.0, s.tel(0.0, fuel_kg=70.0, **measured))], 0.0, 0, 70.0
    for _ in range(6):
        for k in range(1, 100):
            frames.append((t + k, s.tel(t + k, laps=laps, last=100.0 if laps else None, fuel_kg=fuel - k * 0.0175, **measured)))
        t, laps, fuel = t + 100.0, laps + 1, fuel - 1.75
        frames.append((t, s.tel(t, laps=laps, last=100.0, fuel_kg=fuel, **measured)))
    feed(an, frames)
    inp = build_inputs(an.snapshot(t), LiveAssumptions(total_laps=40, tyre_model_calibrated=True))
    assert inp.status == ELIGIBLE, inp.reasons
    result = run_live_strategy(setup, inp)
    assert result.status == ELIGIBLE and result.pit_window is not None
    assert result.recommended_pit_lap == result.pit_window["lowest_predicted_time_lap"]
    assert result.recommended_pit_lap > 6


def test_model_is_anchored_to_observed_pace(setup):
    from racesim.schemas import RaceSnapshot, Stint
    from racesim.integrations.acc.live_strategy import _anchor_to_observed_pace, _prepare_setup
    from racesim.strategy_optimizer import StrategyEvaluator

    inp = build_inputs(snapshot(lap_time=123.0), LiveAssumptions(**FULL))
    snap = RaceSnapshot(current_lap=6, tyre_compound="MEDIUM", tyre_age_laps=6, fuel_kg=60.0, stops_completed=0)
    anchored, offset = _anchor_to_observed_pace(_prepare_setup(setup, inp), snap, 123.0)
    stint = Stint(start_lap=7, end_lap=7, tyre_compound="MEDIUM", starting_tyre_age_laps=6)
    assert StrategyEvaluator(anchored, snap).stint_arrays(stint)["lap_time_s"][0] == pytest.approx(123.0)
    assert offset != 0


def test_unknown_compound_blocks(setup):
    inp = build_inputs(snapshot(), LiveAssumptions(**{**FULL, "tyre_compound": "DRY"}))
    result = run_live_strategy(setup, inp)
    assert result.status == BLOCKED and any("no model" in r for r in result.reasons)


def test_configuration_conflict_blocks_instead_of_raising(setup):
    inp = build_inputs(snapshot(n_laps=3), LiveAssumptions(**{**FULL, "total_laps": 5}))
    result = run_live_strategy(setup, inp)
    assert result.status == BLOCKED and result.candidates == []


def test_synthetic_setup_is_unchanged_by_live_analysis(setup):
    before = setup.model_dump_json()
    run_live_strategy(setup, build_inputs(snapshot(), LiveAssumptions(**FULL)))
    assert setup.model_dump_json() == before
