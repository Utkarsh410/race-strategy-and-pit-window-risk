"""Phase 3: rolling-horizon replanning (deterministic, n_sims=0 unless stated)."""

import numpy as np
import pandas as pd
import pytest

from racesim.config import with_overrides
from racesim.race_simulator import simulate_race
from racesim.replanning import ObservedRaceState, ReplanningEngine, RivalObservation, run_rolling_race, update_rival_estimate
from racesim.schemas import RaceEvent, Strategy

PLAN = Strategy.from_plan(60, [26], ["MEDIUM", "HARD"])


def state_from_run(setup, strategy, lap, **over):
    """Observed state at the end of ``lap`` taken from a deterministic simulate_race run."""
    r = simulate_race(setup, strategy)
    done = [p for p in strategy.planned_pit_laps if p <= lap]
    seq = [strategy.planned_compounds[0]] + [strategy.planned_compounds[i + 1] for i in range(len(done))]
    fut = [p for p in strategy.planned_pit_laps if p > lap]
    kw = dict(current_lap=lap, elapsed_s=float(r.elapsed_history[lap - 1]), tyre_compound=seq[-1],
              tyre_age_laps=int(r.tyre_age_history[lap]), tyre_wear_laps=float(r.tyre_age_history[lap]),
              fuel_kg=float(r.fuel_history[lap]), compound_sequence=tuple(seq), completed_pit_laps=tuple(done),
              plan_pit_laps=tuple(fut), plan_compounds=tuple(strategy.planned_compounds[len(done) + 1:]))
    kw.update(over)
    return ObservedRaceState(**kw), r


@pytest.fixture
def no_rival_setup(setup):
    return with_overrides(setup, replanning={"include_rivals": False, "monte_carlo_simulations": 0})


def test_shared_model_consistency_with_simulator(no_rival_setup):
    """Engine's remaining-time prediction for the current plan equals simulate_race minus elapsed."""
    st, r = state_from_run(no_rival_setup, PLAN, 20)
    dec = ReplanningEngine(no_rival_setup).decide(st)
    plan = next(a for a in dec.record.alternatives if a.label == "current plan")
    assert plan.deterministic_remaining_s == pytest.approx(r.predicted_race_time_s - st.elapsed_s, abs=1e-6)


def test_history_preserved_and_only_future_optimised(no_rival_setup):
    st, r = state_from_run(no_rival_setup, Strategy.from_plan(60, [15, 40], ["SOFT", "MEDIUM", "HARD"]), 20)
    dec = ReplanningEngine(no_rival_setup).decide(st)
    assert all(a.strategy_id.startswith("L21:") for a in dec.record.alternatives)
    assert dec.record.full_race_plan["completed_pit_laps"] == [15]
    assert dec.record.full_race_plan["compounds"][:2] == ["SOFT", "MEDIUM"]
    best = next(a for a in dec.record.alternatives if a.strategy_id == dec.record.recommended_strategy_id)
    assert dec.predicted_elapsed_s[-1] == pytest.approx(st.elapsed_s + best.deterministic_remaining_s)
    assert dec.record.simulated_time_s == st.elapsed_s


def test_rolling_race_time_accounting(setup):
    eng = ReplanningEngine(setup, n_sims=0)
    res = run_rolling_race(setup, PLAN, events=[RaceEvent(event_type="safety_car", start_lap=18, duration_laps=4)], engine=eng)
    laps = res.laps
    assert len(laps) == 60 and laps["elapsed_s"].iloc[-1] == pytest.approx(res.total_race_time_s)
    # lap times include dirty-air losses; SC bunching is a separate, explicit adjustment
    total = laps["lap_time_s"].sum() + laps["pit_time_s"].sum() + laps["safety_car_bunching_s"].sum()
    assert total == pytest.approx(res.total_race_time_s)
    assert sorted(laps.loc[laps.pit_time_s > 0, "lap"]) == res.own_pit_laps


def test_replanning_after_safety_car(setup):
    eng = ReplanningEngine(setup, n_sims=0)
    res = run_rolling_race(setup, PLAN, events=[RaceEvent(event_type="safety_car", start_lap=18, duration_laps=4)], engine=eng)
    rec = next(r for r in res.decision_log.records if r.lap == 18)
    assert rec.replanned and any("GREEN -> SC" in t for t in rec.triggers)
    assert rec.action == "BOX_THIS_LAP" and rec.recommended_pit_lap == 19
    assert 19 in res.own_pit_laps
    assert res.total_race_time_s < res.static_plan_race_time_s
    assert any("SC active" in f for f in rec.influencing_factors)


def test_replanning_after_rain(setup):
    eng = ReplanningEngine(setup, n_sims=0)
    res = run_rolling_race(setup, PLAN, events=[RaceEvent(event_type="rain", start_lap=30, duration_laps=12, severity=0.8)], engine=eng)
    assert any("rain started" in t for r in res.decision_log.records for t in r.triggers)
    assert "INTERMEDIATE" in res.own_compounds or "WET" in res.own_compounds
    assert res.total_race_time_s < res.static_plan_race_time_s


def test_replanning_after_rival_pit_stop(setup):
    eng = ReplanningEngine(setup, n_sims=0)
    rivals = (RivalObservation("RIVAL_1", 1.5, "MEDIUM", 20, 0), RivalObservation("RIVAL_2", -3.0, "MEDIUM", 2, 1, (18,)))
    st, _ = state_from_run(setup, PLAN, 20, rivals=rivals)
    eng.decide(st)
    pitted = (RivalObservation("RIVAL_1", 1.5, "HARD", 0, 1, (20,)), rivals[1])
    dec = eng.decide(ObservedRaceState(**{**st.__dict__, "rivals": pitted}))
    assert dec.replanned and any("rival RIVAL_1 pitted" in t for t in dec.record.triggers)
    # the updated rival estimate starts on the observed tyres with the prior's remaining plan dropped
    car = update_rival_estimate(setup, setup.rivals[0], pitted[0], 20, st.elapsed_s)
    assert car.strategy.planned_compounds[0] == "HARD" and car.strategy.number_of_stops == 0
    assert car.start_offset_s == pytest.approx(st.elapsed_s - 1.5)


def test_no_op_when_state_unchanged(no_rival_setup):
    eng = ReplanningEngine(no_rival_setup)
    st, _ = state_from_run(no_rival_setup, PLAN, 10)
    first = eng.decide(st)
    n_opt = eng.optimisations
    second = eng.decide(st)
    assert first.replanned and not second.replanned
    assert eng.optimisations == n_opt
    assert second.record.recommended_strategy_id == first.record.recommended_strategy_id
    assert second.record.triggers == [] and len(eng.log) == 2


def test_infeasible_remaining_strategy(no_rival_setup):
    st, _ = state_from_run(no_rival_setup, PLAN, 20, fuel_kg=5.0)
    dec = ReplanningEngine(no_rival_setup).decide(st)
    assert dec.action == "NO_FEASIBLE_STRATEGY" and dec.strategy is None and dec.record.warnings
    late, _ = state_from_run(no_rival_setup, Strategy.from_plan(60, [], ["MEDIUM"]) if False else PLAN, 58,
                             compound_sequence=("MEDIUM",), completed_pit_laps=(), plan_pit_laps=(), plan_compounds=(),
                             tyre_compound="MEDIUM")
    assert ReplanningEngine(no_rival_setup).decide(late).action == "NO_FEASIBLE_STRATEGY"  # mandatory stop impossible


def test_monte_carlo_decision_reproducible(setup):
    s2 = with_overrides(setup, replanning={"monte_carlo_simulations": 60})
    st, _ = state_from_run(s2, PLAN, 18, track_status=1, neutralisation_laps_elapsed=1)
    a = ReplanningEngine(s2).decide(st).record
    b = ReplanningEngine(s2).decide(st).record
    assert [x.expected_remaining_s for x in a.alternatives] == [x.expected_remaining_s for x in b.alternatives]
    assert a.risk_comparison["basis"] == "monte_carlo" and any("imprecise" in w for w in a.warnings)
    probs = [x.probability_faster_than_recommended for x in a.alternatives]
    assert all(0 <= p <= 1 for p in probs)


def test_risk_aversion_uses_cvar(setup):
    s2 = with_overrides(setup, replanning={"monte_carlo_simulations": 60, "include_rivals": False})
    st, _ = state_from_run(s2, PLAN, 18, track_status=1, neutralisation_laps_elapsed=1)
    rec = ReplanningEngine(s2, risk_aversion=1.0).decide(st).record
    best = min(rec.alternatives, key=lambda a: a.score_s)
    assert best.strategy_id == rec.recommended_strategy_id
    assert best.score_s == pytest.approx(best.cvar95_remaining_s)
