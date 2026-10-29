# RaceSim v2 — repository audit (before changes)

Audit date: 2026-10-29. Baseline commit `f9e2ae6` on `main`; v2 work is on branch `v2-predictive-strategy`.

## Baseline test results

`pytest`: **110 passed** (Python 3.12.14, `.venv`). Reference CLI output (`run_simulation.py --sims 10000 --seed 42`):
recommended 2-stop M-S-S (L25, 43) at 6,166.61 s; pit window 8 / 25 / 44; P(beat baseline) 0.9233; P(beat rivals) 0.6265.
These numbers were re-checked after every engine change in v2 to confirm the existing behaviour is unchanged.

## Existing architecture (as found)

| Layer | Modules | Key interfaces |
|---|---|---|
| Configuration | `schemas.py`, `config.py` | `SimulatorSetup` (frozen Pydantic), `load_setup`, `with_overrides`, `regulation_warnings` |
| Canonical model | `lap_time_model.py` (`LapModelContext`, `predict_lap_time`) + `tyre_model.py`, `fuel_model.py`, `pit_stop_model.py`, `weather_model.py` | Used by simulator, optimiser and Monte Carlo — the single shared lap-time model |
| Engine | `race_simulator.py` | `CarSpec`, `Scenario`, `RaceState` (vectorised), `apply_race_event`, `simulate_lap`, `update_race_state`, `run_race`, `simulate_race` |
| Decisions | `safety_car_model.py` (`update_strategy_after_event`), `strategy_generator.py`, `strategy_optimizer.py` (`StrategyEvaluator`, `find_optimal_strategy`), `pit_window.py` | `RaceSnapshot` supports mid-race starts (start lap, tyre age, fuel, stops, compounds used) |
| Rivals | `rival_model.py` | `evaluate_undercut/overcut`, `predict_rival_pit_window`, `predict_rejoin_position`; car interactions in `race_simulator` (dirty air, SC bunching) |
| Risk | `monte_carlo.py`, `risk_metrics.py`, `sensitivity.py` | `build_scenario`, `run_monte_carlo` (CRN), `RiskReport`, `compare_objectives`, `calculate_strategy_sensitivity` |
| Validation | `data_ingestion.py`, `data_validation.py`, `validation.py` | stint-split validation of linear vs nonlinear tyre model (dict report) |
| Orchestration | `workflow.py` | `run_full_analysis` → `AnalysisBundle`, shared by CLI, report, dashboard |
| Output | `reporting.py`, `plotting.py`, `scripts/`, `app.py` + `dashboard/` (6 tabs) | |

## Integration risks identified

1. `run_race` held the only lap loop; a rolling-horizon controller needs to step it lap by lap → extract the loop body
   into a public `advance_lap` (behaviour-preserving) instead of writing a second loop.
2. `StrategyEvaluator` assumes tyre wear = tyre age and a green remaining race. After a safety car wear < age, so
   candidate *ranking* may use the evaluator but final replanning comparisons must use `run_race` with the observed state.
3. `run_race` derives starting fuel from the strategy; observed fuel / wear / last track status must be injectable
   (`CarSpec` optional fields, defaults leave fresh races unchanged).
4. Weather scenarios always start dry on lap 1; replanning during rain needs a projection from the *observed* wetness.
5. `RiskReport` is `extra="forbid"`; new uncertainty fields must be added as optional fields.
6. Dashboard report built in Section D labelled sensitivity "not run"; v2 must include real results when run.
7. `MonteCarloResult` CSV and `risk_metrics.csv` columns are consumed by users; only additive changes.
8. `resolve_flags` rejects unknown flags; adding a rival-behaviour flag must default to off to keep v1 results identical.

## Files requiring modification

`race_simulator.py` (advance_lap, CarSpec observed state, rival pit-lap shift), `monte_carlo.py` (rival-behaviour
uncertainty, convergence), `risk_metrics.py` (confidence intervals, stability), `schemas.py` (optional fields, new
config), `validation.py` (historical workflow), `weather_model.py` (projection from observed state), `rival_model.py`,
`sensitivity.py`, `workflow.py`, `reporting.py`, `plotting.py`, dashboard modules, scripts, README.
New: `replanning.py`, `decision_log.py`, `uncertainty.py`, `scripts/replan_race.py`, `scripts/validate_model.py`.
