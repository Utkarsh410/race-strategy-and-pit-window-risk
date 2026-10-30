# RaceSim — Race Strategy and Pit-Window Risk Simulator

RaceSim is a modular Python application for planning motorsport race strategy. It
models tyre degradation, fuel load, pit-stop cost, safety cars, weather and rival
cars, finds the fastest feasible strategy and pit window, and uses Monte Carlo
simulation to put numbers on the risk.

> **All bundled data and parameters are SYNTHETIC.** The default configuration
> describes a fictional 60-lap race at a fictional circuit under fictional rules.
> Nothing in this repository is real telemetry, timing data or a real
> championship's regulations. Results are model outputs and do not guarantee a
> race outcome.

---

## Contents

1. [Features](#features)
2. [Architecture](#architecture)
2a. [v2: predictive strategy and race adaptation](#v2-predictive-strategy-and-race-adaptation)
2b. [ACC live telemetry (optional)](#acc-live-telemetry-optional)
3. [Installation](#installation)
4. [Quick start](#quick-start)
5. [Command-line usage](#command-line-usage)
6. [Dashboard](#dashboard)
7. [CSV schema](#csv-schema)
8. [Configuration guide](#configuration-guide)
9. [Example strategy workflow](#example-strategy-workflow)
10. [Mathematical models](#mathematical-models)
11. [Testing](#testing)
12. [Limitations](#limitations)
13. [Roadmap](#roadmap)

---

## Features

| Area | What is implemented |
|---|---|
| Race configuration | Validated Pydantic models loaded from YAML; championship rules in a separate file |
| Lap-time model | Transparent additive model with one canonical implementation shared by every component |
| Tyres | Linear and thresholded-quadratic degradation, warm-up penalty, temperature sensitivity, several compounds, estimation from data with confounding checks |
| Fuel | Burn-off, lap-time effect, feasibility checks, consumption estimation, refuelling only if the rules allow it |
| Pit stops | Explicit entry/transit/exit/stationary/rejoin components; circuit-specific SC and VSC losses; stochastic sampling |
| Pit window | Earliest, lowest-cost and latest feasible lap, cost of every lap, rejection reasons, race-control scenarios |
| Strategy search | Constrained candidate generation with pruning and a candidate budget, plus lap-by-lap local refinement |
| Race simulation | Explicit vectorised race state; deterministic and stochastic modes; multi-car with dirty air and SC bunching |
| Safety car / VSC | Random or manual events, separate SC/VSC effects, reactive pit decisions evaluated (never assumed beneficial) |
| Weather | Stable, rising/cooling temperature, rain onset, wetness evolution, compound suitability (uncalibrated, can be switched off) |
| Rivals | Pace and pit-window prediction, gap accounting, rejoin position, undercut/overcut, sensitivity to the rival's response |
| Monte Carlo | Reproducible named random streams, common random numbers, batching, progress feedback, separate uncertainty sources |
| Risk metrics | Mean, median, std, P05/P50/P90/P95, CVaR95, P(beat baseline), P(beat rivals), expected position, objective comparison |
| Sensitivity | Degradation, pit loss, SC probability, lap variability; detects when the recommendation changes |
| Validation | Stint-level or driver-level held-out validation of linear vs nonlinear tyre models (MAE, RMSE, bias) |
| Outputs | Streamlit dashboard, Plotly charts, CSV export, self-contained HTML report |
| **v2** Dynamic replanning | Rolling-horizon engine: observes the race each lap, re-optimises the remaining race on relevant changes, compares expected time and CVaR, recommends box/extend |
| **v2** Decision log | Structured record of every recommendation (state, alternatives, risk, factors, version, seed, config hash), CSV/JSON/HTML export |
| **v2** Historical validation | Evidence classification (synthetic / historical / unlabelled / mixed), stint-split leakage check, prediction-interval coverage, residual diagnostics, tyre calibration |
| **v2** Track position | Gap evolution with causes, pit rejoin and traffic exposure, named rival responses, safety-car position changes |
| **v2** Uncertainty | Confidence intervals for probabilities and means, convergence diagnostics, CVaR stability, per-source breakdown incl. rival behaviour, simulation precision vs model uncertainty |

## Architecture

```
race-strategy-and-pit-window-risk/
├── app.py                     Streamlit entry point (thin UI layer)
├── config/
│   ├── default_race.yaml      synthetic race, tyres, pit, events, weather, rivals, Monte Carlo
│   └── default_regulations.yaml   fictional championship rules
├── data/{raw,processed,synthetic}/
├── src/racesim/               simulation engine (installable package)
│   ├── schemas.py             Pydantic configs + dataclass records, project-wide conventions
│   ├── config.py              YAML loading, SimulatorSetup, overrides, rule warnings
│   ├── data_ingestion.py      CSV / DataFrame loading, export
│   ├── data_validation.py     validation, flagging, cleaning, stint identification
│   ├── synthetic_data.py      deterministic SYNTHETIC lap data
│   ├── lap_time_model.py      canonical lap-time model + pace analysis
│   ├── tyre_model.py          degradation, warm-up, tyre life, estimation
│   ├── fuel_model.py          fuel burn, effect, feasibility
│   ├── pit_stop_model.py      pit-cost components and sampling
│   ├── weather_model.py       temperature / wetness scenarios
│   ├── safety_car_model.py    SC/VSC generation and decisions
│   ├── race_simulator.py      lap-by-lap engine (single run, batch, multi-car)
│   ├── strategy_generator.py  candidate generation and constraint checks
│   ├── strategy_optimizer.py  deterministic evaluation, optimisation, undercut/overcut
│   ├── pit_window.py          pit-window calculator
│   ├── rival_model.py         rival analysis
│   ├── monte_carlo.py         Monte Carlo engine
│   ├── risk_metrics.py        risk metrics and objective comparison
│   ├── sensitivity.py         sensitivity analysis
│   ├── validation.py          model calibration and held-out validation
│   ├── workflow.py            end-to-end analysis shared by CLI, report and dashboard (assemble_bundle)
│   ├── replanning.py          v2: rolling-horizon decision engine + rolling race driver
│   ├── decision_log.py        v2: structured decision records and export
│   ├── uncertainty.py         v2: confidence intervals, convergence, tail stability, source breakdown
│   ├── integrations/acc/      optional read-only ACC telemetry: protocol, parser, client, bridge, receiver, health, logger
│   ├── reporting.py           HTML report and CSV summaries
│   └── plotting.py            Plotly figures
├── dashboard/                 one module per dashboard section (A-H) + shared state/caching
├── docs/V2_AUDIT.md           repository audit made before the v2 changes
├── docs/ACC_LIVE_TELEMETRY_SETUP.md   Windows + macOS setup for live ACC telemetry
├── pages/1_ACC_Live_Telemetry.py      Streamlit page for the live ACC receiver
├── pages/2_ACC_Performance.py         Live lap, stint, pace, fuel/tyre and strategy analysis
├── scripts/                   generate_demo_data.py, run_simulation.py, replan_race.py, validate_model.py
├── tests/                     pytest suite
└── outputs/                   generated CSV and HTML (git-ignored)
```

Design rules:

* **One race model.** `lap_time_model.predict_lap_time` is the only lap-time
  implementation. The deterministic optimiser, the race simulator and the Monte
  Carlo engine all call it, and tests check that the optimiser's predicted time
  equals the simulator's to within 1e-6 s.
* **One engine.** `race_simulator.run_race` simulates *n* races for several cars at
  once, using NumPy arrays over the simulation dimension. A deterministic race is
  the special case with one row and all random inputs set to zero.
* **No global state.** Random numbers come from `numpy.random.SeedSequence`
  streams named by `(seed, batch, stream, car)`. Configuration objects are frozen.
* The dashboard only calls into `racesim`. It contains no model logic.

## v2: predictive strategy and race adaptation

v2 extends the v1 engine without replacing it. The v1 conventions are unchanged:
the pit-loss accounting (lane loss excludes stationary time, cost booked once on
the in-lap) and the single shared lap-time model. The only change to the engine
is that `race_simulator.run_race` now calls a public `advance_lap` for each lap,
so a controller can step the same engine one lap at a time. With all v2 options
left at their defaults, seed-42 results are identical to v1. `docs/V2_AUDIT.md`
records the pre-change audit.

### Dynamic replanning workflow (`racesim.replanning`)

1. **Observe.** At the end of each lap an `ObservedRaceState` captures elapsed
   time, tyre compound, age and wear-equivalent age, fuel, completed stops and
   compounds, the remaining plan, race control (and how many neutralised laps
   have run), wetness and rain, and every rival's gap, tyres and stops.
2. **Trigger.** `detect_triggers` compares the observation with the previous
   prediction. It fires on an SC/VSC start or end, rain starting or stopping, a
   wetness change, a rival pitting, gap/elapsed/fuel deviations beyond
   thresholds, a mismatch between completed stops and the plan, a planned stop
   coming close, or a periodic check. With no trigger the plan is carried forward
   unchanged (a **no-op**, still logged).
3. **Generate.** `find_optimal_strategy` runs from a `RaceSnapshot`, so only laps
   *after* the current one are optimised. The shortlist is: the deterministic
   optimum, the best plan for each stop count, the best plan for each next-pit lap
   in the horizon, the current plan, and wet-tyre plans when the track is wet (or
   plans back to slicks when on wets).
4. **Evaluate.** Every candidate goes through the shared engine (`run_race`) from
   the *exact* observed state: elapsed time as the start offset, observed fuel
   and tyre wear, and the last track status (so the restart loss applies). Known
   race control and weather are projected forward, and rivals are re-estimated
   (prior plan shifted by observed stops, pace re-estimated from recent laps
   using the same model). An optional Monte Carlo over the remaining laps (common
   random numbers, remaining SC length sampled) adds expected time, CVaR95 and
   P(ahead of all rivals).
5. **Decide.** `score = mean + risk_aversion × (CVaR95 − mean)`. The current plan
   is kept unless an alternative beats it by more than
   `plan_change_threshold_s` (hysteresis). The action is `BOX_THIS_LAP`,
   `STAY_OUT` (with the recommended pit lap), `NO_FURTHER_STOP` or
   `NO_FEASIBLE_STRATEGY`.
6. **Log.** A `DecisionRecord` is appended to the `DecisionLog` (see below).

`run_rolling_race` runs a deterministic simulated race with the engine in the
loop. It reports the race time, the static-plan time and the v1 reactive-heuristic
time on the same race. Example (default configuration, seed 42, `--sims 0`):

| Scenario | Replanning | Static plan | v1 reactive heuristic |
|---|---|---|---|
| Safety car laps 18–21 | 6,297.6 s (pit L19 under SC, L39), P1 | 6,331.7 s, P3 | 6,321.0 s |
| Rain laps 30–41 (peak 0.8) | 6,404.6 s (intermediates laps 35–50), P1 | 6,525.3 s, P2 | 6,412.9 s |
| Rain laps 30–41 + Rival A stopping 6 laps early, `--sims 200 --risk-aversion 0.3` | 6,466.7 s (full wets laps 38–45), P1 | 6,529.3 s, P2 | **6,411.3 s** |

These races use the same model as the engine, so they show that the decision
logic works. They are **not** evidence of benefit in a real race. The third row
shows the engine is not always best. It plans using its *configured* forecast of
rain duration and intensity (`weather.rain_duration_laps`, `peak_wetness`). When
the real rain differs, a simple rule that reacts to the current wetness, like the
v1 heuristic, can do better.

Configuration lives in the `replanning:` section of `default_race.yaml`: Monte
Carlo runs per option, risk aversion, horizon, trigger thresholds, rival-update
weight, overdue-stop assumption and the plan-change threshold.

### Decision logs (`racesim.decision_log`)

Each record has: decision id, lap, **simulated race clock** (own elapsed seconds)
and wall-clock UTC time, the full observed race state, action, recommended pit
lap and strategy, the full race plan (completed + future stops), whether it was
re-optimised, triggers, every alternative with deterministic/expected/CVaR
remaining time and deltas, P(faster than recommended), P(ahead of all rivals),
the expected difference versus the current plan, a risk comparison, influencing
factors, warnings, model version, random seed, configuration hash, Monte Carlo
count and risk aversion.

Influencing factors are generated only from computed values: triggers,
configured pit losses, tyre age, fuel margin, the next-best option's delta,
hysteresis and rival probabilities. `DecisionLog.export_csv` writes
`decision_log.csv` and `decision_log_alternatives.csv`; `export_json` and
`to_html` are also available.

### Historical-data validation (`racesim.validation`)

* `load_historical_csv` records provenance (SHA-256, rows, labels). It never
  invents a `data_source`: unlabelled files become `UNLABELLED` unless you pass
  `--data-label`.
* `classify_evidence` returns one of four results:
  * `synthetic_self_consistency`: checks code and parameter recovery only.
  * `historical_out_of_sample`: data labelled as non-synthetic.
  * `unlabelled`: provenance unverified, so not reported as historical evidence.
  * `mixed`: split by label and validated separately, never pooled.
* Splits hold out whole **stints** (or drivers). `check_split_leakage` confirms
  that no stint appears in both sets. If leakage were found, metrics would be
  withheld.
* Metrics: MAE, RMSE, mean bias (predicted − observed), sample counts, and
  **prediction-interval coverage**. The interval is an empirical interval from
  training residuals; its held-out coverage is *measured*, not assumed. On the
  synthetic demo it is 81% for a nominal 90%.
* `analyse_residuals` gives residuals by compound, tyre-age band and driver, the
  residual-versus-age slope, and within-stint lag-1 autocorrelation.
* `calibrate_tyre_model` compares fitted and configured degradation with 95% CIs.
  Overrides are only *recommended* for reliable estimates; the dashboard applies
  them only when you click.
* With too little data the status is `inconclusive` and no figures are shown.
  **No historical race data are bundled** with the project.

### Rival and track-position improvements (`racesim.rival_model`)

These use the simplified elapsed-time model. There is no claim of precise
overtaking prediction, and all thresholds are configurable under `traffic:` and
`race_control:`.

* `gap_evolution`: lap-by-lap gap, plus the pit stop, dirty-air or SC-bunching
  term that moved it.
* `predict_pit_rejoin`: position at rejoin, gaps to the cars ahead and behind,
  cars within `rejoin_traffic_window_s`, and dirty-air time lost over the next
  `traffic_exposure_laps` laps.
* `compare_pit_lap_rejoins`: the same, for alternative laps of one stop.
* `rival_response_scenarios`: outcome of an own pit call when the rival covers,
  pits as planned, or extends.
* `safety_car_position_changes`: order and gaps before the SC, after bunching
  and at the finish, versus a green race.
* Optional **rival-behaviour uncertainty** in Monte Carlo
  (`monte_carlo.include_rival_uncertainty`, off by default): a random rival pace
  offset plus a shift of its pit laps.

### Uncertainty and risk (`racesim.uncertainty`)

* **Simulation precision**: standard errors and 95% CIs for means, Wilson
  intervals for probabilities (P(beat baseline), P(beat rivals)), bootstrap CIs
  for CVaR95, and convergence diagnostics with explicit pass/fail criteria (mean,
  CVaR, probability, paired difference).
* **Tail stability**: runs in the 5% tail, bootstrap CI width and split-half
  agreement.
* **Source breakdown**: one Monte Carlo per source (lap noise, tyre/pace/fuel
  parameters, pit stops, race events + traffic, rival behaviour). It reports the
  spread of the absolute race time *and* of the paired difference to the
  baseline. Lap noise cancels exactly in paired comparisons (common random
  numbers), and safety cars widen absolute times far more than they change the
  decision.
* **Model uncertainty** is reported separately and explicitly: more simulations
  do not make predictions accurate for a real race.
* `RiskReport` gains optional CI fields. They are filled only when computed.

## ACC live telemetry (optional)

A **read-only** connection from Assetto Corsa Competizione on a Windows PC to
RaceSim on a Mac. It covers connecting, validating, monitoring health and logging,
and the *ACC Performance* page adds live lap, stint and pace analysis plus
on-demand strategy analysis. RaceSim works exactly as before without ACC, and the
synthetic mode is unchanged.

* **Windows bridge** (`scripts/acc_windows_bridge.py`): registers with ACC's
  UDP broadcasting API (protocol v4, connection password, empty command
  password). It receives session, timing, position and event messages on
  localhost, normalises them to RaceSim's telemetry schema 1.0 (every field
  records its native source; derived fields are marked), and forwards signed
  JSON (HMAC-SHA256) to the Mac with sequence numbers and heartbeats. Its only
  dependencies are pydantic and PyYAML (`requirements-acc-bridge.txt`).
* **Mac receiver** (`scripts/acc_mac_receiver.py`, or the dashboard page *ACC
  Live Telemetry*):
  * validates size, JSON, schema version, HMAC and payload;
  * detects duplicate, out-of-order and missing packets;
  * reports receiver, bridge, ACC and telemetry-freshness health separately
    (`WAITING_FOR_DATA`, `CONNECTED`, `STALE`, `DISCONNECTED`, `ERROR`);
  * writes one JSON Lines log per run to `outputs/acc_sessions/`.
* **Mock mode** (`scripts/send_mock_acc_telemetry.py`): deterministic data,
  always labelled MOCK, for testing without the game.
* **Protocol sources:** cross-checked against
  [bradfier/acbc](https://github.com/bradfier/acbc) (AGPL-3.0; facts only, no
  code copied) and
  [geniusdex/broawp](https://pkg.go.dev/github.com/geniusdex/broawp/accbroadcast)
  (BSD-3-Clause). The reference is the C# SDK shipped in the game's
  `sdk\broadcasting` folder.
* **Limits:** the UDP broadcasting API has no physics channels (fuel, tyres,
  pedals), so those stay UNAVAILABLE until a Windows shared-memory reader exists,
  and pit recommendations stay disabled. Real-session evidence so far is practice
  at one track; the tests use deterministic synthetic fixtures.

Setup, the network checklist and the first-session test are in
[docs/ACC_LIVE_TELEMETRY_SETUP.md](docs/ACC_LIVE_TELEMETRY_SETUP.md).

## Installation

Requires Python **3.11 or newer** (developed and tested with 3.12). No internet
access is needed after the dependencies are installed, and no API keys are used.

```bash
python -m venv .venv
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Windows (PowerShell or cmd):

```bash
.venv\Scripts\activate
```

Install the package in editable mode with the test dependencies:

```bash
pip install -e ".[dev]"
```

`pip install -r requirements.txt` installs the same dependencies without
installing the package itself. In that case `app.py` adds `src/` to the import
path itself, but the scripts and tests need the editable install.

## Quick start

```bash
python scripts/generate_demo_data.py
```

```bash
streamlit run app.py
```

```bash
python scripts/run_simulation.py --sims 10000
```

```bash
pytest
```

## Command-line usage

### Generate synthetic demo data

```bash
python scripts/generate_demo_data.py --output data/synthetic/synthetic_demo_race.csv --seed 42
```

Options: `--no-random-events` (no random SC/VSC), `--inject-anomalies` (adds one
duplicate row and one missing lap time to demonstrate validation), and
`--race-config` / `--regulations` for other YAML files. The CSV comes with a
`.meta.json` file describing the seed, the strategies used, the SC/VSC laps and
any injected anomalies. Every row has `data_source = SYNTHETIC`.

### Run the full analysis

```bash
python scripts/run_simulation.py --sims 10000 --seed 42 --output-dir outputs --data data/synthetic/synthetic_demo_race.csv
```

The command writes the following to `outputs/`:

| File | Content |
|---|---|
| `strategy_comparison.csv` | Every feasible candidate, ranked by deterministic race time |
| `pit_window.csv` | Cost, feasibility and rejection reasons for every pit lap |
| `recommended_race.csv` | Lap-by-lap deterministic race for the recommended strategy |
| `sc_scenarios.csv` | Shortlisted strategies with a safety car starting on different laps |
| `monte_carlo_results.csv` | Race time of every simulated run for every strategy (`--sims 0` skips this) |
| `risk_metrics.csv` | Risk report for each strategy |
| `race_strategy_report.html` | Self-contained HTML report that opens offline |

v2 options for `run_simulation.py`:

| Option | Effect |
|---|---|
| `--replan` | run the rolling-horizon replanning demonstration (adds report section 11, `decision_log*.csv`, `replanning_laps.csv`) |
| `--sc-lap N --sc-laps K` | safety car in that race |
| `--rain-lap N --rain-laps K --rain-severity W` | rain in that race |
| `--replan-sims N` | Monte Carlo runs per option per decision (0 = deterministic) |
| `--risk-aversion L` | weight on CVaR95 in the decision score |
| `--decision-log PATH` | where to write the decision log CSV |
| `--validate-data PATH [--data-label LABEL]` | held-out validation and calibration (report section 12, `validation_metrics.csv`, `tyre_calibration.csv`) |
| `--uncertainty-breakdown` | per-source spread (`uncertainty_breakdown.csv`) |

`precision.csv` and `convergence.csv` are always written when Monte Carlo runs.

Dedicated scripts:

```bash
python scripts/replan_race.py --sc-lap 18 --sims 0
```

```bash
python scripts/replan_race.py --rain-lap 30 --rain-laps 12 --rival-shift RIVAL_1:-6 --sims 200 --risk-aversion 0.3
```

```bash
python scripts/validate_model.py --data data/synthetic/synthetic_demo_race.csv
```

```bash
python scripts/run_simulation.py --sims 10000 --replan --sc-lap 18 --validate-data data/synthetic/synthetic_demo_race.csv --uncertainty-breakdown
```

Use `--no-rivals` and `--no-sensitivity` to make the run faster. On the
development laptop, 10,000 runs × 5 strategies × 3 cars plus the sensitivity
sweeps finished in about 8 s.

Environment variables (see `.env.example`): `RACESIM_RACE_CONFIG`,
`RACESIM_REGULATIONS_CONFIG`, `RACESIM_LOG_LEVEL`.

## Dashboard

```bash
streamlit run app.py
```

The dashboard opens at <http://localhost:8501>. It has eight sections (A–F from
v1, G–H added in v2), and they all share one model. A model assumption changed in any section applies to every
analysis.

| Section | Contents |
|---|---|
| **A. Race overview** | Race distance, baseline pace, pit loss and stationary-time controls; headline KPIs (recommended strategy, stops, predicted time vs baseline, recommended pit lap); current race state (lap, compound, tyre age, stops); configuration and data-quality warnings; technical assumptions |
| **B. Strategy planner** | Maximum stops, stint limits, mandatory stops, compound rules, permitted compounds, tyre degradation parameters per compound, fuel and weather assumptions; ranked strategy table with the recommended strategy highlighted; delta to baseline; explanation; custom strategy evaluator; CSV download |
| **C. Pit window** | Earliest, recommended and latest lap; cost-by-lap curve with the whole window shaded and the recommended lap marked separately; optional SC/VSC scenario curve; model tyre curves overlaid with observed fuel-corrected laps; safety-car scenario comparison |
| **D. Monte Carlo risk** | Simulation count, lap variability, degradation uncertainty, pit variability, SC/VSC probability, switches for each uncertainty source, rivals on or off; **Run** button with a progress bar; histogram, CDF, paired difference to baseline, percentile and CVaR metrics, objective comparison, low-count warnings, CSV and HTML report downloads. The random seed is set in the sidebar. |
| **E. Rival strategy** | Whole-race gaps and predicted positions against the configured rivals; undercut/overcut evaluator (gap, compounds, tyre ages, rival pace, pit laps); predicted rival pit window; rejoin position; sensitivity to the rival's response lap; the simplifying assumptions are shown |
| **F. Telemetry & validation** | CSV upload or synthetic data; data-quality report with flagged records; lap-time progression with SC/VSC shading; sector comparison; stint pace; degradation estimates against the configured model; pace trend; fuel consumption; held-out validation (MAE, RMSE, bias, predicted vs observed, residuals). **v2:** evidence type with an explicit headline, leakage check, prediction-interval coverage, residual diagnostics, calibration table with an *Apply* button |
| **G. Dynamic replanning** (v2) | Initial plan, safety car, rain and an off-plan rival; Monte Carlo runs per decision and risk aversion; **Run** button; replanning vs static vs v1 heuristic; timeline with pits, neutralisations and re-optimisation points; decision log (filterable) with alternatives, influencing factors and observed state for each decision; CSV downloads |
| **H. Sensitivity & uncertainty** (v2) | Simulation precision table, convergence chart and criteria, CVaR stability (from the Section D run); uncertainty breakdown by source (**Run**); sensitivity sweeps and ranking (**Run**); model-uncertainty statements |

v2 also adds to existing sections: D shows Wilson CIs, the Monte Carlo standard
error and a rival-behaviour switch; E shows gap evolution, rejoin and traffic
exposure for the first stop, SC position changes and named rival responses.

Behaviour notes:

* Monte Carlo runs only when you press **Run**, never on every rerun. If the
  settings change after a run, the old results stay on screen with a warning.
* Cached computations are deterministic for their inputs, and the cache key
  includes the configuration and the seed.
* Uploaded data and calculated results are kept separate.
* Analyses that need missing columns (for example sectors or tyre data) are
  switched off with an explanation instead of failing.
* The HTML report (Section D → *Build HTML report*) goes through the shared
  `workflow.assemble_bundle`. It contains Monte Carlo, sensitivity, uncertainty
  breakdown, replanning and validation **only if they were run in the session
  with the current settings**. Everything else is labelled *not run*.

## CSV schema

Only `lap` and `lap_time_s` are mandatory. Units are fixed by the column names and
are **never inferred**. A column called `lap_time` or `lap_time_ms` is rejected
with a message asking you to convert it and rename it to `lap_time_s`.

| Column | Type / unit | Notes |
|---|---|---|
| `lap` | integer, 1-indexed | **mandatory** |
| `lap_time_s` | seconds | **mandatory**; invalid values are flagged, never replaced |
| `driver` | string | without it, all rows are treated as one driver (warning shown) |
| `sector1_s`, `sector2_s`, `sector3_s` | seconds | enable sector analysis |
| `tyre_compound` | string (e.g. `SOFT`) | case-insensitive |
| `tyre_age_laps` | integer laps | laps completed on the set when the lap starts |
| `fuel_kg` | kilograms | on board at lap start; enables fuel correction |
| `track_status` | `GREEN`, `SC`, `VSC`, `YELLOW`, `RED` | non-green laps are excluded from pace fits |
| `pit_in`, `pit_out` | boolean (`true/false`, `1/0`, `yes/no`) | in-laps and out-laps are excluded from pace fits |
| `traffic_loss_s` | seconds | laps with more than 0.3 s are excluded from pace fits |
| `air_temp_c`, `track_temp_c` | °C | informational |

Validation results include missing values, invalid lap times (missing,
non-numeric, ≤ 0, non-finite, or outside 0.5–3× the median), duplicated
`(driver, lap)` keys, and robust MAD outliers with a likely cause (pit in/out,
SC/VSC, traffic, lap 1, unexplained). Outliers are **flagged, not deleted**. The
cleaned dataset removes only invalid and duplicate rows. Unexplained outliers are
excluded from model fitting but stay in the data.

## Configuration guide

### Units and conventions

* Time is in seconds, fuel in kilograms, laps are 1-indexed integers, and
  temperatures are in °C.
* **Stint intervals are inclusive.** `start_lap=1, end_lap=20` covers 20 laps.
* **A pit stop on lap L:** L is the in-lap and carries the whole pit cost, booked
  once. The out-lap L+1 starts on the new tyres at age 0.
* **Tyre age** is the number of laps completed on the set when the lap starts.
* **Pit-loss convention:** `pit_lane_loss_s` is the loss for driving through the
  pit lane (entry + transit + exit) compared with staying on track, **excluding**
  stationary time. Total green-flag pit cost =
  `pit_lane_loss_s + stationary_stop_time_s (+ rejoin_loss_s)`. The out-lap
  warm-up penalty belongs to the tyre model and is never part of the pit cost.
* `baseline_lap_time_s` is defined for an empty tank, a fresh reference tyre after
  warm-up, a green flag and `reference_track_temperature_c`.

### `config/default_race.yaml`

| Section | Key parameters |
|---|---|
| `race` | distance, baseline pace and variability, pit loss and stationary time, fuel (start, consumption, effect), SC/VSC probability per lap, temperatures, seed, simulation count, planning stint limits, minimum laps remaining after a stop, candidate budget, maximum stops, standing-start loss, track evolution |
| `compounds` | per compound: reference lap time, linear and nonlinear degradation, threshold, warm-up penalty and decay, degradation uncertainty, temperature sensitivity, maximum recommended age, type (`dry`/`intermediate`/`wet`), model (`linear`/`nonlinear`) |
| `pit_stop` | optional entry/transit/exit split (must add up to `pit_lane_loss_s`), stationary variability, slow-stop probability, rejoin loss, **SC and VSC pit-lane losses** (circuit-specific, not a percentage) |
| `race_control` | SC/VSC durations, lap-time factors, tyre-wear and fuel factors, restart losses, event timing rules, bunching gap, reactive-decision window |
| `weather` | scenario, temperature trend, rain onset/probability/duration, wetness dynamics, compound suitability, `wet_weather_enabled` |
| `traffic` | random traffic probability and size, dirty-air threshold and loss, pace needed to overtake |
| `rivals` | pace offset, degradation multiplier, starting gap, pit laps, compounds |
| `monte_carlo` | default and dashboard run counts, batch size, uncertainty-source switches, pace and fuel parameter uncertainty |
| `baseline_strategy` | reference plan used for "probability of beating the baseline" |

The default values are plausible but **illustrative**: 100 s baseline, 60 laps,
20 s pit-lane loss, 2.5 s stationary time, 0.04 s/lap MEDIUM degradation, seed 42.

### `config/default_regulations.yaml`

Configurable rules: minimum/maximum stint length, mandatory and maximum stops,
minimum distinct dry compounds (waived after wet running, if configured),
permitted compounds, refuelling (with capacity, rate and whether it runs at the
same time as the tyre change), driver-time rules (`driver_rules`, minimum and
maximum laps per driver).

A rule set to `null` is **not configured**. The constraint is switched off and a
warning is shown. Refuelling left as `null` is treated as **not permitted**. No
real series' rules are hard-coded. Create one YAML file per championship and pass
it with `--regulations`.

The effective stint limit is the stricter of the planning limit in the race file
and the regulation limit.

## Example strategy workflow

1. `python scripts/generate_demo_data.py` creates a synthetic race.
2. In the dashboard, **F. Telemetry & validation** shows the data-quality warnings
   and compares the estimated degradation with the configured values. Adjust the
   tyre parameters in **B** if the evidence supports it.
3. **B. Strategy planner** ranks every feasible strategy. With the defaults the
   deterministic optimum is a 2-stop MEDIUM–SOFT–SOFT (laps 25 and 43), about 9.2 s
   faster than the 1-stop MEDIUM–HARD baseline. The explanation breaks the
   difference down into pit-cost, tyre and compound-pace parts.
4. **C. Pit window** shows the feasible window for the first stop (laps 8–44) and
   the minimum-cost lap (25). The midpoint is not used. Add an SC scenario to see
   how the best lap moves.
5. **D. Monte Carlo risk** runs 1,000–10,000 races per strategy. With the default
   settings the recommended plan beat the baseline in about 92 % of paired runs.
   The objective table shows that the lowest-variability strategy is a different
   plan.
6. **E. Rival strategy** tests whether an undercut on a rival works for a given gap.
7. Download the CSVs and the HTML report.

(The numbers above come from the shipped synthetic configuration with seed 42.
They will change if you change any parameter.)

## Mathematical models

### Lap time

```
lap_time = ref[c] + pace_offset
         + m · [ k_lin[c] · w + k_nl[c] · max(0, w − w_thr[c])² ]     tyre degradation
         + P_warm[c] · exp(−a / τ[c])                                   out-lap warm-up (stints after a stop)
         + e_fuel · fuel_kg                                             fuel
         + traffic + weather(wetness, type[c])
         − min(evo · (lap − 1), evo_cap) + 1[lap = 1] · standing_start
         + driver_offset + ε,                ε ~ N(0, σ_lap) (0 in deterministic mode)
```

`a` is the tyre age and `w` is the wear-equivalent age, which equals `a` under
green and grows by a fraction of a lap per SC/VSC lap.
`m = (1 + s_T · (T_track − T_ref)) × parameter-uncertainty multiplier × car multiplier`.
Lap times are floored at `0.85 × baseline` so no lap can be physically impossible.
Under an SC or VSC the lap time is `max(green lap, baseline × factor)`, and the
first green lap afterwards adds a restart loss.

### Fuel

`fuel(lap) = max(0, fuel_start − consumption × (lap − 1))`. The effect is
`e_fuel × fuel`. With no refuelling a strategy is infeasible unless
`fuel_start ≥ consumption × laps`. With refuelling each stop adds enough fuel for
the next stint, up to the tank capacity.

### Pit stop

`cost = lane_loss(status) + stationary + rejoin`. The lane loss comes from the
green, SC or VSC input for that lap's status. Stochastic mode uses a truncated
normal lane loss, a truncated normal stationary time and an occasional slow stop.

### Optimisation

Candidates are enumerated (stop count × compound sequence × pit laps) under the
hard constraints, using a coarser lap grid when the candidate budget would
otherwise be exceeded. Each candidate is evaluated with cached cumulative stint
tables, and the best candidate for each compound sequence is refined one lap at a
time by coordinate descent. The objective is minimum expected race time, and it
can be swapped for position- or points-based objectives later.

### Neutralisation decision

Under an SC or VSC the model compares *stay out to the planned lap* with *pit now*
over the laps up to the end of the next stint, using the team's deterministic
model and the circuit's neutralised pit loss. It pits only if the expected gain is
positive and the stint limits are still met.

### Monte Carlo

For each run the model samples: degradation multipliers per compound (lognormal,
mean 1), race-pace bias and fuel-consumption multiplier (parameter uncertainty);
lap noise (lap-to-lap variability); SC/VSC/rain events, pit execution and traffic
(event uncertainty). Reactive calls add strategy uncertainty. Every strategy sees
the same scenarios (common random numbers), so paired comparisons have much less
noise. `decompose_uncertainty` reruns with one source at a time.

### Risk metrics

For race time, lower is better. CVaR95 is the mean of the
`k = ceil(0.05 n)` slowest runs. Win probabilities count ties (|Δ| ≤ 1e-6 s) as
½. P(beat rival) is reported only when rival outcomes were simulated. Otherwise it
is `None`. A warning appears when fewer than 50 runs fall in the 5 % tail.

### Validation

Representative laps are split by **stint** or **driver**. A bounded least-squares
model is fitted:
`lap = intercept[c] + driver[d] + e_fuel·fuel + k_lin[c]·age (+ k_nl[c]·max(0, age − thr[c])²)`.
The fuel effect stays fixed because age and fuel are collinear within a stint.
The model is scored on the held-out stints with MAE, RMSE and bias. If there are
too few held-out laps the report says *inconclusive*.

## Testing

```bash
pytest
```

With coverage:

```bash
pytest --cov=racesim --cov-report=term
```

The suite has 312 tests: the 110 from v1, unchanged, 46 for v2 and 156 for the
ACC integration (`tests/integrations/acc/`). The v1 tests It covers configuration validation, data loading and
validation, tyre/fuel/pit models, the pit window (with a hand-calculated optimum
that is not the midpoint), strategy generation limits, the optimiser
(known-optimum case, agreement with the simulator), race-simulation time
accounting, SC/VSC events and decisions, weather reactions, rival analysis, Monte
Carlo reproducibility and batching, hand-calculated risk metrics and CVaR, plots
with empty inputs, and an end-to-end integration test (synthetic data → optimiser
→ simulator → Monte Carlo → risk report → CSV → HTML).

The v2 tests cover:

* **Replanning** (`test_replanning.py`): after a safety car, rain and a rival pit
  stop; no-op on an unchanged state; infeasible remaining strategies;
  preservation of elapsed time and completed stops; consistency of the shared
  model with `simulate_race`; reproducible Monte Carlo decisions; the risk-aversion
  score.
* **Decision log** (`test_decision_log.py`): fields, CSV/JSON/HTML export, and
  factors that reference computed alternatives.
* **Historical validation** (`test_historical_validation.py`): evidence
  classification, no stint leakage, metrics and coverage, inconclusive results,
  incomplete fields, keeping synthetic and historical separate, calibration,
  provenance.
* **Rivals** (`test_rival_v2.py`): gap accounting against independent runs, gap
  reconciliation with interactions, hand-calculated rejoin, traffic exposure,
  responses, SC position changes.
* **Uncertainty** (`test_uncertainty.py`): Wilson values, CIs in risk reports,
  convergence, tail stability, common-random-number noise reduction, source
  separation, rival-behaviour reproducibility.
* **Reports** (`test_reporting_v2.py`): reports contain exactly the analyses run,
  well-formed HTML, valid CSV fields, equivalent assumptions across strategies.
* **Dashboard** (`test_dashboard.py`): headless `AppTest` covering all 8 tabs,
  Monte Carlo + sensitivity + replanning → report contents, and the calibration
  apply button. These take about 20 s. Skip them with
  `pytest --deselect tests/test_dashboard.py`.

## Limitations

* **Synthetic, uncalibrated parameters.** The defaults show how the method works
  and are not fitted to any real car, tyre or circuit. Validation on synthetic data
  only checks parameter recovery and the code, not real-world accuracy.
* The lap-time model is additive. Tyre temperature dynamics, graining, cliffs past
  the quadratic term, and track-limits or driver-behaviour effects are not modelled.
* Car interactions use **aggregate elapsed time only**: a dirty-air threshold and a
  pace-based overtaking rule. There is no track-position, DRS or blue-flag
  modelling, and position predictions are approximate.
* The SC/VSC hazard is constant per lap. There are no red flags, lapped-car
  procedures or pit-lane closures. Weather is a simple wetness index with
  illustrative compound crossovers.
* The v1 reactive rules cover pitting under SC/VSC and switching tyres for rain.
  The v2 replanning engine re-optimises the remaining race on any trigger,
  including rival stops. It is demonstrated only on deterministic simulated races
  that use the engine's own model, so its benefit in a real race is unvalidated.
* Replanning assumptions: the remaining SC/VSC length (sampled from the configured
  range in Monte Carlo, the mean in deterministic mode) and the remaining rain
  duration are taken from configuration. Rival plans are priors adjusted by
  observed stops. In deterministic mode, near-equal options can still switch on
  periodic checks when dirty-air effects differ by more than the hysteresis
  threshold.
* The engine is only as good as its forecasts of rain duration/intensity and SC
  length. In one tested rain case it finished 55 s behind the v1 reactive
  heuristic, although it still beat the static plan. Plan-change hysteresis
  (default 0.5 s) avoids flip-flopping but can delay a switch by a lap; in the
  deterministic rain example that cost about 10 s.
* Candidate plans come from the green-flag optimiser (plus wet-tyre plans). Plans
  that are only good under the projected conditions but poor under green flag may
  be missed.
* Validation: no historical data are bundled. Prediction intervals come from
  in-sample residuals, so held-out coverage can fall short of nominal (as measured
  on the demo data). Driver offsets for drivers missing from the training data are
  assumed to be 0.
* Confidence intervals and convergence checks describe Monte Carlo precision only.
  The paired-difference intervals are tight because of common random numbers;
  that does not reduce model error.
* The optimiser minimises expected race time. Other objectives are compared from
  Monte Carlo results but are not optimised directly.
* Monte Carlo results are reproducible for the same seed, run count, **batch size**
  and configuration.

## Roadmap

The architecture leaves room for these. None of them are implemented, and there
are no placeholder integrations:

* live timing ingestion and historical Formula 1 timing (through a data adapter
  that outputs the CSV schema above)
* GT3 / endurance configurations: driver-change optimisation, stint and fuel
  constraints (driver rules and refuelling already exist in the schema)
* real telemetry and MDF/MF4 ingestion; video synchronisation
* Bayesian parameter estimation for degradation and pace
* dynamic-programming optimisation over the existing stint-cost tables
* richer traffic and opponent modelling with track position
* position- and championship-points objectives (pluggable `objective` in the optimiser)
* richer decision logs (race-engineer annotations, decision outcomes compared with what happened)
* closed-loop Monte Carlo with the replanning engine in every simulated race (currently deterministic examples only)

## License

MIT — see [LICENSE](LICENSE).
