"""Run the full RaceSim analysis from the command line.

Usage:
    python scripts/run_simulation.py [--sims 10000] [--seed 42] [--output-dir outputs]
                                     [--no-rivals] [--no-sensitivity] [--data data/synthetic/synthetic_demo_race.csv]

Outputs (in --output-dir):
    strategy_comparison.csv   ranked deterministic strategy comparison
    pit_window.csv            cost of every candidate pit lap
    recommended_race.csv      lap-by-lap deterministic race of the recommended strategy
    monte_carlo_results.csv   per-run simulated race times (if --sims > 0)
    risk_metrics.csv          risk report per strategy
    sc_scenarios.csv          safety-car scenario comparison
    race_strategy_report.html full HTML report
    precision.csv / convergence.csv   Monte Carlo precision diagnostics (if --sims > 0)

v2 options:
    --replan                 run a deterministic rolling-horizon race with the replanning engine
      --sc-lap N --sc-laps K   safety car in that race (start lap, duration)
      --rain-lap N --rain-laps K --rain-severity W
      --replan-sims N          Monte Carlo runs per option per decision (default from config; 0 = deterministic)
      --risk-aversion L        score = mean + L * (CVaR95 - mean)
    --decision-log PATH      decision-log CSV path (default <output-dir>/decision_log.csv when --replan)
    --validate-data PATH     run held-out model validation on a lap CSV
      --data-label LABEL       data_source label to apply when the CSV has none
    --uncertainty-breakdown  spread of results by uncertainty source (extra Monte Carlo runs)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

from racesim.config import DEFAULT_RACE_CONFIG, DEFAULT_REGULATIONS_CONFIG, PROJECT_ROOT, load_setup
from racesim.data_ingestion import export_simulation_results, export_strategy_comparison, load_race_csv
from racesim.reporting import export_race_summary, generate_strategy_report
from racesim.workflow import run_full_analysis


def _progress(frac: float, msg: str) -> None:
    sys.stderr.write(f"\r  Monte Carlo {frac:6.1%}  {msg[:60]:<60}")
    if frac >= 1:
        sys.stderr.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--race-config", default=os.environ.get("RACESIM_RACE_CONFIG", str(DEFAULT_RACE_CONFIG)))
    parser.add_argument("--regulations", default=os.environ.get("RACESIM_REGULATIONS_CONFIG", str(DEFAULT_REGULATIONS_CONFIG)))
    parser.add_argument("--sims", type=int, default=None, help="Monte Carlo runs per strategy (0 = skip; default from config)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs"))
    parser.add_argument("--no-rivals", action="store_true")
    parser.add_argument("--no-sensitivity", action="store_true")
    parser.add_argument("--data", default=None, help="optional lap CSV recorded in the report's data provenance")
    parser.add_argument("--replan", action="store_true", help="run the rolling-horizon replanning demonstration")
    parser.add_argument("--sc-lap", type=int, default=None)
    parser.add_argument("--sc-laps", type=int, default=4)
    parser.add_argument("--rain-lap", type=int, default=None)
    parser.add_argument("--rain-laps", type=int, default=12)
    parser.add_argument("--rain-severity", type=float, default=0.8)
    parser.add_argument("--replan-sims", type=int, default=None)
    parser.add_argument("--risk-aversion", type=float, default=None)
    parser.add_argument("--decision-log", default=None)
    parser.add_argument("--validate-data", default=None)
    parser.add_argument("--data-label", default=None)
    parser.add_argument("--uncertainty-breakdown", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=os.environ.get("RACESIM_LOG_LEVEL", "WARNING"), format="%(levelname)s %(name)s: %(message)s")

    setup = load_setup(args.race_config, args.regulations)
    provenance = {"configuration": args.race_config, "regulations": args.regulations, "parameters": setup.race.data_label}
    if args.data:
        ing = load_race_csv(args.data)
        src = ",".join(sorted(ing.data["data_source"].astype(str).unique())) if "data_source" in ing.data.columns else "unlabelled"
        provenance.update({"lap_data": args.data, "lap_data_rows": len(ing.data), "lap_data_source_label": src,
                           "lap_data_validation_warnings": len(ing.report.warnings)})
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"RaceSim: {setup.race.race_name} ({setup.race.data_label})")
    validation_df = None
    if args.validate_data:
        from racesim.data_ingestion import load_historical_csv

        hist, vprov = load_historical_csv(args.validate_data, source_label=args.data_label)
        validation_df = hist.data
        provenance.update({f"validation_{k}": v for k, v in vprov.items()})
    from racesim.workflow import default_replan_events

    events = default_replan_events(setup, sc_lap=args.sc_lap, sc_laps=args.sc_laps, rain_lap=args.rain_lap,
                                   rain_laps=args.rain_laps, rain_severity=args.rain_severity)
    bundle = run_full_analysis(setup, n_sims=args.sims, seed=args.seed, include_rivals=not args.no_rivals,
                               run_sensitivity=not args.no_sensitivity, data_provenance=provenance, progress_callback=_progress,
                               run_replanning=args.replan, replan_events=events, replan_sims=args.replan_sims,
                               risk_aversion=args.risk_aversion, validation_data=validation_df,
                               run_breakdown=args.uncertainty_breakdown)
    best = bundle.optimisation.best
    print(f"Recommended: {best.strategy.strategy_name}  predicted {best.predicted_race_time_s:,.2f} s")
    w = bundle.pit_window
    print(f"Pit window (next stop): earliest {w.earliest_feasible_lap}, recommended {w.recommended_lap}, latest {w.latest_feasible_lap}")
    export_strategy_comparison(bundle.optimisation, out / "strategy_comparison.csv")
    w.table.to_csv(out / "pit_window.csv", index=False)
    export_race_summary(bundle.recommended_simulation, out / "recommended_race.csv")
    bundle.sc_scenarios.to_csv(out / "sc_scenarios.csv", index=False)
    if bundle.monte_carlo is not None:
        export_simulation_results(bundle.monte_carlo, out / "monte_carlo_results.csv")
        names = {sid: s.strategy_name for sid, s in bundle.monte_carlo.strategies.items()}
        risk = pd.DataFrame([{"strategy": names[sid], **r.model_dump(exclude={"warnings"})} for sid, r in bundle.risk_reports.items()])
        risk.to_csv(out / "risk_metrics.csv", index=False)
        print(risk[["strategy", "expected_race_time_s", "standard_deviation_s", "cvar_95_s",
                    "probability_of_beating_baseline", "probability_of_beating_rival"]].to_string(index=False))
    if bundle.precision is not None:
        bundle.precision.to_csv(out / "precision.csv", index=False)
    if bundle.convergence is not None:
        bundle.convergence.to_csv(out / "convergence.csv", index=False)
        print(f"Convergence (simulation precision only): {bundle.convergence_summary['checks']}")
    if bundle.uncertainty_breakdown is not None:
        bundle.uncertainty_breakdown.to_csv(out / "uncertainty_breakdown.csv", index=False)
    if bundle.rolling_race is not None:
        rr = bundle.rolling_race
        log_path = Path(args.decision_log) if args.decision_log else out / "decision_log.csv"
        p1, p2 = rr.decision_log.export_csv(log_path)
        rr.laps.to_csv(out / "replanning_laps.csv", index=False)
        print(f"Replanning: {rr.total_race_time_s:,.2f} s vs static plan {rr.static_plan_race_time_s:,.2f} s "
              f"({rr.gain_vs_static_s:+.2f} s); pit laps {rr.own_pit_laps}; decision log {p1}")
    if bundle.validation is not None:
        print(f"Validation: {bundle.validation.headline}")
        bundle.validation.metrics_table().to_csv(out / "validation_metrics.csv", index=False)
        if bundle.validation.calibration is not None:
            bundle.validation.calibration.table.to_csv(out / "tyre_calibration.csv", index=False)
    generate_strategy_report(bundle, out / "race_strategy_report.html")
    for wmsg in bundle.warnings:
        print(f"WARNING: {wmsg}")
    print(f"Outputs written to {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
