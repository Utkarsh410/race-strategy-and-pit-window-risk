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
    bundle = run_full_analysis(setup, n_sims=args.sims, seed=args.seed, include_rivals=not args.no_rivals,
                               run_sensitivity=not args.no_sensitivity, data_provenance=provenance, progress_callback=_progress)
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
    generate_strategy_report(bundle, out / "race_strategy_report.html")
    for wmsg in bundle.warnings:
        print(f"WARNING: {wmsg}")
    print(f"Outputs written to {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
