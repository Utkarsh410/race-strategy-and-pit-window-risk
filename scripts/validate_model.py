"""Held-out validation and tyre calibration on lap-level data.

Usage:
    python scripts/validate_model.py --data laps.csv [--data-label TEAM_LOGS] [--split stint|driver]
                                     [--interval 0.9] [--seed 0] [--output-dir outputs]

Evidence type is reported explicitly: SYNTHETIC data only check
self-consistency; data without a data_source label are reported as
unlabelled; mixed files are split and validated separately. When there are
too few held-out laps the result is "inconclusive" and no metrics are shown.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from racesim.config import DEFAULT_RACE_CONFIG, DEFAULT_REGULATIONS_CONFIG, PROJECT_ROOT, load_setup
from racesim.data_ingestion import load_historical_csv
from racesim.validation import run_model_validation


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--data-label", default=None, help="data_source label if the CSV has no data_source column")
    ap.add_argument("--race-config", default=os.environ.get("RACESIM_RACE_CONFIG", str(DEFAULT_RACE_CONFIG)))
    ap.add_argument("--regulations", default=os.environ.get("RACESIM_REGULATIONS_CONFIG", str(DEFAULT_REGULATIONS_CONFIG)))
    ap.add_argument("--split", choices=["stint", "driver"], default="stint")
    ap.add_argument("--interval", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs"))
    args = ap.parse_args(argv)
    logging.basicConfig(level=os.environ.get("RACESIM_LOG_LEVEL", "WARNING"), format="%(levelname)s %(name)s: %(message)s")
    setup = load_setup(args.race_config, args.regulations)
    ing, prov = load_historical_csv(args.data, source_label=args.data_label)
    res = run_model_validation(ing.data, setup, method=args.split, seed=args.seed, interval_level=args.interval, provenance=prov)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(res.headline)
    print(f"Evidence type: {res.evidence_type}; status: {res.status}; leakage check: {res.report.get('leakage_check')}")
    for w in res.report.get("warnings", []):
        print(f"WARNING: {w}")
    if not res.metrics_table().empty:
        print(res.metrics_table().to_string(index=False))
        res.metrics_table().to_csv(out / "validation_metrics.csv", index=False)
        res.report["predictions"].to_csv(out / "validation_predictions.csv", index=False)
    if res.calibration is not None:
        res.calibration.table.to_csv(out / "tyre_calibration.csv", index=False)
        print(res.calibration.table[["compound", "configured_linear_s_per_lap", "fitted_linear_s_per_lap",
                                     "standard_error_s_per_lap", "reliable"]].to_string(index=False))
    (out / "validation_summary.json").write_text(json.dumps({"headline": res.headline, "evidence_type": res.evidence_type,
                                                             "status": res.status, "provenance": prov,
                                                             "warnings": res.report.get("warnings", [])}, indent=2, default=str))
    print(f"Outputs written to {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
