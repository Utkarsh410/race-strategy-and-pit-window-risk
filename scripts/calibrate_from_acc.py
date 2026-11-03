"""Calibrate RaceSim from recorded ACC session logs (Mac side).

Usage:
    python scripts/calibrate_from_acc.py                               # all logs in outputs/acc_sessions/
    python scripts/calibrate_from_acc.py --logs outputs/acc_sessions/acc_2026*.jsonl --track "Circuit Paul Ricard"
    python scripts/calibrate_from_acc.py --compound DRY_DEFAULT --fuel-start 60 --fuel-per-lap 2.9

Replays each log through the live lap tracker, keeps only real (acc_udp) sessions,
converts completed laps to RaceSim's lap CSV schema and reports, per track and
parameter, either an estimate or how much more data is needed. Writes:
    acc_calibration_laps.csv        lap data (usable with validate_model.py and dashboard Section F)
    acc_calibration_sufficiency.csv data needed vs available per parameter
    acc_calibration_report.md       full report
Nothing is applied to the configuration automatically.
The UDP feed carries no tyre compound or fuel; --compound / --fuel-* are YOUR declarations
and are recorded as assumptions in the report.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from racesim.config import load_setup  # noqa: E402
from racesim.integrations.acc.calibration import calibrate_from_acc_logs, default_log_paths, report_markdown  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", nargs="*", default=None, help="session .jsonl files (default: all in outputs/acc_sessions)")
    ap.add_argument("--track", default=None, help="only this track (exact ACC track name)")
    ap.add_argument("--compound", default=None, help="declare the tyre compound used (not in the UDP feed)")
    ap.add_argument("--fuel-start", type=float, default=None, help="declared starting fuel (kg)")
    ap.add_argument("--fuel-per-lap", type=float, default=None, help="declared consumption (kg/lap)")
    ap.add_argument("--fuel-effect", type=float, default=None, help="lap-time effect of fuel (s/kg); default from config")
    ap.add_argument("--output-dir", default=str(ROOT / "outputs"))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    paths = [Path(p) for p in args.logs] if args.logs else default_log_paths(ROOT / "outputs" / "acc_sessions")
    if not paths:
        print("No session logs found. Record a session with scripts/acc_mac_receiver.py first.")
        return 1
    if (args.fuel_start is None) != (args.fuel_per_lap is None):
        print("Declare both --fuel-start and --fuel-per-lap, or neither.")
        return 2
    rep = calibrate_from_acc_logs(paths, load_setup(), track=args.track, compound=args.compound, fuel_start_kg=args.fuel_start,
                                  fuel_per_lap_kg=args.fuel_per_lap, fuel_effect_s_per_kg=args.fuel_effect)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rep.lap_data.to_csv(out / "acc_calibration_laps.csv", index=False)
    rep.sufficiency.to_csv(out / "acc_calibration_sufficiency.csv", index=False)
    md = report_markdown(rep)
    (out / "acc_calibration_report.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"Written to {out.resolve()}: acc_calibration_laps.csv, acc_calibration_sufficiency.csv, acc_calibration_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
