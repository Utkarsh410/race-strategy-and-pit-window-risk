#!/usr/bin/env python
"""Replay a recorded ACC session log through the live analysis and write lap, stint and summary exports.

    python scripts/analyse_acc_session.py outputs/acc_sessions/acc_<stamp>_<id>.jsonl

Exports are written next to the log and never overwrite an earlier file.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from racesim.integrations.acc.live_analysis import replay_session_log, write_session_exports  # noqa: E402
from racesim.integrations.acc.session_logger import PHYSICS_CSV_CHANNELS, export_physics_csv  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", type=Path)
    ap.add_argument("--no-html", action="store_true", help="skip the HTML summary")
    ap.add_argument("--physics-csv", action="store_true",
                    help="also write <log>_physics.csv with shared-memory physics samples (refuses to overwrite)")
    ap.add_argument("--physics-channels", default=None,
                    help="comma-separated channels for --physics-csv (default: " + ",".join(PHYSICS_CSV_CHANNELS[:5]) + ",...)")
    args = ap.parse_args()
    if not args.log.exists():
        print(f"Log not found: {args.log}", file=sys.stderr)
        return 2
    analysis, last = replay_session_log(args.log)
    for key in analysis.sessions:
        snap = analysis.snapshot(last, key)
        print(f"{key}: {snap['pace']['laps_total']} laps, {snap['pace']['laps_clean']} clean, {len(snap['stints'])} stint(s), "
              f"median {snap['pace']['median']['value']} s (n={snap['pace']['median']['n']}), flags {snap['flags']}")
    for kind, path in write_session_exports(analysis, args.log.with_suffix(""), html=not args.no_html, now=last).items():
        print(f"{kind}: {path}")
    if args.physics_csv:
        channels = args.physics_channels.split(",") if args.physics_channels else PHYSICS_CSV_CHANNELS
        out = args.log.with_name(args.log.stem + "_physics.csv")
        try:
            path, rows = export_physics_csv(args.log, out, channels)
        except FileExistsError:
            print(f"physics csv: {out} already exists, not overwritten", file=sys.stderr)
            return 1
        print(f"physics csv: {path} ({rows} samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
