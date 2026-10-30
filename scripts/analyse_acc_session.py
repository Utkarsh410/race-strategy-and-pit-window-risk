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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", type=Path)
    ap.add_argument("--no-html", action="store_true", help="skip the HTML summary")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
