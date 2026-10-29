"""Rolling-horizon replanning demonstration on a deterministic simulated race.

Usage:
    python scripts/replan_race.py [--sc-lap 18 --sc-laps 4] [--rain-lap 30 --rain-laps 12]
                                  [--rival-shift RIVAL_1:-8] [--sims 0] [--risk-aversion 0.0]
                                  [--strategy "26:MEDIUM,HARD"] [--output-dir outputs]

The engine observes the race after every lap and re-optimises when something
relevant changes. Writes decision_log.csv, decision_log_alternatives.csv,
decision_log.json and replanning_laps.csv. The simulated race uses the same
model as the engine, so this demonstrates the decision logic; it is not
evidence of real-race benefit.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from racesim.config import DEFAULT_RACE_CONFIG, DEFAULT_REGULATIONS_CONFIG, PROJECT_ROOT, load_setup
from racesim.schemas import Strategy
from racesim.workflow import default_replan_events, run_replanning_demo


def _strategy(text: str, total_laps: int) -> Strategy:
    laps_part, comps_part = text.split(":")
    laps = [int(x) for x in laps_part.split(",") if x.strip()]
    return Strategy.from_plan(total_laps, laps, [c.strip() for c in comps_part.split(",")])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--race-config", default=os.environ.get("RACESIM_RACE_CONFIG", str(DEFAULT_RACE_CONFIG)))
    ap.add_argument("--regulations", default=os.environ.get("RACESIM_REGULATIONS_CONFIG", str(DEFAULT_REGULATIONS_CONFIG)))
    ap.add_argument("--strategy", default=None, help='initial plan "pit,laps:COMPOUND,..." (default: baseline)')
    ap.add_argument("--sc-lap", type=int, default=None)
    ap.add_argument("--sc-laps", type=int, default=4)
    ap.add_argument("--rain-lap", type=int, default=None)
    ap.add_argument("--rain-laps", type=int, default=12)
    ap.add_argument("--rain-severity", type=float, default=0.8)
    ap.add_argument("--rival-shift", action="append", default=[], help="RIVAL_ID:LAPS shifts that rival's real stops (unexpected behaviour)")
    ap.add_argument("--sims", type=int, default=None, help="Monte Carlo runs per option per decision (0 = deterministic)")
    ap.add_argument("--risk-aversion", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs"))
    args = ap.parse_args(argv)
    logging.basicConfig(level=os.environ.get("RACESIM_LOG_LEVEL", "WARNING"), format="%(levelname)s %(name)s: %(message)s")
    setup = load_setup(args.race_config, args.regulations)
    shifts = dict((k, int(v)) for k, v in (x.split(":") for x in args.rival_shift))
    truth = [r.model_copy(update={"pit_laps": [p + shifts.get(r.rival_id, 0) for p in r.pit_laps]}) for r in setup.rivals]
    events = default_replan_events(setup, sc_lap=args.sc_lap, sc_laps=args.sc_laps, rain_lap=args.rain_lap,
                                   rain_laps=args.rain_laps, rain_severity=args.rain_severity)
    strategy = _strategy(args.strategy, setup.race.total_laps) if args.strategy else None
    rr = run_replanning_demo(setup, strategy=strategy, events=events, n_sims=args.sims, risk_aversion=args.risk_aversion,
                             seed=args.seed, truth_rivals=truth)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    p1, p2 = rr.decision_log.export_csv(out / "decision_log.csv")
    rr.decision_log.export_json(out / "decision_log.json")
    rr.laps.to_csv(out / "replanning_laps.csv", index=False)
    print(f"Events: {[f'{e.event_type.value}@{e.start_lap}x{e.duration_laps}' for e in events] or 'none'}")
    print(f"Race time with replanning {rr.total_race_time_s:,.2f} s | static plan {rr.static_plan_race_time_s:,.2f} s | "
          f"v1 reactive heuristic {rr.reactive_policy_race_time_s:,.2f} s")
    print(f"Pit laps {rr.own_pit_laps}, compounds {' > '.join(rr.own_compounds)}, position {rr.finishing_position} "
          f"(static plan {rr.static_finishing_position})")
    for r in rr.decision_log.records:
        if r.replanned:
            print(f"  lap {r.lap:>2}: {r.action:<16} pit {str(r.recommended_pit_lap):>4}  {r.recommended_strategy_name}  <- {'; '.join(r.triggers)}")
    print(f"Decision log: {p1}, {p2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
