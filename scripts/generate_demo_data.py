"""Generate a deterministic SYNTHETIC lap-level demo dataset.

Usage:
    python scripts/generate_demo_data.py [--output data/synthetic/synthetic_demo_race.csv] [--seed 42]
                                         [--no-random-events] [--inject-anomalies]

The CSV is written together with a ``.meta.json`` sidecar describing how it
was generated. All values are synthetic; none represent real telemetry.
"""

from __future__ import annotations

import argparse
import logging
import os

from racesim.config import DEFAULT_RACE_CONFIG, DEFAULT_REGULATIONS_CONFIG, PROJECT_ROOT, load_setup
from racesim.synthetic_data import generate_synthetic_race, write_synthetic_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default=str(PROJECT_ROOT / "data" / "synthetic" / "synthetic_demo_race.csv"))
    parser.add_argument("--race-config", default=os.environ.get("RACESIM_RACE_CONFIG", str(DEFAULT_RACE_CONFIG)))
    parser.add_argument("--regulations", default=os.environ.get("RACESIM_REGULATIONS_CONFIG", str(DEFAULT_REGULATIONS_CONFIG)))
    parser.add_argument("--seed", type=int, default=None, help="random seed (default: simulation_seed from config)")
    parser.add_argument("--no-random-events", action="store_true", help="disable random safety cars / VSC")
    parser.add_argument("--inject-anomalies", action="store_true", help="add one duplicate row and one missing lap time")
    args = parser.parse_args(argv)
    logging.basicConfig(level=os.environ.get("RACESIM_LOG_LEVEL", "INFO"), format="%(levelname)s %(name)s: %(message)s")

    setup = load_setup(args.race_config, args.regulations)
    df, meta = generate_synthetic_race(setup, seed=args.seed, random_events=not args.no_random_events,
                                       inject_anomalies=args.inject_anomalies)
    csv_path, meta_path = write_synthetic_dataset(df, meta, args.output)
    print(f"SYNTHETIC dataset written: {csv_path} ({len(df)} rows, {df['driver'].nunique()} drivers)")
    print(f"Metadata: {meta_path}")
    print(f"Safety-car laps: {meta['safety_car_laps']}, VSC laps: {meta['vsc_laps']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
