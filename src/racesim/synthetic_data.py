"""Deterministic synthetic lap-level race data.

**All data produced here are SYNTHETIC.** They come from RaceSim's own model
with sampled noise and do not represent any real championship, circuit, team
or driver. Every row carries ``data_source = "SYNTHETIC"`` and a metadata
sidecar describes the generating assumptions.

The data have realistic *structure* rather than independent noise: pace
follows tyre age and fuel burn, pit stops create stint transitions, safety
cars slow the field, traffic occasionally costs time, and lap times have
random variation. Pit time is split between the in-lap (pit entry share) and
out-lap (transit, stationary, exit) as timing data would show it; the total is
preserved.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from racesim.config import SimulatorSetup, setup_to_dict
from racesim.monte_carlo import build_scenario, resolve_flags, rng_for
from racesim.race_simulator import CarSpec, run_race
from racesim.safety_car_model import STATUS_LABELS
from racesim.schemas import RaceEvent, Strategy

logger = logging.getLogger(__name__)

SYNTHETIC_LABEL = "SYNTHETIC"
SECTOR_SPLIT = np.array([0.31, 0.37, 0.32])


def default_synthetic_cars(setup: SimulatorSetup) -> list[CarSpec]:
    """Three fictional drivers: the baseline plan plus the configured rival plans."""
    n = setup.race.total_laps
    cars = []
    if setup.baseline_strategy is not None:
        b = setup.baseline_strategy
        cars.append(CarSpec("SYN_DRIVER_A", Strategy.from_plan(n, b.pit_laps, b.compounds), 0.0, 1.0, 0.0, label="Synthetic driver A"))
    letters = "BCDEFG"
    for i, r in enumerate(setup.rivals):
        car = CarSpec.from_rival(r, n)
        cars.append(CarSpec(f"SYN_DRIVER_{letters[i]}", car.strategy, car.pace_offset_s, car.degradation_multiplier,
                            car.start_offset_s, car.lap_time_std_s, f"Synthetic driver {letters[i]}"))
    if not cars:
        cars.append(CarSpec("SYN_DRIVER_A", Strategy.from_plan(n, [n // 2], (setup.dry_compounds * 2)[:2])))
    return cars


def generate_synthetic_race(
    setup: SimulatorSetup,
    *,
    seed: Optional[int] = None,
    cars: Optional[Sequence[CarSpec]] = None,
    events: Optional[Sequence[RaceEvent]] = None,
    random_events: bool = True,
    inject_anomalies: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Generate a lap-level dataset and its metadata.

    ``inject_anomalies`` adds one duplicated row and one missing lap time so
    the validation workflow can be demonstrated; injected rows are listed in
    the metadata.
    """
    seed = setup.race.simulation_seed if seed is None else seed
    cars = list(cars or default_synthetic_cars(setup))
    flags = resolve_flags(setup, {"events": random_events and events is None, "reactive": True})
    scen = build_scenario(setup, 1, seed, flags=flags, manual_events=events)
    out = run_race(setup, cars, scen, record_history=True, reactive=True)
    rng = rng_for(seed, "synthetic_extras")
    n_laps = setup.race.total_laps
    laps = np.arange(1, n_laps + 1)
    names = list(setup.compounds)
    ps = setup.pit_stop
    entry_share = (ps.pit_entry_loss_s / setup.race.pit_lane_loss_s) if ps.has_components and setup.race.pit_lane_loss_s else 0.3
    frames = []
    for car in cars:
        h = out.history[car.car_id]
        recs = out.pit_records[car.car_id][0]
        pit_laps = {p.lap: p for p in recs}
        lap_t = h["lap_time_s"][0].copy()
        pit_in = np.isin(laps, list(pit_laps))
        pit_out = np.isin(laps, [p + 1 for p in pit_laps])
        for lap, p in pit_laps.items():
            lane_in = p.lane_loss_s * entry_share
            lap_t[lap - 1] += lane_in
            if lap < n_laps:
                lap_t[lap] += p.total_cost_s - lane_in
        rand = scen.car_randomness(car.car_id)
        status = [STATUS_LABELS[int(s)] for s in h["status"][0]]
        traffic = np.where(np.array(status) == "GREEN", rand.traffic_s[0], 0.0)
        sec = SECTOR_SPLIT[None, :] * (1 + rng.normal(0, 0.004, (n_laps, 3)))
        sec = sec / sec.sum(axis=1, keepdims=True) * lap_t[:, None]
        air = setup.race.ambient_temperature_c + 0.02 * (laps - 1) + rng.normal(0, 0.15, n_laps)
        frames.append(pd.DataFrame({
            "lap": laps,
            "driver": car.car_id,
            "lap_time_s": np.round(lap_t, 3),
            "sector1_s": np.round(sec[:, 0], 3),
            "sector2_s": np.round(sec[:, 1], 3),
            "sector3_s": np.round(lap_t, 3) - np.round(sec[:, 0], 3) - np.round(sec[:, 1], 3),
            "tyre_compound": [names[i] for i in h["compound_idx"][0]],
            "tyre_age_laps": h["tyre_age_laps"][0].astype(int),
            "fuel_kg": np.round(h["fuel_kg"][0], 3),
            "track_status": status,
            "pit_in": pit_in,
            "pit_out": pit_out,
            "traffic_loss_s": np.round(traffic, 3),
            "air_temp_c": np.round(air, 2),
            "track_temp_c": np.round(scen.track_temperature_c[0], 2),
            "data_source": SYNTHETIC_LABEL,
        }))
    df = pd.concat(frames, ignore_index=True)
    df["sector3_s"] = df["sector3_s"].round(3)
    injected = []
    if inject_anomalies and len(df) > 10:
        dup = df.iloc[[5]].copy()
        df = pd.concat([df, dup], ignore_index=True)
        injected.append({"type": "duplicate_row", "driver": str(dup["driver"].iloc[0]), "lap": int(dup["lap"].iloc[0])})
        df.loc[12, "lap_time_s"] = np.nan
        injected.append({"type": "missing_lap_time", "driver": str(df.loc[12, "driver"]), "lap": int(df.loc[12, "lap"])})
    meta = {
        "data_source": SYNTHETIC_LABEL,
        "disclaimer": "Synthetic data generated by RaceSim. Not real telemetry or timing data; no real "
                      "championship, circuit, team or driver is represented.",
        "seed": seed,
        "race_name": setup.race.race_name,
        "circuit_name": setup.race.circuit_name,
        "drivers": {c.car_id: {"strategy": c.strategy.strategy_name, "pace_offset_s": c.pace_offset_s,
                               "degradation_multiplier": c.degradation_multiplier} for c in cars},
        "pit_time_split": f"in-lap carries {entry_share:.0%} of pit-lane loss; out-lap carries the rest of the pit cost",
        "injected_anomalies": injected,
        "safety_car_laps": int((scen.track_status[0] == 1).sum()),
        "vsc_laps": int((scen.track_status[0] == 2).sum()),
        "configuration": setup_to_dict(setup),
    }
    logger.info("Generated synthetic race: %d rows, %d drivers (seed %d)", len(df), len(cars), seed)
    return df, meta


def write_synthetic_dataset(df: pd.DataFrame, meta: dict, path: str | Path) -> tuple[Path, Path]:
    """Write the CSV plus a ``.meta.json`` sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    meta_path = path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return path, meta_path
