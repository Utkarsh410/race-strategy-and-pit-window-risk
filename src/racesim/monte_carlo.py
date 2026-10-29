"""Monte Carlo engine.

Uncertainty sources are kept separate and can be switched individually:

==========================  ===================================================
Source                      What is sampled
==========================  ===================================================
Parameter uncertainty       per-run tyre degradation multipliers (per compound),
                            race-pace bias, fuel-consumption multiplier
Lap-to-lap variability      independent lap noise ``N(0, lap_time_std)``
Event uncertainty           safety cars / VSC (and random rain if configured),
                            random traffic, pit-stop execution variability
Strategy uncertainty        reactive decisions (SC pit calls, rain tyre calls)
                            taken with the team's deterministic model, plus the
                            spread between candidate strategies
==========================  ===================================================

Random streams are derived from ``numpy.random.SeedSequence`` keyed by
``(seed, batch, stream name, car id)``: no global random state is used, every
stream is independent, and results are reproducible for identical seed,
simulation count and batch size. All strategies in a run share the same
scenario (*common random numbers*), which reduces noise in their differences.
"""

from __future__ import annotations

import logging
import time
import zlib
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from racesim.config import SimulatorSetup
from racesim.lap_time_model import LapModelContext
from racesim.race_simulator import CarOutcome, CarRandomness, CarSpec, Scenario, run_race
from racesim.safety_car_model import generate_race_control_events
from racesim.schemas import RaceEvent, Strategy, TrackStatus, TrafficConfig
from racesim.tyre_model import CompoundTable
from racesim.weather_model import generate_weather_scenario

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]
PIT_SLOTS = 12


def _key(x) -> int:
    return zlib.crc32(str(x).encode()) & 0xFFFFFFFF


def rng_for(seed: int, *keys) -> np.random.Generator:
    """Independent, reproducible generator for a named stream."""
    return np.random.default_rng(np.random.SeedSequence(entropy=int(seed), spawn_key=tuple(_key(k) for k in keys)))


def sample_lap_time(rng: np.random.Generator, n_sims: int, n_laps: int, std_s: float = 1.0) -> np.ndarray:
    """Lap-to-lap random error (s), ``N(0, std_s)``, shape ``(n_sims, n_laps)``."""
    return std_s * rng.standard_normal((n_sims, n_laps))


def sample_tyre_degradation(rng: np.random.Generator, n_sims: int, table: CompoundTable) -> np.ndarray:
    """Per-run degradation-rate multipliers (lognormal, mean 1), shape ``(n_sims, n_compounds)``.

    ``degradation_uncertainty`` is used as the log-scale standard deviation.
    """
    sigma = table.degradation_uncertainty[None, :]
    z = rng.standard_normal((n_sims, len(table)))
    return np.exp(sigma * z - 0.5 * sigma**2)


def sample_pit_stop_time(rng: np.random.Generator, n_sims: int, n_slots: int = PIT_SLOTS) -> dict[str, np.ndarray]:
    """Random draws for pit-lane loss, stationary time and slow-stop events per stop slot."""
    return {
        "lane_z": rng.standard_normal((n_sims, n_slots)),
        "stationary_z": rng.standard_normal((n_sims, n_slots)),
        "slow_u": rng.random((n_sims, n_slots)),
    }


def sample_safety_car_event(setup: SimulatorSetup, rng: np.random.Generator, n_sims: int, *,
                            sc_probability: Optional[float] = None, vsc_probability: Optional[float] = None) -> np.ndarray:
    """Random SC/VSC track-status array, shape ``(n_sims, n_laps)``."""
    return generate_race_control_events(setup, setup.race.total_laps, rng=rng, n_sims=n_sims,
                                        sc_probability=sc_probability, vsc_probability=vsc_probability)


def sample_traffic(rng: np.random.Generator, n_sims: int, n_laps: int, cfg: TrafficConfig) -> np.ndarray:
    """Random traffic loss (s): Bernoulli occurrence times exponential magnitude."""
    hit = rng.random((n_sims, n_laps)) < cfg.traffic_probability_per_lap
    return np.where(hit, rng.exponential(cfg.traffic_mean_loss_s, (n_sims, n_laps)), 0.0)


DEFAULT_FLAGS = ("parameter", "lap_variability", "events", "pit_variability", "traffic", "reactive", "rival_behaviour")


def resolve_flags(setup: SimulatorSetup, overrides: Optional[dict] = None) -> dict[str, bool]:
    mc = setup.monte_carlo
    flags = {
        "parameter": mc.include_parameter_uncertainty,
        "lap_variability": mc.include_lap_variability,
        "events": mc.include_event_uncertainty,
        "pit_variability": mc.include_pit_variability,
        "traffic": mc.include_traffic,
        "reactive": mc.reactive_strategy,
        "rival_behaviour": mc.include_rival_uncertainty,
    }
    for k, v in (overrides or {}).items():
        if k not in flags:
            raise KeyError(f"unknown Monte Carlo flag {k!r}; valid: {list(flags)}")
        flags[k] = bool(v)
    return flags


def build_scenario(
    setup: SimulatorSetup,
    n_sims: int,
    seed: int,
    *,
    batch_index: int = 0,
    flags: Optional[dict[str, bool]] = None,
    manual_events: Optional[Sequence[RaceEvent]] = None,
    sc_probability: Optional[float] = None,
    vsc_probability: Optional[float] = None,
) -> Scenario:
    """Sample one batch of common random inputs."""
    flags = flags or resolve_flags(setup)
    n_laps = setup.race.total_laps
    table = CompoundTable.from_compounds(setup.compounds)
    b = batch_index
    if manual_events is not None:
        status = generate_race_control_events(setup, n_laps, n_sims=n_sims, manual_events=manual_events)
    elif flags["events"]:
        status = sample_safety_car_event(setup, rng_for(seed, b, "race_control"), n_sims,
                                         sc_probability=sc_probability, vsc_probability=vsc_probability)
    else:
        status = np.zeros((n_sims, n_laps), dtype=np.int8)
    weather = generate_weather_scenario(setup.weather, n_laps, setup.race.track_temperature_c,
                                        rain_events=list(manual_events or []),
                                        rng=rng_for(seed, b, "weather") if flags["events"] else None, n_sims=n_sims)
    deg = sample_tyre_degradation(rng_for(seed, b, "degradation"), n_sims, table) if flags["parameter"] else np.ones((n_sims, len(table)))
    mc = setup.monte_carlo

    def factory(car_id: str) -> CarRandomness:
        r = CarRandomness.zeros(n_sims, n_laps, PIT_SLOTS)
        if flags["lap_variability"]:
            r.lap_noise_z = sample_lap_time(rng_for(seed, b, "lap_noise", car_id), n_sims, n_laps)
        if flags["parameter"]:
            g = rng_for(seed, b, "parameters", car_id)
            r.pace_bias_s = mc.baseline_pace_uncertainty_s * g.standard_normal(n_sims)
            r.fuel_multiplier = np.maximum(1.0 + mc.fuel_consumption_uncertainty * g.standard_normal(n_sims), 0.0)
        if flags["pit_variability"]:
            p = sample_pit_stop_time(rng_for(seed, b, "pit", car_id), n_sims)
            r.pit_lane_z, r.pit_stationary_z, r.pit_slow_u = p["lane_z"], p["stationary_z"], p["slow_u"]
        if flags["traffic"]:
            r.traffic_s = sample_traffic(rng_for(seed, b, "traffic", car_id), n_sims, n_laps, setup.traffic)
        if flags.get("rival_behaviour") and car_id != "OWN":
            g = rng_for(seed, b, "rival_behaviour", car_id)
            r.pace_bias_s = r.pace_bias_s + mc.rival_pace_uncertainty_s * g.standard_normal(n_sims)
            j = mc.rival_pit_lap_jitter_laps
            r.pit_lap_shift = g.integers(-j, j + 1, size=n_sims)
        return r

    flags = {**flags, "rival_behaviour": flags.get("rival_behaviour", False)}
    active = [k for k, v in flags.items() if v]
    return Scenario(n_sims, n_laps, status, weather.track_temperature_c, weather.wetness, deg,
                    deterministic=not any(flags[k] for k in ("parameter", "lap_variability", "events", "pit_variability", "traffic")),
                    flags=flags, car_factory=factory, description=f"monte carlo seed={seed} batch={b} sources={active}")


def simulate_strategy_batch(setup: SimulatorSetup, strategy: Strategy, scenario: Scenario, *,
                            rivals: Optional[Sequence[CarSpec]] = None, reactive: Optional[bool] = None,
                            ctx: Optional[LapModelContext] = None) -> dict[str, CarOutcome]:
    """Simulate one strategy (and optional rivals) over every scenario row."""
    cars = [CarSpec("OWN", strategy)] + list(rivals or [])
    react = scenario.flags.get("reactive", True) if reactive is None else reactive
    return run_race(setup, cars, scenario, reactive=react, ctx=ctx).cars


@dataclass
class MonteCarloResult:
    """Per-run outputs needed for distributions; no intermediate objects kept."""

    strategies: dict[str, Strategy]
    race_times: dict[str, np.ndarray]
    n_stops: dict[str, np.ndarray]
    reactive_stops: dict[str, np.ndarray]
    fuel_shortfall: dict[str, np.ndarray]
    pit_cost: dict[str, np.ndarray]
    sc_occurred: np.ndarray
    vsc_occurred: np.ndarray
    rain_occurred: np.ndarray
    own_finish_elapsed: Optional[dict[str, np.ndarray]] = None
    rival_finish_elapsed: Optional[dict[str, dict[str, np.ndarray]]] = None
    positions: Optional[dict[str, np.ndarray]] = None
    metadata: dict = field(default_factory=dict)

    @property
    def n_sims(self) -> int:
        return int(self.metadata.get("n_sims", len(next(iter(self.race_times.values())))))

    def to_dataframe(self) -> pd.DataFrame:
        frames = []
        for sid, t in self.race_times.items():
            d = {"sim_index": np.arange(len(t)), "strategy_id": sid, "strategy_name": self.strategies[sid].strategy_name,
                 "race_time_s": t, "n_stops": self.n_stops[sid], "reactive_stops": self.reactive_stops[sid],
                 "pit_cost_s": self.pit_cost[sid], "fuel_shortfall": self.fuel_shortfall[sid],
                 "safety_car": self.sc_occurred, "vsc": self.vsc_occurred, "rain": self.rain_occurred}
            if self.positions is not None:
                d["finishing_position"] = self.positions[sid]
            frames.append(pd.DataFrame(d))
        df = pd.concat(frames, ignore_index=True)
        df["data_label"] = "SIMULATED (synthetic model output)"
        return df


def run_monte_carlo(
    setup: SimulatorSetup,
    strategies: Sequence[Strategy],
    *,
    n_sims: Optional[int] = None,
    seed: Optional[int] = None,
    batch_size: Optional[int] = None,
    include_rivals: bool = False,
    manual_events: Optional[Sequence[RaceEvent]] = None,
    flags: Optional[dict[str, bool]] = None,
    sc_probability: Optional[float] = None,
    vsc_probability: Optional[float] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> MonteCarloResult:
    """Evaluate strategies over ``n_sims`` sampled races with common random numbers.

    ``race_times`` holds each strategy's *isolated* race time (no car
    interactions) so every strategy is judged under identical assumptions.
    With ``include_rivals`` the configured rivals are simulated in the same
    races and finishing elapsed times / positions (including dirty air and
    safety-car bunching) are stored for rival metrics.
    """
    if not strategies:
        raise ValueError("at least one strategy is required")
    n_sims = n_sims or setup.monte_carlo.default_simulations
    seed = setup.race.simulation_seed if seed is None else seed
    batch_size = batch_size or setup.monte_carlo.batch_size
    resolved = resolve_flags(setup, flags)
    ctx = LapModelContext.from_setup(setup)
    rivals = [CarSpec.from_rival(r, setup.race.total_laps) for r in setup.rivals] if include_rivals else []
    if include_rivals and not rivals:
        logger.warning("include_rivals requested but no rivals are configured")
    uniq = {s.strategy_id: s for s in strategies}
    acc: dict[str, dict[str, list]] = {sid: {k: [] for k in ("t", "stops", "react", "fuel", "pit", "own", "pos")} for sid in uniq}
    rival_acc: dict[str, dict[str, list]] = {sid: {r.car_id: [] for r in rivals} for sid in uniq}
    sc_occ, vsc_occ, rain_occ = [], [], []
    n_batches = int(np.ceil(n_sims / batch_size))
    total_steps = n_batches * len(uniq)
    step = 0
    t0 = time.perf_counter()
    logger.info("Monte Carlo: %d runs x %d strategies, seed %d, %d batch(es)", n_sims, len(uniq), seed, n_batches)
    for b in range(n_batches):
        m = min(batch_size, n_sims - b * batch_size)
        scen = build_scenario(setup, m, seed, batch_index=b, flags=resolved, manual_events=manual_events,
                              sc_probability=sc_probability, vsc_probability=vsc_probability)
        sc_occ.append((scen.track_status == TrackStatus.SAFETY_CAR).any(axis=1))
        vsc_occ.append((scen.track_status == TrackStatus.VSC).any(axis=1))
        rain_occ.append((scen.wetness > 0).any(axis=1))
        for sid, strat in uniq.items():
            out = simulate_strategy_batch(setup, strat, scen, rivals=rivals, reactive=resolved["reactive"], ctx=ctx)
            own = out["OWN"]
            a = acc[sid]
            a["t"].append(own.isolated_race_time_s)
            a["stops"].append(own.n_stops)
            a["react"].append(own.reactive_stops)
            a["fuel"].append(own.fuel_shortfall)
            a["pit"].append(own.pit_cost_total_s)
            if rivals:
                a["own"].append(own.finish_elapsed_s)
                a["pos"].append(own.finishing_position)
                for r in rivals:
                    rival_acc[sid][r.car_id].append(out[r.car_id].finish_elapsed_s)
            step += 1
            if progress_callback is not None:
                progress_callback(step / total_steps, f"batch {b + 1}/{n_batches}: {strat.strategy_name}")
    cat = lambda lst: np.concatenate(lst) if lst else None  # noqa: E731
    elapsed = time.perf_counter() - t0
    logger.info("Monte Carlo finished in %.2f s", elapsed)
    return MonteCarloResult(
        strategies=uniq,
        race_times={sid: cat(a["t"]) for sid, a in acc.items()},
        n_stops={sid: cat(a["stops"]) for sid, a in acc.items()},
        reactive_stops={sid: cat(a["react"]) for sid, a in acc.items()},
        fuel_shortfall={sid: cat(a["fuel"]) for sid, a in acc.items()},
        pit_cost={sid: cat(a["pit"]) for sid, a in acc.items()},
        sc_occurred=cat(sc_occ), vsc_occurred=cat(vsc_occ), rain_occurred=cat(rain_occ),
        own_finish_elapsed={sid: cat(a["own"]) for sid, a in acc.items()} if rivals else None,
        rival_finish_elapsed={sid: {rid: cat(v) for rid, v in d.items()} for sid, d in rival_acc.items()} if rivals else None,
        positions={sid: cat(a["pos"]) for sid, a in acc.items()} if rivals else None,
        metadata={"n_sims": n_sims, "seed": seed, "batch_size": batch_size, "flags": resolved,
                  "manual_events": [e.model_dump(mode="json") for e in manual_events] if manual_events else None,
                  "sc_probability_per_lap": setup.race.safety_car_probability_per_lap if sc_probability is None else sc_probability,
                  "vsc_probability_per_lap": setup.race.vsc_probability_per_lap if vsc_probability is None else vsc_probability,
                  "rivals": [r.car_id for r in rivals], "runtime_s": elapsed,
                  "reproducibility": "identical for the same seed, n_sims, batch_size and configuration"},
    )


def compare_strategy_distributions(result: MonteCarloResult, strategy_a: str, strategy_b: str, *,
                                   tie_tolerance_s: float = 1e-6) -> dict:
    """Paired comparison of two strategies over the same simulated races.

    ``difference = time_a - time_b`` (negative favours A). Ties (|diff| <=
    tolerance) count half towards each strategy.
    """
    a, b = result.race_times[strategy_a], result.race_times[strategy_b]
    d = a - b
    n = len(d)
    ties = np.abs(d) <= tie_tolerance_s
    p_a = float(np.mean((d < -tie_tolerance_s) + 0.5 * ties))
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return {
        "strategy_a": strategy_a, "strategy_b": strategy_b, "n_sims": n,
        "mean_difference_s": float(d.mean()), "std_difference_s": float(d.std(ddof=1)) if n > 1 else float("nan"),
        "mean_difference_ci95_s": (float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)),
        "probability_a_faster": p_a, "tie_fraction": float(ties.mean()),
        "independent_std_s": float(np.sqrt(a.var(ddof=1) + b.var(ddof=1))) if n > 1 else float("nan"),
    }


def decompose_uncertainty(setup: SimulatorSetup, strategy: Strategy, *, n_sims: int = 2000, seed: Optional[int] = None) -> pd.DataFrame:
    """Race-time standard deviation with each uncertainty source enabled alone.

    Sources interact (e.g. degradation uncertainty matters more after a late
    safety car), so the individual variances need not sum to the total.
    """
    sources = {
        "parameter uncertainty": {"parameter": True},
        "lap-to-lap variability": {"lap_variability": True},
        "event uncertainty (SC/VSC/rain)": {"events": True},
        "pit-stop variability": {"pit_variability": True},
        "traffic": {"traffic": True},
    }
    off = {k: False for k in DEFAULT_FLAGS}
    rows = []
    for name, on in sources.items():
        flags = off | on | {"reactive": on.get("events", False) and setup.monte_carlo.reactive_strategy}
        r = run_monte_carlo(setup, [strategy], n_sims=n_sims, seed=seed, flags=flags)
        t = r.race_times[strategy.strategy_id]
        rows.append({"source": name, "std_s": float(t.std(ddof=1)), "variance_s2": float(t.var(ddof=1)), "mean_s": float(t.mean())})
    full = run_monte_carlo(setup, [strategy], n_sims=n_sims, seed=seed).race_times[strategy.strategy_id]
    rows.append({"source": "all sources combined", "std_s": float(full.std(ddof=1)), "variance_s2": float(full.var(ddof=1)), "mean_s": float(full.mean())})
    return pd.DataFrame(rows)
