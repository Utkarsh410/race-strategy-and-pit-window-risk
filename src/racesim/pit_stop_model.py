"""Pit-stop cost model.

A pit stop is decomposed into explicit, non-overlapping components (s)::

    total_pit_cost = pit_lane_loss            (entry + transit + exit, vs racing line)
                   + stationary_time          (tyre change, or max(tyres, refuel))
                   + rejoin_loss              (optional mean rejoin/traffic loss)

* ``pit_lane_loss`` excludes stationary time (project-wide convention).
* The **out-lap tyre warm-up** penalty is NOT part of the pit cost; it is
  applied by the tyre model on the out-lap. This prevents double counting.
* Under a safety car or VSC the pit-lane loss is a separate, circuit-specific
  input (``sc_pit_lane_loss_s`` / ``vsc_pit_lane_loss_s``) measured against a
  neutralised lap, not a universal percentage of the green-flag loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from racesim.schemas import PitStopModelConfig, RaceConfig, RegulationsConfig, TrackStatus

ArrayLike = np.ndarray | float | int


@dataclass(frozen=True)
class PitCostBreakdown:
    """Explicit components of a single pit stop (all in seconds)."""

    track_status: TrackStatus
    pit_entry_s: Optional[float]
    pit_transit_s: Optional[float]
    pit_exit_s: Optional[float]
    pit_lane_loss_s: float
    stationary_s: float
    rejoin_loss_s: float

    @property
    def total_s(self) -> float:
        """Total cost; each component is counted exactly once."""
        return self.pit_lane_loss_s + self.stationary_s + self.rejoin_loss_s

    def as_dict(self) -> dict:
        return {
            "track_status": self.track_status.name,
            "pit_entry_s": self.pit_entry_s,
            "pit_transit_s": self.pit_transit_s,
            "pit_exit_s": self.pit_exit_s,
            "pit_lane_loss_s": self.pit_lane_loss_s,
            "stationary_s": self.stationary_s,
            "rejoin_loss_s": self.rejoin_loss_s,
            "total_s": self.total_s,
        }


def calculate_green_flag_pit_loss(race: RaceConfig, pit: PitStopModelConfig) -> float:
    """Mean green-flag pit-lane loss (s), excluding stationary time."""
    if pit.has_components:
        return float(pit.pit_entry_loss_s + pit.pit_transit_loss_s + pit.pit_exit_loss_s)  # type: ignore[operator]
    return float(race.pit_lane_loss_s)


def calculate_neutralised_pit_loss(pit: PitStopModelConfig, status: TrackStatus | int) -> float:
    """Mean pit-lane loss (s) under SC or VSC, from circuit-specific inputs."""
    status = TrackStatus(int(status))
    if status == TrackStatus.SAFETY_CAR:
        return float(pit.sc_pit_lane_loss_s)
    if status == TrackStatus.VSC:
        return float(pit.vsc_pit_lane_loss_s)
    raise ValueError("neutralised pit loss requested for a green-flag lap")


def mean_lane_loss_by_status(race: RaceConfig, pit: PitStopModelConfig, status: ArrayLike) -> np.ndarray:
    """Vectorised mean pit-lane loss for an array of track states."""
    s = np.asarray(status)
    return np.select(
        [s == TrackStatus.SAFETY_CAR, s == TrackStatus.VSC],
        [pit.sc_pit_lane_loss_s, pit.vsc_pit_lane_loss_s],
        default=calculate_green_flag_pit_loss(race, pit),
    )


def calculate_stationary_time(
    race: RaceConfig,
    pit: PitStopModelConfig,
    fuel_added_kg: ArrayLike = 0.0,
    regulations: Optional[RegulationsConfig] = None,
) -> np.ndarray:
    """Mean stationary time (s).

    Without refuelling this is ``stationary_stop_time_s``. With refuelling the
    tyre change and refuel are concurrent (``max``) or sequential (``sum``)
    according to the regulations.
    """
    fuel = np.asarray(fuel_added_kg, dtype=float)
    tyres = pit.tyre_change_time_s if pit.tyre_change_time_s is not None else race.stationary_stop_time_s
    if regulations is None or not regulations.refuelling_allowed:
        if np.any(fuel > 0):
            raise ValueError("fuel added at a pit stop but refuelling is not permitted by the regulations")
        return np.full(fuel.shape, race.stationary_stop_time_s)
    if not np.any(fuel > 0):
        return np.full(fuel.shape, tyres)
    if regulations.refuel_rate_kg_per_s is None:
        raise ValueError("refuelling requires regulations.refuel_rate_kg_per_s")
    refuel = fuel / regulations.refuel_rate_kg_per_s
    return np.maximum(tyres, refuel) if regulations.refuel_concurrent_with_tyres else tyres + refuel


def calculate_total_pit_cost(
    race: RaceConfig,
    pit: PitStopModelConfig,
    status: TrackStatus | int = TrackStatus.GREEN,
    *,
    fuel_added_kg: float = 0.0,
    regulations: Optional[RegulationsConfig] = None,
) -> PitCostBreakdown:
    """Deterministic (mean) pit cost with explicit components."""
    status = TrackStatus(int(status))
    if status == TrackStatus.GREEN:
        lane = calculate_green_flag_pit_loss(race, pit)
        entry, transit, exit_ = pit.pit_entry_loss_s, pit.pit_transit_loss_s, pit.pit_exit_loss_s
    else:
        lane = calculate_neutralised_pit_loss(pit, status)
        entry = transit = exit_ = None
    stationary = float(calculate_stationary_time(race, pit, fuel_added_kg, regulations))
    return PitCostBreakdown(status, entry, transit, exit_, lane, stationary, pit.rejoin_loss_s)


# ---------------------------------------------------------------------------
# Stochastic sampling (shared by the simulator and Monte Carlo engine)
# ---------------------------------------------------------------------------


def lane_loss_from_normals(race: RaceConfig, pit: PitStopModelConfig, status: ArrayLike, z: ArrayLike) -> np.ndarray:
    """Transform standard normals into pit-lane losses for the given states.

    Truncated below at ``minimum_lane_loss_fraction * mean`` so a sampled stop
    can never be implausibly fast.
    """
    s = np.asarray(status)
    mean = mean_lane_loss_by_status(race, pit, s)
    std = np.where(s == TrackStatus.GREEN, race.pit_loss_std_s, pit.neutralised_pit_loss_std_s)
    return np.maximum(mean + std * np.asarray(z), pit.minimum_lane_loss_fraction * mean)


def stationary_from_normals(
    race: RaceConfig, pit: PitStopModelConfig, z: ArrayLike, u_slow: ArrayLike, mean_stationary_s: ArrayLike | None = None
) -> np.ndarray:
    """Transform random draws into stationary times (with occasional slow stops)."""
    mean = race.stationary_stop_time_s if mean_stationary_s is None else np.asarray(mean_stationary_s)
    base = np.maximum(mean + pit.stationary_std_s * np.asarray(z), pit.minimum_stationary_fraction * np.asarray(mean))
    return base + np.where(np.asarray(u_slow) < pit.slow_stop_probability, pit.slow_stop_extra_s, 0.0)


def sample_pit_stop_time(
    race: RaceConfig,
    pit: PitStopModelConfig,
    rng: np.random.Generator,
    size: int | tuple[int, ...] = 1,
    *,
    status: TrackStatus | int = TrackStatus.GREEN,
    stochastic: bool = True,
) -> dict[str, np.ndarray]:
    """Sample pit-stop components. ``stochastic=False`` returns the mean values."""
    z_lane = rng.standard_normal(size) if stochastic else np.zeros(size)
    z_stat = rng.standard_normal(size) if stochastic else np.zeros(size)
    u_slow = rng.random(size) if stochastic else np.ones(size)
    stat_arr = np.full(size, int(status))
    lane = lane_loss_from_normals(race, pit, stat_arr, z_lane)
    stationary = stationary_from_normals(race, pit, z_stat, u_slow)
    rejoin = np.full(size, pit.rejoin_loss_s)
    return {"lane_loss_s": lane, "stationary_s": stationary, "rejoin_loss_s": rejoin, "total_s": lane + stationary + rejoin}


def calculate_pit_loss_uncertainty(
    race: RaceConfig,
    pit: PitStopModelConfig,
    *,
    status: TrackStatus | int = TrackStatus.GREEN,
    n_samples: int = 20_000,
    seed: int = 0,
) -> dict[str, float]:
    """Summary of the sampled total pit cost distribution (s)."""
    s = sample_pit_stop_time(race, pit, np.random.default_rng(seed), n_samples, status=status)["total_s"]
    mean_cost = calculate_total_pit_cost(race, pit, status).total_s
    return {
        "track_status": TrackStatus(int(status)).name,
        "deterministic_mean_s": mean_cost,
        "sampled_mean_s": float(s.mean()),
        "std_s": float(s.std(ddof=1)),
        "p05_s": float(np.percentile(s, 5)),
        "p50_s": float(np.percentile(s, 50)),
        "p95_s": float(np.percentile(s, 95)),
        "n_samples": n_samples,
    }
