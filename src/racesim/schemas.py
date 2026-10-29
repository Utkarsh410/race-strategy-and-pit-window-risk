"""Validated data structures for RaceSim.

Conventions used throughout the project
---------------------------------------
* **Time** is always in seconds (``_s`` suffix).
* **Fuel mass** is always in kilograms (``_kg`` suffix).
* **Laps** are 1-indexed integers. Lap ``1`` is the first racing lap and lap
  ``total_laps`` is the final lap.
* **Stint intervals are inclusive**: a stint with ``start_lap=1`` and
  ``end_lap=20`` covers laps 1, 2, ..., 20 (20 laps).
* **A pit stop "on lap L"** means the car enters the pit lane at the end of
  lap ``L`` (lap ``L`` is the in-lap). The next stint starts on lap ``L + 1``
  (the out-lap) on the new tyres. The whole pit cost is booked against lap
  ``L`` and recorded separately from the lap's racing time so that it is
  counted exactly once.
* **Tyre age** on a lap is the number of laps already completed on that tyre
  set when the lap starts. A fresh set therefore has age ``0`` on its first lap.
* **Pit-loss convention**: ``pit_lane_loss_s`` is the time lost by driving
  through the pit lane (entry + transit + exit) compared with staying on the
  racing line, **excluding** the stationary service time. The total pit cost is
  ``pit_lane_loss_s + stationary_stop_time_s`` (+ any configured rejoin loss).
  The out-lap tyre warm-up penalty is modelled by the tyre model, never by the
  pit-stop model, so it cannot be double-counted.

Pydantic models are used for user-facing configuration and reports; plain
dataclasses are used for lightweight internal simulation records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class TrackStatus(IntEnum):
    """Race-control state of a lap. Stored as small integers in arrays."""

    GREEN = 0
    SAFETY_CAR = 1
    VSC = 2


class EventType(str, Enum):
    """Kinds of race event that can be specified manually or generated."""

    GREEN = "green"
    SAFETY_CAR = "safety_car"
    VSC = "vsc"
    RAIN = "rain"


CompoundType = Literal["dry", "intermediate", "wet"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# ---------------------------------------------------------------------------
# Configuration models
# ---------------------------------------------------------------------------


class RaceConfig(_Frozen):
    """Race-level configuration.

    Reference conditions for ``baseline_lap_time_s``: an empty fuel tank, a
    fresh reference-compound tyre after warm-up (age 0), green flag, no traffic
    and a track at ``reference_track_temperature_c``. Every additive lap-time
    effect is measured relative to these conditions.

    ``minimum_stint_laps`` / ``maximum_stint_laps`` are *engineering planning*
    limits. Championship hard limits live in :class:`RegulationsConfig`; the
    effective limit is the stricter of the two.
    """

    race_name: str
    circuit_name: str
    total_laps: int = Field(ge=1, le=1000)
    baseline_lap_time_s: float = Field(gt=0, le=3600)
    pit_lane_loss_s: float = Field(
        ge=0, le=600, description="Pit-lane loss excluding stationary time (s)."
    )
    stationary_stop_time_s: float = Field(ge=0, le=600, description="Mean stationary service time (s).")
    pit_loss_std_s: float = Field(ge=0, le=60, description="Std. dev. of pit-lane loss (s).")
    baseline_lap_time_std_s: float = Field(ge=0, le=60, description="Lap-to-lap random variability (s).")
    fuel_start_kg: float = Field(ge=0, le=1000)
    fuel_consumption_kg_per_lap: float = Field(ge=0, le=100)
    fuel_effect_s_per_kg: float = Field(ge=0, le=1.0)
    safety_car_probability_per_lap: float = Field(ge=0, le=1)
    vsc_probability_per_lap: float = Field(ge=0, le=1)
    track_temperature_c: float = Field(ge=-30, le=90)
    ambient_temperature_c: float = Field(ge=-40, le=60)
    simulation_seed: int = Field(ge=0)
    simulation_count: int = Field(ge=1, le=1_000_000)
    minimum_stint_laps: int = Field(ge=1)
    maximum_stint_laps: int = Field(ge=1)
    minimum_pit_laps_remaining: int = Field(
        ge=1, description="A stop is not allowed when fewer laps than this remain after the in-lap."
    )
    maximum_strategy_candidates: int = Field(ge=1, le=200_000)
    # Extensions (documented, all with explicit units)
    reference_track_temperature_c: float = Field(default=35.0, ge=-30, le=90)
    standing_start_loss_s: float = Field(default=0.0, ge=0, le=60)
    track_evolution_s_per_lap: float = Field(default=0.0, ge=0, le=1.0)
    track_evolution_cap_s: float = Field(default=0.0, ge=0, le=10.0)
    maximum_stops: int = Field(default=3, ge=0, le=10)
    minimum_lap_time_fraction: float = Field(
        default=0.85, gt=0, le=1, description="Physical floor for a lap as a fraction of baseline."
    )
    data_label: str = "SYNTHETIC DEMONSTRATION PARAMETERS - not a real circuit"

    @model_validator(mode="after")
    def _check_consistency(self) -> "RaceConfig":
        if self.maximum_stint_laps < self.minimum_stint_laps:
            raise ValueError("maximum_stint_laps must be >= minimum_stint_laps")
        if self.minimum_stint_laps > self.total_laps:
            raise ValueError("minimum_stint_laps cannot exceed total_laps")
        if self.safety_car_probability_per_lap + self.vsc_probability_per_lap > 1:
            raise ValueError("safety-car + VSC probability per lap must not exceed 1")
        if self.minimum_pit_laps_remaining >= self.total_laps:
            raise ValueError("minimum_pit_laps_remaining must be < total_laps")
        return self

    @property
    def total_pit_cost_s(self) -> float:
        """Mean green-flag pit cost: lane loss + stationary time (s)."""
        return self.pit_lane_loss_s + self.stationary_stop_time_s


class TyreModelConfig(_Frozen):
    """Per-compound tyre model parameters.

    ``reference_lap_time_s`` is the compound's lap time under the race reference
    conditions (empty tank, age 0 after warm-up, reference track temperature).
    ``degradation_uncertainty`` is the relative standard deviation of the
    degradation-rate multiplier used for parameter uncertainty in Monte Carlo.
    ``temperature_sensitivity`` is the fractional change in degradation rate per
    degree C of track temperature above the reference (e.g. 0.02 = +2 %/degC).
    """

    compound_name: str
    reference_lap_time_s: float = Field(gt=0)
    linear_degradation_s_per_lap: float = Field(ge=0, le=10)
    nonlinear_degradation_coefficient: float = Field(default=0.0, ge=0, le=10)
    nonlinear_degradation_threshold_laps: float = Field(default=0.0, ge=0)
    warmup_penalty_s: float = Field(default=0.0, ge=0, le=60)
    warmup_decay_laps: float = Field(default=1.0, gt=0, le=50)
    degradation_uncertainty: float = Field(default=0.0, ge=0, le=2)
    temperature_sensitivity: float = Field(default=0.0, ge=-0.5, le=0.5)
    maximum_recommended_age_laps: int = Field(ge=1)
    compound_type: CompoundType = "dry"
    degradation_model: Literal["linear", "nonlinear"] = "nonlinear"
    parameter_source: str = "manual (illustrative)"

    @field_validator("compound_name")
    @classmethod
    def _upper(cls, v: str) -> str:
        v = v.strip().upper()
        if not v:
            raise ValueError("compound_name must not be empty")
        return v


class PitStopModelConfig(_Frozen):
    """Pit-stop component model, calibratable per circuit.

    The optional ``pit_entry_loss_s`` / ``pit_transit_loss_s`` /
    ``pit_exit_loss_s`` components must sum to ``RaceConfig.pit_lane_loss_s``
    when supplied. Neutralised pit-lane losses (safety car and VSC) are
    circuit-specific inputs, not a fixed percentage of the green-flag loss.
    """

    pit_entry_loss_s: Optional[float] = Field(default=None, ge=0)
    pit_transit_loss_s: Optional[float] = Field(default=None, ge=0)
    pit_exit_loss_s: Optional[float] = Field(default=None, ge=0)
    stationary_std_s: float = Field(default=0.3, ge=0)
    slow_stop_probability: float = Field(default=0.0, ge=0, le=1)
    slow_stop_extra_s: float = Field(default=0.0, ge=0)
    rejoin_loss_s: float = Field(default=0.0, ge=0, description="Mean rejoin/traffic loss after exit (s).")
    sc_pit_lane_loss_s: float = Field(ge=0, description="Pit-lane loss relative to a lap behind the SC (s).")
    vsc_pit_lane_loss_s: float = Field(ge=0, description="Pit-lane loss relative to a VSC lap (s).")
    neutralised_pit_loss_std_s: float = Field(default=1.0, ge=0)
    minimum_lane_loss_fraction: float = Field(default=0.7, gt=0, le=1)
    minimum_stationary_fraction: float = Field(default=0.6, gt=0, le=1)
    tyre_change_time_s: Optional[float] = Field(
        default=None, ge=0, description="Tyre-change time when refuelling (defaults to stationary time)."
    )

    @property
    def has_components(self) -> bool:
        return None not in (self.pit_entry_loss_s, self.pit_transit_loss_s, self.pit_exit_loss_s)


class RaceControlConfig(_Frozen):
    """Safety-car / VSC event model parameters (all durations in laps)."""

    sc_duration_min_laps: int = Field(default=3, ge=1)
    sc_duration_max_laps: int = Field(default=6, ge=1)
    vsc_duration_min_laps: int = Field(default=1, ge=1)
    vsc_duration_max_laps: int = Field(default=3, ge=1)
    sc_lap_time_factor: float = Field(default=1.40, ge=1)
    vsc_lap_time_factor: float = Field(default=1.30, ge=1)
    sc_tyre_wear_factor: float = Field(default=0.3, ge=0, le=1)
    vsc_tyre_wear_factor: float = Field(default=0.4, ge=0, le=1)
    sc_fuel_factor: float = Field(default=0.6, ge=0, le=1)
    vsc_fuel_factor: float = Field(default=0.7, ge=0, le=1)
    sc_restart_loss_s: float = Field(default=1.0, ge=0)
    vsc_restart_loss_s: float = Field(default=0.3, ge=0)
    earliest_event_lap: int = Field(default=2, ge=1)
    no_new_event_final_laps: int = Field(default=2, ge=0)
    cooldown_laps: int = Field(default=3, ge=0)
    sc_bunching_gap_s: float = Field(default=0.8, ge=0, description="Gap between cars after SC bunching (s).")
    react_to_neutralisation: bool = True
    neutralisation_pit_advance_max_laps: int = Field(default=12, ge=0)

    @model_validator(mode="after")
    def _check(self) -> "RaceControlConfig":
        if self.sc_duration_max_laps < self.sc_duration_min_laps:
            raise ValueError("sc_duration_max_laps < sc_duration_min_laps")
        if self.vsc_duration_max_laps < self.vsc_duration_min_laps:
            raise ValueError("vsc_duration_max_laps < vsc_duration_min_laps")
        return self


class WeatherConfig(_Frozen):
    """Deliberately simple, uncalibrated weather model.

    ``wetness`` is a dimensionless 0..1 track-wetness index (0 = dry,
    1 = fully wet). Wet-weather pace parameters are illustrative only; set
    ``wet_weather_enabled: false`` if no calibrated values are available.
    """

    scenario: Literal["stable", "rising_temperature", "cooling_temperature", "rain_onset"] = "stable"
    temperature_change_c_per_lap: float = Field(default=0.0, ge=-5, le=5)
    wet_weather_enabled: bool = True
    rain_probability_per_race: float = Field(default=0.0, ge=0, le=1)
    rain_onset_lap: Optional[int] = Field(default=None, ge=1)
    rain_duration_laps: int = Field(default=10, ge=1)
    wetness_increase_per_lap: float = Field(default=0.15, ge=0, le=1)
    drying_rate_per_lap: float = Field(default=0.05, ge=0, le=1)
    peak_wetness: float = Field(default=0.6, ge=0, le=1)
    rain_track_cooling_c_per_lap: float = Field(default=0.5, ge=0)
    dry_tyre_wet_penalty_s: float = Field(default=25.0, ge=0, description="Penalty at wetness 1 on dry tyres (s).")
    intermediate_optimal_wetness: float = Field(default=0.45, ge=0, le=1)
    intermediate_mismatch_penalty_s: float = Field(default=8.0, ge=0)
    wet_optimal_wetness: float = Field(default=0.85, ge=0, le=1)
    wet_mismatch_penalty_s: float = Field(default=10.0, ge=0)
    react_to_rain: bool = True
    calibration_status: str = "UNCALIBRATED - illustrative only"


class TrafficConfig(_Frozen):
    """Simplified traffic model (no track-position data)."""

    traffic_probability_per_lap: float = Field(default=0.08, ge=0, le=1)
    traffic_mean_loss_s: float = Field(default=0.6, ge=0)
    dirty_air_threshold_s: float = Field(default=1.0, ge=0)
    dirty_air_loss_s_per_lap: float = Field(default=0.4, ge=0)
    overtake_pace_delta_s: float = Field(
        default=0.8, ge=0, description="Pace advantage needed to pass a car on track (s/lap)."
    )


class RivalConfig(_Frozen):
    """A rival car with a configurable pace offset and planned strategy."""

    rival_id: str
    name: str = "Rival"
    pace_offset_s: float = Field(default=0.0, ge=-20, le=20, description="Negative = faster (s/lap).")
    degradation_multiplier: float = Field(default=1.0, ge=0, le=5)
    start_gap_s: float = Field(default=0.0, description="Rival elapsed-time offset at start; negative = ahead (s).")
    pit_laps: list[int] = Field(default_factory=list)
    compounds: list[str] = Field(min_length=1)
    starting_tyre_age_laps: int = Field(default=0, ge=0)
    lap_time_std_s: Optional[float] = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _check(self) -> "RivalConfig":
        if len(self.compounds) != len(self.pit_laps) + 1:
            raise ValueError("rival compounds must have len(pit_laps) + 1 entries")
        if sorted(self.pit_laps) != list(self.pit_laps) or len(set(self.pit_laps)) != len(self.pit_laps):
            raise ValueError("rival pit_laps must be strictly increasing")
        return self


class MonteCarloConfig(_Frozen):
    """Monte Carlo engine settings and uncertainty-source switches."""

    default_simulations: int = Field(default=10_000, ge=1)
    dashboard_simulations: int = Field(default=1_000, ge=1)
    batch_size: int = Field(default=5_000, ge=1)
    include_parameter_uncertainty: bool = True
    include_lap_variability: bool = True
    include_event_uncertainty: bool = True
    include_pit_variability: bool = True
    include_traffic: bool = True
    reactive_strategy: bool = True
    baseline_pace_uncertainty_s: float = Field(default=0.15, ge=0, description="Std. dev. of race pace bias (s/lap).")
    fuel_consumption_uncertainty: float = Field(default=0.02, ge=0, le=1, description="Relative std. dev.")
    tail_warning_min_samples: int = Field(default=50, ge=1)


class DriverRule(_Frozen):
    driver_id: str
    minimum_laps: int = Field(default=0, ge=0)
    maximum_laps: Optional[int] = Field(default=None, ge=1)


class RegulationsConfig(_Frozen):
    """Championship rules. ``None`` means *not configured*: the constraint is
    disabled and a warning is produced by :func:`regulation_warnings`."""

    championship_name: str
    minimum_stint_laps: Optional[int] = Field(default=None, ge=1)
    maximum_stint_laps: Optional[int] = Field(default=None, ge=1)
    mandatory_pit_stops: Optional[int] = Field(default=None, ge=0)
    maximum_pit_stops: Optional[int] = Field(default=None, ge=0)
    minimum_distinct_dry_compounds: Optional[int] = Field(default=None, ge=1)
    wet_running_waives_compound_rule: bool = True
    permitted_compounds: Optional[list[str]] = None
    refuelling_allowed: Optional[bool] = None
    fuel_capacity_kg: Optional[float] = Field(default=None, gt=0)
    refuel_rate_kg_per_s: Optional[float] = Field(default=None, gt=0)
    refuel_concurrent_with_tyres: bool = True
    driver_rules: Optional[list[DriverRule]] = None
    notes: str = ""

    @field_validator("permitted_compounds")
    @classmethod
    def _upper(cls, v: Optional[list[str]]) -> Optional[list[str]]:
        return None if v is None else [c.strip().upper() for c in v]


class BaselineStrategyConfig(_Frozen):
    """Reference plan used for 'probability of beating the baseline'."""

    pit_laps: list[int]
    compounds: list[str]

    @field_validator("compounds")
    @classmethod
    def _upper(cls, v: list[str]) -> list[str]:
        return [c.strip().upper() for c in v]


# ---------------------------------------------------------------------------
# Strategy structures
# ---------------------------------------------------------------------------


class Stint(_Frozen):
    """A stint over the **inclusive** lap interval ``[start_lap, end_lap]``."""

    start_lap: int = Field(ge=1)
    end_lap: int = Field(ge=1)
    tyre_compound: str
    starting_tyre_age_laps: int = Field(default=0, ge=0)
    driver_id: str = "DRIVER_1"

    @field_validator("tyre_compound")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.strip().upper()

    @model_validator(mode="after")
    def _check(self) -> "Stint":
        if self.end_lap < self.start_lap:
            raise ValueError(f"stint end_lap {self.end_lap} < start_lap {self.start_lap}")
        return self

    @property
    def length_laps(self) -> int:
        return self.end_lap - self.start_lap + 1


class Strategy(_Frozen):
    """A race (or remaining-race) strategy made of contiguous stints.

    ``planned_pit_laps[k]`` is the in-lap of stop ``k`` and equals
    ``stints[k].end_lap``; ``planned_compounds[k]`` is ``stints[k].tyre_compound``.
    """

    strategy_id: str
    strategy_name: str
    stints: list[Stint] = Field(min_length=1)
    planned_pit_laps: list[int]
    planned_compounds: list[str]
    description: str = ""

    @model_validator(mode="after")
    def _check(self) -> "Strategy":
        for a, b in zip(self.stints, self.stints[1:]):
            if b.start_lap != a.end_lap + 1:
                raise ValueError("stints must be contiguous (next start_lap = previous end_lap + 1)")
        if list(self.planned_pit_laps) != [s.end_lap for s in self.stints[:-1]]:
            raise ValueError("planned_pit_laps must equal the end laps of all but the final stint")
        if list(self.planned_compounds) != [s.tyre_compound for s in self.stints]:
            raise ValueError("planned_compounds must match stint compounds")
        return self

    @property
    def number_of_stops(self) -> int:
        return len(self.planned_pit_laps)

    @property
    def start_lap(self) -> int:
        return self.stints[0].start_lap

    @property
    def end_lap(self) -> int:
        return self.stints[-1].end_lap

    @property
    def stint_lengths(self) -> list[int]:
        return [s.length_laps for s in self.stints]

    @classmethod
    def from_plan(
        cls,
        total_laps: int,
        pit_laps: list[int],
        compounds: list[str],
        *,
        start_lap: int = 1,
        starting_tyre_age_laps: int = 0,
        driver_ids: Optional[list[str]] = None,
        strategy_name: Optional[str] = None,
        description: str = "",
    ) -> "Strategy":
        """Build a strategy from in-laps and a compound sequence."""
        compounds = [c.strip().upper() for c in compounds]
        if len(compounds) != len(pit_laps) + 1:
            raise ValueError("compounds must contain len(pit_laps) + 1 entries")
        bounds = [start_lap - 1, *pit_laps, total_laps]
        if any(b2 <= b1 for b1, b2 in zip(bounds, bounds[1:])):
            raise ValueError(f"pit laps {pit_laps} must be strictly increasing within the race distance")
        drivers = driver_ids or ["DRIVER_1"] * len(compounds)
        stints = [
            Stint(
                start_lap=bounds[i] + 1,
                end_lap=bounds[i + 1],
                tyre_compound=compounds[i],
                starting_tyre_age_laps=starting_tyre_age_laps if i == 0 else 0,
                driver_id=drivers[i],
            )
            for i in range(len(compounds))
        ]
        sid = strategy_identifier(pit_laps, compounds, start_lap)
        name = strategy_name or default_strategy_name(pit_laps, compounds)
        return cls(
            strategy_id=sid,
            strategy_name=name,
            stints=stints,
            planned_pit_laps=list(pit_laps),
            planned_compounds=compounds,
            description=description,
        )


def strategy_identifier(pit_laps: list[int], compounds: list[str], start_lap: int = 1) -> str:
    """Deterministic identifier, e.g. ``L1:MEDIUM>25>HARD``."""
    parts = [compounds[0]]
    for lap, comp in zip(pit_laps, compounds[1:]):
        parts += [str(lap), comp]
    return f"L{start_lap}:" + ">".join(parts)


def default_strategy_name(pit_laps: list[int], compounds: list[str]) -> str:
    stops = len(pit_laps)
    label = "no-stop" if stops == 0 else f"{stops}-stop"
    seq = "-".join(c[0] if c not in ("INTERMEDIATE", "WET") else c[:3] for c in compounds)
    laps = ", ".join(str(p) for p in pit_laps)
    return f"{label} {seq}" + (f" (L{laps})" if pit_laps else "")


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


class RaceEvent(_Frozen):
    """A race-control or weather event.

    ``start_lap`` is the first affected lap; the event affects the inclusive
    interval ``[start_lap, start_lap + duration_laps - 1]``. ``severity`` is a
    0..1 intensity used by rain (peak wetness); it is informational for
    SC/VSC.
    """

    event_type: EventType
    start_lap: int = Field(ge=1)
    duration_laps: int = Field(ge=1)
    severity: float = Field(default=1.0, ge=0, le=1)
    event_metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def end_lap(self) -> int:
        return self.start_lap + self.duration_laps - 1


# ---------------------------------------------------------------------------
# Snapshot of a live race (for pit-window analysis mid-race)
# ---------------------------------------------------------------------------


class RaceSnapshot(_Frozen):
    """State at the end of ``current_lap`` (0 = before the start)."""

    current_lap: int = Field(default=0, ge=0)
    tyre_compound: Optional[str] = None
    tyre_age_laps: int = Field(default=0, ge=0)
    fuel_kg: Optional[float] = Field(default=None, ge=0)
    stops_completed: int = Field(default=0, ge=0)
    compounds_used: list[str] = Field(default_factory=list)

    @field_validator("tyre_compound")
    @classmethod
    def _upper(cls, v: Optional[str]) -> Optional[str]:
        return None if v is None else v.strip().upper()


# ---------------------------------------------------------------------------
# Simulation records (lightweight dataclasses)
# ---------------------------------------------------------------------------


@dataclass
class PitStopRecord:
    lap: int
    compound_from: str
    compound_to: str
    lane_loss_s: float
    stationary_s: float
    rejoin_loss_s: float
    total_cost_s: float
    track_status: str
    reason: str = "planned"
    fuel_added_kg: float = 0.0


@dataclass
class StintRecord:
    stint_index: int
    start_lap: int
    end_lap: int
    compound: str
    starting_tyre_age_laps: int
    laps: int
    racing_time_s: float
    mean_lap_time_s: float


@dataclass
class SimulationResult:
    """Detailed outcome of a single simulated race for one car.

    ``predicted_race_time_s == lap_time_history.sum() + sum(pit costs)``
    (lap times exclude pit costs, which are recorded in ``pit_stops``).
    """

    strategy_id: str
    predicted_race_time_s: float
    pit_stops: list[PitStopRecord]
    stint_records: list[StintRecord]
    lap_time_history: np.ndarray
    fuel_history: np.ndarray
    tyre_age_history: np.ndarray
    event_history: list[str]
    compound_history: list[str] = field(default_factory=list)
    pit_time_history: np.ndarray = field(default_factory=lambda: np.zeros(0))
    elapsed_history: np.ndarray = field(default_factory=lambda: np.zeros(0))
    finishing_position: Optional[int] = None
    simulation_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_pit_cost_s(self) -> float:
        return float(sum(p.total_cost_s for p in self.pit_stops))

    def to_dataframe(self) -> pd.DataFrame:
        """Lap-by-lap table of the simulated race."""
        n = len(self.lap_time_history)
        start = int(self.simulation_metadata.get("start_lap", 1))
        return pd.DataFrame(
            {
                "lap": np.arange(start, start + n),
                "lap_time_s": self.lap_time_history,
                "pit_time_s": self.pit_time_history if len(self.pit_time_history) == n else np.zeros(n),
                "elapsed_s": self.elapsed_history if len(self.elapsed_history) == n else np.cumsum(self.lap_time_history),
                "fuel_kg_at_lap_start": self.fuel_history,
                "tyre_age_laps": self.tyre_age_history,
                "tyre_compound": self.compound_history if len(self.compound_history) == n else [""] * n,
                "track_status": self.event_history,
            }
        )


class RiskReport(BaseModel):
    """Risk metrics for a race-time distribution. **Lower race time is better.**

    Probabilities are "higher is better" for the strategy being assessed.
    Optional fields are ``None`` when the analysis was not performed; they are
    never filled with substitute values.
    """

    model_config = ConfigDict(extra="forbid")

    strategy_id: Optional[str] = None
    expected_race_time_s: float
    median_race_time_s: float
    standard_deviation_s: float
    percentile_05_s: float
    percentile_50_s: float
    percentile_90_s: float
    percentile_95_s: float
    cvar_95_s: float = Field(description="Mean of the worst 5% race times (s); lower is better.")
    probability_of_beating_baseline: Optional[float] = None
    probability_of_beating_rival: Optional[float] = None
    expected_finishing_position: Optional[float] = None
    number_of_simulations: int
    warnings: list[str] = Field(default_factory=list)
