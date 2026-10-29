"""Configuration loading, merging and validation.

A :class:`SimulatorSetup` bundles every model assumption used by the
simulator. It is built from two YAML files:

* ``config/default_race.yaml`` - race, tyres, pit, events, weather, rivals,
  Monte Carlo settings and the baseline strategy.
* ``config/default_regulations.yaml`` - championship rules.

All numeric values are in the units documented in :mod:`racesim.schemas`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from racesim.schemas import (
    BaselineStrategyConfig,
    MonteCarloConfig,
    PitStopModelConfig,
    RaceConfig,
    RaceControlConfig,
    RegulationsConfig,
    RivalConfig,
    TrafficConfig,
    TyreModelConfig,
    WeatherConfig,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RACE_CONFIG = PROJECT_ROOT / "config" / "default_race.yaml"
DEFAULT_REGULATIONS_CONFIG = PROJECT_ROOT / "config" / "default_regulations.yaml"

PathLike = Union[str, Path]


class SimulatorSetup(BaseModel):
    """Complete, validated set of model assumptions for one analysis."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    race: RaceConfig
    compounds: dict[str, TyreModelConfig]
    pit_stop: PitStopModelConfig
    race_control: RaceControlConfig = Field(default_factory=RaceControlConfig)
    weather: WeatherConfig = Field(default_factory=WeatherConfig)
    traffic: TrafficConfig = Field(default_factory=TrafficConfig)
    rivals: list[RivalConfig] = Field(default_factory=list)
    monte_carlo: MonteCarloConfig = Field(default_factory=MonteCarloConfig)
    regulations: RegulationsConfig
    baseline_strategy: Optional[BaselineStrategyConfig] = None
    driver_offsets_s: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _cross_validate(self) -> "SimulatorSetup":
        for key, comp in self.compounds.items():
            if key.upper() != comp.compound_name:
                raise ValueError(f"compound key {key!r} does not match compound_name {comp.compound_name!r}")
        if not any(c.compound_type == "dry" for c in self.compounds.values()):
            raise ValueError("at least one dry compound must be configured")
        ps = self.pit_stop
        if ps.has_components:
            total = ps.pit_entry_loss_s + ps.pit_transit_loss_s + ps.pit_exit_loss_s  # type: ignore[operator]
            if abs(total - self.race.pit_lane_loss_s) > 1e-6:
                raise ValueError(
                    "pit entry + transit + exit components must sum to race.pit_lane_loss_s "
                    f"({total:.3f} != {self.race.pit_lane_loss_s:.3f})"
                )
        permitted = self.regulations.permitted_compounds
        if permitted is not None:
            unknown = [c for c in permitted if c not in self.compounds]
            if unknown:
                raise ValueError(f"permitted compounds without a tyre model: {unknown}")
        known = set(self.compounds)
        for rival in self.rivals:
            missing = [c for c in rival.compounds if c.upper() not in known]
            if missing:
                raise ValueError(f"rival {rival.rival_id} uses unknown compounds {missing}")
        if self.baseline_strategy is not None:
            missing = [c for c in self.baseline_strategy.compounds if c not in known]
            if missing:
                raise ValueError(f"baseline strategy uses unknown compounds {missing}")
        return self

    # Convenience accessors -------------------------------------------------
    @property
    def compound_names(self) -> list[str]:
        return list(self.compounds)

    @property
    def dry_compounds(self) -> list[str]:
        """Dry compounds allowed by the regulations (all dry ones if unconfigured)."""
        permitted = self.regulations.permitted_compounds
        return [
            name
            for name, c in self.compounds.items()
            if c.compound_type == "dry" and (permitted is None or name in permitted)
        ]

    @property
    def effective_min_stint_laps(self) -> int:
        reg = self.regulations.minimum_stint_laps
        return max(self.race.minimum_stint_laps, reg or 1)

    @property
    def effective_max_stint_laps(self) -> int:
        reg = self.regulations.maximum_stint_laps
        return min(self.race.maximum_stint_laps, reg if reg is not None else self.race.maximum_stint_laps)

    @property
    def refuelling_allowed(self) -> bool:
        """Refuelling is only permitted if the regulations explicitly allow it."""
        return bool(self.regulations.refuelling_allowed)


def _read_yaml(path: PathLike) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"configuration file {path} must contain a mapping at the top level")
    return data


def load_race_config(path: PathLike = DEFAULT_RACE_CONFIG) -> RaceConfig:
    """Load only the ``race`` section of a race YAML file."""
    return RaceConfig(**_read_yaml(path)["race"])


def load_regulations(path: PathLike = DEFAULT_REGULATIONS_CONFIG) -> RegulationsConfig:
    """Load a regulations YAML file."""
    data = _read_yaml(path)
    return RegulationsConfig(**data.get("regulations", data))


def build_setup(race_data: dict[str, Any], regulations: RegulationsConfig | dict[str, Any]) -> SimulatorSetup:
    """Validate a race-config mapping and regulations into a :class:`SimulatorSetup`."""
    data = dict(race_data)
    compounds_raw = data.pop("compounds", {})
    compounds: dict[str, Any] = {}
    for key, value in compounds_raw.items():
        value = dict(value)
        value.setdefault("compound_name", key)
        compounds[key.strip().upper()] = value
    if isinstance(regulations, dict):
        regulations = RegulationsConfig(**regulations.get("regulations", regulations))
    return SimulatorSetup(compounds=compounds, regulations=regulations, **data)


def load_setup(
    race_path: PathLike = DEFAULT_RACE_CONFIG,
    regulations_path: PathLike = DEFAULT_REGULATIONS_CONFIG,
) -> SimulatorSetup:
    """Load and validate the full simulator configuration."""
    setup = build_setup(_read_yaml(race_path), load_regulations(regulations_path))
    for warning in regulation_warnings(setup):
        logger.warning(warning)
    logger.info("Loaded configuration %s (%s)", setup.race.race_name, setup.race.data_label)
    return setup


def setup_to_dict(setup: SimulatorSetup) -> dict[str, Any]:
    """JSON-serialisable representation (used for reports and cache keys)."""
    return setup.model_dump(mode="json")


def with_overrides(setup: SimulatorSetup, **sections: dict[str, Any]) -> SimulatorSetup:
    """Return a re-validated copy with selected fields overridden.

    Example: ``with_overrides(setup, race={"total_laps": 50})``. Compound
    overrides are given per compound: ``compounds={"SOFT": {...}}``.
    """
    data = setup_to_dict(setup)
    for section, updates in sections.items():
        if section not in data:
            raise KeyError(f"unknown configuration section {section!r}")
        if section == "compounds":
            for name, comp_updates in updates.items():
                data["compounds"][name.upper()].update(comp_updates)
        elif isinstance(data[section], dict) and isinstance(updates, dict):
            data[section].update(updates)
        else:
            data[section] = updates
    regs = data.pop("regulations")
    return SimulatorSetup(regulations=RegulationsConfig(**regs), **data)


def with_baseline_lap_time(setup: SimulatorSetup, baseline_lap_time_s: float) -> SimulatorSetup:
    """Change the baseline pace, shifting every compound reference by the same delta.

    This keeps the relative compound offsets constant so the baseline is not
    applied twice (once as the race baseline and again in compound references).
    """
    delta = baseline_lap_time_s - setup.race.baseline_lap_time_s
    comp_updates = {
        name: {"reference_lap_time_s": c.reference_lap_time_s + delta} for name, c in setup.compounds.items()
    }
    return with_overrides(setup, race={"baseline_lap_time_s": baseline_lap_time_s}, compounds=comp_updates)


def regulation_warnings(setup: SimulatorSetup) -> list[str]:
    """Warnings for rules that are not configured and therefore disabled."""
    regs = setup.regulations
    warnings: list[str] = []
    checks = {
        "minimum_stint_laps": "minimum stint length (planning limit from race config still applies)",
        "maximum_stint_laps": "maximum stint length (planning limit from race config still applies)",
        "mandatory_pit_stops": "mandatory pit stops",
        "minimum_distinct_dry_compounds": "compound-usage rule",
        "permitted_compounds": "permitted-compound list (all configured compounds allowed)",
        "driver_rules": "driver-time constraints",
    }
    for attr, label in checks.items():
        if getattr(regs, attr) is None:
            warnings.append(f"Regulation not configured: {label} - constraint DISABLED.")
    if regs.refuelling_allowed is None:
        warnings.append("Refuelling rule not configured - refuelling is treated as NOT permitted.")
    if regs.refuelling_allowed and (regs.fuel_capacity_kg is None or regs.refuel_rate_kg_per_s is None):
        warnings.append("Refuelling allowed but fuel_capacity_kg / refuel_rate_kg_per_s missing.")
    if not setup.weather.wet_weather_enabled:
        warnings.append("Wet-weather model disabled: rain scenarios are ignored.")
    return warnings
