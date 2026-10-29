"""Deliberately simple, configurable weather model.

State per lap:

* ``track_temperature_c`` - affects tyre degradation rate only (through the
  tyre model's ``temperature_sensitivity``) to avoid double counting.
* ``wetness`` - dimensionless 0..1 index. Rain raises it by
  ``wetness_increase_per_lap`` up to the event's peak; without rain it decays
  by ``drying_rate_per_lap``.

Compound suitability is a quadratic mismatch penalty around each wet
compound's optimal wetness and a linear penalty for dry tyres. **These
parameters are uncalibrated illustrations.** They do not represent real tyre
behaviour; disable the wet model (``wet_weather_enabled: false``) when no
calibrated values exist.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from racesim.schemas import EventType, RaceEvent, WeatherConfig

logger = logging.getLogger(__name__)


@dataclass
class WeatherScenario:
    """Per-lap weather. Arrays are ``(n_laps,)`` or ``(n_sims, n_laps)``."""

    track_temperature_c: np.ndarray
    wetness: np.ndarray
    raining: np.ndarray
    description: str
    warnings: list[str] = field(default_factory=list)


def _temperature_rate(cfg: WeatherConfig) -> float:
    rate = abs(cfg.temperature_change_c_per_lap)
    return {"rising_temperature": rate, "cooling_temperature": -rate}.get(cfg.scenario, 0.0)


def update_track_conditions(
    track_temperature_c: np.ndarray | float,
    wetness: np.ndarray | float,
    raining: np.ndarray | bool,
    cfg: WeatherConfig,
    *,
    peak_wetness: np.ndarray | float | None = None,
    temperature_change_c_per_lap: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance track temperature and wetness by one lap."""
    raining = np.asarray(raining, dtype=bool)
    peak = cfg.peak_wetness if peak_wetness is None else np.asarray(peak_wetness)
    temp = np.asarray(track_temperature_c, dtype=float) + temperature_change_c_per_lap
    temp = temp - np.where(raining, cfg.rain_track_cooling_c_per_lap, 0.0)
    wet = np.asarray(wetness, dtype=float)
    wet = np.where(
        raining,
        np.maximum(wet, np.minimum(peak, wet + cfg.wetness_increase_per_lap)),
        np.maximum(0.0, wet - cfg.drying_rate_per_lap),
    )
    return temp, np.clip(wet, 0.0, 1.0)


def generate_weather_scenario(
    cfg: WeatherConfig,
    n_laps: int,
    start_track_temperature_c: float,
    *,
    rain_events: Optional[Sequence[RaceEvent]] = None,
    rng: Optional[np.random.Generator] = None,
    n_sims: Optional[int] = None,
) -> WeatherScenario:
    """Build a per-lap weather scenario.

    Rain comes from (in priority order) explicit ``RAIN`` events, the
    deterministic ``rain_onset`` scenario, or - when ``rng`` is supplied - a
    random onset with probability ``rain_probability_per_race`` per race.
    """
    shape = (n_sims, n_laps) if n_sims else (n_laps,)
    rows = n_sims or 1
    raining = np.zeros((rows, n_laps), dtype=bool)
    peak = np.full(rows, cfg.peak_wetness)
    warnings: list[str] = []
    desc = cfg.scenario

    events = [e for e in (rain_events or []) if e.event_type == EventType.RAIN]
    if events:
        for ev in events:
            raining[:, ev.start_lap - 1 : min(ev.end_lap, n_laps)] = True
            peak[:] = ev.severity
        desc = f"manual rain ({len(events)} event(s))"
    elif cfg.scenario == "rain_onset":
        onset = cfg.rain_onset_lap or max(2, n_laps // 2)
        raining[:, onset - 1 : min(onset - 1 + cfg.rain_duration_laps, n_laps)] = True
        desc = f"rain onset lap {onset}"
    elif rng is not None and cfg.rain_probability_per_race > 0 and n_laps > 6:
        has_rain = rng.random(rows) < cfg.rain_probability_per_race
        onset = rng.integers(2, n_laps - 4, size=rows)
        peak = np.where(has_rain, rng.uniform(0.3, 1.0, size=rows) * cfg.peak_wetness, peak)
        idx = np.arange(n_laps)[None, :]
        raining = has_rain[:, None] & (idx >= onset[:, None] - 1) & (idx < onset[:, None] - 1 + cfg.rain_duration_laps)
        desc = f"random rain (p={cfg.rain_probability_per_race:.2f}/race)"

    if raining.any() and not cfg.wet_weather_enabled:
        warnings.append("Rain requested but the wet-weather model is disabled; wetness forced to 0.")
        logger.warning(warnings[-1])
        raining[:] = False

    temp = np.empty((rows, n_laps))
    wet = np.empty((rows, n_laps))
    t, w = np.full(rows, start_track_temperature_c, dtype=float), np.zeros(rows)
    rate = _temperature_rate(cfg)
    for lap in range(n_laps):
        if lap > 0:
            t, _ = update_track_conditions(t, w, raining[:, lap - 1], cfg, temperature_change_c_per_lap=rate)
        _, w = update_track_conditions(t, w, raining[:, lap], cfg, peak_wetness=peak)
        temp[:, lap], wet[:, lap] = t, w
    if not n_sims:
        temp, wet, raining = temp[0], wet[0], raining[0]
    return WeatherScenario(temp.reshape(shape), wet.reshape(shape), raining.reshape(shape), desc, warnings)


def calculate_weather_pace_effect(wetness: np.ndarray | float, compound_type_code: np.ndarray | int, cfg: WeatherConfig) -> np.ndarray:
    """Lap-time penalty (s) from track wetness for a compound type.

    type 0 (dry): ``dry_tyre_wet_penalty_s * wetness``;
    type 1/2 (intermediate/wet): ``mismatch * ((wetness - optimum) / 0.5)**2``.
    Returns zeros when the wet model is disabled.
    """
    w = np.asarray(wetness, dtype=float)
    t = np.asarray(compound_type_code)
    if not cfg.wet_weather_enabled:
        return np.zeros(np.broadcast(w, t).shape)
    dry = cfg.dry_tyre_wet_penalty_s * w
    inter = cfg.intermediate_mismatch_penalty_s * ((w - cfg.intermediate_optimal_wetness) / 0.5) ** 2
    wet = cfg.wet_mismatch_penalty_s * ((w - cfg.wet_optimal_wetness) / 0.5) ** 2
    return np.select([t == 0, t == 1], [dry, inter], default=wet)


def estimate_wet_tyre_suitability(
    cfg: WeatherConfig,
    reference_lap_times_s: dict[str, tuple[float, int]],
    wetness_levels: Sequence[float] = tuple(np.round(np.linspace(0, 1, 11), 2)),
) -> pd.DataFrame:
    """Predicted fresh-tyre lap time per compound across wetness levels.

    ``reference_lap_times_s`` maps compound name -> (reference lap time, type code).
    The ``best_compound`` column marks the fastest compound at each wetness.
    """
    rows = []
    for w in wetness_levels:
        best_name, best_time = None, np.inf
        for name, (ref, code) in reference_lap_times_s.items():
            t = float(ref + calculate_weather_pace_effect(w, code, cfg))
            rows.append({"wetness": float(w), "compound": name, "predicted_lap_time_s": t})
            if t < best_time:
                best_name, best_time = name, t
        for r in rows[-len(reference_lap_times_s):]:
            r["best_compound"] = best_name
    return pd.DataFrame(rows)
