"""Sensitivity analysis: how model assumptions change the recommendation.

Sensitivity answers "what if the assumption were different?" - it is **not**
a statistical confidence interval. Each scenario re-runs the deterministic
optimiser with a modified assumption (and optionally a Monte Carlo
evaluation) and records the optimal pit lap, number of stops, predicted race
time, probability of outperforming the baseline and tail risk.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from racesim.config import SimulatorSetup, with_overrides
from racesim.monte_carlo import run_monte_carlo
from racesim.risk_metrics import calculate_cvar, calculate_probability_of_beating_baseline
from racesim.schemas import Strategy
from racesim.strategy_optimizer import find_optimal_strategy

logger = logging.getLogger(__name__)


@dataclass
class SensitivityResult:
    parameter: str
    unit: str
    base_value: float
    table: pd.DataFrame

    @property
    def recommendation_changes(self) -> int:
        ids = self.table["recommended_strategy_id"].tolist()
        return int(sum(a != b for a, b in zip(ids, ids[1:])))


def _scale_degradation(setup: SimulatorSetup, m: float) -> SimulatorSetup:
    comp = {name: {"linear_degradation_s_per_lap": c.linear_degradation_s_per_lap * m,
                   "nonlinear_degradation_coefficient": c.nonlinear_degradation_coefficient * m}
            for name, c in setup.compounds.items()}
    return with_overrides(setup, compounds=comp)


def _set_pit_loss(setup: SimulatorSetup, value: float) -> SimulatorSetup:
    ps = setup.pit_stop
    pit_updates = {}
    if ps.has_components and setup.race.pit_lane_loss_s > 0:
        f = value / setup.race.pit_lane_loss_s
        pit_updates = {"pit_entry_loss_s": ps.pit_entry_loss_s * f, "pit_transit_loss_s": ps.pit_transit_loss_s * f,
                       "pit_exit_loss_s": ps.pit_exit_loss_s * f}
    return with_overrides(setup, race={"pit_lane_loss_s": value}, pit_stop=pit_updates)


def _set_sc(setup: SimulatorSetup, value: float) -> SimulatorSetup:
    return with_overrides(setup, race={"safety_car_probability_per_lap": value})


def _set_lap_std(setup: SimulatorSetup, value: float) -> SimulatorSetup:
    return with_overrides(setup, race={"baseline_lap_time_std_s": value})


def _scale_deg_uncertainty(setup: SimulatorSetup, m: float) -> SimulatorSetup:
    comp = {name: {"degradation_uncertainty": c.degradation_uncertainty * m} for name, c in setup.compounds.items()}
    return with_overrides(setup, compounds=comp)


def _set_pit_std(setup: SimulatorSetup, value: float) -> SimulatorSetup:
    return with_overrides(setup, race={"pit_loss_std_s": value})


PARAMETERS: dict[str, tuple[Callable[[SimulatorSetup, float], SimulatorSetup], str, Callable[[SimulatorSetup], float]]] = {
    "tyre_degradation_multiplier": (_scale_degradation, "x", lambda s: 1.0),
    "pit_lane_loss_s": (_set_pit_loss, "s", lambda s: s.race.pit_lane_loss_s),
    "safety_car_probability_per_lap": (_set_sc, "per lap", lambda s: s.race.safety_car_probability_per_lap),
    "baseline_lap_time_std_s": (_set_lap_std, "s", lambda s: s.race.baseline_lap_time_std_s),
    "degradation_uncertainty_scale": (_scale_deg_uncertainty, "x", lambda s: 1.0),
    "pit_loss_std_s": (_set_pit_std, "s", lambda s: s.race.pit_loss_std_s),
}

# Parameters that only change random variation: the deterministic recommendation cannot change,
# so they are evaluated through Monte Carlo metrics.
UNCERTAINTY_PARAMETERS = ("safety_car_probability_per_lap", "baseline_lap_time_std_s", "degradation_uncertainty_scale",
                          "pit_loss_std_s")


def _baseline(setup: SimulatorSetup) -> Optional[Strategy]:
    b = setup.baseline_strategy
    return Strategy.from_plan(setup.race.total_laps, b.pit_laps, b.compounds) if b else None


def calculate_strategy_sensitivity(
    setup: SimulatorSetup,
    parameter: str,
    values: Sequence[float],
    *,
    monte_carlo: bool = False,
    n_sims: int = 500,
    seed: Optional[int] = None,
    max_stops: Optional[int] = None,
) -> SensitivityResult:
    """Re-optimise for each value of ``parameter``.

    Columns: value, recommended strategy, number of stops, first pit lap,
    predicted race time and (with ``monte_carlo``) the recommended strategy's
    mean race time, probability of beating the baseline and CVaR95.
    """
    if parameter not in PARAMETERS:
        raise KeyError(f"unknown sensitivity parameter {parameter!r}; choose from {list(PARAMETERS)}")
    apply, unit, base_fn = PARAMETERS[parameter]
    base_strategy = _baseline(setup)
    rows = []
    for v in values:
        s = apply(setup, float(v))
        opt = find_optimal_strategy(s, max_stops=max_stops)
        row = {"value": float(v), "recommended_strategy": None, "recommended_strategy_id": None, "stops": None,
               "first_pit_lap": None, "predicted_race_time_s": None}
        if opt.best is not None:
            st = opt.best.strategy
            row.update(recommended_strategy=st.strategy_name, recommended_strategy_id=st.strategy_id,
                       stops=st.number_of_stops, first_pit_lap=st.planned_pit_laps[0] if st.planned_pit_laps else None,
                       predicted_race_time_s=opt.best.predicted_race_time_s)
            if opt.baseline is not None:
                row["deterministic_gain_vs_baseline_s"] = opt.baseline.predicted_race_time_s - opt.best.predicted_race_time_s
            if monte_carlo:
                strategies = [st] + ([base_strategy] if base_strategy and base_strategy.strategy_id != st.strategy_id else [])
                mc = run_monte_carlo(s, strategies, n_sims=n_sims, seed=seed)
                t = mc.race_times[st.strategy_id]
                row["mc_mean_race_time_s"] = float(t.mean())
                row["mc_cvar95_s"] = calculate_cvar(t)
                if base_strategy is not None and base_strategy.strategy_id != st.strategy_id:
                    row["mc_p_beat_baseline"] = calculate_probability_of_beating_baseline(t, mc.race_times[base_strategy.strategy_id])
        rows.append(row)
    table = pd.DataFrame(rows)
    if not table.empty:
        ref = table["recommended_strategy_id"].iloc[int(np.argmin(np.abs(table["value"] - base_fn(setup))))]
        table["recommendation_changed"] = table["recommended_strategy_id"] != ref
    logger.info("Sensitivity %s: %d values", parameter, len(values))
    return SensitivityResult(parameter, unit, base_fn(setup), table)


def vary_tyre_degradation(setup: SimulatorSetup, multipliers: Sequence[float] = (0.6, 0.8, 1.0, 1.2, 1.4, 1.6), **kw) -> SensitivityResult:
    """Scale every compound's linear and nonlinear degradation coefficients."""
    return calculate_strategy_sensitivity(setup, "tyre_degradation_multiplier", multipliers, **kw)


def vary_pit_loss(setup: SimulatorSetup, values_s: Optional[Sequence[float]] = None, **kw) -> SensitivityResult:
    """Vary the green-flag pit-lane loss (s, excluding stationary time)."""
    base = setup.race.pit_lane_loss_s
    values_s = values_s or [round(base * f, 2) for f in (0.6, 0.8, 1.0, 1.2, 1.4)]
    return calculate_strategy_sensitivity(setup, "pit_lane_loss_s", values_s, **kw)


def vary_safety_car_probability(setup: SimulatorSetup, values: Sequence[float] = (0.0, 0.006, 0.012, 0.024, 0.04), **kw) -> SensitivityResult:
    """Vary the per-lap SC probability (affects Monte Carlo metrics only)."""
    kw.setdefault("monte_carlo", True)
    return calculate_strategy_sensitivity(setup, "safety_car_probability_per_lap", values, **kw)


def vary_lap_time_variability(setup: SimulatorSetup, values_s: Sequence[float] = (0.0, 0.2, 0.35, 0.6, 1.0), **kw) -> SensitivityResult:
    """Vary lap-to-lap variability (affects Monte Carlo metrics only)."""
    kw.setdefault("monte_carlo", True)
    return calculate_strategy_sensitivity(setup, "baseline_lap_time_std_s", values_s, **kw)


def rank_sensitive_parameters(results: Sequence[SensitivityResult]) -> pd.DataFrame:
    """Rank parameters by recommendation changes, then by spread of outcomes."""
    rows = []
    for r in results:
        t = r.table
        rt = t["predicted_race_time_s"].dropna()
        laps = t["first_pit_lap"].dropna()
        rows.append({
            "parameter": r.parameter,
            "values_tested": len(t),
            "recommendation_changes": r.recommendation_changes,
            "distinct_recommendations": int(t["recommended_strategy_id"].nunique()),
            "stops_range": f"{int(t['stops'].min())}-{int(t['stops'].max())}" if t["stops"].notna().any() else "n/a",
            "first_pit_lap_range_laps": float(laps.max() - laps.min()) if len(laps) else None,
            "race_time_range_s": float(rt.max() - rt.min()) if len(rt) else None,
            "mc_cvar_range_s": float(t["mc_cvar95_s"].max() - t["mc_cvar95_s"].min()) if "mc_cvar95_s" in t else None,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["recommendation_changes", "distinct_recommendations", "first_pit_lap_range_laps"],
                          ascending=False).reset_index(drop=True)


def vary_uncertain_parameters(setup: SimulatorSetup, strategy: Strategy, baseline: Strategy, *, n_sims: int = 500,
                              seed: Optional[int] = None,
                              ranges: Optional[dict[str, Sequence[float]]] = None) -> pd.DataFrame:
    """Sensitivity of risk metrics to the *size* of each uncertainty source.

    For each parameter value the same two strategies are compared under common
    random numbers. Columns: P(beat baseline), mean paired difference, CVaR95
    of the strategy and the CVaR difference to the baseline.
    """
    from racesim.uncertainty import paired_probability

    ranges = ranges or {
        "degradation_uncertainty_scale": (0.0, 1.0, 2.0, 3.0),
        "pit_loss_std_s": (0.0, setup.race.pit_loss_std_s, 2 * setup.race.pit_loss_std_s, 4 * setup.race.pit_loss_std_s),
        "safety_car_probability_per_lap": (0.0, setup.race.safety_car_probability_per_lap, 2 * setup.race.safety_car_probability_per_lap),
        "baseline_lap_time_std_s": (0.0, setup.race.baseline_lap_time_std_s, 2 * setup.race.baseline_lap_time_std_s),
    }
    rows = []
    for param, values in ranges.items():
        apply, unit, _ = PARAMETERS[param]
        for v in values:
            mc = run_monte_carlo(apply(setup, float(v)), [strategy, baseline], n_sims=n_sims, seed=seed)
            a, b = mc.race_times[strategy.strategy_id], mc.race_times[baseline.strategy_id]
            rows.append({"parameter": param, "unit": unit, "value": float(v), "p_beat_baseline": paired_probability(a, b),
                         "mean_paired_delta_s": float((a - b).mean()), "cvar95_s": calculate_cvar(a),
                         "cvar95_delta_vs_baseline_s": calculate_cvar(a) - calculate_cvar(b)})
    return pd.DataFrame(rows)
