"""Risk metrics for race-time distributions.

Direction conventions
---------------------
* Race times, percentiles, standard deviation and CVaR: **lower is better**.
* Probabilities of beating a baseline or rival: **higher is better**.
* Expected finishing position: **lower is better**.

CVaR definition
---------------
``CVaR_95`` is the mean of the worst 5 % of race-time outcomes, i.e. the mean
of the ``k = ceil(0.05 * n)`` largest race times. With ``n = 100`` it is the
mean of the 5 slowest races.

Ties
----
When two outcomes are equal within ``tie_tolerance_s`` each side receives
half a win, so ``P(A beats B) + P(B beats A) = 1``.
"""

from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from racesim.schemas import RiskReport

DEFAULT_PERCENTILES = (5, 50, 90, 95)


def _arr(x) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.size == 0:
        raise ValueError("empty race-time sample")
    if not np.all(np.isfinite(a)):
        raise ValueError("race-time sample contains non-finite values")
    return a


def calculate_expected_race_time(race_times_s: Sequence[float]) -> float:
    """Mean race time (s). Lower is better."""
    return float(_arr(race_times_s).mean())


def calculate_race_time_percentiles(race_times_s: Sequence[float], percentiles: Sequence[float] = DEFAULT_PERCENTILES) -> dict[float, float]:
    """Race-time percentiles (s) using linear interpolation."""
    a = _arr(race_times_s)
    return {p: float(np.percentile(a, p)) for p in percentiles}


def calculate_race_time_std(race_times_s: Sequence[float]) -> float:
    """Sample standard deviation (ddof = 1) in seconds; 0 for a single sample."""
    a = _arr(race_times_s)
    return float(a.std(ddof=1)) if a.size > 1 else 0.0


def calculate_cvar(race_times_s: Sequence[float], alpha: float = 0.95) -> float:
    """Mean of the worst ``(1 - alpha)`` fraction of race times (s).

    ``k = ceil((1 - alpha) * n)`` largest values are averaged (at least one).
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    a = np.sort(_arr(race_times_s))
    k = max(1, math.ceil(round((1 - alpha) * a.size, 9)))
    return float(a[-k:].mean())


def _p_beat(a: np.ndarray, b: np.ndarray, tol: float) -> float:
    d = a - b
    return float(np.mean((d < -tol) + 0.5 * (np.abs(d) <= tol)))


def calculate_probability_of_beating_baseline(race_times_s: Sequence[float], baseline_times_s: Sequence[float], *,
                                               tie_tolerance_s: float = 1e-6) -> float:
    """Paired probability that the strategy is faster than the baseline (ties = 1/2)."""
    a, b = _arr(race_times_s), _arr(baseline_times_s)
    if a.shape != b.shape:
        raise ValueError("strategy and baseline samples must be paired (same length)")
    return _p_beat(a, b, tie_tolerance_s)


def calculate_probability_of_beating_rival(own_finish_s: Sequence[float], rival_finish_s: Optional[Sequence[float]], *,
                                           tie_tolerance_s: float = 1e-6) -> Optional[float]:
    """Probability of finishing ahead of a rival; ``None`` when no rival outcomes exist."""
    if rival_finish_s is None:
        return None
    return _p_beat(_arr(own_finish_s), _arr(rival_finish_s), tie_tolerance_s)


def calculate_strategy_risk_report(
    race_times_s: Sequence[float],
    *,
    strategy_id: Optional[str] = None,
    baseline_times_s: Optional[Sequence[float]] = None,
    own_finish_s: Optional[Sequence[float]] = None,
    rival_finish_s: Optional[Mapping[str, Sequence[float]] | Sequence[float]] = None,
    finishing_positions: Optional[Sequence[float]] = None,
    min_tail_samples: int = 50,
    tie_tolerance_s: float = 1e-6,
) -> RiskReport:
    """Complete risk report for one strategy.

    The rival probability is reported only if rival outcomes are supplied;
    for several rivals it is the probability of finishing ahead of **all**
    of them. Nothing is substituted for missing analyses.
    """
    a = _arr(race_times_s)
    pct = calculate_race_time_percentiles(a)
    warnings: list[str] = []
    tail_n = math.ceil(0.05 * a.size)
    if tail_n < min_tail_samples:
        warnings.append(f"Only {tail_n} sample(s) in the 5% tail (n={a.size}); CVaR and P95 are unstable. "
                        f"Use at least {min_tail_samples * 20} simulations for a stable tail estimate.")
    p_base = None
    if baseline_times_s is not None:
        p_base = calculate_probability_of_beating_baseline(a, baseline_times_s, tie_tolerance_s=tie_tolerance_s)
    p_rival = None
    if rival_finish_s is not None:
        own = _arr(own_finish_s if own_finish_s is not None else a)
        if isinstance(rival_finish_s, Mapping):
            if rival_finish_s:
                ahead_all = np.ones(own.size, dtype=float)
                for r in rival_finish_s.values():
                    d = own - _arr(r)
                    ahead_all *= (d < -tie_tolerance_s) + 0.5 * (np.abs(d) <= tie_tolerance_s)
                p_rival = float(ahead_all.mean())
        else:
            p_rival = calculate_probability_of_beating_rival(own, rival_finish_s, tie_tolerance_s=tie_tolerance_s)
    return RiskReport(
        strategy_id=strategy_id,
        expected_race_time_s=float(a.mean()),
        median_race_time_s=float(np.median(a)),
        standard_deviation_s=calculate_race_time_std(a),
        percentile_05_s=pct[5], percentile_50_s=pct[50], percentile_90_s=pct[90], percentile_95_s=pct[95],
        cvar_95_s=calculate_cvar(a, 0.95),
        probability_of_beating_baseline=p_base,
        probability_of_beating_rival=p_rival,
        expected_finishing_position=float(np.mean(finishing_positions)) if finishing_positions is not None else None,
        number_of_simulations=int(a.size),
        warnings=warnings,
    )


def risk_reports_from_monte_carlo(result, baseline_id: Optional[str] = None) -> dict[str, RiskReport]:
    """Risk report for every strategy in a :class:`MonteCarloResult`."""
    out = {}
    base = result.race_times.get(baseline_id) if baseline_id else None
    min_tail = 50
    for sid, t in result.race_times.items():
        out[sid] = calculate_strategy_risk_report(
            t, strategy_id=sid,
            baseline_times_s=base if (base is not None and sid != baseline_id) else None,
            own_finish_s=result.own_finish_elapsed[sid] if result.own_finish_elapsed else None,
            rival_finish_s=result.rival_finish_elapsed[sid] if result.rival_finish_elapsed else None,
            finishing_positions=result.positions[sid] if result.positions else None,
            min_tail_samples=min_tail,
        )
    return out


OBJECTIVES = {
    "Best expected race time": ("expected_race_time_s", "min"),
    "Lowest variability (std)": ("standard_deviation_s", "min"),
    "Lowest downside risk (CVaR 95)": ("cvar_95_s", "min"),
    "Highest P(beat baseline)": ("probability_of_beating_baseline", "max"),
    "Highest P(beat all rivals)": ("probability_of_beating_rival", "max"),
    "Best expected finishing position": ("expected_finishing_position", "min"),
}


def compare_objectives(reports: Mapping[str, RiskReport], names: Optional[Mapping[str, str]] = None) -> pd.DataFrame:
    """Which strategy wins under each objective (objectives may disagree).

    Objectives whose metric is unavailable for every strategy are reported as
    unavailable rather than filled in.
    """
    rows = []
    for label, (attr, sense) in OBJECTIVES.items():
        vals = {sid: getattr(r, attr) for sid, r in reports.items() if getattr(r, attr) is not None}
        if not vals:
            rows.append({"objective": label, "best_strategy": "unavailable", "value": None, "direction": sense})
            continue
        sid = (min if sense == "min" else max)(vals, key=vals.get)
        rows.append({"objective": label, "best_strategy": (names or {}).get(sid, sid), "strategy_id": sid,
                     "value": vals[sid], "direction": "lower is better" if sense == "min" else "higher is better"})
    return pd.DataFrame(rows)
