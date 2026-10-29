"""Uncertainty diagnostics: simulation precision versus model uncertainty.

Two different questions are kept apart throughout:

* **Simulation precision** - how much a Monte Carlo estimate would change if
  the simulation were repeated with another seed (standard errors, confidence
  intervals, convergence, tail stability). It shrinks as runs are added.
* **Model uncertainty** - how wrong the model itself may be (uncalibrated
  parameters, simplifications, sensitivity of the recommendation to
  assumptions, validation status). **It does not shrink with more runs.** A
  narrow confidence interval only means the *model's* answer is estimated
  precisely, not that the real race will behave that way.

Probability intervals use the Wilson score interval. CVaR intervals use a
percentile bootstrap. All resampling uses a fixed, explicit seed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from racesim.risk_metrics import calculate_cvar

Z95 = 1.959963984540054


def wilson_interval(p_hat: float, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a proportion (handles p = 0 or 1)."""
    if n <= 0:
        raise ValueError("n must be positive")
    denom = 1 + z * z / n
    centre = (p_hat + z * z / (2 * n)) / denom
    half = z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def mean_confidence_interval(x: Sequence[float], z: float = Z95) -> tuple[float, float, float]:
    """(mean, standard error, half-width) of the sample mean."""
    a = np.asarray(x, float)
    se = float(a.std(ddof=1) / math.sqrt(a.size)) if a.size > 1 else float("nan")
    return float(a.mean()), se, z * se


def bootstrap_interval(x: Sequence[float], statistic=calculate_cvar, *, n_boot: int = 400, seed: int = 0,
                       level: float = 0.95) -> tuple[float, float, float]:
    """(bootstrap standard error, lower, upper) of ``statistic`` by resampling runs."""
    a = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    stats = np.array([statistic(a[rng.integers(0, a.size, a.size)]) for _ in range(n_boot)])
    lo, hi = np.quantile(stats, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(stats.std(ddof=1)), float(lo), float(hi)


def paired_probability(a: Sequence[float], b: Sequence[float], tol: float = 1e-6) -> float:
    d = np.asarray(a, float) - np.asarray(b, float)
    return float(np.mean((d < -tol) + 0.5 * (np.abs(d) <= tol)))


@dataclass
class TailStability:
    cvar95_s: float
    bootstrap_se_s: float
    ci95_s: tuple[float, float]
    split_half_difference_s: float
    tail_samples: int
    stable: bool
    reasons: list[str] = field(default_factory=list)


def assess_tail_stability(x: Sequence[float], *, tolerance_s: Optional[float] = None, min_tail_samples: int = 50,
                          n_boot: int = 400, seed: int = 0) -> TailStability:
    """Is the CVaR95 estimate stable?

    Stable requires at least ``min_tail_samples`` runs in the 5 % tail, a
    bootstrap 95 % CI half-width within ``tolerance_s`` and the two halves of
    the sample agreeing within ``2 * tolerance_s``. The default tolerance is
    10 % of the race-time standard deviation, because tail estimates of a
    wide distribution cannot be pinned to an absolute second.
    """
    a = np.asarray(x, float)
    if tolerance_s is None:
        tolerance_s = max(0.1 * float(a.std(ddof=1)), 0.05) if a.size > 1 else 0.05
    tail = math.ceil(0.05 * a.size)
    se, lo, hi = bootstrap_interval(a, n_boot=n_boot, seed=seed)
    half = a.size // 2
    split = abs(calculate_cvar(a[:half]) - calculate_cvar(a[half:])) if half >= 20 else float("nan")
    reasons = []
    if tail < min_tail_samples:
        reasons.append(f"only {tail} runs in the 5% tail (< {min_tail_samples})")
    if (hi - lo) / 2 > tolerance_s:
        reasons.append(f"bootstrap CI half-width {(hi - lo) / 2:.2f} s > {tolerance_s} s")
    if not math.isnan(split) and split > 2 * tolerance_s:
        reasons.append(f"split-half CVaR difference {split:.2f} s > {2 * tolerance_s} s")
    return TailStability(calculate_cvar(a), se, (lo, hi), split, tail, not reasons, reasons)


def convergence_diagnostics(x: Sequence[float], baseline: Optional[Sequence[float]] = None, *, n_points: int = 20,
                            min_n: int = 50) -> pd.DataFrame:
    """Running estimates as the number of runs grows (in the simulated order).

    Columns: ``n``, running mean with its 95 % CI, running CVaR95 and, when a
    paired baseline is given, running P(beat baseline) with its Wilson CI.
    Flat curves with narrowing intervals indicate convergence of the
    *simulation*; they say nothing about model accuracy.
    """
    a = np.asarray(x, float)
    b = None if baseline is None else np.asarray(baseline, float)
    ns = np.unique(np.linspace(min(min_n, a.size), a.size, n_points).astype(int))
    rows = []
    for n in ns:
        m, se, hw = mean_confidence_interval(a[:n])
        row = {"n": int(n), "mean_s": m, "mean_ci_low_s": m - hw, "mean_ci_high_s": m + hw, "cvar95_s": calculate_cvar(a[:n])}
        if b is not None:
            p = paired_probability(a[:n], b[:n])
            lo, hi = wilson_interval(p, int(n))
            dm, _, dhw = mean_confidence_interval(a[:n] - b[:n])
            row.update(p_beat_baseline=p, p_ci_low=lo, p_ci_high=hi, paired_delta_s=dm,
                       paired_delta_ci_low_s=dm - dhw, paired_delta_ci_high_s=dm + dhw)
        rows.append(row)
    return pd.DataFrame(rows)


def convergence_summary(diag: pd.DataFrame, *, mean_tolerance_s: float = 5.0, probability_tolerance: float = 0.02,
                        cvar_tolerance_s: float = 10.0, paired_delta_tolerance_s: float = 0.5) -> dict:
    """Pass/fail of explicit convergence criteria on the final estimates.

    * mean race time: 95 % CI half-width <= ``mean_tolerance_s``
    * CVaR95: range over the last 25 % of runs <= ``cvar_tolerance_s``
    * P(beat baseline): Wilson CI half-width <= ``probability_tolerance``
    * paired difference to baseline: 95 % CI half-width <= ``paired_delta_tolerance_s``

    Absolute race times are dominated by safety cars that hit every strategy,
    so the paired criteria are the decision-relevant ones. Criteria are
    simulation-precision checks only.
    """
    last = diag.iloc[-1]
    q = diag[diag["n"] >= 0.75 * last["n"]]
    out = {"n": int(last["n"]), "mean_ci_half_width_s": float((last["mean_ci_high_s"] - last["mean_ci_low_s"]) / 2),
           "cvar_change_last_quarter_s": float(q["cvar95_s"].max() - q["cvar95_s"].min())}
    checks = {"mean": out["mean_ci_half_width_s"] <= mean_tolerance_s,
              "cvar": out["cvar_change_last_quarter_s"] <= cvar_tolerance_s}
    if "p_beat_baseline" in diag.columns:
        out["probability_ci_half_width"] = float((last["p_ci_high"] - last["p_ci_low"]) / 2)
        out["paired_delta_ci_half_width_s"] = float((last["paired_delta_ci_high_s"] - last["paired_delta_ci_low_s"]) / 2)
        checks["probability"] = out["probability_ci_half_width"] <= probability_tolerance
        checks["paired_delta"] = out["paired_delta_ci_half_width_s"] <= paired_delta_tolerance_s
    out["checks"] = checks
    out["converged"] = all(checks.values())
    out["criteria"] = {"mean_tolerance_s": mean_tolerance_s, "probability_tolerance": probability_tolerance,
                       "cvar_tolerance_s": cvar_tolerance_s, "paired_delta_tolerance_s": paired_delta_tolerance_s}
    return out


def simulation_precision_table(result, baseline_id: Optional[str] = None, *, n_boot: int = 300, seed: int = 0) -> pd.DataFrame:
    """Monte Carlo precision of every headline estimate in a :class:`MonteCarloResult`."""
    rows = []
    base = result.race_times.get(baseline_id) if baseline_id else None
    for sid, t in result.race_times.items():
        m, se, hw = mean_confidence_interval(t)
        cse, clo, chi = bootstrap_interval(t, n_boot=n_boot, seed=seed)
        row = {"strategy_id": sid, "strategy": result.strategies[sid].strategy_name, "n": int(t.size), "mean_s": m,
               "mean_se_s": se, "mean_ci95_s": f"[{m - hw:,.2f}, {m + hw:,.2f}]", "cvar95_s": calculate_cvar(t),
               "cvar95_bootstrap_se_s": cse, "cvar95_ci95_s": f"[{clo:,.2f}, {chi:,.2f}]"}
        if base is not None and sid != baseline_id:
            p = paired_probability(t, base)
            lo, hi = wilson_interval(p, t.size)
            d = t - base
            dm, dse, dhw = mean_confidence_interval(d)
            row.update(p_beat_baseline=p, p_beat_baseline_ci95=f"[{lo:.3f}, {hi:.3f}]", mean_paired_delta_s=dm,
                       paired_delta_ci95_s=f"[{dm - dhw:+.2f}, {dm + dhw:+.2f}]")
        if result.rival_finish_elapsed:
            own = result.own_finish_elapsed[sid]
            ahead = np.all([own < r for r in result.rival_finish_elapsed[sid].values()], axis=0)
            p = float(ahead.mean())
            lo, hi = wilson_interval(p, own.size)
            row.update(p_beat_all_rivals=p, p_beat_all_rivals_ci95=f"[{lo:.3f}, {hi:.3f}]")
        rows.append(row)
    return pd.DataFrame(rows)


SOURCES = {
    "lap-to-lap variability": {"lap_variability": True},
    "tyre/pace/fuel parameter uncertainty": {"parameter": True},
    "pit-stop variability": {"pit_variability": True},
    "race events (SC/VSC/rain) + traffic": {"events": True, "traffic": True},
    "rival behaviour": {"rival_behaviour": True},
}


def uncertainty_breakdown(setup, strategy, baseline=None, *, n_sims: int = 1000, seed: Optional[int] = None,
                          include_rivals: bool = False) -> pd.DataFrame:
    """Each uncertainty source enabled alone, plus all together.

    Reports the spread of the absolute race time *and* of the paired
    difference to the baseline. Sources such as safety cars widen absolute
    race times for every strategy alike, so their effect on the *decision*
    (paired difference) can be much smaller. Rival behaviour only affects
    rival outcomes (P(beat rivals)); it is skipped without rivals.
    Sources interact, so variances need not add up.
    """
    from racesim.monte_carlo import DEFAULT_FLAGS, run_monte_carlo

    off = {k: False for k in DEFAULT_FLAGS}
    strategies = [strategy] + ([baseline] if baseline is not None and baseline.strategy_id != strategy.strategy_id else [])
    rows = []
    items = list(SOURCES.items()) + [("all sources combined", {k: True for k in DEFAULT_FLAGS if k != "reactive"})]
    for name, on in items:
        if name == "rival behaviour" and not include_rivals:
            continue
        flags = off | on | {"reactive": bool(on.get("events")) and setup.monte_carlo.reactive_strategy}
        r = run_monte_carlo(setup, strategies, n_sims=n_sims, seed=seed, flags=flags, include_rivals=include_rivals)
        t = r.race_times[strategy.strategy_id]
        row = {"source": name, "race_time_std_s": float(t.std(ddof=1))}
        if len(strategies) > 1:
            d = t - r.race_times[baseline.strategy_id]
            row.update(paired_delta_std_s=float(d.std(ddof=1)), mean_paired_delta_s=float(d.mean()),
                       p_beat_baseline=paired_probability(t, r.race_times[baseline.strategy_id]))
        if include_rivals and r.rival_finish_elapsed:
            own = r.own_finish_elapsed[strategy.strategy_id]
            row["p_beat_all_rivals"] = float(np.all([own < x for x in r.rival_finish_elapsed[strategy.strategy_id].values()], axis=0).mean())
        rows.append(row)
    return pd.DataFrame(rows)


MODEL_UNCERTAINTY_STATEMENT = (
    "Monte Carlo confidence intervals measure simulation precision only. They narrow as runs are added but do not "
    "reduce model uncertainty: parameter values, the additive lap-time model, the event hazard and the rival/traffic "
    "simplifications may all differ from reality. A large number of simulations does not make the prediction "
    "accurate for a real race."
)


def model_uncertainty_summary(setup, *, sensitivity_ranking: Optional[pd.DataFrame] = None,
                              validation_status: Optional[str] = None, breakdown: Optional[pd.DataFrame] = None) -> list[str]:
    """Plain statements about model (not simulation) uncertainty, built from available results only."""
    out = [MODEL_UNCERTAINTY_STATEMENT]
    sources = {c.parameter_source for c in setup.compounds.values()}
    out.append(f"Tyre parameters source: {', '.join(sorted(sources))}. Data label: {setup.race.data_label}.")
    out.append(f"Weather model: {setup.weather.calibration_status}.")
    out.append("Validation: " + (validation_status or "not run in this analysis") + ".")
    if sensitivity_ranking is not None and not sensitivity_ranking.empty:
        top = sensitivity_ranking.iloc[0]
        out.append(f"Most influential assumption tested: {top['parameter']} ({int(top['recommendation_changes'])} recommendation "
                   f"change(s) across {int(top['values_tested'])} values).")
    else:
        out.append("Sensitivity analysis: not run in this analysis.")
    if breakdown is not None and not breakdown.empty and "paired_delta_std_s" in breakdown.columns:
        b = breakdown[breakdown["source"] != "all sources combined"].sort_values("paired_delta_std_s", ascending=False)
        if not b.empty:
            out.append(f"Largest source of decision-relevant spread (paired vs baseline): {b.iloc[0]['source']} "
                       f"({b.iloc[0]['paired_delta_std_s']:.2f} s std).")
    return out
