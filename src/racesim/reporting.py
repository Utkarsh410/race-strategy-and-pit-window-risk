"""HTML report and CSV summary generation.

Reports only contain values that were actually calculated. Sections whose
analysis was not run say so explicitly. Plotly is embedded inline once so the
report opens offline.
"""

from __future__ import annotations

import datetime as _dt
import html
import json
import logging
from pathlib import Path
from typing import Iterable, Mapping, Optional

import pandas as pd

from racesim import __version__
from racesim.config import setup_to_dict
from racesim.formatting import format_lap_time
from racesim.plotting import (
    plot_convergence,
    plot_cumulative_distribution,
    plot_replanning_timeline,
    plot_uncertainty_breakdown,
    plot_delta_distribution,
    plot_lap_time_progression,
    plot_monte_carlo_distribution,
    plot_pit_window_costs,
    plot_safety_car_scenarios,
    plot_sensitivity,
    plot_strategy_comparison,
    plot_tyre_degradation,
)
from racesim.schemas import RiskReport, SimulationResult
from racesim.tyre_model import compare_tyre_compounds

logger = logging.getLogger(__name__)

LIMITATIONS = [
    "All default parameters are SYNTHETIC and illustrative; they are not calibrated to any real circuit, car or tyre.",
    "Lap-time model is additive and transparent; interactions such as tyre temperature dynamics are not modelled.",
    "Tyre degradation follows a linear or thresholded quadratic law; real cliffs and graining are not captured.",
    "Car-to-car effects use aggregate elapsed times only (dirty-air threshold, pace-based overtaking); no track position data.",
    "Safety-car / VSC events use a constant per-lap hazard; no red flags, lapped-car procedures or pit-lane closures.",
    "Weather model is deliberately simple and UNCALIBRATED; wet-tyre crossover values are illustrative.",
    "Reactive decisions use the team's deterministic model and can be wrong ex post (strategy uncertainty).",
    "Deterministic optimum minimises expected race time only; it does not guarantee a result or the lowest risk.",
    "Monte Carlo results are reproducible for an identical seed, simulation count, batch size and configuration.",
    "Monte Carlo confidence intervals measure simulation precision only; more runs do not make predictions accurate for a real race.",
    "Rolling-horizon replanning is demonstrated on deterministic simulated races that use the same model as the engine; "
    "real-race benefit is unvalidated. The remaining SC length and rain duration are assumed from configuration.",
    "Validation on SYNTHETIC data only checks self-consistency; no historical data are bundled with the project.",
]

_CSS = """
:root{--bg:#fcfcfb;--ink:#0b0b0b;--muted:#52514e;--line:#e6e5e1;--accent:#2a78d6;--good:#1baf7a;--warn:#b07800}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:36px 0 10px;padding-top:12px;border-top:1px solid var(--line)}
h3{font-size:15px;margin:18px 0 6px}.muted{color:var(--muted)}.banner{background:#fff4d6;border:1px solid #f0d58a;padding:10px 14px;border-radius:8px;margin:14px 0}
table{border-collapse:collapse;width:100%;margin:8px 0 16px;font-size:13px;font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}th{color:var(--muted);font-weight:600}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin:12px 0}
.kpi{border:1px solid var(--line);border-radius:10px;padding:10px 12px}.kpi .v{font-size:20px;font-weight:650}.kpi .l{font-size:12px;color:var(--muted)}
pre{white-space:pre-wrap;background:#f3f2ef;padding:10px;border-radius:8px;font-size:12px}
ul{margin:6px 0 12px 20px;padding:0}.table-wrap{overflow-x:auto}
"""


def _esc(x) -> str:
    return html.escape(str(x))


def _table(df: Optional[pd.DataFrame], max_rows: int = 30, float_fmt: str = "{:,.3f}") -> str:
    if df is None or df.empty:
        return "<p class='muted'>Not available.</p>"
    d = df.head(max_rows).copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else float_fmt.format(v))
    head = "".join(f"<th>{_esc(c)}</th>" for c in d.columns)
    body = "".join("<tr>" + "".join(f"<td>{_esc(v)}</td>" for v in row) + "</tr>" for row in d.itertuples(index=False))
    more = f"<p class='muted'>Showing {max_rows} of {len(df)} rows.</p>" if len(df) > max_rows else ""
    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>{more}"


def _kv(items: Mapping[str, object]) -> str:
    return _table(pd.DataFrame({"item": list(items), "value": [str(v) for v in items.values()]}))


class _FigureEmbedder:
    def __init__(self) -> None:
        self.first = True

    def __call__(self, fig) -> str:
        out = fig.to_html(full_html=False, include_plotlyjs="inline" if self.first else False,
                          config={"displaylogo": False, "responsive": True})
        self.first = False
        return out


def _fmt(v: Optional[float], unit: str = " s", digits: int = 2) -> str:
    return "not computed" if v is None else f"{v:,.{digits}f}{unit}"


def generate_risk_report(risk_reports: Mapping[str, RiskReport], names: Optional[Mapping[str, str]] = None,
                         metadata: Optional[dict] = None) -> str:
    """HTML fragment with the risk-metric table (lower race times are better)."""
    if not risk_reports:
        return "<p class='muted'>Monte Carlo analysis was not run.</p>"
    rows = []
    for sid, r in risk_reports.items():
        rows.append({
            "strategy": (names or {}).get(sid, sid), "mean_s": r.expected_race_time_s, "median_s": r.median_race_time_s,
            "std_s": r.standard_deviation_s, "p05_s": r.percentile_05_s, "p90_s": r.percentile_90_s, "p95_s": r.percentile_95_s,
            "cvar95_s": r.cvar_95_s,
            "p_beat_baseline": "n/a (is baseline or not run)" if r.probability_of_beating_baseline is None else f"{r.probability_of_beating_baseline:.3f}",
            "p_beat_all_rivals": "n/a (no rival model)" if r.probability_of_beating_rival is None else f"{r.probability_of_beating_rival:.3f}",
            "exp_position": "n/a" if r.expected_finishing_position is None else f"{r.expected_finishing_position:.2f}",
            "n": r.number_of_simulations,
        })
    meta = ""
    if metadata:
        meta = (f"<p class='muted'>Simulations per strategy: {metadata.get('n_sims')}; seed: {metadata.get('seed')}; "
                f"batch size: {metadata.get('batch_size')}; uncertainty sources: {_esc(metadata.get('flags'))}.</p>")
    warn = sorted({w for r in risk_reports.values() for w in r.warnings})
    warn_html = "".join(f"<div class='banner'>{_esc(w)}</div>" for w in warn)
    return (meta + warn_html + _table(pd.DataFrame(rows)) +
            "<p class='muted'>Race-time metrics: lower is better. CVaR95 = mean of the worst 5% of simulated race times. "
            "Probabilities: higher is better; ties count one half.</p>")


def generate_strategy_report(bundle, path: Optional[str | Path] = None, *, title: Optional[str] = None) -> str:
    """Full HTML report for an :class:`AnalysisBundle`; optionally written to ``path``."""
    setup = bundle.setup
    race = setup.race
    opt = bundle.optimisation
    fig = _FigureEmbedder()
    names = {s.strategy_id: s.strategy_name for s in bundle.shortlist}
    best = opt.best
    title = title or f"RaceSim strategy report - {race.race_name}"
    parts = [f"<h1>{_esc(title)}</h1>",
             f"<p class='muted'>Generated {_dt.datetime.now().strftime('%Y-%m-%d %H:%M')} by RaceSim {__version__}.</p>",
             f"<div class='banner'><b>{_esc(race.data_label)}</b>. Results are model outputs, not measurements, and do not guarantee any race outcome.</div>",
             "<p><b>Analyses executed in this report:</b> " + _esc(", ".join(bundle.analyses_run) or "none") +
             ". Sections for analyses that were not executed are labelled <i>not run</i>.</p>"]

    # 1. Race configuration
    parts.append("<h2>1. Race configuration</h2>")
    parts.append(_kv({"Race": race.race_name, "Circuit": race.circuit_name, "Total laps": race.total_laps,
                      "Baseline lap time": format_lap_time(race.baseline_lap_time_s), "Championship rules": setup.regulations.championship_name,
                      "Simulation seed": race.simulation_seed,
                      "Monte Carlo runs": bundle.monte_carlo.n_sims if bundle.monte_carlo else "not run"}))
    # 2. Assumptions
    parts.append("<h2>2. Input assumptions</h2>")
    parts.append(_kv({
        "Pit-lane loss (excl. stationary)": f"{race.pit_lane_loss_s:.2f} s", "Stationary time": f"{race.stationary_stop_time_s:.2f} s",
        "Total green-flag pit cost": f"{race.total_pit_cost_s:.2f} s",
        "SC / VSC pit-lane loss": f"{setup.pit_stop.sc_pit_lane_loss_s:.2f} s / {setup.pit_stop.vsc_pit_lane_loss_s:.2f} s",
        "Fuel start / consumption / effect": f"{race.fuel_start_kg:.1f} kg / {race.fuel_consumption_kg_per_lap:.3f} kg per lap / {race.fuel_effect_s_per_kg:.3f} s per kg",
        "Lap-time variability": f"{race.baseline_lap_time_std_s:.2f} s",
        "SC / VSC probability per lap": f"{race.safety_car_probability_per_lap:.4f} / {race.vsc_probability_per_lap:.4f}",
        "Effective stint limits": f"{setup.effective_min_stint_laps}-{setup.effective_max_stint_laps} laps",
        "Refuelling": "permitted" if setup.refuelling_allowed else "not permitted",
        "Weather": f"{setup.weather.scenario} ({setup.weather.calibration_status})",
    }))
    comp_rows = pd.DataFrame([{"compound": c.compound_name, "type": c.compound_type, "reference_lap": format_lap_time(c.reference_lap_time_s),
                               "linear_s_per_lap": c.linear_degradation_s_per_lap, "nonlinear_coef": c.nonlinear_degradation_coefficient,
                               "threshold_laps": c.nonlinear_degradation_threshold_laps, "warmup_s": c.warmup_penalty_s,
                               "max_age_laps": c.maximum_recommended_age_laps, "source": c.parameter_source}
                              for c in setup.compounds.values()])
    parts.append("<h3>Tyre compounds</h3>" + _table(comp_rows))
    parts.append(fig(plot_tyre_degradation(compare_tyre_compounds(setup.compounds, max_age_laps=45,
                                                                  reference_temperature_c=race.reference_track_temperature_c,
                                                                  track_temperature_c=race.track_temperature_c, limit_to_recommended_age=True))))
    if bundle.warnings:
        parts.append("<h3>Configuration and analysis warnings</h3><ul>" + "".join(f"<li>{_esc(w)}</li>" for w in bundle.warnings) + "</ul>")

    # 3. Recommended strategy
    parts.append("<h2>3. Recommended strategy</h2>")
    if best is None:
        parts.append("<p>No feasible strategy satisfies the constraints.</p>")
    else:
        s = best.strategy
        parts.append("<div class='kpis'>" + "".join(
            f"<div class='kpi'><div class='v'>{_esc(v)}</div><div class='l'>{_esc(l)}</div></div>" for l, v in [
                ("Strategy", s.strategy_name), ("Predicted race time", f"{best.predicted_race_time_s:,.2f} s"),
                ("Stops", s.number_of_stops), ("Pit laps", ", ".join(map(str, s.planned_pit_laps))),
                ("Total pit cost", f"{best.total_pit_cost_s:.1f} s"),
                ("Gain vs baseline", _fmt(opt.baseline.predicted_race_time_s - best.predicted_race_time_s if opt.baseline else None))]) + "</div>")
        parts.append(f"<pre>{_esc(opt.explanation)}</pre>")
        if bundle.recommended_simulation is not None:
            df = bundle.recommended_simulation.to_dataframe()
            parts.append(fig(plot_lap_time_progression(predicted=df.assign(strategy=s.strategy_name),
                                                       title="Predicted lap times - recommended strategy (deterministic)")))
    # 4. Alternatives
    parts.append("<h2>4. Alternative strategies</h2>")
    parts.append(f"<p class='muted'>{opt.candidates_evaluated} candidates evaluated with the same deterministic model; "
                 f"pit-lap grid steps by stop count: {_esc(opt.lap_steps)}.</p>")
    parts.append(fig(plot_strategy_comparison(opt.ranking, highlight_id=best.strategy.strategy_id if best else None)))
    cols = ["rank", "strategy_name", "pit_laps", "compounds", "stint_lengths", "race_time_s", "total_pit_cost_s", "delta_to_best_s"]
    if "delta_to_baseline_s" in opt.ranking.columns:
        cols.append("delta_to_baseline_s")
    parts.append(_table(opt.ranking[cols] if not opt.ranking.empty else opt.ranking, max_rows=15))
    # 5. Pit window
    w = bundle.pit_window
    parts.append("<h2>5. Pit-window analysis</h2>")
    parts.append(_kv({"Stops considered": w.stops_considered, "Earliest feasible lap": w.earliest_feasible_lap,
                      "Recommended (minimum-cost) lap": w.recommended_lap, "Latest feasible lap": w.latest_feasible_lap,
                      "Window midpoint (not used for selection)": None if w.earliest_feasible_lap is None else (w.earliest_feasible_lap + w.latest_feasible_lap) / 2}))
    parts.append(fig(plot_pit_window_costs(w.table, recommended_lap=w.recommended_lap, earliest_lap=w.earliest_feasible_lap,
                                           latest_lap=w.latest_feasible_lap)))
    # 6. Risk
    parts.append("<h2>6. Risk metrics (Monte Carlo)</h2>")
    mc = bundle.monte_carlo
    parts.append(generate_risk_report(bundle.risk_reports, names, mc.metadata if mc else None))
    if mc is not None:
        parts.append(fig(plot_monte_carlo_distribution(mc.race_times, names)))
        parts.append(fig(plot_cumulative_distribution(mc.race_times, names)))
        from racesim.workflow import baseline_strategy, paired_deltas

        base = baseline_strategy(setup)
        if base is not None:
            parts.append(fig(plot_delta_distribution(paired_deltas(mc, base.strategy_id), names)))
        if bundle.objectives is not None:
            parts.append("<h3>Objectives can recommend different strategies</h3>" + _table(bundle.objectives))
        parts.append("<h3>Simulation precision (Monte Carlo error)</h3>")
        parts.append("<p class='muted'>These intervals describe how precisely the <i>model's</i> answer has been estimated. "
                     "They are not a measure of real-world accuracy.</p>")
        parts.append(_table(bundle.precision.drop(columns=["strategy_id"]) if bundle.precision is not None else None))
        if bundle.convergence is not None:
            parts.append(fig(plot_convergence(bundle.convergence)))
            cs = bundle.convergence_summary or {}
            parts.append(_kv({"Converged (all criteria)": cs.get("converged"), "Checks": cs.get("checks"),
                              "Criteria": cs.get("criteria")}))
        if bundle.tail_stability:
            parts.append("<h3>Downside-risk (CVaR95) stability</h3>")
            parts.append(_table(pd.DataFrame([{"strategy": names.get(sid, sid), "cvar95_s": t.cvar95_s,
                                               "bootstrap_se_s": t.bootstrap_se_s, "ci95_s": f"[{t.ci95_s[0]:,.2f}, {t.ci95_s[1]:,.2f}]",
                                               "split_half_difference_s": t.split_half_difference_s, "tail_runs": t.tail_samples,
                                               "stable": t.stable, "reasons": "; ".join(t.reasons)}
                                              for sid, t in bundle.tail_stability.items()])))
    parts.append("<h3>Model uncertainty</h3><ul>" + "".join(f"<li>{_esc(l)}</li>" for l in bundle.model_uncertainty) + "</ul>")
    parts.append("<h3>Uncertainty breakdown by source</h3>")
    if bundle.uncertainty_breakdown is None:
        parts.append("<p class='muted'>Uncertainty breakdown: not run.</p>")
    else:
        parts.append(fig(plot_uncertainty_breakdown(bundle.uncertainty_breakdown)) + _table(bundle.uncertainty_breakdown))
    # 7. SC scenarios
    parts.append("<h2>7. Safety-car scenario analysis</h2>")
    parts.append("<p class='muted'>Deterministic races with one safety car (4 laps) at different laps; the team may bring a "
                 "planned stop forward if the model expects a gain.</p>")
    parts.append(fig(plot_safety_car_scenarios(bundle.sc_scenarios, value_col="delta_to_best_in_scenario_s",
                                               title="Race time vs best strategy in each scenario")))
    parts.append(_table(bundle.sc_scenarios.drop(columns=["strategy_id"]), max_rows=40))
    if bundle.sc_conditioning is not None and not bundle.sc_conditioning.empty:
        parts.append("<h3>Monte Carlo runs split by safety-car occurrence</h3>" + _table(bundle.sc_conditioning.drop(columns=["strategy_id"])))
    # 8. Sensitivity
    parts.append("<h2>8. Sensitivity results</h2>")
    if not bundle.sensitivity:
        parts.append("<p class='muted'>Sensitivity analysis was not run.</p>")
    else:
        parts.append("<p class='muted'>Sensitivity shows how the recommendation responds to changed assumptions; "
                     "it is not a statistical confidence measure.</p>")
        parts.append(_table(bundle.sensitivity_ranking))
        for r in bundle.sensitivity:
            y = "mc_cvar95_s" if "mc_cvar95_s" in r.table.columns and r.parameter in ("safety_car_probability_per_lap", "baseline_lap_time_std_s") else "predicted_race_time_s"
            parts.append(fig(plot_sensitivity(r.table, r.parameter, r.unit, y=y)))
            parts.append(_table(r.table.drop(columns=["recommended_strategy_id"])))
    # 9. Limitations
    parts.append("<h2>9. Model limitations</h2><ul>" + "".join(f"<li>{_esc(l)}</li>" for l in LIMITATIONS) + "</ul>")
    # 10. Provenance
    parts.append("<h2>10. Data provenance</h2>")
    parts.append(_kv(bundle.data_provenance))
    parts.append("<h3>Full configuration</h3><pre>" + _esc(json.dumps(setup_to_dict(setup), indent=2)) + "</pre>")

    # 11-12. v2 sections (after the v1 sections, whose numbering is unchanged)
    parts.append("<h2>11. Dynamic replanning and decision log</h2>")
    rr = bundle.rolling_race
    if rr is None:
        parts.append("<p class='muted'>Dynamic replanning: not run.</p>")
    else:
        parts.append(_kv({"Events in simulated race": ", ".join(f"{e.event_type.value} lap {e.start_lap} ({e.duration_laps} laps)"
                                                               for e in rr.events) or "none",
                          "Race time with replanning": f"{rr.total_race_time_s:,.2f} s",
                          "Race time, static plan": f"{rr.static_plan_race_time_s:,.2f} s",
                          "Race time, v1 reactive SC/rain heuristic": f"{rr.reactive_policy_race_time_s:,.2f} s",
                          "Gain vs static plan": f"{rr.gain_vs_static_s:+.2f} s",
                          "Pit laps / compounds": f"{rr.own_pit_laps} / {' > '.join(rr.own_compounds)}",
                          "Finishing position (replanning / static)": f"{rr.finishing_position} / {rr.static_finishing_position}"}))
        parts.append("<p class='muted'>Deterministic example: the simulated race and the engine's predictions use the same model, "
                     "so the gain shows the decision logic working, not a forecast of real-race benefit.</p>")
        parts.append(fig(plot_replanning_timeline(rr.laps, rr.decision_log.to_dataframe())))
        log = rr.decision_log
        replanned = [r for r in log.records if r.replanned]
        from racesim.decision_log import DecisionLog

        parts.append(f"<p class='muted'>{len(log)} decision points; {len(replanned)} re-optimisations shown "
                     "(carried-forward decisions are in the CSV export).</p>")
        parts.append(DecisionLog(log.configuration, replanned).to_html())
    parts.append("<h2>12. Model validation</h2>")
    v = bundle.validation
    if v is None:
        parts.append("<p class='muted'>Model validation: not run.</p>")
    else:
        parts.append(f"<div class='banner'>{_esc(v.headline)}</div>")
        parts.append(_kv({"Evidence type": v.evidence_type, "Status": v.status,
                          "Split method": v.report.get("method"), "Leakage check": v.report.get("leakage_check")}))
        parts.append(_table(v.metrics_table()))
        for w in v.report.get("warnings", []):
            parts.append(f"<p class='muted'>Warning: {_esc(w)}</p>")
        if v.calibration is not None:
            parts.append("<h3>Tyre degradation calibration (not applied automatically)</h3>" + _table(v.calibration.table))
        ra = v.report.get("residual_analysis") or {}
        if "diagnostics" in ra:
            parts.append("<h3>Residual diagnostics</h3>" + _table(ra["diagnostics"]))

    doc = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>{_esc(title)}</title><style>{_CSS}</style></head><body><main>{''.join(parts)}</main></body></html>")
    if path is not None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(doc, encoding="utf-8")
        logger.info("Wrote HTML report to %s", p)
    return doc


def export_strategy_summary(optimisation, path: str | Path) -> Path:
    """CSV of the ranked deterministic strategy comparison."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    optimisation.ranking.to_csv(p, index=False)
    return p


def export_race_summary(result: SimulationResult | Iterable[SimulationResult], path: str | Path) -> Path:
    """CSV of lap-by-lap race simulation(s) with strategy identifiers."""
    results = [result] if isinstance(result, SimulationResult) else list(result)
    frames = [r.to_dataframe().assign(strategy_id=r.strategy_id, predicted_race_time_s=r.predicted_race_time_s) for r in results]
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(p, index=False)
    return p


def generate_live_session_report(snapshots: Iterable[dict], path: Optional[str | Path] = None) -> str:
    """Post-session summary of ACC telemetry analysis, one section per session (from ``LiveAnalysis.snapshot``)."""
    from racesim.integrations.acc.live_analysis import OVERVIEW_LAP_TIME_ROWS

    def with_lap_times(rows: list, column: str) -> pd.DataFrame:
        df = pd.DataFrame(rows)
        if column in df.columns:
            df.insert(df.columns.get_loc(column), column.removesuffix("_s"), df.pop(column).map(format_lap_time))
        return df
    parts = []
    for snap in snapshots:
        if not snap.get("available"):
            continue
        pace = snap["pace"]
        stats = [{"statistic": name, "value": format_lap_time(pace[name]["value"]), "valid_laps": pace[name]["n"]} for name in ("best", "mean", "median")]
        std = pace["std"]["value"]
        stats.append({"statistic": "std", "value": None if std is None else f"{std:.3f} s", "valid_laps": pace["std"]["n"]})
        stats += [{"statistic": f"rolling {w}", "value": format_lap_time(r["value"]), "valid_laps": r["n"]} for w, r in pace["rolling"].items()]
        source = "MOCK data - not ACC" if snap.get("mock") else "Recorded ACC telemetry"
        parts.append(
            f"<h2>Session {_esc(snap['session_key'])}</h2><p class='banner'>{_esc(source)}. Lap times are measured; classifications and "
            "statistics are derived from telemetry.</p>"
            + _kv({r["channel"]: "UNAVAILABLE" if r["value"] is None else format_lap_time(r["value"]) if r["channel"] in OVERVIEW_LAP_TIME_ROWS
                   else f"{r['value']} {r['unit']}".strip() for r in snap["overview"]})
            + "<h3>Pace (clean-lap candidates only)</h3>" + _table(pd.DataFrame(stats))
            + f"<p class='muted'>{_esc(pace['trend_note'])}</p>"
            + "<h3>Laps</h3>" + _table(with_lap_times(snap["laps"], "lap_time_s"), max_rows=500)
            + "<h3>Stints</h3>" + _table(pd.DataFrame(snap["stints"]))
            + "<h3>Stint pace</h3>" + _table(with_lap_times(pace["stints"], "median_s"))
            + "<h3>Pit transitions</h3>" + _table(pd.DataFrame(snap["transitions"]))
            + "<h3>Channels not supplied</h3>" + _table(pd.DataFrame(snap["missing_channels"]))
            + "<h3>Data-quality flags</h3>" + _kv(snap["flags"] or {"none": 0})
            + ("<h3>Strategy snapshots (MODEL PREDICTION)</h3>" + _table(pd.DataFrame(
                [{k: s.get(k) for k in ("status", "laps_completed_at_run", "recommended_pit_lap", "pit_window")} for s in snap["strategy_snapshots"]]))
               if snap["strategy_snapshots"] else ""))
    body = "".join(parts) or "<p class='muted'>No telemetry sessions were recorded.</p>"
    doc = (f"<!doctype html><html><head><meta charset='utf-8'><title>ACC session summary</title><style>{_CSS}</style></head>"
           f"<body><main><h1>ACC session summary</h1>{body}</main></body></html>")
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "x", encoding="utf-8") as fh:
            fh.write(doc)
    return doc
