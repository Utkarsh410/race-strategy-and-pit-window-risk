"""Section D - Monte Carlo risk analysis and report download."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.common import DashboardContext, base_setup, build_report_html, cached_optimisation, json_key, show, ss, to_csv_bytes
from racesim.monte_carlo import run_monte_carlo
from racesim.plotting import plot_cumulative_distribution, plot_delta_distribution, plot_monte_carlo_distribution
from racesim.risk_metrics import compare_objectives, risk_reports_from_monte_carlo
from racesim.workflow import baseline_strategy, monte_carlo_sc_conditioning, paired_deltas, shortlist_strategies

LOW_COUNT = 1000


def _controls(ctx: DashboardContext) -> dict:
    base = base_setup()
    st.subheader("Simulation settings")
    c = st.columns(4)
    n_sims = c[0].number_input("Simulations per strategy", 50, 50_000, value=base.monte_carlo.dashboard_simulations, step=250,
                               help="10,000 is the benchmark; lower counts are faster but noisier.")
    c[1].metric("Random seed (sidebar)", int(ss("mc_seed", base.race.simulation_seed)), help="Same seed + settings = identical results.")
    ss("ov_lap_std", base.race.baseline_lap_time_std_s)
    c[2].number_input("Lap-time variability (s)", 0.0, 5.0, step=0.05, key="ov_lap_std")
    ss("ov_deg_unc_scale", 1.0)
    c[3].number_input("Degradation uncertainty (x config)", 0.0, 5.0, step=0.1, key="ov_deg_unc_scale",
                      help="Scales each compound's relative degradation-rate uncertainty.")
    c = st.columns(4)
    ss("ov_pit_std", base.race.pit_loss_std_s)
    c[0].number_input("Pit-lane loss std (s)", 0.0, 10.0, step=0.1, key="ov_pit_std")
    ss("ov_stationary_std", base.pit_stop.stationary_std_s)
    c[1].number_input("Stationary time std (s)", 0.0, 5.0, step=0.05, key="ov_stationary_std")
    ss("ov_sc_prob", base.race.safety_car_probability_per_lap)
    c[2].number_input("SC probability per lap", 0.0, 0.2, step=0.002, format="%.3f", key="ov_sc_prob")
    ss("ov_vsc_prob", base.race.vsc_probability_per_lap)
    c[3].number_input("VSC probability per lap", 0.0, 0.2, step=0.002, format="%.3f", key="ov_vsc_prob")
    with st.expander("Uncertainty sources"):
        cc = st.columns(7)
        flags = {
            "parameter": cc[0].checkbox("Parameter", True, help="Degradation, pace bias, fuel-consumption draws per run."),
            "lap_variability": cc[1].checkbox("Lap-to-lap", True),
            "events": cc[2].checkbox("SC / VSC / rain", True),
            "pit_variability": cc[3].checkbox("Pit execution", True),
            "traffic": cc[4].checkbox("Traffic", True),
            "reactive": cc[5].checkbox("Reactive calls", True, help="Strategy uncertainty: SC pit calls and rain tyre calls."),
            "rival_behaviour": cc[6].checkbox("Rival behaviour", ctx.setup.monte_carlo.include_rival_uncertainty,
                                              help="Random rival pace offset and pit-lap shifts (affects rival outcomes only)."),
        }
    include_rivals = st.checkbox("Simulate configured rivals (enables P(beat rival) and finishing position)", value=bool(ctx.setup.rivals),
                                 disabled=not ctx.setup.rivals)
    return {"n_sims": int(n_sims), "seed": int(st.session_state["mc_seed"]), "flags": flags, "include_rivals": include_rivals}


def render(ctx: DashboardContext) -> None:
    params = _controls(ctx)
    opt = cached_optimisation(ctx.setup_json)
    if opt.best is None:
        st.error("No feasible strategy to evaluate.")
        return
    short = shortlist_strategies(opt, ctx.setup)
    names = {s.strategy_id: s.strategy_name for s in short}
    st.caption("Strategies evaluated (common random numbers): " + "; ".join(names.values()))
    if params["n_sims"] < LOW_COUNT:
        st.warning(f"{params['n_sims']} simulations: tail metrics (P95, CVaR95) rest on only "
                   f"{max(1, params['n_sims'] // 20)} runs and are unstable. Use >= {LOW_COUNT} for decisions.")
    key = json_key({"setup": ctx.setup_json, "strategies": sorted(names), **params})
    stored = st.session_state.get("mc_store")
    if st.button("Run Monte Carlo", type="primary"):
        bar = st.progress(0.0, text="Starting...")
        mc = run_monte_carlo(ctx.setup, short, n_sims=params["n_sims"], seed=params["seed"], flags=params["flags"],
                             include_rivals=params["include_rivals"], progress_callback=lambda f, m: bar.progress(min(f, 1.0), text=m))
        bar.empty()
        st.session_state["mc_store"] = stored = {"key": key, "mc": mc, "setup": ctx.setup_json}
    if stored is None:
        st.info("Press **Run Monte Carlo** to simulate. Simulations are not re-run automatically on every change.")
        return
    if stored["key"] != key:
        st.warning("Settings changed since the last run - results below are for the previous settings. Press Run to update.")
    mc = stored["mc"]
    names = {sid: s.strategy_name for sid, s in mc.strategies.items()}
    base = baseline_strategy(ctx.setup)
    base_id = base.strategy_id if base is not None and base.strategy_id in mc.race_times else None
    reports = risk_reports_from_monte_carlo(mc, baseline_id=base_id)
    rec_id = opt.best.strategy.strategy_id if opt.best.strategy.strategy_id in reports else next(iter(reports))
    r = reports[rec_id]
    st.subheader(f"Recommended strategy: {names[rec_id]}  (n = {r.number_of_simulations:,}, seed {mc.metadata['seed']})")
    k = st.columns(4)
    k[0].metric("Expected race time", f"{r.expected_race_time_s:,.2f} s")
    k[1].metric("Median race time", f"{r.median_race_time_s:,.2f} s")
    k[2].metric("Standard deviation", f"{r.standard_deviation_s:,.2f} s")
    k[3].metric("CVaR 95 (mean of worst 5%)", f"{r.cvar_95_s:,.2f} s")
    k = st.columns(4)
    k[0].metric("P05 / P50", f"{r.percentile_05_s:,.1f} / {r.percentile_50_s:,.1f} s")
    k[1].metric("P90 / P95", f"{r.percentile_90_s:,.1f} / {r.percentile_95_s:,.1f} s")
    ci = lambda t: "" if t is None else f" [{t[0]:.1%}, {t[1]:.1%}]"  # noqa: E731
    k[2].metric("P(beat baseline) [95% CI]", "n/a" if r.probability_of_beating_baseline is None else
                f"{r.probability_of_beating_baseline:.1%}{ci(r.probability_of_beating_baseline_ci95)}",
                help="Paired over identical simulated races; ties count 1/2. Wilson interval = simulation precision only.")
    k[3].metric("P(beat all rivals) [95% CI]", "n/a (rivals not simulated)" if r.probability_of_beating_rival is None else
                f"{r.probability_of_beating_rival:.1%}{ci(r.probability_of_beating_rival_ci95)}")
    st.caption(f"Monte Carlo standard error of the mean: {r.monte_carlo_standard_error_s:.2f} s. Confidence intervals describe "
               "simulation precision, not real-world accuracy (see Section H).")
    for w in r.warnings:
        st.warning(w)
    st.caption("Absolute race-time spread is dominated by safety cars, which slow every strategy; "
               "the paired difference to the baseline is the strategy-relevant quantity.")
    show(plot_monte_carlo_distribution(mc.race_times, names))
    c1, c2 = st.columns(2)
    with c1:
        show(plot_cumulative_distribution(mc.race_times, names))
    with c2:
        show(plot_delta_distribution(paired_deltas(mc, base_id), names) if base_id else
             plot_delta_distribution({}, names))
    rows = []
    for sid, rep in reports.items():
        d = rep.model_dump(exclude={"warnings", "strategy_id"})
        rows.append({"strategy": names[sid], **d, "mean_stops": float(mc.n_stops[sid].mean()),
                     "reactive_call_rate": float((mc.reactive_stops[sid] > 0).mean()),
                     "fuel_shortfall_rate": float(mc.fuel_shortfall[sid].mean())})
    table = pd.DataFrame(rows)
    st.subheader("Risk metrics by strategy (race times: lower is better)")
    st.dataframe(table, hide_index=True)
    st.subheader("Different objectives can favour different strategies")
    st.dataframe(compare_objectives(reports, names).drop(columns=["strategy_id"], errors="ignore"), hide_index=True)
    cond = monte_carlo_sc_conditioning(mc)
    with st.expander("Runs split by safety-car occurrence"):
        st.dataframe(cond.drop(columns=["strategy_id"]), hide_index=True)
    with st.expander("Run metadata"):
        st.json(mc.metadata)

    st.subheader("Downloads")
    d1, d2, d3 = st.columns(3)
    d1.download_button("Risk metrics (CSV)", to_csv_bytes(table), "risk_metrics.csv", "text/csv")
    d2.download_button("Per-run results (CSV)", to_csv_bytes(mc.to_dataframe()), "monte_carlo_results.csv", "text/csv")
    if d3.button("Build HTML report"):
        with st.spinner("Building report from the analyses run in this session..."):
            st.session_state["report_html"] = build_report_html(ctx)
    if "report_html" in st.session_state:
        d3.download_button("Download HTML report", st.session_state["report_html"].encode("utf-8"), "race_strategy_report.html", "text/html")
    st.caption("The report includes Monte Carlo (this tab), sensitivity and uncertainty breakdown (Section H), dynamic replanning "
               "(Section G) and model validation (Section F) only if they have been run in this session.")
