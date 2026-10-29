"""Section H - sensitivity analysis and uncertainty diagnostics."""

from __future__ import annotations

import streamlit as st

from dashboard.common import DashboardContext, cached_optimisation, show, to_csv_bytes
from racesim.plotting import plot_convergence, plot_sensitivity, plot_uncertainty_breakdown
from racesim.sensitivity import rank_sensitive_parameters
from racesim.uncertainty import (
    MODEL_UNCERTAINTY_STATEMENT, assess_tail_stability, convergence_diagnostics, convergence_summary,
    model_uncertainty_summary, simulation_precision_table, uncertainty_breakdown,
)
from racesim.workflow import baseline_strategy, run_sensitivity_suite


def render(ctx: DashboardContext) -> None:
    setup = ctx.setup
    st.info(MODEL_UNCERTAINTY_STATEMENT)

    st.header("Simulation precision (from the Section D run)")
    store = st.session_state.get("mc_store")
    if store is None or store.get("setup") != ctx.setup_json:
        st.caption("Not available: run the Monte Carlo simulation in Section D with the current settings.")
    else:
        mc = store["mc"]
        base = baseline_strategy(setup)
        base_id = base.strategy_id if base is not None and base.strategy_id in mc.race_times else None
        st.dataframe(simulation_precision_table(mc, base_id).drop(columns=["strategy_id"]), hide_index=True)
        opt = cached_optimisation(ctx.setup_json)
        rec = opt.best.strategy.strategy_id if opt.best and opt.best.strategy.strategy_id in mc.race_times else next(iter(mc.race_times))
        diag = convergence_diagnostics(mc.race_times[rec], mc.race_times[base_id] if base_id and base_id != rec else None)
        summ = convergence_summary(diag)
        c1, c2 = st.columns([3, 2])
        with c1:
            show(plot_convergence(diag))
        with c2:
            st.metric("Converged (all criteria)", "yes" if summ["converged"] else "no")
            st.json({"checks": summ["checks"], "criteria": summ["criteria"],
                     "mean_ci_half_width_s": round(summ["mean_ci_half_width_s"], 3),
                     "paired_delta_ci_half_width_s": summ.get("paired_delta_ci_half_width_s")})
        stab = {mc.strategies[sid].strategy_name: assess_tail_stability(t, n_boot=200) for sid, t in mc.race_times.items()}
        st.markdown("**Downside-risk (CVaR95) stability**")
        st.dataframe([{"strategy": k, "cvar95_s": v.cvar95_s, "bootstrap_se_s": v.bootstrap_se_s, "ci95_s": v.ci95_s,
                       "split_half_difference_s": v.split_half_difference_s, "tail_runs": v.tail_samples, "stable": v.stable,
                       "reasons": "; ".join(v.reasons)} for k, v in stab.items()], hide_index=True)

    st.header("Uncertainty breakdown by source")
    st.caption("Each source enabled alone. 'Paired difference' is the decision-relevant spread versus the baseline; "
               "sources that hit every strategy alike (e.g. safety cars, lap noise) widen absolute race times far more. "
               "Rival behaviour changes only rival outcomes, so its effect appears in the p_beat_all_rivals column, not in the bars.")
    c = st.columns(3)
    n_b = c[0].number_input("Runs per source", 100, 5000, value=500, step=100)
    rivals = c[1].checkbox("Include rivals (adds rival behaviour)", value=bool(setup.rivals), disabled=not setup.rivals)
    if c[2].button("Run breakdown"):
        opt = cached_optimisation(ctx.setup_json)
        with st.spinner("Running one Monte Carlo per uncertainty source..."):
            df = uncertainty_breakdown(setup, opt.best.strategy, baseline_strategy(setup), n_sims=int(n_b),
                                       seed=setup.race.simulation_seed, include_rivals=rivals)
        st.session_state["breakdown_store"] = {"result": df, "setup": ctx.setup_json}
    bstore = st.session_state.get("breakdown_store")
    breakdown = bstore["result"] if bstore and bstore.get("setup") == ctx.setup_json else None
    if breakdown is not None:
        show(plot_uncertainty_breakdown(breakdown))
        st.dataframe(breakdown, hide_index=True)

    st.header("Sensitivity to model assumptions")
    st.caption("Re-optimises for each assumption value. This shows how the recommendation responds to assumptions; "
               "it is not a statistical confidence measure.")
    c = st.columns(2)
    sims = c[0].number_input("Monte Carlo runs for SC / variability sweeps", 100, 3000, value=300, step=100)
    if c[1].button("Run sensitivity analysis", type="primary"):
        with st.spinner("Running sensitivity sweeps..."):
            res = run_sensitivity_suite(setup, sensitivity_sims=int(sims), seed=setup.race.simulation_seed)
        st.session_state["sens_store"] = {"result": res, "setup": ctx.setup_json}
    sstore = st.session_state.get("sens_store")
    sens = sstore["result"] if sstore and sstore.get("setup") == ctx.setup_json else None
    if sens is None:
        st.caption("Not run with the current settings.")
    else:
        ranking = rank_sensitive_parameters(sens)
        st.dataframe(ranking, hide_index=True)
        for r in sens:
            y = "mc_cvar95_s" if "mc_cvar95_s" in r.table.columns and r.parameter in ("safety_car_probability_per_lap",
                                                                                       "baseline_lap_time_std_s") else "predicted_race_time_s"
            show(plot_sensitivity(r.table, r.parameter, r.unit, y=y))
        st.download_button("Sensitivity ranking (CSV)", to_csv_bytes(ranking), "sensitivity_ranking.csv", "text/csv")

    st.header("Model uncertainty")
    vstore = st.session_state.get("valid_store")
    vstat = None
    if vstore and vstore.get("setup") == ctx.setup_json:
        v = vstore["result"]
        vstat = f"{v.status} ({v.evidence_type})"
    lines = model_uncertainty_summary(setup, sensitivity_ranking=rank_sensitive_parameters(sens) if sens else None,
                                      validation_status=vstat, breakdown=breakdown)
    for line in lines[1:]:
        st.markdown(f"- {line}")
