"""Section G - dynamic (rolling-horizon) strategy replanning and the decision log."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from dashboard.common import DashboardContext, cached_optimisation, json_key, show, to_csv_bytes
from racesim.plotting import plot_replanning_timeline
from racesim.replanning import __doc__ as REPLANNING_DOC
from racesim.schemas import Strategy
from racesim.workflow import baseline_strategy, default_replan_events, run_replanning_demo


def _controls(ctx: DashboardContext) -> dict:
    setup = ctx.setup
    n = setup.race.total_laps
    opt = cached_optimisation(ctx.setup_json)
    options = {}
    base = baseline_strategy(setup)
    if base is not None:
        options[f"Baseline: {base.strategy_name}"] = base
    if opt.best is not None:
        options[f"Pre-race optimum: {opt.best.strategy.strategy_name}"] = opt.best.strategy
    st.subheader("Simulated race")
    c = st.columns(3)
    start = c[0].selectbox("Initial plan", list(options), help="The plan the car starts the race on.")
    sims = c[1].number_input("Monte Carlo runs per option per decision", 0, 2000, value=0, step=50,
                             help="0 = deterministic comparison (fast). Larger values compare expected time and CVaR95.")
    risk = c[2].slider("Risk aversion", 0.0, 1.0, float(setup.replanning.risk_aversion), 0.05,
                       help="score = mean + lambda x (CVaR95 - mean). Only used when Monte Carlo runs > 0.")
    c = st.columns(4)
    use_sc = c[0].checkbox("Safety car", value=True)
    sc_lap = c[1].number_input("SC start lap", 2, n - 1, value=min(18, n - 2), disabled=not use_sc)
    sc_len = c[2].number_input("SC length (laps)", 1, 10, value=4, disabled=not use_sc)
    use_rain = c[3].checkbox("Rain", value=False)
    c = st.columns(4)
    rain_lap = c[0].number_input("Rain start lap", 2, n - 1, value=min(30, n - 2), disabled=not use_rain)
    rain_len = c[1].number_input("Rain length (laps)", 1, n, value=12, disabled=not use_rain)
    rain_sev = c[2].slider("Rain peak wetness", 0.1, 1.0, 0.8, 0.05, disabled=not use_rain)
    shifts = {}
    if setup.rivals:
        rv = c[3].selectbox("Rival deviating from its plan", ["none"] + [r.rival_id for r in setup.rivals],
                            help="The real rival pits earlier/later than the engine's prior expects.")
        if rv != "none":
            shifts[rv] = st.slider(f"{rv} stop shift (laps)", -10, 10, -6)
    events = default_replan_events(setup, sc_lap=int(sc_lap) if use_sc else None, sc_laps=int(sc_len),
                                   rain_lap=int(rain_lap) if use_rain else None, rain_laps=int(rain_len), rain_severity=float(rain_sev))
    return {"strategy": options[start], "n_sims": int(sims), "risk": float(risk), "events": events, "shifts": shifts}


def render(ctx: DashboardContext) -> None:
    setup = ctx.setup
    st.caption("A rolling-horizon engine observes the race after every lap and re-optimises the remaining race when something "
               "relevant changes (race control, weather, rival stops, deviations from its prediction, approaching stop). "
               "Completed laps and stops are never re-optimised.")
    with st.expander("How it works and its simplifying assumptions"):
        st.markdown(REPLANNING_DOC)
    p = _controls(ctx)
    truth = [r.model_copy(update={"pit_laps": [x + p["shifts"].get(r.rival_id, 0) for x in r.pit_laps]}) for r in setup.rivals]
    key = json_key({"setup": ctx.setup_json, "strategy": p["strategy"].strategy_id, "n": p["n_sims"], "risk": p["risk"],
                    "events": [e.model_dump(mode="json") for e in p["events"]], "shifts": p["shifts"]})
    stored = st.session_state.get("replan_store")
    if st.button("Run rolling-horizon race", type="primary"):
        with st.spinner("Simulating the race lap by lap with the decision engine in the loop..."):
            try:
                rr = run_replanning_demo(setup, strategy=p["strategy"], events=p["events"], n_sims=p["n_sims"],
                                         risk_aversion=p["risk"], truth_rivals=truth)
            except ValueError as exc:  # e.g. a rival shift that makes its plan invalid
                st.error(f"Could not run this scenario: {exc}")
                return
        st.session_state["replan_store"] = stored = {"key": key, "result": rr, "setup": ctx.setup_json}
    if stored is None:
        st.info("Press **Run rolling-horizon race**. The run is not repeated automatically on every change.")
        return
    if stored["key"] != key:
        st.warning("Settings changed since the last run - results below are for the previous settings.")
    rr = stored["result"]
    k = st.columns(4)
    k[0].metric("Race time with replanning", f"{rr.total_race_time_s:,.2f} s",
                delta=f"{-rr.gain_vs_static_s:+.2f} s vs static plan", delta_color="inverse")
    k[1].metric("Static plan", f"{rr.static_plan_race_time_s:,.2f} s")
    k[2].metric("v1 reactive heuristic", f"{rr.reactive_policy_race_time_s:,.2f} s",
                help="The v1 built-in SC/rain reaction rules on the same race.")
    k[3].metric("Position (replanning / static)", f"P{rr.finishing_position} / P{rr.static_finishing_position}")
    st.caption(f"Stops on laps {rr.own_pit_laps}; compounds {' > '.join(rr.own_compounds)}. Deterministic example: the simulated "
               "race uses the same model as the engine, so this demonstrates the decision logic, not real-race benefit.")
    log_df = rr.decision_log.to_dataframe()
    show(plot_replanning_timeline(rr.laps, log_df))

    st.subheader("Strategy decision log")
    only = st.toggle("Show re-optimisations only", value=True)
    view = log_df[log_df["replanned"]] if only else log_df
    st.dataframe(view[["decision_id", "lap", "simulated_time_s", "action", "recommended_pit_lap", "recommended_strategy",
                       "replanned", "triggers", "expected_difference_vs_current_plan_s"]], hide_index=True)
    replanned = [r for r in rr.decision_log.records if r.replanned]
    if replanned:
        choice = st.selectbox("Inspect decision", [f"Lap {r.lap}: {r.action} ({r.recommended_strategy_name})" for r in replanned])
        rec = replanned[[f"Lap {r.lap}: {r.action} ({r.recommended_strategy_name})" for r in replanned].index(choice)]
        c1, c2 = st.columns([3, 2])
        with c1:
            alts = pd.DataFrame([a.__dict__ for a in rec.alternatives])
            st.markdown("**Alternatives evaluated (remaining race, seconds)**")
            st.dataframe(alts.drop(columns=["strategy_id"]), hide_index=True)
        with c2:
            st.markdown("**Influencing factors (computed values)**")
            for f in rec.influencing_factors:
                st.markdown(f"- {f}")
            for w in rec.warnings:
                st.warning(w)
            with st.expander("Observed race state"):
                st.json(rec.race_state)
            st.caption(f"Model {rec.model_version}; seed {rec.random_seed}; configuration {rec.configuration_hash}; "
                       f"{rec.monte_carlo_simulations} Monte Carlo runs per option; risk aversion {rec.risk_aversion}.")
    d = st.columns(3)
    d[0].download_button("Decision log (CSV)", to_csv_bytes(log_df), "decision_log.csv", "text/csv")
    d[1].download_button("Alternatives (CSV)", to_csv_bytes(rr.decision_log.alternatives_dataframe()), "decision_log_alternatives.csv", "text/csv")
    d[2].download_button("Lap history (CSV)", to_csv_bytes(rr.laps), "replanning_laps.csv", "text/csv")
