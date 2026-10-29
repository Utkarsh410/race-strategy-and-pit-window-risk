"""Section C - pit-window analysis and safety-car scenario comparison."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from dashboard.common import DashboardContext, cached_optimisation, cached_pit_window, cached_sc_table, show, to_csv_bytes
from racesim.plotting import plot_pit_window_costs, plot_safety_car_scenarios, plot_tyre_degradation
from racesim.tyre_model import compare_tyre_compounds
from racesim.workflow import shortlist_strategies


def render(ctx: DashboardContext) -> None:
    setup = ctx.setup
    n = setup.race.total_laps
    opt = cached_optimisation(ctx.setup_json, ctx.snapshot_json if ctx.snapshot.current_lap else "")
    default_k = max(opt.best.number_of_stops, 1) if opt.best else 1
    st.caption("From the race start" if ctx.snapshot.current_lap == 0 else
               f"From the end of lap {ctx.snapshot.current_lap} on {ctx.snapshot.tyre_compound} aged {ctx.snapshot.tyre_age_laps} laps (set in Section A).")
    c = st.columns([1, 1, 1, 1])
    k = c[0].number_input("Remaining stops to plan", 1, max(setup.race.maximum_stops, 1), value=min(default_k, max(setup.race.maximum_stops, 1)), step=1)
    use_sc = c[1].checkbox("Add a race-control scenario", value=False)
    kind = c[2].selectbox("Type", ["safety_car", "vsc"], disabled=not use_sc)
    start = max(ctx.snapshot.current_lap + 1, 2)
    sc_lap = c[3].number_input("Starts on lap", start, n, value=min(max(start, n // 3), n), disabled=not use_sc)
    events = [{"event_type": kind, "start_lap": int(sc_lap), "duration_laps": int(min(4, n - sc_lap + 1))}] if use_sc else []
    pw = cached_pit_window(ctx.setup_json, ctx.snapshot_json, (int(k),), json.dumps(events))
    m = st.columns(4)
    m[0].metric("Earliest feasible lap", pw.earliest_feasible_lap or "-")
    m[1].metric("Recommended pit lap", pw.recommended_lap or "-", help="Lowest predicted race time among feasible laps (not the window midpoint).")
    m[2].metric("Latest feasible lap", pw.latest_feasible_lap or "-")
    if opt.baseline is not None and pw.recommended_race_time_s is not None and ctx.snapshot.current_lap == 0:
        m[3].metric("Benefit vs baseline", f"{opt.baseline.predicted_race_time_s - pw.recommended_race_time_s:+.2f} s",
                    help="Baseline race time minus recommended race time; positive = recommended is faster.")
    else:
        m[3].metric("Benefit vs baseline", "n/a", help="Baseline comparison is defined for a full race from lap 1.")
    if pw.recommended_strategy is not None:
        st.info(f"Recommended continuation: **{pw.recommended_strategy.strategy_name}** - predicted "
                f"{pw.recommended_race_time_s:,.2f} s. The shaded band is the whole feasible window; the star is the recommended lap.")
    for w in pw.warnings:
        st.warning(w)
    show(plot_pit_window_costs(pw.table, recommended_lap=pw.recommended_lap, earliest_lap=pw.earliest_feasible_lap,
                               latest_lap=pw.latest_feasible_lap))
    with st.expander("Per-lap costs, feasibility flags and rejection reasons"):
        st.dataframe(pw.table.drop(columns=["strategy_id"]), hide_index=True)
        st.download_button("Download pit-window table (CSV)", to_csv_bytes(pw.table), "pit_window.csv", "text/csv")

    comp = ctx.snapshot.tyre_compound
    curves = compare_tyre_compounds(setup.compounds, max_age_laps=50, reference_temperature_c=setup.race.reference_track_temperature_c,
                                    track_temperature_c=setup.race.track_temperature_c, limit_to_recommended_age=True)
    show(plot_tyre_degradation(curves, observed=ctx.lap_data, fuel_effect_s_per_kg=setup.race.fuel_effect_s_per_kg,
                               title="Tyre degradation: model curves and observed (fuel-corrected) laps"
                               + (f" - current set {comp}" if comp else "")))

    st.subheader("Safety-car scenario comparison")
    if ctx.snapshot.current_lap:
        st.caption("Scenario comparison is evaluated for full races from lap 1.")
    full_opt = cached_optimisation(ctx.setup_json)
    if full_opt.best is None:
        st.info("No feasible strategy for scenario comparison.")
        return
    short = shortlist_strategies(full_opt, setup, max_count=4)
    cc = st.columns(3)
    dur = cc[0].slider("Neutralisation length (laps)", 1, 8, 4)
    kind2 = cc[1].selectbox("Neutralisation type", ["safety_car", "vsc"], key="sc_table_kind")
    laps_default = [max(2, round(n * f)) for f in (0.15, 0.3, 0.45, 0.6, 0.75)]
    laps = cc[2].multiselect("Start laps", list(range(2, n)), default=laps_default)
    table = cached_sc_table(ctx.setup_json, json.dumps([s.model_dump(mode="json") for s in short]), tuple(sorted(laps)), dur, kind2)
    show(plot_safety_car_scenarios(table, value_col="delta_to_best_in_scenario_s",
                                   title="Race time vs best strategy in each scenario (lower is better)"))
    st.caption("Each strategy may bring its next stop forward under neutralisation only when the model expects a gain; "
               "stopping under a safety car is not assumed to be beneficial.")
    st.dataframe(pd.DataFrame(table).drop(columns=["strategy_id"]), hide_index=True)
