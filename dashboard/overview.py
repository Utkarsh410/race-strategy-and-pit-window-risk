"""Section A - race overview: configuration, headline numbers, current race state."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.common import DashboardContext, cached_optimisation, cached_pit_window, ss


def _controls(ctx: DashboardContext) -> None:
    r = ctx.setup.race
    st.subheader("Race controls")
    c1, c2, c3, c4 = st.columns(4)
    ss("ov_total_laps", r.total_laps)
    c1.number_input("Race distance (laps)", 10, 300, step=1, key="ov_total_laps", help="Integer laps. Rival and baseline pit laps scale with distance.")
    ss("ov_baseline", r.baseline_lap_time_s)
    c2.number_input("Baseline lap time (s)", 30.0, 600.0, step=0.1, key="ov_baseline",
                    help="Empty tank, fresh reference tyre, reference track temperature. All compound references shift with it.")
    ss("ov_pit_loss", r.pit_lane_loss_s)
    c3.number_input("Pit-lane loss (s)", 0.0, 120.0, step=0.5, key="ov_pit_loss",
                    help="Entry + transit + exit loss versus staying on track, EXCLUDING stationary time.")
    ss("ov_stationary", r.stationary_stop_time_s)
    c4.number_input("Stationary time (s)", 0.0, 60.0, step=0.1, key="ov_stationary", help="Mean time stationary in the box.")
    st.caption(f"Total green-flag pit cost = {r.pit_lane_loss_s:.1f} + {r.stationary_stop_time_s:.1f} = "
               f"{r.total_pit_cost_s:.1f} s (counted once per stop; out-lap warm-up handled by the tyre model).")


def _snapshot_controls(ctx: DashboardContext) -> None:
    n = ctx.setup.race.total_laps
    st.subheader("Current race state")
    st.caption("Set a lap > 0 to analyse the pit window from a point in the race (used by Section C).")
    c1, c2, c3, c4 = st.columns(4)
    ss("snap_lap", 0)
    if st.session_state["snap_lap"] > n - 1:
        st.session_state["snap_lap"] = 0
    c1.number_input("Current lap (completed)", 0, n - 1, step=1, key="snap_lap")
    dry = ctx.setup.dry_compounds
    ss("snap_compound", dry[min(1, len(dry) - 1)])
    if st.session_state["snap_compound"] not in dry:
        st.session_state["snap_compound"] = dry[0]
    c2.selectbox("Current compound", dry, key="snap_compound", disabled=ctx.snapshot.current_lap == 0)
    ss("snap_age", max(int(st.session_state["snap_lap"]), 0))
    c3.number_input("Current tyre age (laps)", 0, 200, step=1, key="snap_age", disabled=ctx.snapshot.current_lap == 0)
    ss("snap_stops", 0)
    c4.number_input("Stops completed", 0, 10, step=1, key="snap_stops", disabled=ctx.snapshot.current_lap == 0)
    ss("snap_used", [])
    st.multiselect("Other compounds already used", dry, key="snap_used", disabled=ctx.snapshot.current_lap == 0)


def render(ctx: DashboardContext) -> None:
    setup, r = ctx.setup, ctx.setup.race
    _controls(ctx)
    opt = cached_optimisation(ctx.setup_json)
    best = opt.best
    pw = cached_pit_window(ctx.setup_json, ctx.snapshot_json, (max(best.number_of_stops, 1),) if best else (1,))
    st.subheader("Headline")
    k = st.columns(4)
    k[0].metric("Total laps", r.total_laps)
    k[1].metric("Baseline lap time", f"{r.baseline_lap_time_s:.3f} s")
    k[2].metric("Current lap", ctx.snapshot.current_lap)
    k[3].metric("Current tyre age", f"{ctx.snapshot.tyre_age_laps} laps" if ctx.snapshot.current_lap else "new (pre-race)")
    k = st.columns(4)
    if best is None:
        st.error("No feasible strategy satisfies the configured constraints. Relax stint limits, stops or compound rules.")
    else:
        delta = (best.predicted_race_time_s - opt.baseline.predicted_race_time_s) if opt.baseline else None
        k[0].metric("Recommended strategy", best.strategy.strategy_name)
        k[1].metric("Planned stops", best.number_of_stops)
        k[2].metric("Predicted race time", f"{best.predicted_race_time_s:,.2f} s",
                    delta=None if delta is None else f"{delta:+.2f} s vs baseline", delta_color="inverse",
                    help="Deterministic prediction (no SC, traffic or variability). Lower is better.")
        k[3].metric("Recommended next pit lap", pw.recommended_lap if pw.recommended_lap else "none feasible",
                    help=f"Minimum-cost lap within the feasible window {pw.earliest_feasible_lap}-{pw.latest_feasible_lap}"
                         + (" from the current race state." if ctx.snapshot.current_lap else "."))
    _snapshot_controls(ctx)

    st.subheader("Warnings")
    msgs = list(ctx.config_warnings)
    if ctx.data is not None:
        msgs += [f"Data: {w}" for w in ctx.data.report.warnings]
        msgs += [f"Data error: {e}" for e in ctx.data.report.errors]
    if not msgs:
        st.success("No configuration or data-quality warnings.")
    for m in msgs:
        st.warning(m)
    with st.expander("Race configuration (all values)"):
        st.dataframe(pd.DataFrame({"parameter": list(r.model_dump()), "value": [str(v) for v in r.model_dump().values()]}),
                     hide_index=True)
    with st.expander("Technical assumptions"):
        st.markdown(
            "- **Units:** seconds, kilograms, integer 1-indexed laps; stint intervals are inclusive.\n"
            "- **Pit stop on lap L:** in-lap L carries the full pit cost; the out-lap is L+1 on new tyres (age 0).\n"
            "- **Lap time** = compound reference + tyre degradation & warm-up + fuel effect + traffic + weather + "
            "track evolution / standing start + driver + random error (0 in deterministic mode). Each effect applied once.\n"
            "- **Neutralised laps** run at baseline x SC/VSC factor with reduced tyre wear and fuel use.\n"
            f"- **Rules:** {setup.regulations.championship_name}; refuelling "
            f"{'permitted' if setup.refuelling_allowed else 'not permitted'}.")
