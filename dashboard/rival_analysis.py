"""Section E - rival strategy: gap, rival pit window, undercut / overcut, rival response."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from dashboard.common import DashboardContext, cached_optimisation, show
from racesim.plotting import SERIES, _layout, plot_rival_gaps
from racesim.race_simulator import simulate_race
from racesim.rival_model import RIVAL_ASSUMPTIONS, evaluate_overcut, evaluate_rival_response, evaluate_undercut, predict_rival_pit_window
from racesim.schemas import RaceSnapshot


@st.cache_data(show_spinner="Predicting rival pit window...", max_entries=16)
def _rival_window(setup_json: str, idx: int, current_lap: int, compound: str, age: int):
    from dashboard.common import setup_from_json

    setup = setup_from_json(setup_json)
    rival = setup.rivals[idx]
    snap = RaceSnapshot(current_lap=current_lap, tyre_compound=compound, tyre_age_laps=age) if current_lap else None
    return predict_rival_pit_window(setup, rival, snap, n_stops=[1])


def render(ctx: DashboardContext) -> None:
    setup = ctx.setup
    n = setup.race.total_laps
    with st.expander("Simplified assumptions (read before using these numbers)", expanded=True):
        st.markdown("\n".join(f"- {a}" for a in RIVAL_ASSUMPTIONS) +
                    "\n- Position predictions come from elapsed time only; they are not wheel-to-wheel predictions.")
    if not setup.rivals:
        st.info("No rivals configured in the race YAML; a custom rival can still be analysed below.")
    st.subheader("Whole-race comparison with configured rivals")
    opt = cached_optimisation(ctx.setup_json)
    if opt.best is not None and setup.rivals:
        res = simulate_race(setup, opt.best.strategy, rivals=setup.rivals)
        gaps = {setup.rivals[i].name: g for i, g in enumerate(res.simulation_metadata["gap_history_s"].values())}
        show(plot_rival_gaps(gaps, list(range(1, n + 1)), title=f"Gap: own car on {opt.best.strategy.strategy_name}"))
        c = st.columns(len(setup.rivals) + 1)
        c[0].metric("Own predicted position", res.finishing_position)
        for i, (rid, info) in enumerate(res.simulation_metadata["rivals"].items()):
            c[i + 1].metric(setup.rivals[i].name, f"P{info['finishing_position']}",
                            help=f"Plan {setup.rivals[i].pit_laps} on {setup.rivals[i].compounds}")
        st.caption("Deterministic race. Includes dirty-air and safety-car bunching interactions; see Section D for probabilities.")

    st.subheader("Undercut / overcut analysis")
    c = st.columns(4)
    names = [r.name for r in setup.rivals] + ["Custom rival"]
    pick = c[0].selectbox("Rival", names)
    rv = setup.rivals[names.index(pick)] if pick != "Custom rival" else None
    current = c[1].number_input("Current lap (completed)", 1, n - 5, value=min(max(n // 3 - 2, 1), n - 5))
    gap = c[2].number_input("Current gap to rival (s, + = own car behind)", -30.0, 30.0, value=1.5, step=0.1)
    pace = c[3].number_input("Rival pace offset (s/lap, - = faster)", -3.0, 3.0, value=float(rv.pace_offset_s) if rv else 0.0, step=0.05)
    dry = setup.dry_compounds
    c = st.columns(4)
    own_c = c[0].selectbox("Own current compound", dry, index=min(1, len(dry) - 1))
    own_age = c[1].number_input("Own tyre age (laps)", 0, 100, value=int(current))
    new_c = c[2].selectbox("Own next compound", dry, index=len(dry) - 1)
    deg = c[3].number_input("Rival degradation multiplier", 0.2, 3.0, value=float(rv.degradation_multiplier) if rv else 1.0, step=0.05)
    c = st.columns(4)
    r_c = c[0].selectbox("Rival current compound", dry, index=dry.index(rv.compounds[0]) if rv and rv.compounds[0] in dry else min(1, len(dry) - 1))
    r_age = c[1].number_input("Rival tyre age (laps)", 0, 100, value=int(current))
    r_new = c[2].selectbox("Rival next compound", dry, index=len(dry) - 1)
    own_pit = c[3].number_input("Own pit lap", current + 1, n - setup.race.minimum_pit_laps_remaining, value=min(current + 2, n - 4))

    if rv is not None:
        try:
            win = _rival_window(ctx.setup_json, names.index(pick), int(current), r_c, int(r_age))
            st.info(f"Predicted rival pit window (same model, rival degradation x{rv.degradation_multiplier:.2f}): "
                    f"laps {win.earliest_feasible_lap}-{win.latest_feasible_lap}, most likely lap {win.recommended_lap}.")
        except Exception as exc:  # e.g. infeasible state
            st.warning(f"Rival pit window unavailable: {exc}")
    rival_pit = st.slider("Rival pit lap (response)", int(current) + 1, n - setup.race.minimum_pit_laps_remaining,
                          value=min(int(own_pit) + 2, n - setup.race.minimum_pit_laps_remaining))
    kw = dict(current_lap=int(current), gap_s=float(gap), own_compound=own_c, own_tyre_age_laps=int(own_age), own_new_compound=new_c,
              rival_compound=r_c, rival_tyre_age_laps=int(r_age), rival_new_compound=r_new, rival_pace_offset_s=float(pace),
              rival_degradation_multiplier=float(deg))
    if own_pit == rival_pit:
        st.info("Own and rival pit on the same lap: neither an undercut nor an overcut. Change one of the laps.")
        return
    fn = evaluate_undercut if own_pit < rival_pit else evaluate_overcut
    res = fn(setup, own_pit_lap=int(own_pit), rival_pit_lap=int(rival_pit), **kw)
    m = st.columns(4)
    m[0].metric("Current gap", f"{gap:+.2f} s")
    m[1].metric(f"Gap after both stops ({res.kind})", f"{res.gap_after_both_stops_s:+.2f} s",
                delta=f"{res.gain_s:+.2f} s gained", delta_color="normal")
    m[2].metric("Own rejoin position vs rival", f"P{res.own_rejoin['position']}",
                help="Comparing elapsed times at the end of the own in-lap (simplified).")
    m[3].metric("Predicted outcome", "Own car ahead" if res.successful else "Rival stays ahead")
    st.caption(f"Pace-only {res.kind} gain (no traffic, same model): {res.pace_only_gain_s:+.2f} s. " + " ".join(res.notes))
    show(plot_rival_gaps(res.gap_history[["lap", "gap_s"]].rename(columns={"gap_s": "Gap to rival"}),
                         title=f"Gap evolution - own pit L{own_pit}, rival pit L{rival_pit}"))

    st.subheader("Sensitivity to the rival's response")
    lo = int(current) + 1
    hi = n - setup.race.minimum_pit_laps_remaining
    laps = list(range(max(lo, int(own_pit) - 6), min(hi, int(own_pit) + 8) + 1))
    resp = evaluate_rival_response(setup, own_pit_lap=int(own_pit), rival_pit_laps=laps, **{k: v for k, v in kw.items()})
    if resp.empty:
        st.info("No valid rival responses in range.")
        return
    fig = go.Figure(go.Scatter(x=resp["rival_pit_lap"], y=resp["gap_after_both_stops_s"], mode="lines+markers",
                               line={"width": 2, "color": SERIES[0]}, marker={"size": 9}, name="Gap after both stops (model)",
                               hovertemplate="Rival pits L%{x}<br>Gap %{y:+.2f} s<extra></extra>"))
    fig.add_hline(y=0, line={"color": "#52514e", "width": 1, "dash": "dash"})
    fig.add_vline(x=int(own_pit), line={"color": SERIES[1], "width": 1, "dash": "dot"}, annotation_text="own pit lap")
    show(_layout(fig, "Gap after both stops vs rival pit lap (negative = own car ahead)", "Rival pit lap", "Gap (s)", legend=False))
    st.dataframe(resp, hide_index=True)
