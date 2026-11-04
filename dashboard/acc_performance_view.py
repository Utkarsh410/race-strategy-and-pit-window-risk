"""ACC Performance view (rendered by ``pages/2_ACC_Performance.py``).

Observations and model outputs are kept apart: telemetry panels refresh in a
fragment and read a snapshot from the receiver's thread-safe analysis, while the
strategy model runs only when its button is pressed. The receiver is the single
process-wide instance from :func:`dashboard.acc_live_view.receiver_manager`, so
reruns and page changes never start a second one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import streamlit as st

from dashboard import acc_performance_data as data
from dashboard.acc_live_view import _config, _receiver_controls, receiver_manager
from dashboard.common import show
from racesim.config import PROJECT_ROOT, load_setup
from racesim.integrations.acc.live_analysis import LiveAnalysis, replay_session_log
from racesim.integrations.acc.live_strategy import BLOCKED, ELIGIBLE, LiveAssumptions, build_inputs, run_live_strategy
from racesim.plotting import plot_lap_time_progression

STRATEGY_KEY = "acc_strategy_result"


def _banner(snap: dict) -> None:
    if snap.get("mock"):
        st.error("MOCK DATA - NOT LIVE ACC TELEMETRY.")
    elif snap.get("replay"):
        st.warning("REPLAY - recorded session log, not live telemetry.")
    elif snap.get("stale"):
        st.warning(f"STALE DATA: last telemetry {snap.get('age_s', 0):.1f} s ago ({snap['data_state']}). Values below are not current.")
    elif snap.get("available"):
        st.success("Receiving live ACC telemetry.")


def _session_panel(snap: dict) -> None:
    st.dataframe(data.overview_table(snap), hide_index=True, width="stretch")


def _lap_panel(snap: dict) -> None:
    st.markdown("**Pace statistics** - clean-lap candidates only, each with its sample size")
    st.dataframe(data.pace_display(data.pace_table(snap)), hide_index=True, width="stretch")
    counts = snap["pace"]["classification_counts"]
    if counts:
        st.caption("Lap classifications: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    st.markdown("**Recent laps** - lap times are LIVE MEASUREMENT; classifications are DERIVED FROM TELEMETRY")
    laps = data.recent_laps_table(snap)
    if laps.empty:
        st.info("No completed laps yet. ACC does not count the out-lap, so the first timed lap is the first to appear.")
    else:
        st.dataframe(data.with_lap_times(laps, ("lap_time_s",)), hide_index=True, width="stretch")
        chart = laps.dropna(subset=["lap_time_s"]).rename(columns={"lap_number": "lap"})[["lap", "lap_time_s"]]
        if len(chart) > 1:
            show(plot_lap_time_progression(chart, title="Lap times, all timed laps"))
    st.markdown("**Sector comparison**")
    sectors = data.sector_table(snap)
    if sectors["n"].sum() == 0:
        st.caption("UNAVAILABLE: no sector times received. They need the updated Windows bridge (see the setup guide).")
    else:
        st.dataframe(sectors, hide_index=True, width="stretch")


def _stint_panel(snap: dict) -> None:
    stints = data.stint_table(snap)
    if stints.empty:
        st.info("No stint yet: waiting for a valid car location.")
    else:
        st.dataframe(data.with_lap_times(stints, ("median_s",)), hide_index=True, width="stretch")
    st.caption(snap["pace"]["trend_note"])
    if snap["transitions"]:
        st.markdown("**Pit transitions** (DERIVED FROM TELEMETRY; a pit visit does not prove a tyre change)")
        st.dataframe(snap["transitions"], hide_index=True, width="stretch")


def _fuel_tyre_panel(snap: dict) -> None:
    st.markdown("**Fuel**")
    st.dataframe(data.fuel_table(snap), hide_index=True, width="stretch")
    for flag in snap["fuel"]["flags"]:
        st.warning(f"Fuel reading: {flag}")
    st.markdown("**Tyres** (measurements only; no tyre-model values appear here)")
    st.dataframe(data.tyre_table(snap), hide_index=True, width="stretch")
    if snap["missing_channels"]:
        st.warning("Channels not supplied by this connection: " + ", ".join(c["channel"] for c in snap["missing_channels"])
                   + ". ACC's UDP broadcasting API does not carry them; they need a future Windows shared-memory reader.")


def _assumption_form(setup) -> LiveAssumptions:
    st.markdown("**Configured assumptions** - leave blank if unknown. Nothing is filled in for you.")
    c = st.columns(3)
    def opt_number(col, label, key, **kw) -> Optional[float]:
        return col.number_input(label, value=None, key=f"acc_a_{key}", placeholder="not stated", **kw)
    total = opt_number(c[0], "Race distance (laps)", "laps", min_value=1, max_value=1000, step=1)
    compound = c[0].selectbox("Current tyre compound", setup.compound_names, index=None, placeholder="not stated", key="acc_a_compound")
    age = opt_number(c[1], "Current tyre age (laps)", "age", min_value=0, max_value=1000, step=1)
    fuel = opt_number(c[1], "Current fuel (kg)", "fuel", min_value=0.0, max_value=500.0, step=1.0)
    burn = opt_number(c[2], "Fuel use (kg/lap)", "burn", min_value=0.0, max_value=100.0, step=0.1)
    pit = opt_number(c[2], "Pit-lane loss (s)", "pit", min_value=0.0, max_value=600.0, step=1.0)
    stop = opt_number(c[2], "Stationary time (s)", "stop", min_value=0.0, max_value=600.0, step=0.5)
    calibrated = st.checkbox("I have calibrated the tyre model in the configuration for this car and track", key="acc_a_cal",
                             help="The default configuration is synthetic. Without this, no pit lap is ever recommended.")
    return LiveAssumptions(total_laps=None if total is None else int(total), tyre_compound=compound, tyre_age_laps=None if age is None else int(age),
                           fuel_kg=fuel, fuel_kg_per_lap=burn, pit_lane_loss_s=pit, stationary_stop_time_s=stop, tyre_model_calibrated=calibrated)


def _strategy_panel(analysis: LiveAnalysis, snap: dict) -> None:
    st.caption("Everything in this panel except the input table is MODEL PREDICTION. RaceSim only reports analysis and sends nothing to the game.")
    setup = load_setup()
    assumptions = _assumption_form(setup)
    inputs = build_inputs(snap, assumptions)
    st.markdown(f"**Readiness: {inputs.status}**")
    for reason in inputs.reasons:
        st.caption(f"- {reason}")
    st.dataframe(data.strategy_inputs_table(inputs.rows()), hide_index=True, width="stretch")
    if st.button("Run strategy analysis", disabled=inputs.status == BLOCKED, help="Runs on demand only, never per packet."):
        with st.spinner("Running the optimiser and pit-window analysis..."):
            result = run_live_strategy(setup, inputs).as_dict()
        analysis.record_strategy(result)
        st.session_state[STRATEGY_KEY] = result
    result = st.session_state.get(STRATEGY_KEY)
    if not result or result.get("session_key") != snap.get("session_key"):
        return
    ran_at, laps_now = result.get("laps_completed_at_run"), snap.get("laps_completed")
    st.markdown(f"**MODEL PREDICTION** - run at {ran_at} completed laps"
                + (f" ({laps_now - ran_at} lap(s) ago)" if None not in (ran_at, laps_now) else ""))
    if result["status"] == BLOCKED:
        for reason in result["reasons"]:
            st.error(reason)
        return
    if result["status"] == ELIGIBLE and result["recommended_pit_lap"] is not None:
        st.info(f"Lowest predicted time pits on lap {result['recommended_pit_lap']}.")
    else:
        st.warning("Scenario analysis only: no pit stop is recommended, because required inputs are assumed or unreliable.")
    st.dataframe(data.candidates_table(result), hide_index=True, width="stretch")
    window = result.get("pit_window")
    if window:
        k = st.columns(3)
        k[0].metric("Earliest feasible pit lap", window["earliest_feasible_lap"])
        k[1].metric("Latest feasible pit lap", window["latest_feasible_lap"])
        k[2].metric("Lowest predicted time at lap", window["lowest_predicted_time_lap"])
    else:
        st.caption("Pit window: the lowest predicted time has no further stop.")
    for warning in result["warnings"]:
        st.warning(warning)
    with st.expander("Model assumptions"):
        for assumption in result["assumptions"]:
            st.markdown(f"- {assumption}")


def _panels(snap: dict) -> None:
    tabs = st.tabs(["Session overview", "Lap performance", "Stint analysis", "Fuel and tyres"])
    for tab, panel in zip(tabs, (_session_panel, _lap_panel, _stint_panel, _fuel_tyre_panel)):
        with tab:
            panel(snap)


def _session_choice(analysis: LiveAnalysis) -> Optional[str]:
    keys = list(analysis.sessions)
    if len(keys) < 2:
        return None
    return st.selectbox("Session", list(reversed(keys)), help="Sessions are analysed separately; a restart starts a new one.")


def _render_live(refresh_s: float) -> None:
    mgr = receiver_manager()
    analysis = mgr.analysis
    if analysis is None:
        st.info("Receiver not started. Use **Start** in the sidebar. Nothing is shown until real messages arrive.")
        return
    key = _session_choice(analysis)

    @st.fragment(run_every=refresh_s)
    def live() -> None:
        snap = analysis.snapshot(session_key=key)
        _banner(snap)
        if not snap["available"]:
            st.info("Waiting for telemetry.")
            return
        _panels(snap)

    live()
    st.divider()
    st.subheader("Strategy analysis")
    _strategy_panel(analysis, analysis.snapshot(session_key=key))
    if mgr.exports:
        st.caption("Session exports: " + ", ".join(str(p) for p in mgr.exports.values()))


def _render_replay() -> None:
    st.warning("REPLAY MODE - a recorded session log run through the live analysis, not live telemetry.")
    log_dir = PROJECT_ROOT / "outputs" / "acc_sessions"
    files = sorted(log_dir.glob("*.jsonl"), reverse=True) if log_dir.exists() else []
    if not files:
        st.info("No session logs found in outputs/acc_sessions.")
        return
    path: Path = st.selectbox("Session log", files, format_func=lambda p: p.name)
    analysis, last = _cached_replay(str(path), path.stat().st_mtime)
    if not analysis.sessions:
        st.info("This log contains no telemetry.")
        return
    key = _session_choice(analysis)
    snap = analysis.snapshot(last, key)
    _banner(snap)
    _panels(snap)
    st.download_button("Download laps CSV", data.recent_laps_table(snap, n=100_000).to_csv(index=False).encode(), f"{snap['session_key']}_laps.csv", "text/csv")
    st.download_button("Download stints CSV", data.stint_table(snap).to_csv(index=False).encode(), f"{snap['session_key']}_stints.csv", "text/csv")


@st.cache_resource(max_entries=4, show_spinner="Replaying session log...")
def _cached_replay(path: str, mtime: float) -> tuple[LiveAnalysis, float]:
    return replay_session_log(path)


def render() -> None:
    st.title("ACC Performance")
    st.caption("Read-only lap, stint and strategy analysis of live ACC telemetry. Labels: LIVE MEASUREMENT, DERIVED FROM TELEMETRY, "
               "MODEL PREDICTION, CONFIGURED ASSUMPTION, UNAVAILABLE, STALE DATA.")
    cfg, cfg_err = _config()
    if cfg_err:
        st.warning(cfg_err)
    if st.sidebar.radio("Mode", ["Live receiver", "Replay a recorded session log"]).startswith("Replay"):
        _render_replay()
        return
    _receiver_controls(cfg, receiver_manager())
    _render_live(st.sidebar.slider("Refresh interval (s)", 0.5, 5.0, 1.0, 0.5))
