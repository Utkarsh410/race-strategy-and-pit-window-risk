"""Section B - strategy planner: constraints, tyre/fuel assumptions, ranked strategies."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.common import DashboardContext, base_setup, cached_optimisation, show, ss, to_csv_bytes
from dashboard.theme import HIGHLIGHT
from racesim.plotting import plot_strategy_comparison, plot_tyre_degradation
from racesim.schemas import Strategy
from racesim.strategy_optimizer import simulate_strategy
from racesim.tyre_model import compare_tyre_compounds


def _assumption_controls(ctx: DashboardContext) -> None:
    base = base_setup()
    r = ctx.setup.race
    st.subheader("Constraints")
    c = st.columns(5)
    ss("ov_max_stops", base.race.maximum_stops)
    c[0].number_input("Maximum stops", 0, 6, step=1, key="ov_max_stops")
    ss("ov_min_stint", base.race.minimum_stint_laps)
    c[1].number_input("Min stint (laps)", 1, 100, step=1, key="ov_min_stint", help="Planning limit; regulations may be stricter.")
    ss("ov_max_stint", base.race.maximum_stint_laps)
    c[2].number_input("Max stint (laps)", 1, 300, step=1, key="ov_max_stint")
    ss("ov_mandatory", base.regulations.mandatory_pit_stops or 0)
    c[3].number_input("Mandatory stops", 0, 5, step=1, key="ov_mandatory")
    ss("ov_distinct", base.regulations.minimum_distinct_dry_compounds or 1)
    c[4].number_input("Min distinct dry compounds", 1, 3, step=1, key="ov_distinct")
    ss("ov_compounds", list(base.dry_compounds))
    st.multiselect("Permitted dry compounds", [n for n, cc in base.compounds.items() if cc.compound_type == "dry"],
                   key="ov_compounds", help="Wet compounds stay available for rain reactions.")
    if not st.session_state["ov_compounds"]:
        st.warning("Select at least one dry compound.")

    with st.expander("Tyre degradation parameters", expanded=False):
        st.caption("Linear: loss = k * age. Nonlinear adds c * max(0, age - threshold)^2. Values are illustrative unless calibrated.")
        for name, comp in base.compounds.items():
            if comp.compound_type != "dry":
                continue
            cc = st.columns(4)
            ss(f"ov_lin_{name}", comp.linear_degradation_s_per_lap)
            cc[0].number_input(f"{name} linear (s/lap)", 0.0, 2.0, step=0.005, format="%.3f", key=f"ov_lin_{name}")
            ss(f"ov_nl_{name}", comp.nonlinear_degradation_coefficient)
            cc[1].number_input(f"{name} nonlinear (s/lap^2)", 0.0, 0.5, step=0.001, format="%.4f", key=f"ov_nl_{name}")
            ss(f"ov_thr_{name}", comp.nonlinear_degradation_threshold_laps)
            cc[2].number_input(f"{name} threshold (laps)", 0.0, 200.0, step=1.0, key=f"ov_thr_{name}")
            ss(f"ov_age_{name}", comp.maximum_recommended_age_laps)
            cc[3].number_input(f"{name} max age (laps)", 1, 300, step=1, key=f"ov_age_{name}")
        ss("ov_track_temp", base.race.track_temperature_c)
        st.number_input("Track temperature (degC)", -10.0, 80.0, step=0.5, key="ov_track_temp",
                        help=f"Affects degradation rate via each compound's temperature sensitivity (reference {r.reference_track_temperature_c} degC).")
    with st.expander("Fuel assumptions", expanded=False):
        cc = st.columns(4)
        ss("ov_fuel_cons", base.race.fuel_consumption_kg_per_lap)
        cc[0].number_input("Consumption (kg/lap)", 0.0, 20.0, step=0.05, key="ov_fuel_cons", help="Explicit assumption unless measured.")
        ss("ov_fuel_effect", base.race.fuel_effect_s_per_kg)
        cc[1].number_input("Fuel effect (s/kg)", 0.0, 0.5, step=0.005, format="%.3f", key="ov_fuel_effect")
        ss("ov_auto_fuel", True)
        cc[2].checkbox("Auto starting fuel", key="ov_auto_fuel", help="Race distance x consumption + 5 kg reserve.")
        ss("ov_fuel_start", base.race.fuel_start_kg)
        cc[3].number_input("Starting fuel (kg)", 0.0, 500.0, step=1.0, key="ov_fuel_start", disabled=st.session_state["ov_auto_fuel"])
        st.caption(f"Effective starting fuel: {r.fuel_start_kg:.1f} kg. Refuelling "
                   f"{'permitted' if ctx.setup.refuelling_allowed else 'not permitted by the regulations'}.")
    with st.expander("Weather scenario", expanded=False):
        cc = st.columns(4)
        ss("ov_weather", base.weather.scenario)
        cc[0].selectbox("Scenario", ["stable", "rising_temperature", "cooling_temperature", "rain_onset"], key="ov_weather")
        ss("ov_temp_rate", 0.2)
        cc[1].number_input("Temp change (degC/lap)", 0.0, 3.0, step=0.05, key="ov_temp_rate")
        ss("ov_rain_lap", max(2, r.total_laps // 2))
        cc[2].number_input("Rain onset lap", 1, r.total_laps, step=1, key="ov_rain_lap")
        ss("ov_wet_enabled", base.weather.wet_weather_enabled)
        cc[3].checkbox("Wet-weather model enabled", key="ov_wet_enabled")
        st.caption(f"Weather model status: {ctx.setup.weather.calibration_status}. Deterministic weather applies to all strategies equally.")


def render(ctx: DashboardContext) -> None:
    _assumption_controls(ctx)
    opt = cached_optimisation(ctx.setup_json)
    st.subheader("Ranked strategies (deterministic, same model for all)")
    if opt.best is None:
        st.error("No feasible strategy. Rejection examples: " + "; ".join(r for _, r in opt.rejected[:5]))
        return
    best_id = opt.best.strategy.strategy_id
    c1, c2 = st.columns([1, 1])
    one_per_pattern = c1.toggle("Best timing per compound sequence only", value=True,
                                help="Hide near-identical pit-lap variants of the same compound sequence.")
    top_n = c2.slider("Rows", 5, 50, 15)
    df = opt.ranking.copy()
    if one_per_pattern:
        df = df.drop_duplicates("compounds")
    base_t = opt.baseline.predicted_race_time_s if opt.baseline else None
    cols = ["rank", "strategy_name", "pit_laps", "compounds", "stint_lengths", "race_time_s", "total_pit_cost_s",
            "tyre_time_s", "feasible"] + (["delta_to_baseline_s"] if "delta_to_baseline_s" in df.columns else [])
    view = df[cols + ["strategy_id"]].head(top_n)
    styled = view.style.apply(lambda row: [f"background-color:{HIGHLIGHT}; font-weight:600" if row["strategy_id"] == best_id else ""
                                           for _ in row], axis=1).format(precision=2)
    st.dataframe(styled, hide_index=True, column_config={"strategy_id": None})
    st.caption(f"{opt.candidates_evaluated} candidates evaluated; pit-lap grid steps by stop count {opt.lap_steps} "
               f"(then refined lap-by-lap). Recommended strategy highlighted. Baseline: "
               f"{opt.baseline.strategy.strategy_name if opt.baseline else 'not configured'}.")
    show(plot_strategy_comparison(view, highlight_id=best_id, baseline_time_s=base_t, top_n=top_n,
                                  title="Predicted race time vs baseline" if base_t else "Predicted race time vs best"))
    with st.expander("Why is this strategy recommended?", expanded=True):
        st.text(opt.explanation)
    st.download_button("Download strategy comparison (CSV)", to_csv_bytes(opt.ranking), "strategy_comparison.csv", "text/csv")

    st.subheader("Evaluate a custom strategy")
    c = st.columns([2, 2, 1])
    laps_txt = c[0].text_input("Pit laps (comma separated)", value=", ".join(map(str, opt.best.strategy.planned_pit_laps)))
    comps_txt = c[1].text_input("Compounds (one more than pit laps)", value=", ".join(opt.best.strategy.planned_compounds))
    if c[2].button("Evaluate", width="stretch"):
        try:
            laps = [int(x) for x in laps_txt.replace(" ", "").split(",") if x]
            comps = [x.strip().upper() for x in comps_txt.split(",") if x.strip()]
            strat = Strategy.from_plan(ctx.setup.race.total_laps, laps, comps)
            ev = simulate_strategy(ctx.setup, strat)
            if ev.feasible:
                st.success(f"{strat.strategy_name}: {ev.predicted_race_time_s:,.2f} s "
                           f"({ev.predicted_race_time_s - opt.best.predicted_race_time_s:+.2f} s vs recommended).")
            else:
                st.error(f"Infeasible: {'; '.join(ev.reasons)} (predicted {ev.predicted_race_time_s:,.2f} s if allowed).")
        except (ValueError, KeyError) as exc:
            st.error(f"Invalid strategy: {exc}")

    st.subheader("Tyre model")
    curves = compare_tyre_compounds({n: c for n, c in ctx.setup.compounds.items() if n in ctx.setup.dry_compounds},
                                    max_age_laps=50, reference_temperature_c=ctx.setup.race.reference_track_temperature_c,
                                    track_temperature_c=ctx.setup.race.track_temperature_c, limit_to_recommended_age=True)
    show(plot_tyre_degradation(curves, title="Modelled degradation at the configured track temperature"))
    rejected = pd.DataFrame(opt.rejected[:200], columns=["candidate", "reason"])
    with st.expander(f"Rejected candidates ({len(opt.rejected)} recorded)"):
        st.dataframe(rejected, hide_index=True)
