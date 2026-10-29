"""Reusable Plotly figures.

Every figure has a title, axis labels with units and a legend where more than
one series is shown. Observations are drawn as markers and model predictions
as lines so the two are never confused. Empty or incomplete inputs return a
placeholder figure explaining what is missing instead of raising.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Validated categorical order (fixed slots, never cycled).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
NEUTRAL = "#8a8984"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e6e5e1"
SURFACE = "#fcfcfb"
STATUS_FILL = {"SC": "rgba(237,161,0,0.18)", "VSC": "rgba(98,80,214,0.14)"}


def color_for(i: int) -> str:
    """Categorical colour by slot; beyond eight series the neutral is used (fold into 'Other')."""
    return SERIES[i] if i < len(SERIES) else NEUTRAL


def _layout(fig: go.Figure, title: str, x: str, y: str, *, height: int = 420, legend: bool = True) -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0.0, "xanchor": "left", "font": {"size": 16, "color": TEXT}},
        xaxis_title=x, yaxis_title=y, height=height, template="plotly_white",
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font={"color": MUTED, "size": 12},
        margin={"l": 60, "r": 20, "t": 60, "b": 50}, showlegend=legend, hovermode="closest",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "xanchor": "right", "x": 1.0},
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


def empty_figure(title: str, message: str) -> go.Figure:
    """Placeholder figure stating why a chart cannot be drawn."""
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper",
                       font={"size": 13, "color": MUTED})
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _layout(fig, title, "", "", height=260, legend=False)


def _shade_status(fig: go.Figure, laps: Sequence[int], status: Sequence[str]) -> None:
    status = list(status)
    i = 0
    while i < len(status):
        s = status[i]
        if s in STATUS_FILL:
            j = i
            while j < len(status) and status[j] == s:
                j += 1
            fig.add_vrect(x0=laps[i] - 0.5, x1=laps[j - 1] + 0.5, fillcolor=STATUS_FILL[s], line_width=0,
                          annotation_text=s, annotation_position="top left", annotation_font_color=MUTED)
            i = j
        else:
            i += 1


def plot_lap_time_progression(observed: Optional[pd.DataFrame] = None, predicted: Optional[pd.DataFrame] = None,
                              *, title: str = "Lap-time progression") -> go.Figure:
    """Observed lap times (markers, per driver) and predicted lap times (line).

    ``observed`` needs ``lap`` and ``lap_time_s`` (optional ``driver``,
    ``track_status``); ``predicted`` needs ``lap`` and ``lap_time_s`` and may
    hold several strategies in a ``strategy`` column.
    """
    if (observed is None or observed.empty) and (predicted is None or predicted.empty):
        return empty_figure(title, "No lap data available.")
    fig = go.Figure()
    k = 0
    if observed is not None and not observed.empty:
        groups = observed.groupby("driver") if "driver" in observed.columns else [("Observed", observed)]
        for name, g in groups:
            g = g.sort_values("lap")
            fig.add_trace(go.Scatter(x=g["lap"], y=g["lap_time_s"], mode="markers", name=f"{name} (observed)",
                                     marker={"size": 8, "color": color_for(k), "line": {"width": 1, "color": SURFACE}},
                                     hovertemplate="Lap %{x}<br>%{y:.3f} s<extra>%{fullData.name}</extra>"))
            k += 1
        if "track_status" in observed.columns:
            first = observed.drop_duplicates("lap").sort_values("lap")
            _shade_status(fig, first["lap"].tolist(), first["track_status"].astype(str).str.upper().tolist())
    if predicted is not None and not predicted.empty:
        groups = predicted.groupby("strategy") if "strategy" in predicted.columns else [("Predicted", predicted)]
        for name, g in groups:
            fig.add_trace(go.Scatter(x=g["lap"], y=g["lap_time_s"], mode="lines", name=f"{name} (model)",
                                     line={"width": 2, "color": color_for(k)},
                                     hovertemplate="Lap %{x}<br>%{y:.3f} s<extra>%{fullData.name}</extra>"))
            k += 1
        if observed is None and "track_status" in predicted.columns:
            first = predicted.drop_duplicates("lap")
            _shade_status(fig, first["lap"].tolist(), first["track_status"].tolist())
    return _layout(fig, title, "Lap", "Lap time (s)")


def plot_tyre_degradation(model_curves: Optional[pd.DataFrame] = None, observed: Optional[pd.DataFrame] = None,
                          *, title: str = "Tyre degradation by compound", fuel_effect_s_per_kg: Optional[float] = None) -> go.Figure:
    """Model degradation curves (lines) with optional observed fuel-corrected laps (markers).

    ``model_curves`` is the output of :func:`compare_tyre_compounds`. Observed
    points are plotted relative to each stint's best lap so they share the
    curves' "time loss vs fresh tyre" axis.
    """
    if (model_curves is None or model_curves.empty) and (observed is None or observed.empty):
        return empty_figure(title, "No tyre data or model curves available.")
    fig = go.Figure()
    order = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]
    comps = []
    for df in (model_curves, observed):
        if df is not None and not df.empty:
            col = "compound" if "compound" in df.columns else "tyre_compound"
            comps += [c for c in df[col].astype(str).unique() if c not in comps]
    comps.sort(key=lambda c: order.index(c) if c in order else 99)
    cmap = {c: color_for(i) for i, c in enumerate(comps)}
    if model_curves is not None and not model_curves.empty:
        for c, g in model_curves.groupby("compound"):
            fig.add_trace(go.Scatter(x=g["tyre_age_laps"], y=g["degradation_s"], mode="lines", name=f"{c} (model)",
                                     line={"width": 2, "color": cmap[c]},
                                     hovertemplate="Age %{x} laps<br>+%{y:.3f} s<extra>%{fullData.name}</extra>"))
    if observed is not None and not observed.empty and {"tyre_age_laps", "tyre_compound", "lap_time_s"} <= set(observed.columns):
        from racesim.data_validation import assign_stint_ids, representative_lap_mask

        obs = assign_stint_ids(observed[representative_lap_mask(observed)])
        y = obs["lap_time_s"].astype(float)
        if fuel_effect_s_per_kg is not None and "fuel_kg" in obs.columns:
            y = y - fuel_effect_s_per_kg * obs["fuel_kg"]
        obs = obs.assign(_y=y - y.groupby(obs["stint_id"]).transform("min"))
        for c, g in obs.groupby("tyre_compound"):
            fig.add_trace(go.Scatter(x=g["tyre_age_laps"], y=g["_y"], mode="markers", name=f"{c} (observed)",
                                     marker={"size": 8, "color": cmap.get(c, NEUTRAL), "opacity": 0.55,
                                             "line": {"width": 1, "color": SURFACE}},
                                     hovertemplate="Age %{x} laps<br>+%{y:.3f} s<extra>%{fullData.name}</extra>"))
    return _layout(fig, title, "Tyre age (laps)", "Time loss vs fresh tyre (s)")


def plot_pit_window_costs(window: pd.DataFrame, *, recommended_lap: Optional[int] = None,
                          earliest_lap: Optional[int] = None, latest_lap: Optional[int] = None,
                          title: str = "Predicted cost by pit lap") -> go.Figure:
    """Predicted race-time delta to the best lap for every candidate in-lap.

    The feasible window is shaded; the recommended (minimum-cost) lap is a
    distinct marker so it is not confused with the window.
    """
    if window is None or window.empty or not window["feasible"].any():
        return empty_figure(title, "No feasible pit laps under the configured constraints.")
    feas = window[window["feasible"]]
    fig = go.Figure()
    if earliest_lap is not None and latest_lap is not None:
        fig.add_vrect(x0=earliest_lap - 0.5, x1=latest_lap + 0.5, fillcolor="rgba(42,120,214,0.07)", line_width=0,
                      annotation_text="feasible window", annotation_position="top left", annotation_font_color=MUTED)
    fig.add_trace(go.Scatter(x=feas["pit_lap"], y=feas["delta_to_best_s"], mode="lines+markers", name="Green-flag cost (model)",
                             line={"width": 2, "color": SERIES[0]}, marker={"size": 8},
                             customdata=feas[["best_continuation"]],
                             hovertemplate="Pit lap %{x}<br>+%{y:.2f} s vs best<br>%{customdata[0]}<extra></extra>"))
    if "scenario_delta_to_best_s" in feas.columns:
        fig.add_trace(go.Scatter(x=feas["pit_lap"], y=feas["scenario_delta_to_best_s"], mode="lines+markers",
                                 name="With race-control scenario (model)", line={"width": 2, "color": SERIES[1], "dash": "dot"},
                                 marker={"size": 8}, hovertemplate="Pit lap %{x}<br>+%{y:.2f} s<extra></extra>"))
    if recommended_lap is not None:
        r = feas[feas["pit_lap"] == recommended_lap]
        fig.add_trace(go.Scatter(x=r["pit_lap"], y=r["delta_to_best_s"], mode="markers+text", name="Recommended lap",
                                 marker={"size": 20, "symbol": "star", "color": SERIES[2], "line": {"width": 2, "color": SURFACE}},
                                 text=[f"Lap {recommended_lap}"], textposition="top center", textfont={"color": TEXT}))
    return _layout(fig, title, "In-lap of next stop", "Race time vs best pit lap (s)")


def plot_strategy_comparison(ranking: pd.DataFrame, *, top_n: int = 12, highlight_id: Optional[str] = None,
                             baseline_time_s: Optional[float] = None,
                             title: str = "Strategy comparison (deterministic)") -> go.Figure:
    """Horizontal bars of predicted race time relative to the best (or baseline) strategy."""
    if ranking is None or ranking.empty:
        return empty_figure(title, "No feasible strategies to compare.")
    df = ranking.head(top_n).iloc[::-1]
    ref = baseline_time_s if baseline_time_s is not None else df["race_time_s"].min()
    delta = df["race_time_s"] - ref
    colors = [SERIES[2] if sid == highlight_id else SERIES[0] for sid in df["strategy_id"]]
    fig = go.Figure(go.Bar(x=delta, y=df["strategy_name"], orientation="h", marker={"color": colors, "line": {"width": 2, "color": SURFACE}},
                           text=[f"{d:+.2f} s" for d in delta], textposition="outside", textfont={"color": TEXT},
                           customdata=df[["race_time_s", "total_pit_cost_s"]],
                           hovertemplate="%{y}<br>Race %{customdata[0]:.2f} s<br>Pit cost %{customdata[1]:.1f} s<extra></extra>",
                           showlegend=False))
    label = "baseline" if baseline_time_s is not None else "best strategy"
    return _layout(fig, title, f"Predicted race time vs {label} (s)", "", height=max(320, 32 * len(df) + 120), legend=False)


def plot_monte_carlo_distribution(race_times: Mapping[str, np.ndarray], names: Optional[Mapping[str, str]] = None, *,
                                  title: str = "Monte Carlo race-time distribution", nbins: int = 60) -> go.Figure:
    """Overlaid histograms of simulated race times."""
    if not race_times:
        return empty_figure(title, "Run the Monte Carlo simulation to see the distribution.")
    fig = go.Figure()
    allv = np.concatenate([np.asarray(v) for v in race_times.values()])
    size = (allv.max() - allv.min()) / nbins if allv.max() > allv.min() else 1.0
    for i, (sid, t) in enumerate(race_times.items()):
        fig.add_trace(go.Histogram(x=t, name=(names or {}).get(sid, sid), opacity=0.55, marker_color=color_for(i),
                                   xbins={"start": allv.min(), "end": allv.max() + size, "size": size},
                                   hovertemplate="%{x:.1f} s<br>%{y} runs<extra>%{fullData.name}</extra>"))
    fig.update_layout(barmode="overlay")
    n = len(next(iter(race_times.values())))
    return _layout(fig, f"{title} (n = {n:,} per strategy)", "Simulated race time (s)", "Runs")


def plot_cumulative_distribution(race_times: Mapping[str, np.ndarray], names: Optional[Mapping[str, str]] = None, *,
                                 title: str = "Cumulative race-time distribution") -> go.Figure:
    """Empirical CDF per strategy (left/higher curve = faster)."""
    if not race_times:
        return empty_figure(title, "Run the Monte Carlo simulation to see the distribution.")
    fig = go.Figure()
    for i, (sid, t) in enumerate(race_times.items()):
        x = np.sort(np.asarray(t))
        y = np.arange(1, len(x) + 1) / len(x)
        step = max(1, len(x) // 800)
        fig.add_trace(go.Scatter(x=x[::step], y=y[::step], mode="lines", name=(names or {}).get(sid, sid),
                                 line={"width": 2, "color": color_for(i)},
                                 hovertemplate="%{x:.1f} s<br>P(race time <= x) = %{y:.3f}<extra>%{fullData.name}</extra>"))
    return _layout(fig, title, "Simulated race time (s)", "Cumulative probability")


def plot_delta_distribution(deltas: Mapping[str, np.ndarray], names: Optional[Mapping[str, str]] = None, *,
                            title: str = "Paired race-time difference vs baseline", nbins: int = 60) -> go.Figure:
    """Histogram of paired differences (strategy - baseline); negative = faster than baseline."""
    if not deltas:
        return empty_figure(title, "No baseline comparison available.")
    fig = go.Figure()
    for i, (sid, d) in enumerate(deltas.items()):
        fig.add_trace(go.Histogram(x=d, name=(names or {}).get(sid, sid), opacity=0.55, marker_color=color_for(i), nbinsx=nbins,
                                   hovertemplate="%{x:.2f} s<br>%{y} runs<extra>%{fullData.name}</extra>"))
    fig.add_vline(x=0, line={"color": MUTED, "width": 1, "dash": "dash"}, annotation_text="baseline", annotation_font_color=MUTED)
    fig.update_layout(barmode="overlay")
    return _layout(fig, title, "Strategy minus baseline race time (s)", "Runs")


def plot_safety_car_scenarios(table: pd.DataFrame, *, value_col: str = "mean_race_time_s",
                              title: str = "Safety-car scenario comparison") -> go.Figure:
    """Grouped bars: strategy (colour) x scenario (x-axis). ``table`` columns: scenario, strategy, value_col."""
    if table is None or table.empty:
        return empty_figure(title, "No safety-car scenario results available.")
    fig = go.Figure()
    for i, (strat, g) in enumerate(table.groupby("strategy", sort=False)):
        fig.add_trace(go.Bar(x=g["scenario"], y=g[value_col], name=strat, marker={"color": color_for(i), "line": {"width": 2, "color": SURFACE}},
                             hovertemplate="%{x}<br>%{y:.2f} s<extra>%{fullData.name}</extra>"))
    fig.update_layout(barmode="group")
    label = "Delta to best in scenario (s)" if "delta" in value_col else "Race time (s)"
    return _layout(fig, title, "Scenario", label)


def plot_rival_gaps(gap_history: pd.DataFrame | Mapping[str, Sequence[float]], laps: Optional[Sequence[int]] = None, *,
                    title: str = "Gap to rival") -> go.Figure:
    """Gap (own - rival elapsed, s) per lap; positive = own car behind."""
    if gap_history is None or len(gap_history) == 0:
        return empty_figure(title, "No rival simulation available.")
    fig = go.Figure()
    if isinstance(gap_history, pd.DataFrame):
        series = {c: gap_history[c].to_numpy() for c in gap_history.columns if c not in ("lap",)}
        laps = gap_history["lap"].to_numpy() if "lap" in gap_history.columns else laps
    else:
        series = dict(gap_history)
    for i, (name, g) in enumerate(series.items()):
        x = laps if laps is not None else np.arange(1, len(g) + 1)
        fig.add_trace(go.Scatter(x=x, y=g, mode="lines", name=str(name), line={"width": 2, "color": color_for(i)},
                                 hovertemplate="Lap %{x}<br>Gap %{y:+.2f} s<extra>%{fullData.name}</extra>"))
    fig.add_hline(y=0, line={"color": MUTED, "width": 1, "dash": "dash"})
    return _layout(fig, title + " (positive = own car behind)", "Lap", "Gap (s)", legend=len(series) > 1)


def plot_sensitivity(table: pd.DataFrame, parameter: str, unit: str = "", *, y: str = "predicted_race_time_s",
                     title: Optional[str] = None) -> go.Figure:
    """Outcome vs assumption value; marker colour shows the recommended strategy."""
    title = title or f"Sensitivity to {parameter}"
    if table is None or table.empty or y not in table.columns:
        return empty_figure(title, "No sensitivity results available.")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=table["value"], y=table[y], mode="lines", line={"width": 2, "color": NEUTRAL},
                             showlegend=False, hoverinfo="skip"))
    for i, (name, g) in enumerate(table.groupby("recommended_strategy", sort=False)):
        fig.add_trace(go.Scatter(x=g["value"], y=g[y], mode="markers", name=str(name),
                                 marker={"size": 11, "color": color_for(i), "line": {"width": 2, "color": SURFACE}},
                                 hovertemplate="%{x}<br>%{y:.2f}<extra>%{fullData.name}</extra>"))
    ylab = {"predicted_race_time_s": "Optimal predicted race time (s)", "mc_cvar95_s": "CVaR 95 (s)",
            "mc_p_beat_baseline": "P(beat baseline)", "first_pit_lap": "First pit lap"}.get(y, y)
    return _layout(fig, title, f"{parameter} ({unit})" if unit else parameter, ylab)


def plot_model_residuals(predictions: pd.DataFrame, *, title: str = "Model validation: predicted vs observed") -> go.Figure:
    """Predicted vs observed held-out lap times with the identity line."""
    if predictions is None or predictions.empty:
        return empty_figure(title, "Validation inconclusive or not run: no held-out predictions.")
    fig = go.Figure()
    lo = float(min(predictions["lap_time_s"].min(), predictions["predicted_lap_time_s"].min()))
    hi = float(max(predictions["lap_time_s"].max(), predictions["predicted_lap_time_s"].max()))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect prediction",
                             line={"width": 1, "color": MUTED, "dash": "dash"}))
    for i, (m, g) in enumerate(predictions.groupby("model")):
        fig.add_trace(go.Scatter(x=g["lap_time_s"], y=g["predicted_lap_time_s"], mode="markers", name=f"{m} model",
                                 marker={"size": 8, "color": color_for(i), "opacity": 0.7, "line": {"width": 1, "color": SURFACE}},
                                 hovertemplate="Observed %{x:.3f} s<br>Predicted %{y:.3f} s<extra>%{fullData.name}</extra>"))
    return _layout(fig, title, "Observed lap time (s)", "Predicted lap time (s)")


def plot_residuals_by_age(predictions: pd.DataFrame, *, title: str = "Prediction residuals by tyre age") -> go.Figure:
    """Residual (predicted - observed, s) against tyre age per model."""
    if predictions is None or predictions.empty:
        return empty_figure(title, "No held-out predictions.")
    fig = go.Figure()
    for i, (m, g) in enumerate(predictions.groupby("model")):
        fig.add_trace(go.Scatter(x=g["tyre_age_laps"], y=g["residual_s"], mode="markers", name=f"{m} model",
                                 marker={"size": 8, "color": color_for(i), "opacity": 0.7}))
    fig.add_hline(y=0, line={"color": MUTED, "width": 1, "dash": "dash"})
    return _layout(fig, title, "Tyre age (laps)", "Residual: predicted - observed (s)")


def plot_sector_comparison(sector_stats: pd.DataFrame, *, title: str = "Sector comparison (median, representative laps)") -> go.Figure:
    """Grouped bars of median sector times per driver, relative to the fastest driver."""
    cols = [c for c in ("sector1_s_median", "sector2_s_median", "sector3_s_median") if c in sector_stats.columns]
    if sector_stats is None or sector_stats.empty or not cols:
        return empty_figure(title, "Sector columns (sector1_s..sector3_s) are not available in this dataset.")
    fig = go.Figure()
    for i, (_, row) in enumerate(sector_stats.iterrows()):
        vals = [row[c] - sector_stats[c].min() for c in cols]
        fig.add_trace(go.Bar(x=[c.split("_")[0].replace("sector", "Sector ") for c in cols], y=vals, name=str(row["driver"]),
                             marker={"color": color_for(i), "line": {"width": 2, "color": SURFACE}},
                             hovertemplate="%{x}<br>+%{y:.3f} s<extra>%{fullData.name}</extra>"))
    fig.update_layout(barmode="group")
    return _layout(fig, title, "Sector", "Median time vs fastest driver (s)")


def plot_stint_pace(stints: pd.DataFrame, *, title: str = "Stint pace (median lap time)") -> go.Figure:
    """Median lap time per stint (dot plot, coloured by compound)."""
    if stints is None or stints.empty:
        return empty_figure(title, "Stint information unavailable.")
    fig = go.Figure()
    comps = list(dict.fromkeys(stints["compound"].astype(str)))
    for i, c in enumerate(comps):
        g = stints[stints["compound"].astype(str) == c]
        fig.add_trace(go.Scatter(x=g["stint_id"], y=g["median_lap_time_s"], name=c, mode="markers",
                                 marker={"size": 12, "color": color_for(i), "line": {"width": 2, "color": SURFACE}},
                                 customdata=g[["laps", "first_lap", "last_lap"]],
                                 hovertemplate="%{x}<br>%{y:.3f} s<br>%{customdata[0]} laps (L%{customdata[1]}-%{customdata[2]})<extra>%{fullData.name}</extra>"))
    return _layout(fig, title, "Stint", "Median lap time (s)")
