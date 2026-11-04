"""Race-dash style visuals for shared-memory physics (ACC Live Telemetry page).

Two figures built from one physics sample: a driver dash (RPM arc, speed,
gear, pedals, fuel, water temperature) and a top-down car with the four
corners (tyre core temperature, pressure, brake temperature). No Streamlit
here, so both can be tested directly.

Rules carried over from the rest of the page:

* An unavailable value is drawn grey and labelled UNAVAILABLE, never as 0.
* MOCK and STALE samples carry a large watermark and the label in the title.
* The colour bands are display aids, not validated per-car targets; they are
  constants below so they can be tuned for a car.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import plotly.graph_objects as go

from dashboard.theme import BEIGE, CARBON_700, CARBON_800, CARBON_900, LAVENDER, LINE, LINE_STRONG, MUTED, SUBTLE, WHITE

GREEN, AMBER, RED, BLUE = "#1baf7a", "#eda100", "#e34948", "#2a78d6"
OFF = "#3a3a42"  # unavailable

# (upper bound, colour) pairs, read in order; display aids only.
TYRE_TEMP_BANDS = ((65.0, BLUE), (95.0, GREEN), (105.0, AMBER), (math.inf, RED))      # degC, core
TYRE_PRESSURE_BANDS = ((25.5, BLUE), (28.5, GREEN), (29.5, AMBER), (math.inf, RED))    # psi, hot
BRAKE_TEMP_BANDS = ((150.0, BLUE), (650.0, GREEN), (800.0, AMBER), (math.inf, RED))    # degC, disc
WATER_TEMP_BANDS = ((70.0, BLUE), (105.0, GREEN), (115.0, AMBER), (math.inf, RED))     # degC
TYRE_BAR_RANGE = (20.0, 130.0)
SHIFT_FRACTION = (0.85, 0.95)  # of max RPM: amber from the first, red from the second

WHEEL_NAMES = ("FL", "FR", "RL", "RR")


def _num(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def band_colour(value: Optional[float], bands: Sequence[tuple[float, str]]) -> str:
    if value is None:
        return OFF
    return next(colour for upper, colour in bands if value < upper)


def wheel(channels: dict, name: str, i: int) -> Optional[float]:
    values = channels.get(name)
    return _num(values[i]) if isinstance(values, (list, tuple)) and len(values) == 4 else None


def _watermark(fig: go.Figure, label: str) -> None:
    if label.startswith(("MOCK", "STALE")):
        fig.add_annotation(text=label.split(" ")[0], x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                           textangle=-18, font={"size": 64, "color": "rgba(237,161,0,0.22)", "family": "Space Grotesk"})


def _layout(fig: go.Figure, title: str, height: int) -> go.Figure:
    fig.update_layout(title={"text": title, "x": 0.0, "xanchor": "left", "font": {"size": 14, "color": WHITE}},
                      height=height, margin={"l": 10, "r": 10, "t": 44, "b": 10}, paper_bgcolor=CARBON_900,
                      plot_bgcolor=CARBON_900, font={"color": MUTED, "family": "Inter"}, showlegend=False)
    return fig


def _gauge(fig: go.Figure, value: Optional[float], title: str, rng: tuple[float, float], domain: dict, *,
           suffix: str = "", bands: Optional[Sequence[tuple[float, str]]] = None, colour: str = LAVENDER, big: bool = False,
           decimals: int = 0, bar_bands: Optional[Sequence[tuple[float, str]]] = None) -> None:
    lo, hi = rng
    steps = []
    if bands:
        edge = lo
        for upper, c in bands:
            top = min(upper, hi)
            if top > edge:
                steps.append({"range": [edge, top], "color": c})
            edge = top
    bar_bands = bar_bands or bands
    bar = colour if not bar_bands or value is None else band_colour(value, bar_bands)
    fig.add_trace(go.Indicator(
        mode="gauge+number" if value is not None else "gauge", value=value if value is not None else lo, domain=domain,
        number={"suffix": suffix, "valueformat": f",.{decimals}f", "font": {"size": 54 if big else 24, "color": WHITE}},
        title={"text": title, "font": {"size": 13 if big else 11, "color": SUBTLE}},
        gauge={"shape": "angular", "axis": {"range": [lo, hi], "tickcolor": LINE_STRONG, "tickfont": {"size": 10, "color": SUBTLE}},
               "bar": {"color": bar if value is not None else "rgba(0,0,0,0)", "thickness": 0.28 if steps else 0.6},
               "bgcolor": CARBON_800, "borderwidth": 0, "steps": [{**s, "color": _tint(s["color"])} for s in steps]}))
    if value is None:
        x, y = sum(domain["x"]) / 2, domain["y"][0] + (domain["y"][1] - domain["y"][0]) * 0.18
        fig.add_annotation(text="UNAVAILABLE", x=x, y=y, xref="paper", yref="paper", showarrow=False,
                           font={"size": 13 if big else 10, "color": SUBTLE})


def _tint(colour: str, alpha: float = 0.55) -> str:
    h = colour.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def driver_dash(channels: dict, label: str) -> go.Figure:
    """RPM arc in the middle with speed and gear below it; pedals left, fuel and water temperature right."""
    c = channels
    fig = go.Figure()
    rpm, max_rpm = _num(c.get("rpm")), _num(c.get("max_rpm"))
    top = max_rpm or max(9000.0, math.ceil((rpm or 0) / 1000) * 1000)
    shift = None if max_rpm is None else ((max_rpm * SHIFT_FRACTION[0], CARBON_700), (max_rpm * SHIFT_FRACTION[1], AMBER), (math.inf, RED))
    bar = None if shift is None else ((shift[0][0], LAVENDER),) + shift[1:]
    _gauge(fig, rpm, "RPM" if max_rpm else "RPM (no max RPM: no shift bands)", (0, top),
           {"x": [0.26, 0.74], "y": [0.3, 1.0]}, bands=shift, bar_bands=bar, big=True)
    throttle, brake = _num(c.get("throttle")), _num(c.get("brake"))
    _gauge(fig, None if throttle is None else throttle * 100, "Throttle", (0, 100), {"x": [0.0, 0.2], "y": [0.56, 1.0]},
           suffix=" %", colour=GREEN)
    _gauge(fig, None if brake is None else brake * 100, "Brake", (0, 100), {"x": [0.0, 0.2], "y": [0.04, 0.48]},
           suffix=" %", colour=RED)
    fuel, max_fuel = _num(c.get("fuel_remaining")), _num(c.get("max_fuel"))
    _gauge(fig, fuel, "Fuel (raw unit)", (0, max_fuel or max(100.0, math.ceil((fuel or 0) / 10) * 10)),
           {"x": [0.8, 1.0], "y": [0.56, 1.0]}, colour=BEIGE, decimals=1)
    _gauge(fig, _num(c.get("water_temp")), "Water °C", (40, 130), {"x": [0.8, 1.0], "y": [0.04, 0.48]},
           bands=WATER_TEMP_BANDS)
    speed, gear = _num(c.get("speed_kmh")), c.get("gear")
    fig.add_annotation(text="UNAVAILABLE" if speed is None else f"{speed:,.0f}", x=0.43, y=0.08, xref="paper", yref="paper",
                       showarrow=False, font={"size": 46 if speed is not None else 16, "color": WHITE, "family": "Space Grotesk"})
    fig.add_annotation(text="km/h", x=0.43, y=-0.05, xref="paper", yref="paper", showarrow=False, font={"size": 11, "color": SUBTLE})
    fig.add_annotation(text="-" if gear is None else str(gear), x=0.6, y=0.08, xref="paper", yref="paper", showarrow=False,
                       font={"size": 46 if gear is not None else 16, "color": LAVENDER, "family": "Space Grotesk"})
    fig.add_annotation(text="gear (raw ACC index)", x=0.6, y=-0.05, xref="paper", yref="paper", showarrow=False,
                       font={"size": 11, "color": SUBTLE})
    _watermark(fig, label)
    fig = _layout(fig, f"Driver dash | {label}", 390)
    fig.update_layout(margin={"l": 10, "r": 10, "t": 44, "b": 30})
    return fig


def _corner(fig: go.Figure, i: int, c: dict) -> None:
    """One wheel: tyre block coloured by core temp, a temperature bar, and the readings beside it."""
    left, front = i in (0, 2), i in (0, 1)
    temp, press, brake = (wheel(c, "tyre_core_temperature", i), wheel(c, "wheel_pressure", i), wheel(c, "brake_temperature", i))
    y0 = 58 if front else 12
    tx0 = 34 if left else 60            # tyre block x
    bx0 = 29 if left else 68            # temperature bar x
    rect = {"type": "rect", "line": {"width": 0}}
    fig.add_shape(**rect, x0=tx0, x1=tx0 + 6, y0=y0, y1=y0 + 26, fillcolor=band_colour(temp, TYRE_TEMP_BANDS), layer="above")
    fig.add_shape(type="rect", x0=tx0, x1=tx0 + 6, y0=y0, y1=y0 + 26, line={"color": LINE_STRONG, "width": 1})
    # brake disc: a thin block on the inner side of the tyre
    dx0 = tx0 + 6.6 if left else tx0 - 1.6
    fig.add_shape(**rect, x0=dx0, x1=dx0 + 1, y0=y0 + 6, y1=y0 + 20, fillcolor=band_colour(brake, BRAKE_TEMP_BANDS))
    lo, hi = TYRE_BAR_RANGE
    fig.add_shape(type="rect", x0=bx0, x1=bx0 + 3, y0=y0, y1=y0 + 26, fillcolor=CARBON_800, line={"color": LINE, "width": 1})
    if temp is not None:
        frac = min(max((temp - lo) / (hi - lo), 0.0), 1.0)
        fig.add_shape(**rect, x0=bx0, x1=bx0 + 3, y0=y0, y1=y0 + 26 * frac, fillcolor=band_colour(temp, TYRE_TEMP_BANDS))
    xt, anchor = (27, "right") if left else (73, "left")
    lines = [
        (WHEEL_NAMES[i], 25, 11, SUBTLE),
        ("UNAVAILABLE" if temp is None else f"{temp:.0f} °C", 17, 13 if temp is None else 24, SUBTLE if temp is None else WHITE),
        ("pressure UNAVAILABLE" if press is None else f"{press:.2f} psi", 8, 13, band_colour(press, TYRE_PRESSURE_BANDS) if press is not None else SUBTLE),
        ("brake UNAVAILABLE" if brake is None else f"brake {brake:.0f} °C", 1, 12, band_colour(brake, BRAKE_TEMP_BANDS) if brake is not None else SUBTLE),
    ]
    for text, dy, size, colour in lines:
        fig.add_annotation(text=text, x=xt, y=y0 + dy, xref="x", yref="y", showarrow=False, xanchor=anchor, yanchor="middle",
                           font={"size": size, "color": colour, "family": "Space Grotesk" if size > 20 else "Inter"})


def tyre_corners(channels: dict, label: str) -> go.Figure:
    """Top-down car with tyre core temperature, pressure and brake temperature at each corner."""
    fig = go.Figure()
    # body: nose at the top
    fig.add_shape(type="path", path="M 44 96 Q 50 100 56 96 L 58 70 L 58 14 Q 50 4 42 14 L 42 70 Z",
                  fillcolor=CARBON_700, line={"color": LINE_STRONG, "width": 1.5})
    fig.add_shape(type="path", path="M 45 66 Q 50 70 55 66 L 54.5 50 Q 50 47 45.5 50 Z", fillcolor=CARBON_800, line={"color": LINE, "width": 1})
    for i in range(4):
        _corner(fig, i, channels)
    compound = channels.get("tyre_compound")
    fig.add_annotation(text=f"compound<br><b>{compound or 'UNAVAILABLE'}</b>", x=50, y=32, xref="x", yref="y", showarrow=False,
                       font={"size": 11, "color": SUBTLE})
    fig.update_xaxes(range=[0, 100], visible=False, fixedrange=True)
    fig.update_yaxes(range=[0, 100], visible=False, fixedrange=True)
    _watermark(fig, label)
    legend = ("Tyre block and bar: core temperature. Inner strip: brake disc. "
              f"<span style='color:{BLUE}'>cold</span> / <span style='color:{GREEN}'>window</span> / "
              f"<span style='color:{AMBER}'>warm</span> / <span style='color:{RED}'>hot</span> (display bands, not car targets)")
    fig.add_annotation(text=legend, x=0.0, y=-0.02, xref="paper", yref="paper", showarrow=False, xanchor="left",
                       font={"size": 10, "color": SUBTLE})
    fig = _layout(fig, f"Tyres and brakes | {label}", 460)
    fig.update_layout(margin={"l": 10, "r": 10, "t": 44, "b": 28})
    return fig
