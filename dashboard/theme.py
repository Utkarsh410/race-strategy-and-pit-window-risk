"""Veltrix Motorsport brand theme for the dashboard.

Palette and fonts come from the CSS variables on https://www.veltrixmotorsport.com
(checked 2026-11-03): a carbon-black dark theme with a lavender accent and a beige
secondary accent, Space Grotesk for display text and Inter for body text. The
Streamlit side lives in ``.streamlit/config.toml``. This module adds the CSS
Streamlit cannot set, the sidebar wordmark, and :func:`brand_figure`, which turns
the light figures from :mod:`racesim.plotting` dark for the dashboard only. The
exported HTML report keeps its light figures.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import streamlit as st

from racesim import plotting

CARBON = "#0B0B0D"        # --color-carbon
CARBON_900 = "#111114"    # --color-carbon-900
CARBON_800 = "#17171B"    # --color-carbon-800
CARBON_700 = "#222228"    # --color-carbon-700
WHITE = "#FFFFFF"
MUTED = "rgba(255,255,255,0.64)"    # --color-muted (#ffffffa3)
SUBTLE = "rgba(255,255,255,0.50)"   # --color-subtle
LINE = "rgba(255,255,255,0.10)"     # --color-line
LINE_STRONG = "rgba(255,255,255,0.20)"
LAVENDER = "#B9A7E8"      # --color-lavender
BEIGE = "#E7DDCF"         # --color-beige
HIGHLIGHT = "rgba(185,167,232,0.18)"  # lavender tint for highlighted table rows

# Light colours used by racesim.plotting -> their dark equivalents (exact-match only).
_FIGURE_COLOURS = {
    plotting.SURFACE: CARBON_900, plotting.TEXT: WHITE, plotting.MUTED: MUTED, plotting.GRID: LINE,
    plotting.NEUTRAL: "#A09FA6",
}

_CSS = f"""
<style>
.block-container {{padding-top: 1.6rem; max-width: 1400px;}}
h1, h2, h3 {{letter-spacing: -0.01em;}}
div[data-testid="stMetric"] {{border: 1px solid {LINE}; border-radius: 0.25rem; padding: 8px 12px; background: {CARBON_800};}}
div[data-testid="stMetricLabel"] p {{color: {MUTED}; text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.72rem;}}
div[data-testid="stMetricValue"] {{font-variant-numeric: tabular-nums; font-size: 1.35rem;}}
button[data-baseweb="tab"] p {{text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.8rem;}}
.synthetic-banner {{background: rgba(237,161,0,0.12); border: 1px solid rgba(237,161,0,0.45); border-left: 3px solid #eda100;
  border-radius: 0.25rem; padding: 6px 12px; margin-bottom: 8px; font-size: 0.9rem; color: {WHITE};}}
.veltrix-wordmark {{font-family: "Space Grotesk", sans-serif; font-weight: 600; letter-spacing: 0.22em; text-transform: uppercase;
  font-size: 0.85rem; color: {WHITE}; padding: 2px 0 2px 10px; border-left: 2px solid {LAVENDER}; margin-bottom: 0.4rem;}}
.veltrix-wordmark span {{display: block; color: {SUBTLE}; letter-spacing: 0.12em; font-size: 0.68rem; font-weight: 500;}}
</style>
"""


def apply_brand(product: str = "RaceSim") -> None:
    """Inject the brand CSS and the sidebar wordmark (call once per page, after ``set_page_config``)."""
    st.markdown(_CSS, unsafe_allow_html=True)
    st.sidebar.markdown(f"<div class='veltrix-wordmark'>Veltrix Motorsport<span>{product}</span></div>",
                        unsafe_allow_html=True)


def _recolour(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _recolour(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_recolour(v) for v in obj]
    if isinstance(obj, str):
        return _FIGURE_COLOURS.get(obj, obj)
    return obj


def brand_figure(fig: go.Figure) -> go.Figure:
    """A dark, brand-coloured copy of a light :mod:`racesim.plotting` figure (data and series colours unchanged)."""
    out = go.Figure(_recolour(fig.to_dict()))
    out.update_layout(template="plotly_dark", hoverlabel={"bgcolor": CARBON_700, "bordercolor": LINE_STRONG,
                                                          "font": {"color": WHITE}})
    out.update_xaxes(zerolinecolor=LINE_STRONG)
    out.update_yaxes(zerolinecolor=LINE_STRONG)
    return out
