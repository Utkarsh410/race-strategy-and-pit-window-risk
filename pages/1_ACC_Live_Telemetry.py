"""ACC Live Telemetry page (Streamlit multipage). The main RaceSim dashboard remains app.py."""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dashboard.acc_live_view import render  # noqa: E402

st.set_page_config(page_title="RaceSim - ACC Live Telemetry", page_icon=":material/sensors:", layout="wide")
try:
    render()
except Exception as exc:  # telemetry problems must never crash the dashboard
    st.error(f"ACC telemetry view error: {exc}")
