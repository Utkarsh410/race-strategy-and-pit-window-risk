"""ACC Performance page (Streamlit multipage). The main RaceSim dashboard remains app.py."""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dashboard.acc_performance_view import render  # noqa: E402
from dashboard.theme import apply_brand  # noqa: E402

st.set_page_config(page_title="RaceSim - ACC Performance", page_icon=":material/speed:", layout="wide")
apply_brand("RaceSim - ACC Performance")
try:
    render()
except Exception as exc:  # telemetry problems must never crash the dashboard
    st.error(f"ACC performance view error: {exc}")
