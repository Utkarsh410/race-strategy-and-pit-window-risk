"""Phase 6: dashboard workflows exercised headlessly with Streamlit's AppTest."""

from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


def _app():
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=600).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _click(at, label):
    btn = [b for b in at.button if b.label == label]
    assert btn, f"button {label!r} not found"
    btn[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _set_number(at, label, value):
    [n for n in at.number_input if n.label == label][0].set_value(value)


def test_all_sections_render():
    at = _app()
    assert len(at.tabs) == 8
    assert not at.error, [e.value for e in at.error]
    labels = [t.label for t in at.tabs]
    assert labels[:6] == ["A. Race overview", "B. Strategy planner", "C. Pit window", "D. Monte Carlo risk",
                          "E. Rival strategy", "F. Telemetry & validation"]


def test_report_contains_only_executed_analyses():
    at = _app()
    _set_number(at, "Simulations per strategy", 200)
    _click(at, "Run Monte Carlo")
    assert any("P(beat baseline)" in m.label for m in at.metric)
    _click(at, "Build HTML report")
    html = at.session_state["report_html"]
    assert "Sensitivity analysis was not run" in html and "Dynamic replanning: not run" in html
    assert "Simulation precision" in html and "Monte Carlo" in html.split("Analyses executed")[1][:300]

    _set_number(at, "Monte Carlo runs for SC / variability sweeps", 100)
    _click(at, "Run sensitivity analysis")
    _click(at, "Run rolling-horizon race")
    assert at.session_state["replan_store"]["result"].decision_log.records
    _click(at, "Build HTML report")
    html = at.session_state["report_html"]
    assert "Sensitivity analysis was not run" not in html and "tyre_degradation_multiplier" in html
    assert "Dynamic replanning: not run" not in html and "Race time with replanning" in html
    assert "12. Model validation" in html and "SYNTHETIC self-consistency" in html


def test_apply_calibration_updates_model():
    at = _app()
    before = at.session_state["ov_lin_HARD"]
    _click(at, "Apply fitted linear degradation to the model")
    after = at.session_state["ov_lin_HARD"]
    assert after != before
    assert not at.error, [e.value for e in at.error]


def _render_with_lap_only_upload():
    """Every section with lap data like a MoTeC upload: only lap and lap_time_s."""
    import logging

    import pandas as pd
    import streamlit as st

    from dashboard import (overview, pit_window_view, replanning_view, risk_analysis, rival_analysis, strategy_planner,
                           telemetry_analysis, uncertainty_view)
    from dashboard.common import make_context
    from racesim.data_ingestion import load_race_dataframe
    from racesim.data_validation import clean_lap_times

    ctx = make_context()
    laps = pd.DataFrame({"lap": range(1, 21), "lap_time_s": [110.0 + 0.05 * i for i in range(20)]})
    ctx.data = load_race_dataframe(laps, source="Uploaded MoTeC file: test.csv")
    ctx.data_label = ctx.data.source
    ctx.cleaning = clean_lap_times(ctx.data.data)
    failed = []
    for section in (overview, strategy_planner, pit_window_view, risk_analysis, rival_analysis, telemetry_analysis,
                    replanning_view, uncertainty_view):
        try:
            section.render(ctx)
        except Exception as exc:  # collect every failure, not just the first
            logging.getLogger("test").exception("section failed")
            failed.append(f"{section.__name__}: {type(exc).__name__}: {exc}")
    st.session_state["failed_sections"] = failed


def test_sections_render_with_lap_only_upload():
    at = AppTest.from_function(_render_with_lap_only_upload, default_timeout=600).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["failed_sections"] == []
