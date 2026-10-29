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
