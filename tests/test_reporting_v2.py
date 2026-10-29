"""Phase 8: reports contain exactly the analyses run; exports have valid fields; HTML is well formed."""

from html.parser import HTMLParser

import pandas as pd
import pytest

from racesim.reporting import generate_strategy_report
from racesim.synthetic_data import generate_synthetic_race
from racesim.workflow import assemble_bundle, default_replan_events, run_full_analysis, run_replanning_demo, run_sensitivity_suite


class _Checker(HTMLParser):
    VOID = {"meta", "br", "hr", "img", "input", "link", "source", "wbr", "col", "area", "base", "embed", "param", "track"}

    def __init__(self):
        super().__init__()
        self.stack, self.errors, self.h2 = [], [], []
        self._in_h2 = False

    def handle_starttag(self, tag, attrs):
        if tag not in self.VOID:
            self.stack.append(tag)
        self._in_h2 = tag == "h2"

    def handle_endtag(self, tag):
        if tag in self.VOID:
            return
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"unexpected </{tag}> (open: {self.stack[-3:]})")
            if tag in self.stack:
                while self.stack and self.stack.pop() != tag:
                    pass
        else:
            self.stack.pop()
        self._in_h2 = False

    def handle_data(self, data):
        if self._in_h2:
            self.h2.append(data.strip())


def _check(html):
    c = _Checker()
    c.feed(html)
    return c


def test_minimal_report_labels_unrun_analyses(setup):
    b = assemble_bundle(setup)
    html = generate_strategy_report(b)
    c = _check(html)
    assert not c.errors, c.errors[:5]
    assert c.h2[:10] == ["1. Race configuration", "2. Input assumptions", "3. Recommended strategy", "4. Alternative strategies",
                         "5. Pit-window analysis", "6. Risk metrics (Monte Carlo)", "7. Safety-car scenario analysis",
                         "8. Sensitivity results", "9. Model limitations", "10. Data provenance"]
    for marker in ("Monte Carlo analysis was not run", "Sensitivity analysis was not run", "Dynamic replanning: not run",
                   "Model validation: not run", "Uncertainty breakdown: not run"):
        assert marker in html
    assert b.analyses_run == ["deterministic optimisation", "pit window", "safety-car scenarios"]


def test_full_report_contains_actual_results(setup, tmp_path):
    df, _ = generate_synthetic_race(setup)
    sens = run_sensitivity_suite(setup, sensitivity_sims=60)
    rr = run_replanning_demo(setup, events=default_replan_events(setup, sc_lap=18), n_sims=0)
    from racesim.monte_carlo import run_monte_carlo
    from racesim.strategy_optimizer import find_optimal_strategy
    from racesim.validation import run_model_validation
    from racesim.workflow import shortlist_strategies

    opt = find_optimal_strategy(setup)
    mc = run_monte_carlo(setup, shortlist_strategies(opt, setup), n_sims=300, seed=1)
    b = assemble_bundle(setup, optimisation=opt, monte_carlo=mc, sensitivity=sens, rolling_race=rr,
                        validation=run_model_validation(df, setup))
    html = generate_strategy_report(b, tmp_path / "r.html")
    c = _check(html)
    assert not c.errors, c.errors[:5]
    assert "11. Dynamic replanning and decision log" in c.h2 and "12. Model validation" in c.h2
    # actual values, not placeholders
    assert f"{rr.total_race_time_s:,.2f} s" in html
    assert sens[0].table["recommended_strategy"].iloc[0] in html
    assert "SYNTHETIC self-consistency" in html and "Simulation precision" in html
    assert "not run" not in html.split("<h2>8. Sensitivity results</h2>")[1].split("<h2>9.")[0]
    assert html.count("plotly") > 5 and (tmp_path / "r.html").exists()


def test_cli_style_exports_have_valid_fields(setup, tmp_path):
    b = run_full_analysis(setup, n_sims=200, run_sensitivity=False, run_replanning=True,
                          replan_events=default_replan_events(setup, sc_lap=18), replan_sims=0)
    p1, p2 = b.decision_log.export_csv(tmp_path / "decisions.csv")
    d = pd.read_csv(p1)
    assert d["lap"].between(1, setup.race.total_laps - 1).all() and d["simulated_time_s"].is_monotonic_increasing
    a = pd.read_csv(p2)
    assert {"label", "strategy_id", "deterministic_remaining_s"} <= set(a.columns) and (a["deterministic_remaining_s"] > 0).all()
    b.precision.to_csv(tmp_path / "precision.csv", index=False)
    pr = pd.read_csv(tmp_path / "precision.csv")
    assert {"mean_s", "mean_se_s", "cvar95_s", "cvar95_ci95_s"} <= set(pr.columns) and (pr["mean_se_s"] > 0).all()
    risk = pd.DataFrame([{**r.model_dump(exclude={"warnings"})} for r in b.risk_reports.values()])
    assert {"mean_race_time_ci95_s", "monte_carlo_standard_error_s"} <= set(risk.columns)


def test_strategy_comparisons_use_equivalent_assumptions(setup):
    """All shortlisted strategies in a Monte Carlo run share one scenario (same SC/VSC per run)."""
    b = run_full_analysis(setup, n_sims=150, run_sensitivity=False, include_rivals=False)
    mc = b.monte_carlo
    lens = {len(t) for t in mc.race_times.values()}
    assert lens == {150} and len(mc.sc_occurred) == 150
    # paired differences are much less noisy than absolute times only if scenarios are common
    ids = list(mc.race_times)
    d = mc.race_times[ids[0]] - mc.race_times[ids[1]]
    assert d.std() < mc.race_times[ids[0]].std()
