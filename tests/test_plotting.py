import numpy as np
import pandas as pd
import pytest

from racesim import plotting as P


@pytest.mark.parametrize("fig", [
    P.plot_lap_time_progression(None, None), P.plot_tyre_degradation(None, None),
    P.plot_pit_window_costs(pd.DataFrame({"feasible": []})), P.plot_strategy_comparison(pd.DataFrame()),
    P.plot_monte_carlo_distribution({}), P.plot_cumulative_distribution({}), P.plot_delta_distribution({}),
    P.plot_safety_car_scenarios(pd.DataFrame()), P.plot_rival_gaps(pd.DataFrame()),
    P.plot_sensitivity(pd.DataFrame(), "x"), P.plot_model_residuals(pd.DataFrame()),
    P.plot_sector_comparison(pd.DataFrame()), P.plot_stint_pace(pd.DataFrame()),
])
def test_empty_inputs_give_placeholder(fig):
    assert fig.layout.title.text and fig.layout.annotations


def test_charts_have_titles_and_units(lap_df):
    f = P.plot_lap_time_progression(lap_df)
    assert "(s)" in f.layout.yaxis.title.text and f.layout.xaxis.title.text == "Lap"
    m = P.plot_monte_carlo_distribution({"a": np.random.default_rng(0).normal(100, 1, 200)}, {"a": "A"})
    assert "(s)" in m.layout.xaxis.title.text and "n = 200" in m.layout.title.text
    assert "observed" in f.data[0].name
