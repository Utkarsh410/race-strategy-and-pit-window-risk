import math

import pandas as pd
import pytest

from racesim.formatting import format_lap_time, lap_time_ticks
from racesim.plotting import plot_lap_time_progression


@pytest.mark.parametrize("seconds, text", [
    (110.485, "1:50.485"), (83.0, "1:23.000"), (58.123, "0:58.123"), (600.0, "10:00.000"),
    (59.9996, "1:00.000"), (3723.456, "1:02:03.456"), (0.0, "0:00.000"), (-1.5, "-0:01.500"),
])
def test_format_lap_time(seconds, text):
    assert format_lap_time(seconds) == text


def test_format_lap_time_unavailable_is_none_never_zero():
    for v in (None, math.nan, math.inf, "n/a"):
        assert format_lap_time(v) is None


def test_format_lap_time_decimals():
    assert format_lap_time(112.36, 1) == "1:52.4" and format_lap_time(120.0, 0) == "2:00"


def test_lap_time_ticks_cover_data_in_race_format():
    ticks, text = lap_time_ticks([110.5, 113.4, 117.6])
    assert ticks[0] <= 110.5 and ticks[-1] >= 117.6 and len(ticks) <= 8
    assert all(":" in t for t in text) and text[ticks.index(112.0)] == "1:52"
    assert lap_time_ticks([]) == ([], [])


def test_lap_time_chart_uses_race_format_axis_and_hover():
    fig = plot_lap_time_progression(pd.DataFrame({"lap": [1, 2, 3], "lap_time_s": [110.485, 112.0, 113.25]}))
    assert all(":" in t for t in fig.layout.yaxis.ticktext)
    assert list(fig.data[0].customdata) == ["1:50.485", "1:52.000", "1:53.250"]
