"""Pace statistics, insufficient samples, fuel consumption and tyre channels."""

import statistics

import pytest

from racesim.integrations.acc.lap_tracker import CompletedLap, LapClass, Stint
from racesim.integrations.acc.monitoring import fuel_status, missing_channels, tyre_status
from racesim.integrations.acc.pace_analysis import TREND_LABEL, analyse_pace, linear_trend, rolling_pace


def lap(n, t=120.0, cls=LapClass.CLEAN, stint=1, sectors=None, fuel=None):
    start, end = fuel if fuel else (None, None)
    return CompletedLap("s", n, t, cls, stint, n, 1, cls in (LapClass.IN_LAP, LapClass.OUT_LAP), sectors, fuel_start_kg=start, fuel_end_kg=end)


def test_statistics_with_sample_sizes():
    times = [120.0, 121.0, 122.0, 119.0]
    p = analyse_pace([lap(i + 1, t) for i, t in enumerate(times)])
    assert p["best"] == {"value": 119.0, "n": 4, "needed": 1}
    assert p["mean"]["value"] == pytest.approx(statistics.fmean(times))
    assert p["median"]["value"] == pytest.approx(120.5)
    assert p["std"]["value"] == pytest.approx(statistics.stdev(times)) and p["std"]["n"] == 4


def test_only_clean_laps_enter_statistics_but_all_are_counted():
    laps = [lap(1, 120.0), lap(2, 90.0, LapClass.INVALID), lap(3, 200.0, LapClass.UNCERTAIN), lap(4, 140.0, LapClass.IN_LAP), lap(5, 122.0)]
    p = analyse_pace(laps)
    assert p["best"]["value"] == 120.0 and p["mean"]["n"] == 2 and p["laps_total"] == 5
    assert p["classification_counts"]["INVALID"] == 1
    assert p["latest_completed"]["lap_number"] == 5


def test_rolling_needs_a_full_window():
    assert rolling_pace([120.0, 121.0], 3).value is None
    assert rolling_pace([120.0, 121.0], 3).n == 2
    assert rolling_pace([1.0, 120.0, 121.0, 122.0], 3).value == pytest.approx(121.0)


@pytest.mark.parametrize("n,expected", [(2, (None, None, None)), (3, (True, None, None)), (5, (True, True, None)), (10, (True, True, True))])
def test_rolling_windows_by_sample_count(n, expected):
    r = analyse_pace([lap(i + 1) for i in range(n)])["rolling"]
    assert tuple(None if r[w]["value"] is None else True for w in (3, 5, 10)) == expected


def test_empty_and_single_lap_produce_no_false_numbers():
    empty = analyse_pace([])
    assert empty["best"]["value"] is None and empty["mean"]["n"] == 0 and empty["latest_completed"] is None
    single = analyse_pace([lap(1)])
    assert single["std"]["value"] is None and single["std"]["n"] == 1 and single["std"]["needed"] == 2


def test_delta_to_best():
    p = analyse_pace([lap(1, 120.0), lap(2, 121.5)])
    assert p["delta_to_best"] == [{"lap_number": 1, "delta_s": 0.0}, {"lap_number": 2, "delta_s": 1.5}]


def test_stint_comparison_and_trend_labels():
    laps = [lap(i, 120.0 + 0.1 * i, stint=1) for i in range(1, 7)] + [lap(i, 119.0, stint=2) for i in range(7, 9)]
    rows = analyse_pace(laps)["stints"]
    assert rows[0]["pace_trend_s_per_lap"] == pytest.approx(0.1) and rows[0]["pace_trend_label"] == TREND_LABEL
    assert rows[1]["pace_trend_s_per_lap"] is None and "insufficient" in rows[1]["pace_trend_label"]
    assert rows[1]["median_delta_to_previous_stint_s"] == pytest.approx(119.0 - rows[0]["median_s"])


def test_trend_is_never_called_tyre_degradation_as_a_fact():
    note = analyse_pace([])["trend_note"]
    assert "not a calibrated tyre-degradation estimate" in note and "fuel load" in note and "traffic" in note


def test_sector_trends():
    laps = [lap(i, 120.0, sectors=[40.0 + i * 0.1, 40.0, 40.0]) for i in range(1, 4)]
    rows = analyse_pace(laps)["sector_trends"]
    assert rows[0]["n"] == 3 and rows[0]["trend_s_per_lap"] == pytest.approx(0.1) and rows[0]["best_s"] == pytest.approx(40.1)


def test_sector_trend_needs_three_laps():
    rows = analyse_pace([lap(1, sectors=[40.0, 40.0, 40.0])])["sector_trends"]
    assert rows[0]["n"] == 1 and rows[0]["trend_s_per_lap"] is None and rows[1]["median_s"] == 40.0


def test_linear_trend_undefined_cases():
    assert linear_trend([1], [1]) is None and linear_trend([1, 1], [1, 2]) is None


def test_fuel_consumption_from_consecutive_laps_reports_laps_used():
    laps = [lap(1, fuel=(50.0, 47.5)), lap(2, fuel=(47.5, 45.0)), lap(3, fuel=(45.0, 42.5))]
    f = fuel_status(laps, 42.5, pace_s=120.0, pace_n=3)
    assert f["burn_kg_per_lap"] == pytest.approx(2.5) and f["burn_laps_used"] == 3
    assert f["laps_remaining_estimate"] == pytest.approx(17.0) and f["time_remaining_estimate_s"] == pytest.approx(2040.0)


def test_fuel_estimate_needs_enough_laps():
    f = fuel_status([lap(1, fuel=(50.0, 47.5)), lap(2, fuel=(47.5, 45.0))], 45.0, pace_s=120.0, pace_n=3)
    assert f["burn_kg_per_lap"] is None and f["laps_remaining_estimate"] is None and f["burn_laps_used"] == 2


def test_laps_remaining_without_pace_has_no_time_estimate():
    laps = [lap(i, fuel=(50.0 - i, 49.0 - i)) for i in range(3)]
    f = fuel_status(laps, 40.0, pace_s=None)
    assert f["laps_remaining_estimate"] == pytest.approx(40.0) and f["time_remaining_estimate_s"] is None


def test_fuel_flags_implausible_readings_and_excludes_them():
    laps = [lap(1, fuel=(50.0, 47.5)), lap(2, fuel=(47.5, 49.0)), lap(3, fuel=(49.0, 10.0)), lap(4, fuel=(10.0, 7.5)), lap(5, fuel=(7.5, 5.0))]
    f = fuel_status(laps, 5.0)
    assert f["burn_laps_used"] == 3 and f["burn_kg_per_lap"] == pytest.approx(2.5)
    assert any("fuel_not_decreasing" in x for x in f["flags"]) and any("implausible_fuel_drop" in x for x in f["flags"])


def test_sudden_fuel_change_is_flagged():
    laps = [lap(i, fuel=(50.0 - 3 * i, 47.5 - 3 * i)) for i in range(4)] + [lap(5, fuel=(30.0, 24.0))]
    f = fuel_status(laps, 24.0)
    assert any("sudden_fuel_change" in x for x in f["flags"]) and f["burn_laps_used"] == 4


def test_pit_laps_do_not_count_toward_fuel_consumption():
    f = fuel_status([lap(1, cls=LapClass.IN_LAP, fuel=(10.0, 60.0))], 60.0)
    assert f["burn_laps_used"] == 0 and f["flags"] == []


def test_stale_fuel_gives_no_laps_remaining():
    laps = [lap(i, fuel=(50.0 - i, 49.0 - i)) for i in range(3)]
    assert fuel_status(laps, 40.0, stale=True)["laps_remaining_estimate"] is None


def test_no_fuel_channel_means_nothing_is_estimated():
    f = fuel_status([lap(1), lap(2), lap(3)], None, pace_s=120.0, pace_n=3)
    assert f["latest_fuel_kg"] is None and f["burn_kg_per_lap"] is None and f["history"] == []


def test_tyre_status_passes_through_only_supplied_channels():
    assert tyre_status({}, [])["compound"] is None
    t = tyre_status({"tyre_compound": "DRY", "tyre_pressures_psi": [27.0, 27.1, 27.2, 27.3]},
                    [Stint("s", 1, "x", tyre_compound="DRY", tyre_age_start_laps=0), Stint("s", 2, "pit_exit", tyre_compound="WET")])
    assert t["compound"] == "DRY" and t["temperatures_c"] is None and t["by_stint"][1]["compound"] == "WET"


def test_missing_channels_name_the_future_reader():
    rows = missing_channels({})
    assert {r["field"] for r in rows} >= {"fuel_kg", "tyre_compound", "tyre_pressures_psi", "tyre_temperatures_c"}
    assert all("shared-memory" in r["requires"] for r in rows)
    assert "fuel_kg" not in {r["field"] for r in missing_channels({"fuel_kg": 10.0})}
