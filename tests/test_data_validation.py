import numpy as np
import pandas as pd
import pytest

from racesim.data_ingestion import DataValidationError, export_simulation_results, load_race_csv, load_race_dataframe
from racesim.data_validation import (
    assign_stint_ids, calculate_data_quality_report, clean_lap_times, identify_duplicate_laps, identify_invalid_lap_times,
    identify_lap_time_outliers, identify_missing_values, validate_race_data,
)


def test_valid_csv_loads(tmp_path, lap_df):
    p = tmp_path / "laps.csv"
    lap_df.to_csv(p, index=False)
    res = load_race_csv(p)
    assert res.report.valid
    assert len(res.data) == len(lap_df)
    assert res.data["pit_in"].dtype.name in ("bool", "boolean")
    pd.testing.assert_frame_equal(res.raw, pd.read_csv(p))  # original preserved


def test_missing_mandatory_column_clear_error(lap_df):
    with pytest.raises(DataValidationError, match="lap_time_s"):
        load_race_dataframe(lap_df.drop(columns=["lap_time_s"]))


def test_ambiguous_unit_column_not_inferred(lap_df):
    df = lap_df.rename(columns={"lap_time_s": "lap_time_ms"})
    with pytest.raises(DataValidationError, match="not inferred"):
        load_race_dataframe(df)


def test_invalid_lap_times_flagged_not_replaced(lap_df):
    df = lap_df.copy()
    df.loc[3, "lap_time_s"] = -5
    df.loc[4, "lap_time_s"] = np.nan
    df.loc[5, "lap_time_s"] = 10_000
    inv = identify_invalid_lap_times(df)
    assert set(inv.index) == {3, 4, 5}
    cl = clean_lap_times(df)
    assert cl.annotated.loc[4, "lap_time_s"] != cl.annotated.loc[4, "lap_time_s"]  # still NaN, not filled
    assert 3 not in cl.cleaned.index and len(cl.annotated) == len(df)


def test_non_numeric_lap_time_flagged(lap_df):
    df = lap_df.astype({"lap_time_s": object})
    df.loc[2, "lap_time_s"] = "DNF"
    res = load_race_dataframe(df)
    assert res.report.invalid_lap_time_count == 1
    assert any("non-numeric" in w for w in res.report.warnings)


def test_duplicates_detected(lap_df):
    df = pd.concat([lap_df, lap_df.iloc[[0]]], ignore_index=True)
    assert len(identify_duplicate_laps(df)) == 2
    assert clean_lap_times(df).summary["duplicates_removed"] == 1


def test_outliers_flagged_with_cause_not_deleted(lap_df):
    df = lap_df.copy()
    df.loc[df["pit_in"], "lap_time_s"] += 20
    df.loc[15, "lap_time_s"] += 15  # unexplained
    out = identify_lap_time_outliers(df)
    assert "pit in-lap" in set(out["likely_cause"])
    assert "unexplained" in set(out["likely_cause"])
    cl = clean_lap_times(df)
    assert len(cl.cleaned) == len(df)
    assert cl.annotated.loc[15, "exclude_from_model"]


def test_optional_fields_missing_do_not_crash(lap_df):
    df = lap_df[["lap", "lap_time_s"]]
    rep = validate_race_data(df)
    assert rep.valid
    assert "driver" in rep.missing_optional_columns
    q = calculate_data_quality_report(df)
    assert q["validation"]["valid"]
    assert len(identify_missing_values(df)) == 2


def test_stint_ids(lap_df):
    s = assign_stint_ids(lap_df)
    assert s.groupby("driver")["stint_id"].nunique().tolist() == [2, 2]


def test_export(tmp_path, lap_df):
    p = export_simulation_results(lap_df, tmp_path / "x.csv")
    assert p.exists() and len(pd.read_csv(p)) == len(lap_df)
