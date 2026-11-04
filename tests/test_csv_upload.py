import io

import pytest

from racesim.data_ingestion import DataValidationError, load_race_dataframe, read_lap_csv
from racesim.formatting import parse_lap_time

MOTEC = b'''"Format","MoTeC CSV File",,,"Workbook",""
"Venue","monza",,,"Worksheet",""
"Vehicle","991 GT3 R 2016",,,"Vehicle Desc",""
"Driver","",,,"Engine ID",""
"Device","ADL"
"Comment","",,,"Session",""
"Log Date","14-04-2026",,,"Origin Time","0.000","s"
"Log Time","15:30:33",,,"Start Time","0.000","s"
"Sample Rate","200.000","Hz",,"End Time","500.000","s"
"Duration","500.000","s",,"Start Distance","0","m"
"Range","entire outing",,,"End Distance","20000","m"
"Beacon Markers","110.962 226.789 342.038 459.212 "


"Time","Distance","SPEED"
"s","m","m/s"


"0.000","0","0"
'''


def _load(data: bytes, name: str = "upload.csv"):
    df, label, notes = read_lap_csv(io.BytesIO(data), name)
    return load_race_dataframe(df, source=label), label, notes


def test_parse_lap_time():
    assert parse_lap_time("1:50.485") == 110.485 and parse_lap_time("1:02:03.456") == 3723.456
    assert parse_lap_time("0:58.1") == 58.1 and parse_lap_time("2:00") == 120.0
    for bad in ("110.485", "1:75.0", "abc", None, ""):
        assert parse_lap_time(bad) is None


def test_motec_export_gives_complete_laps_from_beacons():
    res, label, notes = _load(MOTEC, "monza.csv")
    assert res.report.valid and "MoTeC" in label and "monza" in label
    assert list(res.data["lap"]) == [1, 2, 3]
    assert list(res.data["lap_time_s"]) == pytest.approx([115.827, 115.249, 117.174])
    assert "out-lap" in notes[0] and "partial lap" in notes[0]


def test_motec_without_two_beacons_is_reported():
    data = MOTEC.replace(b'"110.962 226.789 342.038 459.212 "', b'"110.962"')
    with pytest.raises(ValueError, match="fewer than two beacon markers"):
        read_lap_csv(io.BytesIO(data))


def test_race_format_lap_times_are_converted():
    res, _, _ = _load(b"lap,lap_time_s\n1,1:50.485\n2,1:51.002\n3,110.9\n")
    assert list(res.data["lap_time_s"]) == [110.485, 111.002, 110.9]


def test_malformed_race_format_is_flagged_not_dropped_silently():
    res, _, _ = _load(b"lap,lap_time_s\n1,1:50.485\n2,1:75.0\n3,oops\n")
    assert res.data["lap_time_s"].isna().sum() == 2
    assert any("2 non-numeric" in w for w in res.report.warnings)


def test_lap_time_header_accepted_only_for_race_format():
    res, _, _ = _load(b"Lap,Lap Time,Driver\n1,1:50.485,A\n2,1:51.002,A\n")
    assert list(res.data["lap_time_s"]) == [110.485, 111.002] and res.data["driver"].tolist() == ["A", "A"]
    for data in (b"lap,lap_time\n1,110.4\n", b"lap,Lap Time\n1,1:50.485\n2,110.9\n"):
        with pytest.raises(DataValidationError, match="unknown units"):
            _load(data)


def test_lap_number_alias_from_acc_lap_export():
    res, _, _ = _load(b"session_key,lap_number,lap_time_s,tyre_compound\nk,1,110.4,DRY\nk,2,111.0,DRY\n")
    assert res.report.valid and list(res.data["lap"]) == [1, 2]
    assert any("lap_number" in w for w in res.report.warnings)


@pytest.mark.parametrize("data", [b"lap;lap_time_s\n1;110.4\n2;111.0\n", b"lap\tlap_time_s\n1\t110.4\n2\t111.0\n",
                                  "﻿lap,lap_time_s\n1,110.4\n2,111.0\n".encode()])
def test_separators_and_bom(data):
    res, _, _ = _load(data)
    assert res.report.valid and list(res.data["lap_time_s"]) == [110.4, 111.0]
