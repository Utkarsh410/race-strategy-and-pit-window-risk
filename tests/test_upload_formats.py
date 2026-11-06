"""Every upload format, built here as small synthetic files (no real telemetry is bundled)."""

import io
import json
import struct

import numpy as np
import pandas as pd
import pytest

from racesim.data_ingestion import UPLOAD_TYPES, load_race_dataframe, read_lap_file
from racesim.integrations.motec import LD_CHANNEL, LD_HEADER


def _load(buf, name):
    df, label, notes = read_lap_file(buf, name)
    return load_race_dataframe(df, source=label), label, notes


def _ld_file(crossings, end_s, rate=50, venue=b"TestTrack", channel=b"TIME") -> bytes:
    """Minimal MoTeC .ld: header + one float32 lap-timer channel that resets at each crossing."""
    t = np.arange(int(end_s * rate)) / rate
    last = np.zeros_like(t)
    for c in crossings:
        last[t >= c] = c
    timer = (t - last).astype("<f4")
    head_size, chan_size = struct.calcsize(LD_HEADER), struct.calcsize(LD_CHANNEL)
    meta_ptr, data_ptr = head_size, head_size + chan_size
    head = struct.pack(LD_HEADER, 0x40, meta_ptr, data_ptr, 0, 0, 0, 0, 0, b"ADL", 0, 0, 1, b"01/01/2026", b"10:00:00",
                       b"", b"", venue, 0, b"")
    chan = struct.pack(LD_CHANNEL, 0, 0, data_ptr, len(timer), 0, 0x07, 4, rate, 0, 1, 1, 0, channel, b"T", b"s")
    return head + chan + timer.tobytes()


def _ibt_file(laps, *, fuel_l=80.0, burn_l=2.0, tick=60, kg_per_l="0.750") -> bytes:
    """Minimal iRacing .ibt: LapCompleted, LapLastLapTime, FuelLevel, OnPitRoad and a session YAML."""
    variables = [("LapCompleted", 2, 0), ("LapLastLapTime", 4, 4), ("FuelLevel", 4, 8), ("OnPitRoad", 1, 12)]
    buf_len = 16
    rows = []
    fuel, last = fuel_l, 0.0
    for completed, lap_time in enumerate(list(laps) + [None]):
        n = int((lap_time or 30.0) * tick)
        for i in range(n):
            pit = completed == 0 and i < tick  # leaves the pit lane at the start of lap 1
            rows.append(struct.pack("<iff?3x", completed, last, fuel - burn_l * i / n, pit))
        fuel -= burn_l
        last = lap_time or last
    yaml = ("---\nWeekendInfo:\n TrackDisplayName: Test Circuit\nDriverInfo:\n DriverCarFuelKgPerLtr: " + kg_per_l
            + "\n Drivers:\n - UserName: Someone Real\n   CarScreenName: Test Car\n...\n").encode() + b"\0"
    vh_off = 144
    si_off = vh_off + 144 * len(variables)
    buf_off = si_off + len(yaml)
    header = struct.pack("<10i2i", 2, 1, tick, 1, len(yaml), si_off, len(variables), vh_off, 1, buf_len, 0, 0)
    header += struct.pack("<4i", len(rows), buf_off, 0, 0) + bytes(112 - 64)
    sub = struct.pack("<qddii", 0, 0.0, 0.0, len(laps), len(rows))
    vh = b"".join(struct.pack("<iii?3x", t, off, 1, False) + name.encode().ljust(32, b"\0") + bytes(64) + bytes(32)
                  for name, t, off in variables)
    return header + sub + vh + yaml + b"".join(rows)


def test_upload_types_listed():
    assert {"csv", "xlsx", "json", "jsonl", "parquet", "ld", "ibt"} <= set(UPLOAD_TYPES)


def test_motec_ld_laps_from_lap_timer():
    data = _ld_file(crossings=[50.0, 150.0, 251.5, 353.76], end_s=373.0)
    res, label, notes = _load(io.BytesIO(data), "test.ld")
    assert "MoTeC .ld" in label and "TestTrack" in label
    assert list(res.data["lap_time_s"]) == pytest.approx([100.0, 101.5, 102.26], abs=0.001)
    assert "out-lap" in notes[0]


def test_motec_ld_without_lap_timer_is_reported():
    with pytest.raises(ValueError, match="no running lap-timer channel"):
        read_lap_file(io.BytesIO(_ld_file([50.0, 150.0, 250.0], 260.0, channel=b"SPEED")), "test.ld")


def test_iracing_ibt_laps_fuel_and_pit_flags():
    res, label, notes = _load(io.BytesIO(_ibt_file([100.25, 101.5, 99.75])), "spa.ibt")
    d = res.data
    assert "iRacing" in label and "Test Circuit" in label
    assert list(d["lap"]) == [1, 2, 3] and list(d["lap_time_s"]) == [100.25, 101.5, 99.75]
    assert list(d["fuel_kg"]) == pytest.approx([60.0, 58.5, 57.0])  # litres x 0.75 at lap start
    assert list(d["pit_out"]) == [True, False, False] and not d["pit_in"].any()
    assert "Someone Real" not in d.to_csv() + label + " ".join(notes)


def test_iracing_without_fuel_density_writes_no_fuel():
    res, _, notes = _load(io.BytesIO(_ibt_file([100.0, 101.0], kg_per_l="unknown")), "x.ibt")
    assert "fuel_kg" not in res.data.columns and any("not converted" in n for n in notes)


def test_wrong_binary_is_reported():
    with pytest.raises(ValueError, match="does not look like"):
        read_lap_file(io.BytesIO(b"not a log"), "x.ibt")


@pytest.mark.parametrize("name, payload", [
    ("laps.json", json.dumps([{"lap": 1, "lap_time_s": "1:50.485"}, {"lap": 2, "lap_time_s": 111.0}]).encode()),
    ("laps.json", json.dumps({"laps": [{"lap": 1, "lap_time_s": 110.485}, {"lap": 2, "lap_time_s": 111.0}]}).encode()),
    ("laps.jsonl", b'{"lap": 1, "lap_time_s": 110.485}\n{"lap": 2, "lap_time_s": 111.0}\n'),
    ("laps.tsv", b"lap\tlap_time_s\n1\t110.485\n2\t111.0\n"),
])
def test_lap_tables(name, payload):
    res, _, _ = _load(io.BytesIO(payload), name)
    assert res.report.valid and list(res.data["lap_time_s"]) == [110.485, 111.0]


def test_parquet():
    buf = io.BytesIO()
    pd.DataFrame({"lap": [1, 2], "lap_time_s": [110.485, 111.0]}).to_parquet(buf)
    buf.seek(0)
    res, _, _ = _load(buf, "laps.parquet")
    assert list(res.data["lap_time_s"]) == [110.485, 111.0]


def test_bad_json_shape_is_reported():
    with pytest.raises(ValueError, match="list of lap objects"):
        read_lap_file(io.BytesIO(b'{"rows": 3}'), "laps.json")


def test_excel_without_openpyxl_explains_how_to_fix(monkeypatch):
    import builtins

    real = builtins.__import__

    def no_openpyxl(name, *a, **kw):
        if name == "openpyxl":
            raise ImportError(name)
        return real(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", no_openpyxl)
    with pytest.raises(ValueError, match="openpyxl"):
        read_lap_file(io.BytesIO(b"PK\x03\x04"), "laps.xlsx")
