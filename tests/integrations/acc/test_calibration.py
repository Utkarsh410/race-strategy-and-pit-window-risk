"""ACC log -> RaceSim lap data -> calibration.

Fixtures are deterministic test data (formula + seeded noise, or the
live_fixtures telemetry script); they are not recorded ACC sessions.
"""

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from live_fixtures import BASE, Script, drive
from racesim.integrations.acc.calibration import (
    DATA_SOURCE, UNKNOWN_COMPOUND, calibrate_from_acc_logs, calibrate_lap_data, report_markdown, session_logs_to_lap_data,
)
from racesim.integrations.acc.session_logger import SessionLogger
from racesim.validation import classify_evidence


def write_log(tmp_path, frames, *, mock=False, name="log"):
    lg = SessionLogger(tmp_path, prefix=name)
    lg.open()
    for i, (t, payload) in enumerate(frames, start=1):
        env = {"message_type": "telemetry", "schema_version": "1.0", "sequence": i, "sent_at_utc": payload["timestamp_utc"],
               "sender_id": "test", "sender_run_id": "r", "mock": mock, "payload": payload}
        lg.log(env, received_at_utc=BASE + timedelta(seconds=t))
    lg.close()
    return lg.path


def pit_start_session(script, lap_times):
    frames = [(float(t), script.tel(float(t), location="PITLANE", speed=0)) for t in range(0, 3)]
    frames += [(3.0, script.tel(3.0, location="PIT_EXIT", speed=60))]
    return frames + drive(script, 3.0, 0, lap_times)


def test_conversion_from_logs(tmp_path):
    real = write_log(tmp_path, pit_start_session(Script(session_id="s1", track="Circuit Test"), [121.0, 120.5, 120.6, 120.8]))
    mock = write_log(tmp_path, pit_start_session(Script(session_id="m1", source="mock", track="MOCK"), [100.0, 100.0]),
                     mock=True, name="mock")
    res = session_logs_to_lap_data([real, mock])
    df = res.laps
    assert res.provenance["mock_sessions_skipped"] == 1 and res.provenance["sessions_used"] == 1
    assert set(df["track_name"]) == {"Circuit Test"} and (df["data_source"] == DATA_SOURCE).all()
    assert classify_evidence(df) == "historical_out_of_sample"
    assert (df["tyre_compound"] == UNKNOWN_COMPOUND).all() and df["fuel_kg"].isna().all()   # not invented
    assert df["lap"].min() >= 2                                                             # ACC counter + 1
    known = df.dropna(subset=["tyre_age_laps"])
    assert not known.empty and (known["tyre_age_laps"] >= 0).all()
    assert set(df["acc_lap_class"]) <= {"CLEAN_CANDIDATE", "INVALID", "OUT_LAP", "IN_LAP", "UNCERTAIN", "INCOMPLETE"}
    assert (df["exclude_from_model"] == (df["acc_lap_class"] != "CLEAN_CANDIDATE")).all()


def formula_laps(sessions=6, laps=20, deg=0.05, pit_lane=10.0, out_extra=15.0, seed=0, fuel=True, compound="MEDIUM", pit=True):
    """Deterministic lap table with known degradation and pit loss (test data, not ACC)."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(sessions):
        for i in range(laps):
            fuel_kg = 60 - 2.5 * i
            lap_time = 120 + 0.2 * s + deg * i + 0.03 * fuel_kg + rng.normal(0, 0.15)
            rows.append(dict(lap=i + 2, driver=f"sess{s}", track_name="Circuit Test", session_type="PRACTICE", lap_time_s=lap_time,
                             tyre_compound=compound, tyre_age_laps=i, fuel_kg=fuel_kg if fuel else None, pit_in=False, pit_out=False,
                             acc_lap_class="CLEAN_CANDIDATE", acc_stint_id=1, exclude_from_model=False, data_source=DATA_SOURCE))
        if not pit:
            continue
        base = 120 + 0.2 * s + 0.03 * 10
        rows.append(dict(lap=laps + 2, driver=f"sess{s}", track_name="Circuit Test", session_type="PRACTICE",
                         lap_time_s=base + pit_lane, tyre_compound=compound, tyre_age_laps=laps, fuel_kg=10.0 if fuel else None,
                         pit_in=True, pit_out=False, acc_lap_class="IN_LAP", acc_stint_id=1, exclude_from_model=True, data_source=DATA_SOURCE))
        rows.append(dict(lap=laps + 3, driver=f"sess{s}", track_name="Circuit Test", session_type="PRACTICE",
                         lap_time_s=base + out_extra, tyre_compound=compound, tyre_age_laps=0, fuel_kg=10.0 if fuel else None,
                         pit_in=False, pit_out=True, acc_lap_class="OUT_LAP", acc_stint_id=2, exclude_from_model=True, data_source=DATA_SOURCE))
    return pd.DataFrame(rows)


def test_sufficient_data_gives_estimates(setup):
    rep = calibrate_lap_data(formula_laps(), setup, fuel_effect_s_per_kg=0.03)
    s = rep.sufficiency.set_index("parameter")
    assert (s["status"] == "estimated").all(), s[["status", "available"]]
    assert s.loc["linear tyre degradation", "value"] == pytest.approx(0.05, abs=0.02)
    assert s.loc["lap-to-lap variability (robust std)", "value"] < 0.5
    # median clean lap is mid-stint, so the pit-loss cross-check is near pit_lane + out_extra minus that offset
    assert 15 < s.loc["pit loss (in-lap + out-lap - 2 x clean lap)", "value"] < 30
    v = rep.validation["Circuit Test"]
    assert v.status == "conclusive" and v.evidence_type == "historical_out_of_sample"
    assert rep.any_estimate and "Circuit Test" in report_markdown(rep)


def test_insufficient_real_like_data_produces_no_values(setup):
    df = formula_laps(sessions=1, laps=4, compound=UNKNOWN_COMPOUND, fuel=False, pit=False)
    rep = calibrate_lap_data(df, setup)
    assert not rep.any_estimate and rep.sufficiency["value"].isna().all()
    assert (rep.sufficiency["still_needed"] > 0).sum() >= 4
    assert any("configuration is unchanged" in w for w in rep.warnings)


def test_unknown_fuel_is_never_corrected_with_synthetic_rate(setup):
    rep = calibrate_lap_data(formula_laps(fuel=False), setup)
    est = rep.calibration["Circuit Test"]["MEDIUM"]
    assert not est.fuel_corrected and not est.reliable                       # confounded, flagged
    note = rep.sufficiency.set_index("parameter").loc["linear tyre degradation", "note"]
    assert "confounded" in note
    declared = calibrate_lap_data(formula_laps(fuel=False), setup, fuel_per_lap_kg=2.5, fuel_effect_s_per_kg=0.03)
    assert declared.calibration["Circuit Test"]["MEDIUM"].fuel_corrected  # declared rate is used, and only then


def test_empty_logs(tmp_path, setup):
    p = write_log(tmp_path, [])
    rep = calibrate_from_acc_logs([p], setup)
    assert rep.lap_data.empty and rep.warnings
