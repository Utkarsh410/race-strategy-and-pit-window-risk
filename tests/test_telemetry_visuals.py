from dashboard.telemetry_visuals import (BRAKE_TEMP_BANDS, GREEN, OFF, RED, TYRE_TEMP_BANDS, band_colour, driver_dash,
                                         tyre_corners)

FULL = {"rpm": 6221.0, "max_rpm": 7800.0, "throttle": 0.92, "brake": 0.0, "speed_kmh": 199.4, "gear": 5, "fuel_remaining": 62.3,
        "max_fuel": 120.0, "water_temp": 88.0, "tyre_core_temperature": [84.0, 91.0, 79.0, 97.0],
        "wheel_pressure": [27.6, 27.9, 27.1, 29.7], "brake_temperature": [455.0, 470.0, 320.0, 820.0], "tyre_compound": "dry_compound"}


def _texts(fig):
    return [a.text for a in fig.layout.annotations]


def test_band_colour_and_unavailable_is_grey():
    assert band_colour(85.0, TYRE_TEMP_BANDS) == GREEN and band_colour(820.0, BRAKE_TEMP_BANDS) == RED
    assert band_colour(None, TYRE_TEMP_BANDS) == OFF


def test_driver_dash_shows_values_and_label():
    fig = driver_dash(FULL, "LIVE - MEASURED")
    values = {t.title.text: t.value for t in fig.data}
    assert values["RPM"] == 6221.0 and values["Throttle"] == 92.0 and values["Water °C"] == 88.0
    assert "199" in _texts(fig) and "5" in _texts(fig) and "LIVE - MEASURED" in fig.layout.title.text
    assert not any(t in ("MOCK", "STALE") for t in _texts(fig))


def test_driver_dash_never_draws_missing_values_as_zero():
    fig = driver_dash({"rpm": 4200.0}, "LIVE - MEASURED")
    shown = {t.title.text: t for t in fig.data}
    assert shown["Throttle"].mode == "gauge" and "number" not in shown["Throttle"].mode
    assert _texts(fig).count("UNAVAILABLE") >= 5  # throttle, brake, fuel, water, speed
    assert any("no shift bands" in t for t in shown)  # max RPM unknown: no invented shift point


def test_tyre_corners_readings_and_missing_wheel():
    channels = {**FULL, "tyre_core_temperature": [84.0, None, 79.0, 97.0], "wheel_pressure": None}
    texts = _texts(tyre_corners(channels, "LIVE - MEASURED"))
    assert "84 °C" in texts and "97 °C" in texts and "brake 820 °C" in texts
    assert texts.count("UNAVAILABLE") == 1 and texts.count("pressure UNAVAILABLE") == 4


def test_mock_and_stale_are_watermarked():
    for label, mark in (("MOCK - synthetic", "MOCK"), ("STALE - not current", "STALE")):
        assert mark in _texts(driver_dash(FULL, label)) and mark in _texts(tyre_corners(FULL, label))
