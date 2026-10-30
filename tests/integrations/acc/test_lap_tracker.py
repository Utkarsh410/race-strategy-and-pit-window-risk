"""Lap completion, classification, duplicates, ordering and stint transitions."""

from live_fixtures import Script, drive, feed

from racesim.integrations.acc.lap_tracker import LapClass, LapTracker


def _tracker():
    return LapTracker("s")


def test_lap_completes_once_when_counter_moves():
    s, tr = Script(), _tracker()
    laps = feed(tr, drive(s, 0.0, 0, [120.0, 119.0]))
    assert [(x.lap_number, x.lap_time_s) for x in laps] == [(1, 120.0), (2, 119.0)]


def test_repeated_telemetry_does_not_duplicate_a_lap():
    s, tr = Script(), _tracker()
    feed(tr, [(1.0, s.tel(1.0)), (120.0, s.tel(120.0, laps=1, last=119.0))])
    for t in (120.5, 121.0, 121.0):
        assert tr.update(s.tel(t, laps=1, last=119.0), t) == []
    assert len(tr.laps) == 1


def test_out_lap_that_does_not_move_the_counter_does_not_taint_next_lap():
    """Recorded ACC behaviour: leave the pits, cross the line with laps still 0, then the first timed lap."""
    s, tr = Script(), _tracker()
    frames = [(float(t), s.tel(float(t), location="PITLANE", speed=0, current=float(t))) for t in range(1, 5)]
    frames += [(5.0, s.tel(5.0, location="PIT_EXIT", current=5.0))]
    frames += [(float(t), s.tel(float(t), current=float(t))) for t in range(6, 125)]
    frames += [(float(t), s.tel(float(t), current=float(t - 125))) for t in range(125, 245)]
    frames += [(245.0, s.tel(245.0, laps=1, last=120.0, current=0.1))]
    laps = feed(tr, frames)
    assert [(x.lap_number, x.classification) for x in laps] == [(1, LapClass.CLEAN)]
    assert tr.flags["line_crossing_without_lap_count"] == 1
    assert len(tr.stints) == 1 and tr.stints[0].left_pits_at_start


def test_invalid_lap_is_marked_from_acc_flag():
    s, tr = Script(), _tracker()
    laps = feed(tr, drive(s, 0.0, 0, [120.0], valid=False))
    assert laps[0].classification == LapClass.INVALID and "acc_lap_invalid" in laps[0].flags
    assert not laps[0].is_clean


def test_missing_validity_is_uncertain():
    s, tr = Script(), _tracker()
    frames = drive(s, 0.0, 0, [120.0])
    frames[-1][1]["vehicle"].pop("last_lap_valid")
    laps = feed(tr, frames)
    assert laps[0].classification == LapClass.UNCERTAIN and "validity_unknown" in laps[0].flags


def test_slow_lap_is_kept_and_uncertain_not_discarded():
    s, tr = Script(), _tracker()
    laps = feed(tr, drive(s, 0.0, 0, [120.0, 120.5, 119.5, 150.0]))
    slow = laps[-1]
    assert slow.lap_time_s == 150.0 and slow.classification == LapClass.UNCERTAIN and "slow_lap_cause_unknown" in slow.flags
    assert len(tr.laps) == 4


def test_moderately_slower_lap_stays_clean():
    s, tr = Script(), _tracker()
    laps = feed(tr, drive(s, 0.0, 0, [120.0, 120.5, 119.5, 124.0]))
    assert laps[-1].classification == LapClass.CLEAN


def test_slow_lap_before_enough_history_is_not_judged():
    s, tr = Script(), _tracker()
    assert feed(tr, drive(s, 0.0, 0, [120.0, 150.0]))[-1].classification == LapClass.CLEAN


def test_in_lap_from_pit_entry_inside_lap_window():
    s, tr = Script(), _tracker()
    frames = drive(s, 0.0, 0, [120.0])
    frames += [(t, s.tel(t, laps=1, last=120.0)) for t in (121.0, 200.0)]
    frames += [(t, s.tel(t, laps=1, last=120.0, location="PITLANE", speed=60)) for t in (235.0, 239.0)]
    frames += [(240.0, s.tel(240.0, laps=2, last=120.0 + 0.001, location="PITLANE", speed=60))]
    laps = feed(LapTracker("s"), frames)
    assert laps[-1].classification == LapClass.IN_LAP and laps[-1].in_pit


def test_out_lap_when_lap_starts_in_pits():
    s, tr = Script(), _tracker()
    frames = [(0.0, s.tel(0.0, location="PITLANE", speed=0))]
    frames += [(t, s.tel(t, location="PITLANE", speed=50)) for t in (1.0, 2.0)]
    frames += [(3.0, s.tel(3.0, location="PIT_EXIT"))]
    frames += [(t, s.tel(t)) for t in (4.0, 8.0, 12.0, 16.0)]
    frames += [(float(t), s.tel(float(t))) for t in range(20, 121, 4)]
    frames += [(121.0, s.tel(121.0, laps=1, last=121.0))]
    assert feed(tr, frames)[0].classification == LapClass.OUT_LAP


def test_native_out_lap_flag_is_used():
    s, tr = Script(), _tracker()
    frames = drive(s, 0.0, 0, [120.0])
    frames[-1][1]["vehicle"]["last_lap_is_out_lap"] = True
    assert feed(tr, frames)[0].classification == LapClass.OUT_LAP


def test_pit_visit_starts_a_new_stint_and_is_not_a_lap():
    s, tr = Script(), _tracker()
    frames = drive(s, 0.0, 0, [120.0])
    frames += [(125.0, s.tel(125.0, laps=1, last=120.0, location="PIT_ENTRY", speed=80)),
               (130.0, s.tel(130.0, laps=1, last=120.0, location="PITLANE", speed=0)),
               (150.0, s.tel(150.0, laps=1, last=120.0, location="PIT_EXIT", speed=80)),
               (151.0, s.tel(151.0, laps=1, last=120.0, location="TRACK"))]
    feed(tr, frames)
    assert len(tr.laps) == 1
    assert [x["event"] for x in tr.transitions] == ["PIT_ENTRY", "PIT_EXIT"]
    assert tr.transitions[1]["stationary_observed"] is True
    assert len(tr.stints) == 2 and tr.stints[0].end_reason == "pit_visit" and tr.stints[1].start_reason == "pit_exit"
    assert tr.stints[0].lap_numbers == [1]


def test_drive_through_records_no_stationary_stop():
    s, tr = Script(), _tracker()
    feed(tr, [(1.0, s.tel(1.0)), (2.0, s.tel(2.0, location="PITLANE", speed=60)), (3.0, s.tel(3.0, location="TRACK"))])
    assert tr.transitions[1]["stationary_observed"] is False
    assert tr.stints[0].stationary_observed is False


def test_pit_exit_flicker_is_not_a_second_entry():
    s, tr = Script(), _tracker()
    feed(tr, [(1.0, s.tel(1.0)), (2.0, s.tel(2.0, location="PITLANE", speed=60)), (3.0, s.tel(3.0, location="PIT_EXIT")),
              (3.2, s.tel(3.2, location="PITLANE", speed=60)), (3.4, s.tel(3.4, location="PIT_EXIT")), (4.0, s.tel(4.0))])
    assert [x["event"] for x in tr.transitions] == ["PIT_ENTRY", "PIT_EXIT"] and len(tr.stints) == 2


def test_garage_exit_at_start_is_not_a_stint_change():
    s, tr = Script(), _tracker()
    feed(tr, [(0.0, s.tel(0.0, location="PITLANE", speed=0)), (5.0, s.tel(5.0, location="PIT_EXIT")), (6.0, s.tel(6.0))])
    assert len(tr.stints) == 1 and tr.stints[0].start_reason == "session_start_in_pit"


def test_skipped_lap_is_recorded_as_incomplete():
    s, tr = Script(), _tracker()
    feed(tr, [(1.0, s.tel(1.0)), (120.0, s.tel(120.0, laps=1, last=119.0)), (360.0, s.tel(360.0, laps=3, last=118.0))])
    assert [x.lap_number for x in tr.laps] == [1, 2, 3]
    assert tr.laps[1].classification == LapClass.INCOMPLETE and tr.laps[1].lap_time_s is None
    assert "telemetry_missed_lap" in tr.laps[1].flags and tr.laps[2].lap_time_s == 118.0


def test_telemetry_gap_inside_lap_is_uncertain():
    s, tr = Script(), _tracker()
    frames = drive(s, 0.0, 0, [120.0])
    frames = frames[:30] + frames[60:]
    lap = feed(tr, frames)[0]
    assert lap.classification == LapClass.UNCERTAIN and "telemetry_gap_in_lap" in lap.flags


def test_joining_mid_session_invents_no_earlier_laps():
    s, tr = Script(), _tracker()
    feed(tr, [(500.0, s.tel(500.0, laps=4, last=120.0))])
    assert tr.laps == [] and tr.flags["joined_mid_session"] == 1


def test_lap_time_arriving_after_counter_is_matched_to_that_lap():
    s, tr = Script(), _tracker()
    laps = feed(tr, drive(s, 0.0, 0, [120.0])[:-1] + [(120.0, s.tel(120.0, laps=1))])
    assert laps == []
    late = tr.update(s.tel(120.5, laps=1, last=120.0, valid=True), 120.5)
    assert late[0].lap_number == 1 and late[0].lap_time_s == 120.0


def test_lap_time_that_never_updates_is_incomplete():
    s, tr = Script(), _tracker()
    feed(tr, drive(s, 0.0, 0, [120.0]))
    assert tr.update(s.tel(260.0, laps=2, last=120.0), 260.0) == []  # 140 s on the clock, so 120.0 is the old lap's time
    lap = tr.update(s.tel(264.0, laps=2, last=120.0), 264.0)[0]
    assert lap.lap_number == 2 and lap.lap_time_s is None and lap.classification == LapClass.INCOMPLETE
    assert "lap_time_not_updated" in lap.flags


def test_identical_consecutive_lap_times_are_both_recorded():
    s, tr = Script(), _tracker()
    laps = feed(tr, drive(s, 0.0, 0, [120.0, 120.0, 120.0]))
    assert [x.lap_time_s for x in laps] == [120.0, 120.0, 120.0]
    assert all(x.classification == LapClass.CLEAN for x in laps)


def test_lap_time_is_not_credited_to_an_earlier_pending_lap():
    s, tr = Script(), _tracker()
    feed(tr, drive(s, 0.0, 0, [120.0]))
    tr.update(s.tel(260.0, laps=2, last=120.0), 260.0)
    tr.update(s.tel(379.0, laps=3, last=119.0), 260.5)
    assert [(x.lap_number, x.lap_time_s) for x in tr.laps] == [(1, 120.0), (2, None), (3, 119.0)]


def test_race_start_lap_is_uncertain():
    s, tr = Script(session_type="RACE"), _tracker()
    lap = feed(tr, drive(s, 0.0, 0, [130.0]))[0]
    assert lap.classification == LapClass.UNCERTAIN and "race_start_lap" in lap.flags


def test_formation_phase_inside_lap_is_uncertain():
    s, tr = Script(), _tracker()
    frames = drive(s, 0.0, 0, [120.0])
    frames[3][1]["session"]["session_phase"] = "FORMATION_LAP"
    assert "session_phase_not_green" in feed(tr, frames)[0].flags


def test_sectors_from_last_lap_splits_when_they_add_up():
    s, tr = Script(), _tracker()
    lap = feed(tr, drive(s, 0.0, 0, [120.0], last_lap_splits_s=[40.0, 40.0, 40.0]))[0]
    assert lap.sector_times_s == [40.0, 40.0, 40.0] and "lastLap" in lap.sector_source


def test_sectors_are_dropped_if_they_do_not_add_up():
    s, tr = Script(), _tracker()
    lap = feed(tr, drive(s, 0.0, 0, [120.0], last_lap_splits_s=[40.0, 80.0, 120.0]))[0]
    assert lap.sector_times_s is None and "sector_sum_mismatch" in lap.flags


def test_no_sectors_means_unavailable():
    s, tr = Script(), _tracker()
    assert feed(tr, drive(s, 0.0, 0, [120.0]))[0].sector_times_s is None


def test_fuel_recorded_at_lap_start_and_end():
    s, tr = Script(), _tracker()
    frames = [(float(t), s.tel(float(t), fuel_kg=50.0 - t * 0.02)) for t in range(0, 120)]
    frames.append((120.0, s.tel(120.0, laps=1, last=120.0, fuel_kg=47.6)))
    lap = feed(tr, frames)[0]
    assert lap.fuel_start_kg == 50.0 and lap.fuel_end_kg == 47.6


def test_tyre_age_stays_none_without_a_measurement():
    s, tr = Script(), _tracker()
    lap = feed(tr, drive(s, 0.0, 0, [120.0]))[0]
    assert lap.tyre_age_laps is None and lap.tyre_compound is None and lap.stint_lap == 1
