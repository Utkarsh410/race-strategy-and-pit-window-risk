"""Native ACC message fixtures for tests.

These datagrams are MANUALLY CONSTRUCTED from the wire layout cross-checked in
racesim.integrations.acc.protocol (two independent open-source
implementations). They are not captured from a running game, and passing tests
on them does not prove a live ACC session works.

The only byte-exact external fixture is REGISTER_TEST_VECTOR, the expected
encoding published in bradfier/acbc's unit test (facts only; no code copied).
"""

from racesim.integrations.acc.protocol import INT32_NULL, Inbound, Writer

REGISTER_TEST_VECTOR = bytes.fromhex("01 04 09 00 59 6f 75 72 20 6e 61 6d 65 03 00 61 73 64 fa 00 00 00 00 00")


def lap(w: Writer, lap_ms=INT32_NULL, car=0, driver=0, splits=(), invalid=False, valid_best=True, out=False, inl=False) -> Writer:
    w.i32(lap_ms).u16(car).u16(driver).u8(len(splits))
    for s in splits:
        w.i32(s)
    return w.boolean(invalid).boolean(valid_best).boolean(out).boolean(inl)


def registration_result(conn_id=7, success=True, flag=False, error="") -> bytes:
    return Writer().u8(Inbound.REGISTRATION_RESULT).u32(conn_id).boolean(success).boolean(flag).string(error).bytes()


def realtime_update(event=1, session=0, session_type=10, phase=5, session_time_ms=125_000.0, end_ms=1_675_000.0,
                    focused=3, replay=False, ambient=22, track=31) -> bytes:
    w = (Writer().u8(Inbound.REALTIME_UPDATE).u16(event).u16(session).u8(session_type).u8(phase).f32(session_time_ms)
         .f32(end_ms).u32(focused).string("drivable").string("chase").string("Basic HUD").boolean(replay))
    if replay:
        w.f32(1000.0).f32(2000.0)
    w.f32(50_400_000.0).u8(ambient).u8(track).u8(1).u8(0).u8(0)
    return lap(w, 101_234, car=3).bytes()


def car_update(car=3, position=2, laps=4, location=1, last_ms=102_345, best_ms=101_234, current_ms=34_567,
               current_splits=(), spline=0.37) -> bytes:
    w = (Writer().u8(Inbound.REALTIME_CAR_UPDATE).u16(car).u16(0).u8(1).u8(5).f32(10.0).f32(-5.0).f32(0.5).u8(location)
         .u16(212).u16(position).u16(position).u16(position).f32(spline).u16(laps).i32(-350))
    lap(w, best_ms, car=car)
    lap(w, last_ms, car=car, splits=(33_000, 35_000, 34_345))
    lap(w, current_ms, car=car, splits=current_splits)
    return w.bytes()


def entry_list(conn=7, cars=(3, 5)) -> bytes:
    w = Writer().u8(Inbound.ENTRY_LIST).u32(conn).u16(len(cars))
    for c in cars:
        w.u16(c)
    return w.bytes()


def entry_list_car(car=3, number=88) -> bytes:
    w = (Writer().u8(Inbound.ENTRY_LIST_CAR).u16(car).u8(24).string("Test Team").i32(number).u8(0).u8(0).u16(5).u8(1)
         .string("Alex").string("Tester").string("TES").u8(1).u16(5))
    return w.bytes()


def track_data(conn=7, name="Test Track", meters=5793) -> bytes:
    w = Writer().u8(Inbound.TRACK_DATA).i32(conn).string(name).i32(12).i32(meters)
    w.u8(1).string("set1").u8(2).string("cam1").string("cam2")
    w.u8(1).string("Basic HUD")
    return w.bytes()


def broadcasting_event(kind=5, msg="Lap completed", time_ms=125_000, car=3) -> bytes:
    return Writer().u8(Inbound.BROADCASTING_EVENT).u8(kind).string(msg).i32(time_ms).i32(car).bytes()
