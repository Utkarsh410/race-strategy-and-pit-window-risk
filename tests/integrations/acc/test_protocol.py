import pytest

from racesim.integrations.acc import protocol as P
from acc_fixtures import REGISTER_TEST_VECTOR


def test_register_matches_published_test_vector():
    assert P.build_register("Your name", "asd", 250, "") == REGISTER_TEST_VECTOR


def test_register_is_read_only_and_validates_interval():
    data = P.build_register("RaceSim", "pw", 250)
    assert data[0] == P.Outbound.REGISTER_COMMAND_APPLICATION and data[1] == P.PROTOCOL_VERSION
    assert data.endswith(b"\x00\x00")  # empty command password: no control rights requested
    with pytest.raises(ValueError):
        P.build_register("x", "y", 0)


def test_simple_requests():
    assert P.build_unregister(7) == bytes([9, 7, 0, 0, 0])
    assert P.build_request_entry_list(7) == bytes([10, 7, 0, 0, 0])
    assert P.build_request_track_data(7) == bytes([11, 7, 0, 0, 0])
    assert P.CONTROL_COMMAND_TYPES.isdisjoint(int(x) for x in P.Outbound)


def test_reader_bounds_and_strings():
    w = P.Writer().string("héllo").u16(5)
    r = P.Reader(w.bytes())
    assert r.string() == "héllo" and r.u16() == 5 and r.remaining == 0
    with pytest.raises(P.ProtocolError):
        P.Reader(b"\x05\x00ab").string()
    with pytest.raises(P.ProtocolError):
        P.Reader(b"\x02\x00\xff\xfe").string()  # invalid UTF-8


def test_enum_names():
    assert P.enum_name(P.SessionType, 10) == "RACE"
    assert P.enum_name(P.SessionType, 2) == "UNKNOWN_2"
    assert "UNVERIFIED" in P.VERIFICATION["REQUEST_TRACK_DATA layout"]
