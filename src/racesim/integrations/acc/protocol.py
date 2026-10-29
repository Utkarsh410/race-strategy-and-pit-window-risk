"""ACC UDP broadcasting protocol: constants, enums and binary encoding.

Wire format (version 4) as cross-checked against two independent open-source
implementations of the Kunos broadcasting SDK protocol:

* ``bradfier/acbc`` (Rust, AGPL-3.0) - ``src/protocol/{outbound,inbound,parser,acc_enum}.rs``
* ``geniusdex/broawp`` ``accbroadcast`` (Go, BSD-3-Clause) - ``messages.go``,
  ``messageparser.go``, ``messagebuilder.go``, ``constants.go`` at commit ``d7f5ba008f03``

No code was copied from either project; this module is an independent
implementation of the documented wire format. The official reference is the
C# SDK shipped with the game under ``<ACC install>/sdk/broadcasting``.

Encoding: little-endian integers, IEEE-754 float32, booleans as one byte
(non-zero = true), strings as a ``uint16`` byte length followed by UTF-8 bytes.
One message per UDP datagram; the first byte is the message type.

Verification status of each element is recorded in :data:`VERIFICATION` and
documented in ``docs/ACC_LIVE_TELEMETRY_SETUP.md``.
"""

from __future__ import annotations

import struct
from enum import IntEnum

PROTOCOL_VERSION = 4
INT32_NULL = 2**31 - 1  # "no value" for lap and split times (broawp: math.MaxInt32 -> unknown)


class ProtocolError(ValueError):
    """A datagram could not be decoded according to the protocol."""


class UnsupportedMessageType(ProtocolError):
    def __init__(self, type_id: int):
        super().__init__(f"unsupported ACC message type {type_id}")
        self.type_id = type_id


class Outbound(IntEnum):
    """Client -> ACC message types used by RaceSim (read-only subset)."""

    REGISTER_COMMAND_APPLICATION = 1
    UNREGISTER_COMMAND_APPLICATION = 9
    REQUEST_ENTRY_LIST = 10
    REQUEST_TRACK_DATA = 11


# ACC also accepts HUD/camera/replay commands (49, 50, 51, 52, 60). RaceSim never
# builds or sends them; the bridge is read-only.
CONTROL_COMMAND_TYPES = frozenset({49, 50, 51, 52, 60})


class Inbound(IntEnum):
    """ACC -> client message types."""

    REGISTRATION_RESULT = 1
    REALTIME_UPDATE = 2
    REALTIME_CAR_UPDATE = 3
    ENTRY_LIST = 4
    TRACK_DATA = 5
    ENTRY_LIST_CAR = 6
    BROADCASTING_EVENT = 7


class SessionType(IntEnum):
    PRACTICE = 0
    QUALIFYING = 4
    SUPERPOLE = 9
    RACE = 10
    HOTLAP = 11
    HOTSTINT = 12
    HOTLAP_SUPERPOLE = 13
    REPLAY = 14


class SessionPhase(IntEnum):
    NONE = 0
    STARTING = 1
    PRE_FORMATION = 2
    FORMATION_LAP = 3
    PRE_SESSION = 4
    SESSION = 5
    SESSION_OVER = 6
    POST_SESSION = 7
    RESULT_UI = 8


class CarLocation(IntEnum):
    NONE = 0
    TRACK = 1
    PITLANE = 2
    PIT_ENTRY = 3
    PIT_EXIT = 4


class BroadcastingEventType(IntEnum):
    NONE = 0
    GREEN_FLAG = 1
    SESSION_OVER = 2
    PENALTY_MESSAGE = 3
    ACCIDENT = 4
    LAP_COMPLETED = 5
    BEST_SESSION_LAP = 6
    BEST_PERSONAL_LAP = 7


def enum_name(enum_cls: type[IntEnum], value: int) -> str:
    """Enum member name, or ``UNKNOWN_<value>`` for values not in the verified table."""
    try:
        return enum_cls(value).name
    except ValueError:
        return f"UNKNOWN_{value}"


VERIFICATION = {
    "encoding (little-endian, u16-length UTF-8 strings)": "verified: both sources",
    "protocol version 4": "verified: both sources",
    "REGISTER layout": "verified: both sources + acbc byte-exact test vector",
    "UNREGISTER layout": "verified: both sources",
    "REQUEST_ENTRY_LIST layout": "verified: broawp (acbc has no builder)",
    "REQUEST_TRACK_DATA layout": "UNVERIFIED: type id 11 in broawp constants only; layout assumed by analogy "
                                 "with REQUEST_ENTRY_LIST (type byte + u32 connection id). Can be disabled.",
    "REGISTRATION_RESULT layout": "verified: both sources (meaning of the 2nd flag byte differs: "
                                  "acbc 'read_only', broawp 'AllowCommands'; stored raw)",
    "REALTIME_UPDATE layout": "verified: both sources for live sessions; replay block differs between sources "
                              "(broawp/SDK sample: bool + 2 floats is used here)",
    "REALTIME_CAR_UPDATE layout": "verified: both sources (gear encoding disagrees; gear not normalised)",
    "ENTRY_LIST / ENTRY_LIST_CAR layouts": "verified: both sources",
    "TRACK_DATA layout": "single source (acbc); broawp does not parse it",
    "BROADCASTING_EVENT layout": "single source (acbc); car id read as int32",
    "lap info + int32-max null": "layout verified: both sources; null sentinel from broawp",
    "session/sessionEnd/timeOfDay units": "milliseconds as float32 (broawp); meaning of sessionEndTime unverified",
    "enum tables": "session type/phase/location/event: acbc; car location also broawp",
}


class Reader:
    """Bounds-checked little-endian reader."""

    def __init__(self, data: bytes):
        self.data = memoryview(data)
        self.pos = 0

    @property
    def remaining(self) -> int:
        return len(self.data) - self.pos

    def _take(self, n: int) -> memoryview:
        if n < 0 or self.pos + n > len(self.data):
            raise ProtocolError(f"truncated message: need {n} byte(s) at offset {self.pos}, have {self.remaining}")
        out = self.data[self.pos:self.pos + n]
        self.pos += n
        return out

    def _unpack(self, fmt: str):
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self._take(size))[0]

    def u8(self) -> int:
        return self._unpack("<B")

    def i8(self) -> int:
        return self._unpack("<b")

    def u16(self) -> int:
        return self._unpack("<H")

    def i32(self) -> int:
        return self._unpack("<i")

    def u32(self) -> int:
        return self._unpack("<I")

    def f32(self) -> float:
        return self._unpack("<f")

    def boolean(self) -> bool:
        return self.u8() != 0

    def string(self) -> str:
        n = self.u16()
        raw = bytes(self._take(n))
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ProtocolError(f"invalid UTF-8 string at offset {self.pos - n}") from exc


class Writer:
    def __init__(self) -> None:
        self.buf = bytearray()

    def u8(self, v: int) -> "Writer":
        self.buf += struct.pack("<B", v)
        return self

    def u16(self, v: int) -> "Writer":
        self.buf += struct.pack("<H", v)
        return self

    def i32(self, v: int) -> "Writer":
        self.buf += struct.pack("<i", v)
        return self

    def u32(self, v: int) -> "Writer":
        self.buf += struct.pack("<I", v)
        return self

    def f32(self, v: float) -> "Writer":
        self.buf += struct.pack("<f", v)
        return self

    def boolean(self, v: bool) -> "Writer":
        return self.u8(1 if v else 0)

    def string(self, s: str) -> "Writer":
        raw = s.encode("utf-8")
        if len(raw) >= 0xFFFF:
            raise ProtocolError("string too long for a u16 length prefix")
        self.u16(len(raw))
        self.buf += raw
        return self

    def bytes(self) -> bytes:
        return bytes(self.buf)


def build_register(display_name: str, connection_password: str, update_interval_ms: int, command_password: str = "") -> bytes:
    """REGISTER_COMMAND_APPLICATION. RaceSim always sends an empty command password (no control rights requested)."""
    if not 0 < update_interval_ms < 2**31:
        raise ValueError("update interval must be a positive number of milliseconds")
    return (Writer().u8(Outbound.REGISTER_COMMAND_APPLICATION).u8(PROTOCOL_VERSION).string(display_name)
            .string(connection_password).u32(update_interval_ms).string(command_password).bytes())


def build_unregister(connection_id: int) -> bytes:
    return Writer().u8(Outbound.UNREGISTER_COMMAND_APPLICATION).u32(connection_id).bytes()


def build_request_entry_list(connection_id: int) -> bytes:
    return Writer().u8(Outbound.REQUEST_ENTRY_LIST).u32(connection_id).bytes()


def build_request_track_data(connection_id: int) -> bytes:
    """UNVERIFIED layout (see :data:`VERIFICATION`); disable with ``request_track_data: false``."""
    return Writer().u8(Outbound.REQUEST_TRACK_DATA).u32(connection_id).bytes()
