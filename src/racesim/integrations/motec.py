"""Lap times from MoTeC files: an i2 CSV export or a binary .ld log (as written by ACC's MoTeC logger).

From a CSV export only the header block is read: ``Beacon Markers`` holds the lap-line
crossing times (seconds from the start of the outing), so lap ``i`` is the
time between markers ``i-1`` and ``i``. The 200 Hz sample rows are not
parsed, which keeps even a 250 MB export fast.

* The out-lap (start of outing to the first marker) and the final partial
  lap (last marker to end of outing) are not complete laps and are left out.
* A segment that includes a pit stop or garage visit is still a "lap"
  between two markers; RaceSim's cleaning step flags it as an outlier
  rather than this reader guessing.
"""

from __future__ import annotations

import csv
import io
import struct
from dataclasses import dataclass, field
from typing import BinaryIO, Optional

import pandas as pd

MOTEC_MARKER = "MoTeC CSV File"
HEADER_LINES = 20


@dataclass
class MotecHeader:
    fields: dict[str, str]
    beacons_s: list[float]
    duration_s: Optional[float]
    notes: list[str] = field(default_factory=list)

    @property
    def venue(self) -> str:
        return self.fields.get("Venue", "")

    @property
    def vehicle(self) -> str:
        return self.fields.get("Vehicle", "")


def _head_text(buffer: BinaryIO, n_bytes: int = 64 * 1024) -> str:
    pos = buffer.tell()
    head = buffer.read(n_bytes)
    buffer.seek(pos)
    return head.decode("latin-1") if isinstance(head, bytes) else head


def is_motec_csv(buffer: BinaryIO) -> bool:
    first = _head_text(buffer, 256).lstrip("﻿").splitlines()[:1]
    return bool(first) and MOTEC_MARKER in first[0]


def read_motec_header(buffer: BinaryIO) -> MotecHeader:
    lines = _head_text(buffer).splitlines()[:HEADER_LINES]
    fields: dict[str, str] = {}
    for row in csv.reader(io.StringIO("\n".join(lines))):
        # rows hold up to two "name","value" pairs: columns 0/1 and 4/5
        for k, v in ((0, 1), (4, 5)):
            if len(row) > v and row[k].strip():
                fields[row[k].strip()] = row[v].strip()
    beacons = [float(x) for x in fields.get("Beacon Markers", "").split()]
    try:
        duration = float(fields["Duration"])
    except (KeyError, ValueError):
        duration = None
    return MotecHeader(fields=fields, beacons_s=beacons, duration_s=duration)


def motec_laps(buffer: BinaryIO) -> tuple[pd.DataFrame, MotecHeader]:
    """Lap-level frame (``lap``, ``lap_time_s``) from the beacon markers of a MoTeC CSV export."""
    header = read_motec_header(buffer)
    b = header.beacons_s
    if len(b) < 2:
        raise ValueError("MoTeC CSV has fewer than two beacon markers, so it holds no complete lap. "
                         "Export again from MoTeC i2 with the lap beacons included.")
    laps = pd.DataFrame({"lap": range(1, len(b)), "lap_time_s": [round(b2 - b1, 3) for b1, b2 in zip(b, b[1:])]})
    header.notes.append(f"Lap times from {len(b)} MoTeC beacon markers: {len(laps)} complete lap(s). "
                        f"The out-lap ({b[0]:.1f} s from the start of the outing) is left out"
                        + (f", as is the partial lap after the last marker ({header.duration_s - b[-1]:.1f} s)."
                           if header.duration_s is not None and header.duration_s > b[-1] else "."))
    header.notes.append("Only lap times come from this file: no tyre, fuel or sector columns. "
                        "Laps that include a pit stop or garage time are flagged by the cleaning step, not removed.")
    return laps, header


# --- MoTeC .ld binary log -------------------------------------------------------------------------
# Layout as documented by the open-source ldparser project: a fixed header, then a linked list of
# channel descriptors, each pointing at its sample block. Values are (raw / scale * 10**-dec + shift) * mul.
LD_HEADER = "<I4xII20xI24xHHHI8sHHI4x16s16x16s16x64s64s64x64s64x1024xI66x64s126x"
LD_CHANNEL = "<IIIIHHHHhhhh32s8s12s40x"
LAP_TIMER_CHANNELS = ("TIME", "Lap Time", "Running Lap Time")  # ACC writes the running lap time as TIME


def _ld_text(b: bytes) -> str:
    return b.split(b"\0")[0].decode("latin-1").strip()


def is_motec_ld(buffer: BinaryIO) -> bool:
    head = _head_bytes(buffer, struct.calcsize(LD_HEADER))
    if len(head) < struct.calcsize(LD_HEADER):
        return False
    marker, meta_ptr, data_ptr = struct.unpack_from("<I4xII", head, 0)
    return marker == 0x40 and 0 < meta_ptr < 1 << 31 and 0 < data_ptr < 1 << 31


def _head_bytes(buffer: BinaryIO, n: int) -> bytes:
    pos = buffer.tell()
    head = buffer.read(n)
    buffer.seek(pos)
    return head if isinstance(head, bytes) else head.encode("latin-1")


def ld_laps(buffer: BinaryIO) -> tuple[pd.DataFrame, MotecHeader]:
    """Lap-level frame from a MoTeC .ld log, using its running lap-timer channel.

    A lap ends where the timer resets; the lap time is the timer just before
    the reset plus the part of the sample interval left after it, so the
    sampling step does not shorten every lap. The out-lap (before the first
    reset) and the unfinished last lap are left out.
    """
    import numpy as np

    buffer.seek(0)
    raw = buffer.read()
    buffer.seek(0)
    h = struct.unpack_from(LD_HEADER, raw, 0)
    fields = {"Device": _ld_text(h[8]), "Log Date": _ld_text(h[12]), "Log Time": _ld_text(h[13]), "Driver": _ld_text(h[14]),
              "Vehicle": _ld_text(h[15]), "Venue": _ld_text(h[16])}
    timer, freq, names, ptr, seen = None, None, [], h[1], 0
    while ptr and seen < 10_000:
        (_, nxt, dptr, n, _, dta, dt, rate, shift, mul, scale, dec, name, _, _) = struct.unpack_from(LD_CHANNEL, raw, ptr)
        names.append(_ld_text(name))
        if timer is None and names[-1] in LAP_TIMER_CHANNELS and rate > 0:
            dtype = {(0x07, 4): "<f4", (0x07, 2): "<f2", (0, 4): "<i4", (3, 4): "<i4", (5, 4): "<i4",
                     (0, 2): "<i2", (3, 2): "<i2", (5, 2): "<i2"}.get((dta, dt))
            if dtype and scale:
                timer = (np.frombuffer(raw, dtype, n, dptr).astype(float) / scale * 10.0 ** -dec + shift) * mul
                freq = float(rate)
        ptr, seen = nxt, seen + 1
    if timer is None:
        raise ValueError("MoTeC .ld log has no running lap-timer channel (" + ", ".join(LAP_TIMER_CHANNELS)
                         + "), so lap times cannot be read from it alone. Export the laps as a MoTeC CSV instead.")
    resets = np.where(np.diff(timer) < -1.0)[0]
    lap_ends = timer[resets] + 1.0 / freq - timer[resets + 1]
    if len(lap_ends) < 2:
        raise ValueError("MoTeC .ld log holds no complete lap (fewer than two lap-line crossings).")
    times = np.round(lap_ends[1:], 3)  # the first crossing ends the out-lap
    laps = pd.DataFrame({"lap": range(1, len(times) + 1), "lap_time_s": times})
    header = MotecHeader(fields=fields, beacons_s=[], duration_s=len(timer) / freq)
    header.notes.append(f"Lap times from the .ld lap-timer channel at {freq:g} Hz: {len(laps)} complete lap(s). "
                        f"The out-lap ({lap_ends[0]:.1f} s) and the unfinished last lap are left out.")
    header.notes.append(f"{len(names)} channels in the file; only the lap timer is used. No tyre, fuel or sector columns.")
    return laps, header
