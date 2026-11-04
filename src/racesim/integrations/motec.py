"""Lap times from a MoTeC i2 CSV export (as written by ACC's MoTeC logger).

Only the header block is read: ``Beacon Markers`` holds the lap-line
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
