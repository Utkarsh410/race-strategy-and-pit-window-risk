"""Lap times from an iRacing ``.ibt`` telemetry file (the player's car only).

Layout as published in the iRacing SDK (``irsdk_defines.h``): a 112-byte
header, a 32-byte disk sub-header, 144-byte variable headers, a YAML session
string, then fixed-size sample rows at ``tickRate`` Hz.

* Lap ``L`` runs while ``LapCompleted == L - 1``; its time is iRacing's own
  ``LapLastLapTime`` read at the end of the next lap's samples (iRacing
  updates it shortly after the line, not on the crossing tick). Laps with a
  non-positive time (no valid time recorded) are left out.
* ``fuel_kg`` at lap start is ``FuelLevel`` (litres) times the car's
  ``DriverCarFuelKgPerLtr`` from the session YAML; without that value no fuel
  column is written rather than assuming a density.
* ``pit_out`` / ``pit_in``: ``OnPitRoad`` at the first / last sample of the lap.
* Driver names in the session YAML are never copied into the data.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from typing import BinaryIO, Optional

import numpy as np
import pandas as pd

_TYPES = {0: "S1", 1: "?", 2: "<i4", 3: "<u4", 4: "<f4", 5: "<f8"}
_REQUIRED = ("LapCompleted", "LapLastLapTime")


@dataclass
class IbtInfo:
    track: str = ""
    car: str = ""
    tick_rate: int = 0
    notes: list[str] = field(default_factory=list)


def is_ibt(buffer: BinaryIO) -> bool:
    pos = buffer.tell()
    head = buffer.read(144)
    buffer.seek(pos)
    if len(head) < 144:
        return False
    ver, _, tick, _, si_len, si_off, nvars, vh_off, _, buf_len = struct.unpack_from("<10i", head, 0)
    return ver in (1, 2) and 1 <= tick <= 1000 and 0 < nvars < 10_000 and vh_off >= 144 and buf_len > 0 and si_len > 0


def _yaml_value(text: str, key: str) -> Optional[str]:
    m = re.search(rf"^\s*{re.escape(key)}:\s*(.+?)\s*$", text, re.M)
    return m.group(1) if m else None


def ibt_laps(buffer: BinaryIO) -> tuple[pd.DataFrame, IbtInfo]:
    buffer.seek(0)
    raw = buffer.read()
    buffer.seek(0)
    _, _, tick, _, si_len, si_off, nvars, vh_off, _, buf_len = struct.unpack_from("<10i", raw, 0)
    buf_off = struct.unpack_from("<i", raw, 52)[0]
    rows = (len(raw) - buf_off) // buf_len
    variables = {}
    for i in range(nvars):
        o = vh_off + i * 144
        vtype, offset, count = struct.unpack_from("<iii", raw, o)
        name = raw[o + 16:o + 48].split(b"\0")[0].decode("latin-1")
        variables[name] = (_TYPES.get(vtype), offset, count)
    missing = [v for v in _REQUIRED if v not in variables]
    if missing:
        raise ValueError(f"iRacing .ibt file lacks {missing}, so lap times cannot be read from it.")
    block = np.frombuffer(raw, np.uint8, rows * buf_len, buf_off).reshape(rows, buf_len)

    def channel(name: str) -> Optional[np.ndarray]:
        dtype, offset, _ = variables.get(name, (None, 0, 0))
        if dtype is None:
            return None
        size = np.dtype(dtype).itemsize
        return block[:, offset:offset + size].copy().view(dtype).ravel()

    yaml = raw[si_off:si_off + si_len].split(b"\0")[0].decode("latin-1")
    info = IbtInfo(track=_yaml_value(yaml, "TrackDisplayName") or "", car=_yaml_value(yaml, "CarScreenName") or "", tick_rate=tick)
    completed, last = channel("LapCompleted"), channel("LapLastLapTime")
    fuel, on_pit = channel("FuelLevel"), channel("OnPitRoad")
    try:
        kg_per_l = float((_yaml_value(yaml, "DriverCarFuelKgPerLtr") or "").split()[0])
    except (ValueError, IndexError):
        kg_per_l = None

    records = []
    for lap in range(1, int(completed.max()) + 1):
        driving = np.where(completed == lap - 1)[0]     # samples while lap `lap` is being driven
        after = np.where(completed == lap)[0]           # LapLastLapTime for `lap` is valid here
        if driving.size == 0 or after.size == 0:
            continue
        lap_time = float(last[after[-1]])
        if not lap_time > 0:
            continue
        rec = {"lap": lap, "lap_time_s": round(lap_time, 3)}
        if fuel is not None and kg_per_l:
            rec["fuel_kg"] = round(float(fuel[driving[0]]) * kg_per_l, 3)
        if on_pit is not None:
            rec["pit_out"], rec["pit_in"] = bool(on_pit[driving[0]]), bool(on_pit[driving[-1]])
        records.append(rec)
    if not records:
        raise ValueError("iRacing .ibt file holds no completed lap with a recorded time.")
    laps = pd.DataFrame(records)
    info.notes.append(f"Lap times from iRacing's LapLastLapTime at {tick} Hz: {len(laps)} timed lap(s), player's car only.")
    if "fuel_kg" in laps:
        info.notes.append(f"fuel_kg = FuelLevel (litres) x {kg_per_l:g} kg/L from the file's car data, at lap start.")
    else:
        info.notes.append("No fuel column: the file gives no fuel density, and litres are not converted by assumption.")
    sessions = channel("SessionNum")
    if sessions is not None and len(np.unique(sessions)) > 1:
        info.notes.append("The file spans several sessions; laps are numbered across all of them.")
    return laps, info
