"""Race-style formatting of lap times for display (data and exports stay in seconds)."""

from __future__ import annotations

import math
import re
from typing import Any, Optional


def format_lap_time(seconds: Optional[float], decimals: int = 3) -> Optional[str]:
    """``110.485`` -> ``"1:50.485"``; an hour or more -> ``"1:02:03.456"``. None / NaN / inf -> None."""
    if seconds is None:
        return None
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    sign = "-" if value < 0 else ""
    total = round(abs(value), decimals)  # round first so 59.9996 becomes 1:00.000, not 0:60.000
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(int(minutes), 60)
    width = 3 + decimals if decimals else 2
    sec_text = f"{secs:0{width}.{decimals}f}"
    return f"{sign}{hours}:{minutes:02d}:{sec_text}" if hours else f"{sign}{minutes}:{sec_text}"


def lap_time_ticks(values, target: int = 6) -> tuple[list[float], list[str]]:
    """Tick positions and m:ss labels for a lap-time axis covering ``values``."""
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not finite:
        return [], []
    lo, hi = min(finite), max(finite)
    span = max(hi - lo, 0.5)
    step = next((s for s in (0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300) if span / s <= target), 600)
    decimals = 0 if step >= 1 else (2 if step == 0.25 else 1)
    first, last = math.floor(lo / step), math.ceil(hi / step)  # one tick either side; Plotly hides those off-axis
    ticks = [round(i * step, 6) for i in range(first, last + 1)]
    return ticks, [format_lap_time(t, decimals) for t in ticks]


_RACE_TIME = re.compile(r"^\s*(?:(\d+):)?(\d{1,2}):(\d{1,2}(?:\.\d+)?)\s*$")


def parse_lap_time(text: Any) -> Optional[float]:
    """Inverse of :func:`format_lap_time`: ``"1:50.485"`` -> 110.485, ``"1:02:03.456"`` -> 3723.456.

    Only the m:ss / h:mm:ss race format is accepted (its unit is unambiguous); anything else returns None.
    """
    m = _RACE_TIME.match(str(text)) if text is not None else None
    if not m:
        return None
    hours, minutes, seconds = int(m.group(1) or 0), int(m.group(2)), float(m.group(3))
    if seconds >= 60 or (m.group(1) and minutes >= 60):
        return None
    return round(hours * 3600 + minutes * 60 + seconds, 6)
