"""Pace statistics from completed laps (standard library only).

Every statistic carries its sample size, and a statistic with too few valid laps
is ``None`` rather than a number from a smaller window. Only clean-lap candidates
enter the statistics; the laps left out are counted by classification.

A lap-time slope is reported as an *observed pace trend*. Fuel burn, traffic,
track evolution, weather and driver variation all move it, and UDP telemetry
cannot separate them, so it is never presented as tyre degradation.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Optional, Sequence

from racesim.integrations.acc.lap_tracker import CompletedLap

ROLLING_WINDOWS = (3, 5, 10)
MIN_STD_LAPS = 2
MIN_SECTOR_TREND_LAPS = 3
MIN_TREND_LAPS = 5
TREND_LABEL = "OBSERVED PACE TREND - not a calibrated tyre-degradation estimate"
TREND_CONFOUNDERS = ("fuel load", "traffic", "track conditions", "driver variation", "weather", "pit activity")


@dataclass(frozen=True)
class Stat:
    value: Optional[float]
    n: int
    needed: int = 1

    @property
    def sufficient(self) -> bool:
        return self.value is not None


def _stat(values: Sequence[float], fn, needed: int = 1) -> Stat:
    return Stat(fn(values) if len(values) >= needed else None, len(values), needed)


def linear_trend(xs: Sequence[float], ys: Sequence[float]) -> Optional[tuple[float, Optional[float]]]:
    """Ordinary least squares ``(slope, standard error)``; None if the slope is undefined."""
    n = len(xs)
    if n < 2:
        return None
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    if n < 3:
        return slope, None
    rss = sum((y - (my + slope * (x - mx))) ** 2 for x, y in zip(xs, ys))
    return slope, math.sqrt(rss / (n - 2) / sxx)


def rolling_pace(clean_times: Sequence[float], window: int) -> Stat:
    return Stat(statistics.fmean(clean_times[-window:]) if len(clean_times) >= window else None, min(len(clean_times), window), window)


def sector_trends(clean: Sequence[CompletedLap]) -> list[dict]:
    rows = []
    for i in range(3):
        pts = [(lap.lap_number, lap.sector_times_s[i]) for lap in clean
               if lap.sector_times_s and i < len(lap.sector_times_s) and lap.sector_times_s[i] is not None]
        ys = [y for _, y in pts]
        fit = linear_trend([x for x, _ in pts], ys) if len(pts) >= MIN_SECTOR_TREND_LAPS else None
        rows.append({"sector": i + 1, "n": len(pts), "best_s": min(ys) if ys else None, "median_s": statistics.median(ys) if ys else None,
                     "latest_s": ys[-1] if ys else None, "trend_s_per_lap": fit[0] if fit else None})
    return rows


def stint_comparison(clean: Sequence[CompletedLap]) -> list[dict]:
    by_stint: dict[Optional[int], list[CompletedLap]] = {}
    for lap in clean:
        by_stint.setdefault(lap.stint_id, []).append(lap)
    rows, previous = [], None
    for stint_id in sorted(by_stint, key=lambda x: (x is None, x)):
        laps = by_stint[stint_id]
        times = [lap.lap_time_s for lap in laps]
        median = statistics.median(times)
        fit = linear_trend([lap.lap_number for lap in laps], times) if len(laps) >= MIN_TREND_LAPS else None
        rows.append({"stint_id": stint_id, "n": len(times), "best_s": min(times), "median_s": median,
                     "std_s": statistics.stdev(times) if len(times) >= MIN_STD_LAPS else None,
                     "median_delta_to_previous_stint_s": None if previous is None else median - previous,
                     "pace_trend_s_per_lap": fit[0] if fit else None, "pace_trend_stderr_s": fit[1] if fit else None,
                     "pace_trend_label": TREND_LABEL if fit else f"insufficient laps (n={len(times)}, need {MIN_TREND_LAPS})"})
        previous = median
    return rows


def analyse_pace(laps: Sequence[CompletedLap]) -> dict:
    """Pace summary as plain data for dashboards, logs and reports."""
    timed = [lap for lap in laps if lap.lap_time_s is not None]
    clean = [lap for lap in laps if lap.is_clean]
    times = [lap.lap_time_s for lap in clean]
    best = min(times) if times else None
    out = {
        "laps_total": len(laps), "laps_clean": len(clean), "classification_counts": dict(Counter(lap.classification.value for lap in laps)),
        "latest_completed": None if not timed else {"lap_number": timed[-1].lap_number, "lap_time_s": timed[-1].lap_time_s,
                                                    "classification": timed[-1].classification.value},
        "best": asdict(_stat(times, min)), "mean": asdict(_stat(times, statistics.fmean)), "median": asdict(_stat(times, statistics.median)),
        "std": asdict(_stat(times, statistics.stdev, MIN_STD_LAPS)),
        "rolling": {w: asdict(rolling_pace(times, w)) for w in ROLLING_WINDOWS},
        "delta_to_best": [{"lap_number": lap.lap_number, "delta_s": lap.lap_time_s - best} for lap in clean],
        "sector_trends": sector_trends(clean), "stints": stint_comparison(clean),
        "trend_note": f"{TREND_LABEL}. Not separated from: {', '.join(TREND_CONFOUNDERS)}.",
    }
    return out
