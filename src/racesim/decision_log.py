"""Structured strategy decision log.

Every recommendation made by the replanning engine is stored as a
:class:`DecisionRecord`. Records hold only values the engine actually
computed: the observed race state, every alternative that was evaluated with
its deterministic and Monte Carlo results, the triggers that caused the
re-evaluation and the factual inputs behind the recommendation. Explanatory
text is generated from those numbers; nothing is free-form or invented.

``simulated_time_s`` is the race clock (own elapsed time at the end of the
observed lap), not wall-clock time; the wall-clock UTC time of the computation
is stored separately.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import html
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from racesim import __version__

ACTIONS = {
    "BOX_THIS_LAP": "Pit at the end of the next lap.",
    "STAY_OUT": "Extend the stint; pit on the recommended lap.",
    "NO_FURTHER_STOP": "Run to the finish on the current tyres.",
    "NO_FEASIBLE_STRATEGY": "No remaining strategy satisfies the constraints.",
}


def configuration_hash(config: dict) -> str:
    """Short, stable SHA-256 of a configuration dictionary."""
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:12]


@dataclass
class AlternativeRecord:
    """One evaluated option (seconds; 'remaining' = from the next lap to the finish)."""

    label: str
    strategy_id: str
    strategy_name: str
    next_pit_lap: Optional[int]
    deterministic_remaining_s: float
    expected_remaining_s: Optional[float] = None
    cvar95_remaining_s: Optional[float] = None
    std_remaining_s: Optional[float] = None
    score_s: Optional[float] = None
    delta_expected_vs_recommended_s: Optional[float] = None
    delta_cvar_vs_recommended_s: Optional[float] = None
    probability_faster_than_recommended: Optional[float] = None
    probability_ahead_of_all_rivals: Optional[float] = None
    deterministic_position: Optional[int] = None
    feasible: bool = True


@dataclass
class DecisionRecord:
    """A single strategy recommendation and the evidence behind it."""

    decision_id: int
    lap: int
    simulated_time_s: float
    race_state: dict[str, Any]
    action: str
    recommended_pit_lap: Optional[int]
    recommended_strategy_id: Optional[str]
    recommended_strategy_name: Optional[str]
    full_race_plan: dict[str, Any]
    replanned: bool
    triggers: list[str]
    alternatives: list[AlternativeRecord]
    expected_difference_vs_current_plan_s: Optional[float]
    risk_comparison: dict[str, Any]
    influencing_factors: list[str]
    warnings: list[str]
    model_version: str
    random_seed: int
    configuration_hash: str
    monte_carlo_simulations: int
    risk_aversion: float
    wall_clock_utc: str = field(default_factory=lambda: _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"))

    def to_row(self) -> dict[str, Any]:
        """Flat representation for CSV export (lists joined with ' | ')."""
        return {
            "decision_id": self.decision_id, "lap": self.lap, "simulated_time_s": self.simulated_time_s,
            "action": self.action, "recommended_pit_lap": self.recommended_pit_lap,
            "recommended_strategy": self.recommended_strategy_name, "recommended_strategy_id": self.recommended_strategy_id,
            "replanned": self.replanned, "triggers": " | ".join(self.triggers),
            "expected_difference_vs_current_plan_s": self.expected_difference_vs_current_plan_s,
            "alternatives_evaluated": len(self.alternatives),
            "risk_comparison": json.dumps(self.risk_comparison, default=str),
            "influencing_factors": " | ".join(self.influencing_factors), "warnings": " | ".join(self.warnings),
            "race_state": json.dumps(self.race_state, default=str), "full_race_plan": json.dumps(self.full_race_plan, default=str),
            "model_version": self.model_version, "random_seed": self.random_seed, "configuration_hash": self.configuration_hash,
            "monte_carlo_simulations": self.monte_carlo_simulations, "risk_aversion": self.risk_aversion,
            "wall_clock_utc": self.wall_clock_utc,
        }


@dataclass
class DecisionLog:
    """Ordered collection of decision records for one race."""

    configuration: dict[str, Any] = field(default_factory=dict)
    records: list[DecisionRecord] = field(default_factory=list)

    def append(self, record: DecisionRecord) -> None:
        self.records.append(record)

    def __len__(self) -> int:
        return len(self.records)

    @property
    def model_version(self) -> str:
        return __version__

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([r.to_row() for r in self.records])

    def alternatives_dataframe(self) -> pd.DataFrame:
        rows = [{"decision_id": r.decision_id, "lap": r.lap, **asdict(a)} for r in self.records for a in r.alternatives]
        return pd.DataFrame(rows)

    def export_csv(self, path: str | Path) -> tuple[Path, Path]:
        """Write ``<path>`` (one row per decision) and ``<stem>_alternatives.csv``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_csv(path, index=False)
        alt = path.with_name(path.stem + "_alternatives.csv")
        self.alternatives_dataframe().to_csv(alt, index=False)
        return path, alt

    def export_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"model_version": __version__, "configuration": self.configuration,
                                    "records": [asdict(r) for r in self.records]}, indent=2, default=str), encoding="utf-8")
        return path

    def to_html(self) -> str:
        """HTML fragment for reports (escaped)."""
        if not self.records:
            return "<p class='muted'>No strategy decisions were recorded.</p>"
        esc = lambda x: html.escape(str(x))  # noqa: E731
        rows = []
        for r in self.records:
            rows.append(
                "<tr>" + "".join(f"<td>{esc(v)}</td>" for v in (
                    r.lap, f"{r.simulated_time_s:,.1f}", r.action, r.recommended_pit_lap or "-", r.recommended_strategy_name or "-",
                    "yes" if r.replanned else "no (carried forward)", "; ".join(r.triggers) or "-",
                    "-" if r.expected_difference_vs_current_plan_s is None else f"{r.expected_difference_vs_current_plan_s:+.2f} s",
                )) + "</tr>"
                + f"<tr><td></td><td colspan='7' class='muted'>{'<br>'.join(esc(f) for f in r.influencing_factors)}"
                + ("".join(f"<br><b>Warning:</b> {esc(w)}" for w in r.warnings)) + "</td></tr>"
            )
        head = "".join(f"<th>{h}</th>" for h in ("Lap", "Race clock (s)", "Action", "Pit lap", "Strategy", "Re-optimised",
                                                  "Triggers", "vs current plan"))
        first = self.records[0]
        meta = (f"<p class='muted'>Model version {esc(first.model_version)}; seed {first.random_seed}; configuration hash "
                f"{esc(first.configuration_hash)}; Monte Carlo runs per option {first.monte_carlo_simulations}; "
                f"risk aversion {first.risk_aversion}.</p>")
        return meta + f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
