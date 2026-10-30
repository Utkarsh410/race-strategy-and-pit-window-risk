"""Live strategy analysis: telemetry-aware inputs for the existing optimiser.

Inputs are classified as LIVE MEASUREMENT, DERIVED FROM TELEMETRY, CONFIGURED
ASSUMPTION or UNAVAILABLE. A missing input is never passed as zero and never
filled from the synthetic race configuration.

Readiness has three levels:

* ``BLOCKED`` - an input the model cannot run without is unavailable or stale.
* ``SCENARIO_ONLY`` - the model runs, but on assumptions or in a session that
  cannot support advice. Candidates are shown; no pit lap is recommended.
* ``RECOMMENDATION_ELIGIBLE`` - every tyre and fuel input is measured or derived
  in a live race, on a tyre model the user has declared calibrated.

With the UDP-only bridge, fuel and tyres are not measured, so the result is at
most ``SCENARIO_ONLY``. This module only reports analysis; it sends nothing to
the game. ``racesim.config`` and the optimiser are imported inside the functions
that need them, so the Windows bridge keeps its small dependency set.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from racesim.integrations.acc.race_state import Provenance

if TYPE_CHECKING:
    from racesim.config import SimulatorSetup

MIN_PACE_LAPS = 3
MIN_RECOMMENDATION_LAPS = 5
BLOCKED, SCENARIO_ONLY, ELIGIBLE = "BLOCKED", "SCENARIO_ONLY", "RECOMMENDATION_ELIGIBLE"
_OK = {Provenance.LIVE, Provenance.DERIVED, Provenance.CONFIGURED}


@dataclass(frozen=True)
class LiveAssumptions:
    """Values the user states explicitly. ``None`` means not stated; nothing is defaulted."""

    total_laps: Optional[int] = None
    tyre_compound: Optional[str] = None
    tyre_age_laps: Optional[int] = None
    fuel_kg: Optional[float] = None
    fuel_kg_per_lap: Optional[float] = None
    pit_lane_loss_s: Optional[float] = None
    stationary_stop_time_s: Optional[float] = None
    tyre_model_calibrated: bool = False


@dataclass(frozen=True)
class StrategyInput:
    name: str
    value: Any
    category: Provenance
    source: str = ""
    uncertainty: Optional[str] = None
    required: bool = True


@dataclass
class LiveStrategyInputs:
    inputs: dict[str, StrategyInput]
    status: str
    reasons: list[str]
    session_key: Optional[str] = None

    def value(self, name: str) -> Any:
        return self.inputs[name].value

    def rows(self) -> list[dict]:
        return [{"input": i.name, "value": i.value, "category": i.category.value, "source": i.source,
                 "uncertainty": i.uncertainty or "", "required": i.required} for i in self.inputs.values()]


def _measured_or_assumed(name: str, measured: Any, assumed: Any, measured_source: str, category: Provenance = Provenance.LIVE,
                         required: bool = True, uncertainty: Optional[str] = None) -> StrategyInput:
    if measured is not None:
        return StrategyInput(name, measured, category, measured_source, uncertainty, required)
    if assumed is not None:
        return StrategyInput(name, assumed, Provenance.CONFIGURED, "Stated by the user on the dashboard", None, required)
    return StrategyInput(name, None, Provenance.UNAVAILABLE, "Not supplied by the connection or the user", None, required)


def _remaining_laps(snapshot: dict, a: LiveAssumptions, pace_s: Optional[float]) -> StrategyInput:
    completed = snapshot.get("laps_completed")
    if a.total_laps is not None:
        value = None if completed is None else a.total_laps - completed
        return StrategyInput("remaining_laps", value, Provenance.CONFIGURED if value is not None else Provenance.UNAVAILABLE,
                             "Race distance stated by the user minus laps completed")
    remaining_s = snapshot.get("session_time_remaining_s")
    if remaining_s is None or pace_s is None:
        return StrategyInput("remaining_laps", None, Provenance.UNAVAILABLE, "Needs a stated race distance, or time remaining and observed pace")
    return StrategyInput("remaining_laps", math.ceil(remaining_s / pace_s), Provenance.DERIVED,
                         "Session time remaining / median of recent clean laps, rounded up", "+/- 1 lap")


def build_inputs(snapshot: dict, assumptions: Optional[LiveAssumptions] = None) -> LiveStrategyInputs:
    """Classify every input from a :meth:`LiveAnalysis.snapshot` and assess readiness."""
    a = assumptions or LiveAssumptions()
    pace = snapshot.get("pace") or {}
    rolling = (pace.get("rolling") or {}).get(5) or {}
    median = pace.get("median") or {}
    pace_s, pace_n = rolling.get("value"), rolling.get("n", 0)
    if pace_s is None and (median.get("n") or 0) >= MIN_PACE_LAPS:
        pace_s, pace_n = median.get("value"), median.get("n")
    std = (pace.get("std") or {}).get("value")
    fuel, tyres = snapshot.get("fuel") or {}, snapshot.get("tyres") or {}
    items = [
        StrategyInput("laps_completed", snapshot.get("laps_completed"),
                      Provenance.LIVE if snapshot.get("laps_completed") is not None else Provenance.UNAVAILABLE, "REALTIME_CAR_UPDATE.laps"),
        _remaining_laps(snapshot, a, pace_s),
        StrategyInput("observed_pace_s", pace_s, Provenance.DERIVED if pace_s is not None else Provenance.UNAVAILABLE,
                      f"Clean-lap mean or median, n={pace_n} (needs {MIN_PACE_LAPS})",
                      None if std is None or not pace_n else f"+/- {std / math.sqrt(pace_n):.3f} s standard error"),
        _measured_or_assumed("tyre_compound", tyres.get("compound"), a.tyre_compound, "vehicle.tyre_compound"),
        _measured_or_assumed("tyre_age_laps", tyres.get("age_laps"), a.tyre_age_laps, "vehicle.tyre_age_laps"),
        _measured_or_assumed("fuel_kg", fuel.get("latest_fuel_kg"), a.fuel_kg, "vehicle.fuel_kg"),
        _measured_or_assumed("fuel_kg_per_lap", fuel.get("burn_kg_per_lap"), a.fuel_kg_per_lap,
                             f"Median burn over {fuel.get('burn_laps_used', 0)} reliable laps", Provenance.DERIVED),
        StrategyInput("stops_completed", snapshot.get("stops_completed"), Provenance.DERIVED if snapshot.get("stops_completed") is not None else Provenance.UNAVAILABLE,
                      "Pit exits observed in this session"),
        _measured_or_assumed("track_temperature_c", snapshot.get("track_temperature_c"), None, "REALTIME_UPDATE.trackTemp", required=False),
        _measured_or_assumed("ambient_temperature_c", snapshot.get("ambient_temperature_c"), None, "REALTIME_UPDATE.ambientTemp", required=False),
        _measured_or_assumed("pit_lane_loss_s", None, a.pit_lane_loss_s, "", required=False),
        _measured_or_assumed("stationary_stop_time_s", None, a.stationary_stop_time_s, "", required=False),
    ]
    inputs = {i.name: i for i in items}
    status, reasons = _assess(inputs, snapshot, a, pace.get("laps_clean", 0))
    return LiveStrategyInputs(inputs, status, reasons, snapshot.get("session_key"))


def _assess(inputs: dict[str, StrategyInput], snapshot: dict, a: LiveAssumptions, clean_laps: int) -> tuple[str, list[str]]:
    blocked = [f"Required input unavailable: {i.name} ({i.source})" for i in inputs.values() if i.required and i.category not in _OK]
    if snapshot.get("stale"):
        blocked.insert(0, f"Telemetry is {snapshot.get('data_state')}: strategy analysis needs current data.")
    if not snapshot.get("available"):
        blocked.insert(0, "No telemetry has been received.")
    remaining = inputs["remaining_laps"].value
    if remaining is not None and remaining <= 0:
        blocked.append("The stated race distance has already been reached.")
    if blocked:
        return BLOCKED, blocked
    limits = []
    for name in ("tyre_compound", "tyre_age_laps", "fuel_kg", "fuel_kg_per_lap"):
        if inputs[name].category == Provenance.CONFIGURED:
            limits.append(f"{name} is a configured assumption, not a measurement.")
    if snapshot.get("session_type") != "RACE":
        limits.append(f"Session type is {snapshot.get('session_type')}, not RACE.")
    if snapshot.get("session_phase") != "SESSION":
        limits.append(f"Session phase is {snapshot.get('session_phase')}, not SESSION.")
    if snapshot.get("mock") or snapshot.get("replay"):
        limits.append("Data are mock or replayed, not live ACC telemetry.")
    if clean_laps < MIN_RECOMMENDATION_LAPS:
        limits.append(f"Only {clean_laps} clean laps (a recommendation needs {MIN_RECOMMENDATION_LAPS}).")
    if not a.tyre_model_calibrated:
        limits.append("Tyre model parameters are not declared calibrated for this car and track.")
    return (SCENARIO_ONLY, limits) if limits else (ELIGIBLE, [])


@dataclass
class LiveStrategyResult:
    status: str
    reasons: list[str]
    inputs: list[dict]
    candidates: list[dict] = field(default_factory=list)
    pit_window: Optional[dict] = None
    recommended_pit_lap: Optional[int] = None
    assumptions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    label: str = Provenance.MODEL.value
    laps_completed_at_run: Optional[int] = None
    session_key: Optional[str] = None

    def as_dict(self) -> dict:
        return asdict(self)


def _prepare_setup(setup: "SimulatorSetup", inp: LiveStrategyInputs) -> "SimulatorSetup":
    from racesim.config import with_overrides

    completed = inp.value("laps_completed")
    race: dict[str, Any] = {"total_laps": completed + inp.value("remaining_laps"), "fuel_consumption_kg_per_lap": inp.value("fuel_kg_per_lap"),
                            "fuel_start_kg": inp.value("fuel_kg") + inp.value("fuel_kg_per_lap") * completed}
    for key, name in (("track_temperature_c", "track_temperature_c"), ("ambient_temperature_c", "ambient_temperature_c"),
                      ("pit_lane_loss_s", "pit_lane_loss_s"), ("stationary_stop_time_s", "stationary_stop_time_s")):
        if inp.value(name) is not None:
            race[key] = inp.value(name)
    sections: dict[str, Any] = {"race": race, "baseline_strategy": None, "rivals": []}
    if inp.value("pit_lane_loss_s") is not None:
        sections["pit_stop"] = {"pit_entry_loss_s": None, "pit_transit_loss_s": None, "pit_exit_loss_s": None}
    return with_overrides(setup, **sections)


def _anchor_to_observed_pace(setup: "SimulatorSetup", snapshot, observed_s: float) -> tuple["SimulatorSetup", float]:
    """Shift the model so its next-lap prediction on the current tyres equals the observed pace."""
    from racesim.config import with_baseline_lap_time
    from racesim.schemas import Stint
    from racesim.strategy_optimizer import StrategyEvaluator

    lap = snapshot.current_lap + 1
    stint = Stint(start_lap=lap, end_lap=lap, tyre_compound=snapshot.tyre_compound, starting_tyre_age_laps=snapshot.tyre_age_laps)
    model = float(StrategyEvaluator(setup, snapshot).stint_arrays(stint)["lap_time_s"][0])
    offset = observed_s - model
    return with_baseline_lap_time(setup, setup.race.baseline_lap_time_s + offset), offset


def run_live_strategy(setup: "SimulatorSetup", inp: LiveStrategyInputs) -> LiveStrategyResult:
    """Run the existing deterministic optimiser and pit-window analysis on live inputs.

    Call on demand, not per packet. Monte Carlo is not run here because
    ``run_monte_carlo`` simulates a whole race and cannot start from a mid-race state.
    """
    from racesim.pit_window import calculate_pit_window
    from racesim.schemas import RaceSnapshot
    from racesim.strategy_optimizer import MODEL_ASSUMPTIONS, find_optimal_strategy

    result = LiveStrategyResult(inp.status, list(inp.reasons), inp.rows(), session_key=inp.session_key)
    if inp.status == BLOCKED:
        return result
    compound = str(inp.value("tyre_compound")).strip().upper()
    if compound not in setup.compounds:
        result.status = BLOCKED
        result.reasons.append(f"Tyre compound {compound!r} has no model in the configuration ({', '.join(setup.compounds)}).")
        return result
    completed = inp.value("laps_completed")
    result.laps_completed_at_run = completed
    snapshot = RaceSnapshot(current_lap=completed, tyre_compound=compound, tyre_age_laps=inp.value("tyre_age_laps"),
                            fuel_kg=inp.value("fuel_kg"), stops_completed=inp.value("stops_completed"))
    try:
        live_setup = _prepare_setup(setup, inp)
        live_setup, offset = _anchor_to_observed_pace(live_setup, snapshot, inp.value("observed_pace_s"))
    except ValueError as exc:
        result.status = BLOCKED
        result.reasons.append(f"The inputs do not fit the race configuration: {exc}")
        return result
    opt = find_optimal_strategy(live_setup, snapshot)
    result.assumptions = list(MODEL_ASSUMPTIONS) + [
        f"Model pace shifted by {offset:+.3f} s so its next-lap prediction equals the observed pace.",
        f"Compound models, degradation and pit costs come from: {setup.race.data_label}.",
        "Predicted times cover the remaining laps only.",
        "Monte Carlo risk is not included: it cannot start from a mid-race state.",
    ]
    if opt.best is None:
        result.warnings.append("No feasible strategy under the configured constraints.")
        result.status, result.recommended_pit_lap = (BLOCKED, None)
        result.reasons.append("No feasible strategy under the configured constraints.")
        return result
    columns = ["rank", "stops", "pit_laps", "compounds", "race_time_s", "delta_to_best_s"]
    result.candidates = opt.ranking.head(10)[columns].rename(columns={"race_time_s": "predicted_remaining_time_s"}).to_dict("records")
    if opt.best.number_of_stops > 0:
        window = calculate_pit_window(live_setup, snapshot, n_stops=[opt.best.number_of_stops])
        result.warnings += window.warnings
        result.pit_window = {"earliest_feasible_lap": window.earliest_feasible_lap, "latest_feasible_lap": window.latest_feasible_lap,
                             "lowest_predicted_time_lap": window.recommended_lap, "stops_considered": list(window.stops_considered)}
    if inp.status == ELIGIBLE and result.pit_window is not None:
        result.recommended_pit_lap = result.pit_window["lowest_predicted_time_lap"]
    return result
