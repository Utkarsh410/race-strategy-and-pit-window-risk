"""ACC live telemetry view (rendered by ``pages/1_ACC_Live_Telemetry.py``).

* The UDP receiver runs in a background thread owned by a process-wide
  :class:`ReceiverManager` (``st.cache_resource``), so reruns never start a
  second receiver on the same port.
* Only the status fragment refreshes (``st.fragment(run_every=...)``); the
  rest of the dashboard is not re-executed.
* Values are labelled LIVE - MEASURED, DERIVED, UNAVAILABLE, STALE or MOCK.
  Nothing is substituted when no data arrive. Recorded logs are shown only in
  the explicit replay mode and labelled REPLAY.
"""

from __future__ import annotations

import ipaddress
import os
import threading
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from racesim.config import PROJECT_ROOT
from racesim.integrations.acc.auth import load_secret
from racesim.integrations.acc.config import load_acc_config
from racesim.integrations.acc.health import HealthState
from racesim.integrations.acc.live_analysis import LiveAnalysis, write_session_exports
from racesim.integrations.acc.receiver import ReceiverBindError, TelemetryReceiver
from racesim.integrations.acc.session_logger import SessionLogger, flatten_record, read_session_log
from racesim.integrations.acc.state_builder import MOCK, STALE, channel_table

CONFIG_PATH = PROJECT_ROOT / "config" / "acc_bridge.yaml"
STATE_ICON = {HealthState.CONNECTED: ":material/check_circle:", HealthState.WAITING_FOR_DATA: ":material/hourglass_empty:",
              HealthState.STALE: ":material/warning:", HealthState.DISCONNECTED: ":material/cancel:",
              HealthState.NOT_STARTED: ":material/radio_button_unchecked:", HealthState.ERROR: ":material/error:"}


class ReceiverManager:
    """One receiver per server process; thread-safe start/stop."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.receiver: Optional[TelemetryReceiver] = None
        self.analysis: Optional[LiveAnalysis] = None
        self.exports: dict[str, Path] = {}
        self.error: Optional[str] = None

    def start(self, host: str, port: int, *, secret: Optional[bytes], log_dir: Optional[Path], cfg) -> None:
        with self._lock:
            if self.receiver is not None and self.receiver.running:
                return
            logger = None
            if log_dir is not None:
                logger = SessionLogger(log_dir, flush_every_records=cfg.logging.flush_every_records,
                                       flush_interval_s=cfg.logging.flush_interval_seconds,
                                       metadata={"host": host, "port": port, "started_from": "dashboard",
                                                 "authentication": secret is not None})
            rx = TelemetryReceiver(host, port, secret=secret, require_auth=secret is not None, thresholds=cfg.health,
                                   session_logger=logger, max_bytes=cfg.mac_receiver.max_datagram_bytes,
                                   allowed_sender_ips=cfg.mac_receiver.allowed_sender_ips, history_size=cfg.mac_receiver.history_size)
            analysis = LiveAnalysis(thresholds=cfg.health, sink=logger)
            rx.add_listener(analysis.on_message)
            try:
                rx.start()
                self.receiver, self.analysis, self.error, self.exports = rx, analysis, None, {}
            except ReceiverBindError as exc:
                self.error = f"{exc}. Is scripts/acc_mac_receiver.py (or another dashboard) already listening on this port?"

    def stop(self) -> None:
        with self._lock:
            if self.receiver is not None:
                if self.analysis is not None:
                    self.analysis.finish()
                self.receiver.stop()
                log = self.receiver.session_logger
                if self.analysis is not None and log is not None and log.path is not None and not self.exports:
                    self.exports = write_session_exports(self.analysis, log.path.with_suffix(""))


@st.cache_resource
def receiver_manager() -> ReceiverManager:
    return ReceiverManager()


def _config():
    try:
        return load_acc_config(CONFIG_PATH if CONFIG_PATH.exists() else None), None
    except Exception as exc:  # invalid local YAML: fall back to defaults and say so
        return load_acc_config(None), f"config/acc_bridge.yaml is invalid ({exc}); defaults used."


def _fmt(v, unit: str = "") -> str:
    if v is None:
        return "UNAVAILABLE"
    if isinstance(v, float):
        v = f"{v:,.3f}"
    return f"{v} {unit}".strip()


def _receiver_controls(cfg, mgr: ReceiverManager) -> None:
    rx = mgr.receiver
    running = rx is not None and rx.running
    st.sidebar.header("Receiver")
    host = st.sidebar.text_input("Bind address", value=cfg.mac_receiver.host, disabled=running,
                                 help="127.0.0.1 = this Mac only (mock tests). 0.0.0.0 or the Mac's Wi-Fi IP = receive from the Windows bridge.")
    port = st.sidebar.number_input("UDP port", 1, 65535, value=cfg.mac_receiver.port, disabled=running)
    secret_env = cfg.mac_receiver.shared_secret_env
    has_secret = bool(os.environ.get(secret_env))
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        st.sidebar.error("Bind address must be an IP address.")
        return
    insecure = st.sidebar.checkbox("Accept unsigned messages (loopback testing only)", value=not has_secret and loopback,
                                   disabled=running or not loopback,
                                   help=f"Without {secret_env} set, only unsigned loopback testing is possible.")
    log_on = st.sidebar.checkbox("Write session log", value=cfg.logging.session_logging, disabled=running)
    if not has_secret and not insecure:
        st.sidebar.warning(f"Set {secret_env} in the environment before starting Streamlit to authenticate the bridge.")
    c1, c2 = st.sidebar.columns(2)
    if c1.button("Start", disabled=running or (not has_secret and not insecure), type="primary"):
        try:
            secret = None if insecure else load_secret(secret_env, required=True)
        except RuntimeError as exc:
            st.sidebar.error(str(exc))
            return
        out = Path(cfg.logging.output_directory)
        mgr.start(host, int(port), secret=secret, log_dir=(out if out.is_absolute() else PROJECT_ROOT / out) if log_on else None, cfg=cfg)
        st.rerun()
    if c2.button("Stop", disabled=not running):
        mgr.stop()
        st.rerun()
    if mgr.error:
        st.sidebar.error(mgr.error)
    if running:
        st.sidebar.caption(f"Listening on {rx.bound_address[0]}:{rx.bound_address[1]}; authentication "
                           f"{'on' if rx.require_auth else 'OFF (loopback test)'}.")
        if rx.session_logger is not None and rx.session_logger.path is not None:
            st.sidebar.caption(f"Session log: {rx.session_logger.path.relative_to(PROJECT_ROOT) if rx.session_logger.path.is_relative_to(PROJECT_ROOT) else rx.session_logger.path}")


def render_live(mgr: ReceiverManager, refresh_s: float) -> None:
    @st.fragment(run_every=refresh_s)
    def live_panel() -> None:
        rx = mgr.receiver
        if rx is None:
            st.info("Receiver not started. Use **Start** in the sidebar. No data is shown until real messages arrive.")
            return
        h = rx.health_snapshot()
        latest = rx.latest_telemetry()
        mock = bool(rx.latest_is_mock())
        if mock:
            st.error("MOCK DATA - NOT LIVE ACC TELEMETRY. Messages are from scripts/send_mock_acc_telemetry.py.")
        st.subheader(f"{h.overall.value}")
        st.caption("Overall is CONNECTED only when the receiver, the Windows bridge (heartbeats), ACC (as reported by "
                   "the bridge) and the telemetry stream are all fresh.")
        cols = st.columns(4)
        for col, (name, state) in zip(cols, [("Receiver", h.receiver), ("Windows bridge", h.bridge),
                                            ("ACC broadcasting", h.acc), ("Telemetry freshness", h.telemetry)]):
            col.metric(name, state.value, help=None)
            col.markdown(STATE_ICON.get(state, ""))
        c = st.columns(6)
        c[0].metric("Last valid packet", "never" if h.last_valid_age_s is None else f"{h.last_valid_age_s:.1f} s ago")
        c[1].metric("Packets received", h.packets_received)
        c[2].metric("Invalid", h.invalid_packets)
        c[3].metric("Sequence gaps", h.sequence_gaps)
        c[4].metric("Duplicates / out-of-order", f"{h.duplicate_packets} / {h.out_of_order_packets}")
        c[5].metric("Last sequence", "-" if h.last_sequence is None else h.last_sequence)
        hb = rx.latest_heartbeat()
        if hb is not None:
            p, age = hb
            st.caption(f"Bridge reports ACC status **{p['acc_status']}**, {p['native_messages_received']} native ACC messages, "
                       f"{p['messages_forwarded']} forwarded, heartbeat {age:.1f} s ago"
                       + (f"; last error: {p['last_error']}" if p.get("last_error") else "") + ".")
        if h.invalid_reasons:
            st.caption(f"Rejected packets by reason: {h.invalid_reasons}")
        if h.last_error:
            st.error(h.last_error)
        if latest is None:
            st.info("Waiting for telemetry. Nothing is displayed until a valid message arrives.")
            return
        payload, age = latest
        stale = h.telemetry != HealthState.CONNECTED
        if stale and not mock:
            st.warning(f"Telemetry is {h.telemetry.value}: the last message is {age:.1f} s old. Values below are not current.")
        s, v = payload.get("session") or {}, payload.get("vehicle") or {}
        label = MOCK if mock else (STALE if stale else "LIVE - MEASURED")
        st.markdown(f"**Session** ({label})")
        k = st.columns(5)
        k[0].metric("Track", _fmt(s.get("track_name")))
        k[1].metric("Session type", _fmt(s.get("session_type")))
        k[2].metric("Phase", _fmt(s.get("session_phase")))
        k[3].metric("Elapsed", _fmt(s.get("session_time_elapsed_s"), "s"))
        k[4].metric("Remaining", _fmt(s.get("session_time_remaining_s"), "s"),
                    help="Not populated: the meaning of ACC's sessionEndTime is unverified (shown raw in the channel table).")
        st.markdown(f"**Driver and lap** ({label}; current lap and sector are DERIVED)")
        k = st.columns(6)
        k[0].metric("Position", _fmt(v.get("position")))
        k[1].metric("Current lap (derived)", _fmt(v.get("current_lap")))
        k[2].metric("Last lap", _fmt(v.get("last_lap_time_s"), "s"))
        k[3].metric("Best lap", _fmt(v.get("best_lap_time_s"), "s"))
        k[4].metric("Sector (derived)", _fmt(v.get("current_sector")))
        k[5].metric("Car location", _fmt(v.get("car_location")))
        rows = channel_table(payload, mock=mock, stale=stale)
        avail = sum(r["status"] != "UNAVAILABLE" for r in rows)
        st.markdown(f"**Channels** - {avail} of {len(rows)} available. Source: `{payload.get('source')}`, "
                    f"native message `{payload.get('native_message_type')}`, source timestamp {payload.get('timestamp_utc')}, "
                    f"session id `{payload.get('session_id')}`.")
        table = pd.DataFrame(rows)
        table["value"] = table["value"].map(lambda x: "" if x is None else str(x))  # mixed types -> text for Arrow
        st.dataframe(table, hide_index=True)
        if payload.get("data_quality"):
            st.caption("Data-quality flags: " + ", ".join(payload["data_quality"]))
        events = list(rx.events)[-10:]
        if events:
            st.markdown("**Recent events**")
            st.dataframe(pd.DataFrame([{"received_utc": t.isoformat(timespec="seconds"), **{k: e.get(k) for k in
                                        ("event_type", "message", "car_id", "source")}} for e, t in events]), hide_index=True)

    live_panel()


def render_replay() -> None:
    st.warning("REPLAY MODE - showing a recorded session log, not live telemetry.")
    log_dir = PROJECT_ROOT / "outputs" / "acc_sessions"
    files = sorted(log_dir.glob("*.jsonl"), reverse=True) if log_dir.exists() else []
    if not files:
        st.info(f"No session logs found in {log_dir.relative_to(PROJECT_ROOT)}.")
        return
    f = st.selectbox("Session log", files, format_func=lambda p: p.name)
    records, bad = read_session_log(f)
    rows = [r for r in (flatten_record(x) for x in records) if r]
    df = pd.DataFrame(rows)
    mock = bool(df["mock"].any()) if "mock" in df else False
    st.caption(f"{len(rows)} messages; {bad} unreadable line(s) (e.g. cut by an interrupted run)"
               + ("; contains MOCK data" if mock else "") + ".")
    if not df.empty:
        tel = df[df["message_type"] == "telemetry"]
        laps = tel.dropna(subset=["last_lap_time_s"]).drop_duplicates(["laps_completed"])
        if not laps.empty:
            st.line_chart(laps.set_index("laps_completed")["last_lap_time_s"], x_label="Laps completed", y_label="Last lap time (s)")
        st.dataframe(df, hide_index=True)


def render() -> None:
    st.title("ACC Live Telemetry")
    st.caption("Read-only telemetry from Assetto Corsa Competizione via the Windows UDP bridge. Phase 1: connection and "
               "validation only - this page does not feed the strategy engine; lap, stint and strategy analysis are on the "
               "ACC Performance page. Setup: docs/ACC_LIVE_TELEMETRY_SETUP.md.")
    cfg, cfg_err = _config()
    if cfg_err:
        st.warning(cfg_err)
    mode = st.sidebar.radio("Mode", ["Live receiver", "Replay a recorded session log"])
    if mode.startswith("Replay"):
        render_replay()
        return
    mgr = receiver_manager()
    _receiver_controls(cfg, mgr)
    refresh = st.sidebar.slider("Refresh interval (s)", 0.5, 5.0, 1.0, 0.5)
    render_live(mgr, refresh)
