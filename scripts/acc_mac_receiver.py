"""RaceSim telemetry receiver. Run this on the Mac.

Usage:
    export RACESIM_BRIDGE_SECRET='<same shared secret as on the Windows bridge>'
    python scripts/acc_mac_receiver.py --host 0.0.0.0 --port 9100        # listen on all interfaces (LAN)
    python scripts/acc_mac_receiver.py --insecure-no-auth                # loopback mock test (127.0.0.1)

Validates every datagram (size, JSON, schema, HMAC), tracks sequence numbers
and connection health, writes a JSON Lines session log under
outputs/acc_sessions/ and prints a status line every --status-interval
seconds. Stop with Ctrl+C. Only one process can listen on a port: stop this
receiver before starting the dashboard's built-in receiver on the same port.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from racesim.integrations.acc.auth import load_secret  # noqa: E402
from racesim.integrations.acc.config import load_acc_config  # noqa: E402
from racesim.integrations.acc.receiver import ReceiverBindError, TelemetryReceiver  # noqa: E402
from racesim.integrations.acc.live_analysis import LiveAnalysis, write_session_exports  # noqa: E402
from racesim.integrations.acc.session_logger import SessionLogger, export_session_csv  # noqa: E402


def _physics_brief(rx) -> str:
    lp = rx.latest_physics()
    if lp is None:
        return "no samples"
    c = lp[0].get("channels", {})
    return (f"{'MOCK ' if lp[2] else ''}speed={c.get('speed_kmh')} rpm={c.get('rpm')} fuel_raw={c.get('fuel_remaining')} "
            f"tyre_p={c.get('tyre_pressure')} ({len(lp[0].get('available', []))} ch, {lp[1]:.1f}s old)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="YAML file (default: config/acc_bridge.yaml if present)")
    ap.add_argument("--host", default=None, help="bind address (default from config: 127.0.0.1; use 0.0.0.0 or the Wi-Fi IP for LAN)")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--status-interval", type=float, default=2.0)
    ap.add_argument("--duration", type=float, default=None, help="stop automatically after N seconds")
    ap.add_argument("--no-log", action="store_true", help="do not write a session log")
    ap.add_argument("--csv", action="store_true", help="also export the session log to CSV on exit")
    ap.add_argument("--insecure-no-auth", action="store_true", help="accept unsigned messages (loopback testing only)")
    ap.add_argument("--log-level", default=None)
    args = ap.parse_args(argv)
    default_cfg = ROOT / "config" / "acc_bridge.yaml"
    cfg = load_acc_config(args.config or (default_cfg if default_cfg.exists() else None))
    rc = cfg.mac_receiver
    host, port = args.host or rc.host, args.port or rc.port
    logging.basicConfig(level=(args.log_level or cfg.logging.level).upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("acc_mac_receiver")
    try:
        secret = None if args.insecure_no_auth else load_secret(rc.shared_secret_env, required=rc.require_authentication)
    except RuntimeError as exc:
        log.error("%s", exc)
        return 2
    session_logger = None
    if cfg.logging.session_logging and not args.no_log:
        session_logger = SessionLogger(ROOT / cfg.logging.output_directory if not Path(cfg.logging.output_directory).is_absolute()
                                       else cfg.logging.output_directory, flush_every_records=cfg.logging.flush_every_records,
                                       flush_interval_s=cfg.logging.flush_interval_seconds,
                                       metadata={"host": host, "port": port, "authentication": secret is not None})
    rx = TelemetryReceiver(host, port, secret=secret, require_auth=secret is not None, thresholds=cfg.health,
                           session_logger=session_logger, max_bytes=rc.max_datagram_bytes,
                           allowed_sender_ips=rc.allowed_sender_ips, history_size=rc.history_size)
    analysis = LiveAnalysis(thresholds=cfg.health, sink=session_logger)
    rx.add_listener(analysis.on_message)
    try:
        rx.start()
    except ReceiverBindError as exc:
        log.error("%s (is another receiver or the dashboard already using this port?)", exc)
        return 2
    if secret is None:
        log.warning("Authentication DISABLED: accepting unsigned messages. Use only on loopback for testing.")
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    t0 = time.monotonic()
    try:
        while not stop.wait(args.status_interval):
            h = rx.health_snapshot()
            latest = rx.latest_telemetry()
            v = (latest[0].get("vehicle") or {}) if latest else {}
            s = (latest[0].get("session") or {}) if latest else {}
            print(f"[{h.overall.value:<16}] receiver={h.receiver.value} bridge={h.bridge.value} acc={h.acc.value} "
                  f"telemetry={h.telemetry.value} | valid={h.valid_messages} invalid={h.invalid_packets} dup={h.duplicate_packets} "
                  f"ooo={h.out_of_order_packets} gaps={h.sequence_gaps} last_seq={h.last_sequence} | "
                  f"{'MOCK ' if h.mock else ''}track={s.get('track_name')} lap={v.get('current_lap')} pos={v.get('position')} "
                  f"last={v.get('last_lap_time_s')}"
                  + ("" if h.physics.value == "NOT_STARTED" else
                     f" | physics={h.physics.value} reader={h.physics_reader_status} " + _physics_brief(rx)), flush=True)
            if args.duration is not None and time.monotonic() - t0 >= args.duration:
                break
    finally:
        analysis.finish()
        rx.stop()
        if session_logger is not None and session_logger.path is not None:
            log.info("Session log written: %s (%d records)", session_logger.path, session_logger.records_written)
            for kind, path in write_session_exports(analysis, session_logger.path.with_suffix("")).items():
                log.info("Session %s: %s", kind, path)
            if args.csv:
                log.info("CSV export: %s", export_session_csv(session_logger.path, session_logger.path.with_suffix(".csv")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
