"""ACC -> RaceSim telemetry bridge. Run this on the Windows PC that runs ACC.

Usage (PowerShell):
    $env:ACC_CONNECTION_PASSWORD = "<connectionPassword from broadcasting.json>"
    $env:RACESIM_BRIDGE_SECRET  = "<long random shared secret, same on the Mac>"
    python scripts\\acc_windows_bridge.py --config config\\acc_bridge.yaml --target-host <MAC_WIFI_IP>

Options override the YAML file. The bridge is read-only: it registers with
ACC's broadcasting API with an empty command password and never sends HUD,
camera or replay commands. Stop with Ctrl+C.
See docs/ACC_LIVE_TELEMETRY_SETUP.md.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # works without installing RaceSim on the Windows PC

from racesim.integrations.acc.auth import load_secret  # noqa: E402
from racesim.integrations.acc.bridge import AccBridge  # noqa: E402
from racesim.integrations.acc.config import load_acc_config  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="YAML file (default: config/acc_bridge.yaml if present; see config/acc_bridge.example.yaml)")
    ap.add_argument("--acc-port", type=int, default=None, help="ACC updListenerPort (broadcasting.json)")
    ap.add_argument("--target-host", default=None, help="Mac Wi-Fi IPv4 address")
    ap.add_argument("--target-port", type=int, default=None)
    ap.add_argument("--heartbeat-interval", type=float, default=None)
    ap.add_argument("--connection-timeout", type=float, default=None)
    ap.add_argument("--no-track-data-request", action="store_true", help="do not send the (unverified) track-data request")
    ap.add_argument("--log-level", default=None)
    ap.add_argument("--log-file", default=None, help="optional bridge log file")
    ap.add_argument("--insecure-no-auth", action="store_true", help="send unsigned messages (loopback testing only)")
    ap.add_argument("--physics", action="store_true", help="also forward shared-memory physics (pyaccsharedmemory, Windows only)")
    ap.add_argument("--physics-hz", type=float, default=None, help="physics forwarding rate (default 5 Hz)")
    args = ap.parse_args(argv)

    default_cfg = ROOT / "config" / "acc_bridge.yaml"
    cfg = load_acc_config(args.config or (default_cfg if default_cfg.exists() else None))
    upd = {k: v for k, v in {"native_broadcast_port": args.acc_port, "target_host": args.target_host or os.environ.get("RACESIM_MAC_HOST"),
                             "target_port": args.target_port, "heartbeat_interval_s": args.heartbeat_interval,
                             "connection_timeout_s": args.connection_timeout}.items() if v is not None}
    if args.no_track_data_request:
        upd["request_track_data"] = False
    bcfg = cfg.acc_bridge.model_validate({**cfg.acc_bridge.model_dump(), **upd})
    cfg = cfg.model_copy(update={"acc_bridge": bcfg})
    level = (args.log_level or cfg.logging.level).upper()
    handlers = [logging.StreamHandler()] + ([logging.FileHandler(args.log_file, encoding="utf-8")] if args.log_file else [])
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s", handlers=handlers)
    log = logging.getLogger("acc_windows_bridge")
    for problem in cfg.bridge_port_conflicts():
        log.error("Configuration error: %s", problem)
        return 2
    if bcfg.target_host is None:
        log.error("Set the Mac's Wi-Fi IPv4 address with --target-host, RACESIM_MAC_HOST or acc_bridge.target_host")
        return 2
    password = os.environ.get(bcfg.connection_password_env, "")
    if not password:
        log.warning("Environment variable %s is empty: registering with an empty connection password "
                    "(ACC rejects this unless connectionPassword is empty in broadcasting.json)", bcfg.connection_password_env)
    try:
        secret = None if args.insecure_no_auth else load_secret(bcfg.shared_secret_env, required=True)
    except RuntimeError as exc:
        log.error("%s", exc)
        return 2
    if secret is None:
        log.warning("Sending UNSIGNED messages (--insecure-no-auth). Use only for local testing.")
    pcfg = cfg.physics
    if args.physics or args.physics_hz is not None:
        pcfg = pcfg.model_validate({**pcfg.model_dump(), "enabled": True,
                                    **({"forward_hz": args.physics_hz} if args.physics_hz is not None else {})})
    if pcfg.enabled:
        log.info("Physics forwarding ON (%.1f Hz read, %.1f Hz forward). Update the Mac receiver first: older receivers "
                 "reject 'physics' messages (telemetry is unaffected).", pcfg.sample_hz, pcfg.forward_hz)
    bridge = AccBridge(bcfg, connection_password=password, secret=secret, physics=pcfg)
    signal.signal(signal.SIGINT, lambda *_: bridge.stop())
    log.info("ACC broadcasting at %s:%d -> RaceSim receiver at %s:%d", bcfg.acc_host, bcfg.native_broadcast_port,
             bcfg.target_host, bcfg.target_port)
    bridge.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
