"""Send deterministic MOCK telemetry to the RaceSim receiver (no ACC needed).

Usage:
    python scripts/send_mock_acc_telemetry.py --count 20 --interval 0.5 --insecure-no-auth
    RACESIM_BRIDGE_SECRET='...' python scripts/send_mock_acc_telemetry.py --host 127.0.0.1 --port 9100

Every message is flagged mock=true / source="mock" and the track is named
"MOCK TRACK (synthetic)"; the receiver and dashboard label it as mock data.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from racesim.integrations.acc.auth import load_secret  # noqa: E402
from racesim.integrations.acc.bridge_sender import BridgeSender  # noqa: E402
from racesim.integrations.acc.mock import MockTelemetryGenerator  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9100)
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--secret-env", default="RACESIM_BRIDGE_SECRET")
    ap.add_argument("--insecure-no-auth", action="store_true")
    args = ap.parse_args(argv)
    secret = None if args.insecure_no_auth else load_secret(args.secret_env, required=True)
    gen = MockTelemetryGenerator(seed=args.seed, lap_samples=5)
    tx = BridgeSender(args.host, args.port, secret=secret, sender_id="mock-sender", mock=True)
    t0 = time.monotonic()
    try:
        for i in range(args.count):
            tx.send("heartbeat", gen.heartbeat(forwarded=tx.sent, uptime_s=time.monotonic() - t0).model_dump(mode="json"))
            payload = gen.telemetry(tx.sequence + 1)
            tx.send("telemetry", payload.model_dump(mode="json"))
            if "lap_completed" in payload.data_quality:
                tx.send("event", gen.lap_event().model_dump(mode="json"))
            print(f"MOCK sent seq={tx.sequence} lap={payload.vehicle.current_lap} errors={tx.errors}", flush=True)
            time.sleep(args.interval)
    finally:
        tx.close()
    print(f"Done: {tx.sent} MOCK messages sent to {args.host}:{args.port}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
