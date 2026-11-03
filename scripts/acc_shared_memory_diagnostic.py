"""Print live ACC shared-memory channels on the Windows PC (no network).

Run this BEFORE enabling forwarding, with ACC running and a session loaded:
    python scripts\\acc_shared_memory_diagnostic.py                 # refresh every second
    python scripts\\acc_shared_memory_diagnostic.py --once --json   # one sample as JSON
    python scripts/acc_shared_memory_diagnostic.py --mock --count 3  # MOCK source, works on any OS

For each channel it prints the value, the unit and how far that unit is
verified (kunos / conflict / unstated / library). Compare 'unstated' and
'conflict' channels with the in-game displays (e.g. fuel, tyre pressures and
temperatures) and note the result. Missing channels are listed explicitly.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from racesim.integrations.acc.mock import MockSharedMemorySource  # noqa: E402
from racesim.integrations.acc.physics_channels import BY_NAME, CHANNELS  # noqa: E402
from racesim.integrations.acc.shared_memory import PhysicsReader, PyAccSharedMemorySource  # noqa: E402


def fmt(v) -> str:
    if isinstance(v, list):
        return "[" + ", ".join("-" if x is None else f"{x:.3f}" for x in v) + "]"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mock", action="store_true", help="use the MOCK source (no ACC needed; synthetic values)")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--count", type=int, default=None, help="number of refreshes (default: until Ctrl+C)")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--json", action="store_true", help="print JSON instead of a table")
    args = ap.parse_args(argv)
    reader = PhysicsReader(MockSharedMemorySource if args.mock else PyAccSharedMemorySource)
    n = 1 if args.once else args.count
    i = 0
    try:
        while n is None or i < n:
            i += 1
            reader.poll_once()
            time.sleep(0.05)
            reader.poll_once()  # a second read shows whether the packet id advances
            p = reader.build_payload()
            if args.json:
                print(json.dumps(p.model_dump(mode="json"), indent=2))
            else:
                print(f"\n=== {'MOCK - NOT ACC' if args.mock else 'ACC shared memory'} | reader status {p.reader_status} | "
                      f"packet {p.physics_packet_id} | samples {p.samples_read} | errors {p.read_errors}"
                      + (f" | last error: {p.last_error}" if p.last_error else ""))
                if p.channels:
                    print(f"{'channel':<24} {'value':<38} {'unit':<40} unit status")
                    for ch in CHANNELS:
                        if ch.name in p.channels:
                            print(f"{ch.name:<24} {fmt(p.channels[ch.name])[:38]:<38} {ch.unit[:40]:<40} {ch.unit_status}")
                    print(f"available: {len(p.available)}  missing ({len(p.missing)}): {', '.join(p.missing) or 'none'}")
                    print("Check 'conflict'/'unstated' units against the in-game display: "
                          + ", ".join(c.name for c in CHANNELS if c.unit_status in ('conflict', 'unstated') and c.name in p.channels)[:300])
                else:
                    print("No sample yet. Start ACC and load a session (driving or spectating); the main menu has no data.")
            if n is None or i < n:
                time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        reader._close_source()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
