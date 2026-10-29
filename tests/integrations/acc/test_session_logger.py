import json
from datetime import datetime, timezone

from racesim.integrations.acc.session_logger import SessionLogger, export_session_csv, read_session_log

ENV = {"message_type": "telemetry", "sequence": 1, "mock": True, "mac": "x" * 64,
       "payload": {"session_id": "s", "timestamp_utc": "2026-10-29T10:00:00Z", "session": {"track_name": "T"},
                   "vehicle": {"current_lap": 2, "position": 1}}}


def test_creates_unique_files_and_valid_jsonl(tmp_path):
    a = SessionLogger(tmp_path / "logs", metadata={"port": 9100}).__enter__()
    b = SessionLogger(tmp_path / "logs").__enter__()
    assert a.path != b.path and a.path.parent.exists()
    a.log(ENV, received_at_utc=datetime.now(timezone.utc), quality=["out_of_order"])
    a.close(summary={"valid": 1})
    b.close()
    recs, bad = read_session_log(a.path)
    assert bad == 0 and [r["record_type"] for r in recs] == ["session_start", "message", "session_end"]
    assert recs[1]["quality"] == ["out_of_order"]
    for line in a.path.read_text().splitlines():
        json.loads(line)


def test_no_overwrite_and_interrupted_write(tmp_path):
    lg = SessionLogger(tmp_path, flush_every_records=1)
    lg.open()
    lg.log(ENV, received_at_utc=datetime.now(timezone.utc))
    # simulate a crash mid-write: truncated last line, no session_end record
    with open(lg.path, "a", encoding="utf-8") as fh:
        fh.write('{"record_type": "message", "envel')
    recs, bad = read_session_log(lg.path)
    assert bad == 1 and len(recs) == 2
    csv = export_session_csv(lg.path, tmp_path / "out.csv")
    assert "track_name" in csv.read_text().splitlines()[0]
    lg._fh.close()


def test_never_records_secret(tmp_path, monkeypatch):
    monkeypatch.setenv("RACESIM_BRIDGE_SECRET", "super-secret-value-123")
    with SessionLogger(tmp_path) as lg:
        lg.log(ENV, received_at_utc=datetime.now(timezone.utc))
    assert "super-secret-value-123" not in lg.path.read_text()
