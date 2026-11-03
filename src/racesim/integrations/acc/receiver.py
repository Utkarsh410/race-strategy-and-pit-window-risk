"""Mac-side UDP receiver for normalised telemetry from the Windows bridge.

* Runs a background thread with a bounded socket timeout (no busy waiting);
  :meth:`TelemetryReceiver.stop` returns within about one timeout.
* Every datagram is validated (size, UTF-8, JSON, envelope schema, schema
  version, HMAC, payload schema); a bad packet is counted and logged, never fatal.
* Duplicates are dropped; out-of-order packets are counted and kept only if
  they are newer than the latest telemetry (so the "latest" view never goes
  backwards); gaps are counted.
* State is exposed through thread-safe accessors and bounded history.
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
from collections import deque
from datetime import datetime
from typing import Callable, Optional

from pydantic import ValidationError

from racesim.integrations.acc import auth
from racesim.integrations.acc.config import HealthConfig
from racesim.integrations.acc.health import ConnectionHealth, HealthSnapshot
from racesim.integrations.acc.models import MAX_DATAGRAM_BYTES, Envelope, utc_now
from racesim.integrations.acc.session_logger import SessionLogger

logger = logging.getLogger(__name__)


class InvalidMessage(ValueError):
    """A datagram was rejected; ``reason`` is a short machine-friendly category."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


class ReceiverBindError(OSError):
    pass


def validate_datagram(data: bytes, *, secret: Optional[bytes], require_auth: bool,
                      max_bytes: int = MAX_DATAGRAM_BYTES) -> tuple[Envelope, object]:
    """Validate one datagram; returns the envelope and its typed payload or raises :class:`InvalidMessage`."""
    if len(data) > max_bytes:
        raise InvalidMessage("oversized", f"{len(data)} > {max_bytes} bytes")
    try:
        raw = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidMessage("malformed_json", str(exc)[:80]) from None
    if not isinstance(raw, dict):
        raise InvalidMessage("malformed_json", "top level is not an object")
    if raw.get("schema_version") is not None and raw.get("schema_version") not in {"1.0"}:
        raise InvalidMessage("unsupported_schema_version", str(raw.get("schema_version"))[:20])
    if require_auth:
        if secret is None:
            raise InvalidMessage("auth_not_configured")
        if not auth.verify(raw, secret):
            raise InvalidMessage("authentication_failed")
    try:
        env = Envelope.model_validate(raw)
        payload = env.typed_payload()
    except ValidationError as exc:
        first = exc.errors()[0]
        raise InvalidMessage("schema_invalid", f"{'.'.join(map(str, first['loc']))}: {first['msg']}"[:120]) from None
    if env.mock != (getattr(payload, "source", None) == "mock"):
        raise InvalidMessage("schema_invalid", "mock flag does not match payload source")
    return env, payload


class TelemetryReceiver:
    """Background UDP receiver. Use :meth:`start` / :meth:`stop` (or as a context manager)."""

    def __init__(self, host: str = "127.0.0.1", port: int = 9100, *, secret: Optional[bytes] = None,
                 require_auth: bool = True, thresholds: Optional[HealthConfig] = None,
                 session_logger: Optional[SessionLogger] = None, max_bytes: int = MAX_DATAGRAM_BYTES,
                 allowed_sender_ips: Optional[list[str]] = None, history_size: int = 500, socket_timeout_s: float = 0.5):
        if require_auth and secret is None:
            raise ValueError("authentication required but no shared secret supplied")
        self.host, self.port = host, port
        self.secret = secret
        self.require_auth = require_auth
        self.health = ConnectionHealth(thresholds)
        self.session_logger = session_logger
        self.max_bytes = max_bytes
        self.allowed = set(allowed_sender_ips or [])
        self.timeout = socket_timeout_s
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: Optional[tuple[dict, float, datetime, bool]] = None   # payload, monotonic t, utc, mock
        self._latest_heartbeat: Optional[tuple[dict, float]] = None
        self._latest_physics: Optional[tuple[dict, float, bool]] = None
        self._latest_physics_status: Optional[tuple[dict, float]] = None
        self._physics_run: Optional[tuple[str, str]] = None   # sender run the stored physics sample came from
        self.physics_samples_dropped = 0                       # older than the stored sample (same run)
        self.history: deque = deque(maxlen=history_size)
        self.events: deque = deque(maxlen=200)
        self.bound_address: Optional[tuple[str, int]] = None
        self._listeners: list[Callable[..., None]] = []
        self.listener_errors = 0

    def add_listener(self, listener: Callable[..., None]) -> None:
        """Call ``listener(message_type, payload, status, received_t, received_utc, mock)`` on the receiver
        thread for every accepted message. A failing listener is logged and never stops the receiver."""
        self._listeners.append(listener)

    # -- lifecycle --------------------------------------------------------------
    def start(self) -> "TelemetryReceiver":
        if self._thread is not None and self._thread.is_alive():
            return self
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.bind((self.host, self.port))
        except OSError as exc:
            sock.close()
            msg = f"cannot bind {self.host}:{self.port}: {exc}"
            self.health.mark_error(msg)
            raise ReceiverBindError(msg) from exc
        sock.settimeout(self.timeout)
        self._sock = sock
        self.bound_address = sock.getsockname()
        if self.session_logger is not None and self.session_logger.path is None:
            self.session_logger.open()
        self._stop.clear()
        self.health.mark_listening()
        self._thread = threading.Thread(target=self._run, name="acc-telemetry-receiver", daemon=True)
        self._thread.start()
        logger.info("Receiver listening on %s:%d (authentication %s)", *self.bound_address,
                    "required" if self.require_auth else "DISABLED")
        return self

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
        if self._sock is not None:
            self._sock.close()
            self._sock = None
        self.health.mark_stopped()
        if self.session_logger is not None:
            self.session_logger.close(summary=self.health.snapshot().as_dict())

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def __enter__(self) -> "TelemetryReceiver":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                data, addr = self._sock.recvfrom(self.max_bytes + 1)
            except socket.timeout:
                continue
            except OSError as exc:
                if self._stop.is_set():
                    break
                self.health.mark_error(f"socket error: {exc}")
                logger.error("Receiver socket error: %s", exc)
                break
            self.handle_datagram(data, addr)

    # -- processing -------------------------------------------------------------
    def handle_datagram(self, data: bytes, addr: tuple[str, int] = ("127.0.0.1", 0), now: Optional[float] = None) -> Optional[str]:
        """Validate and record one datagram. Returns the classification (``accepted``, ``duplicate``, ...)."""
        now = time.monotonic() if now is None else now
        self.health.record_packet()
        if self.allowed and addr[0] not in self.allowed:
            self.health.record_invalid("sender_not_allowed")
            logger.warning("Rejected packet from unlisted sender %s", addr[0])
            return "rejected"
        try:
            env, payload = validate_datagram(data, secret=self.secret, require_auth=self.require_auth, max_bytes=self.max_bytes)
        except InvalidMessage as exc:
            self.health.record_invalid(exc.reason)
            logger.warning("Rejected packet from %s: %s", addr[0], exc)
            return "rejected"
        status = self.health.check_sequence(env.sender_id, env.sender_run_id, env.sequence)
        if status == "duplicate":
            return "duplicate"
        received_utc = utc_now()
        p = payload.model_dump(mode="json")
        is_physics = env.message_type == "physics"
        self.health.record_valid(env.message_type, now, session_id=p.get("session_id"), mock=env.mock,
                                 acc_status=p.get("acc_status"), acc_packet_age_s=p.get("acc_last_packet_age_s"),
                                 physics_reader_status=p.get("reader_status") if is_physics else None,
                                 physics_has_sample=is_physics and bool(p.get("channels")) and status == "new",
                                 physics_sample_age_s=p.get("sample_age_s") if is_physics else None)
        quality = [] if status == "new" else [status]
        with self._lock:
            if is_physics:
                quality += self._physics_quality(p)
            if env.message_type == "telemetry" and status == "new":
                old_session = (self._latest[0].get("session_id") if self._latest else None)
                self._latest = (p, now, received_utc, env.mock)
                new_session = p.get("session_id")
                if (old_session and new_session != old_session and self._latest_physics is not None
                        and self._latest_physics[0].get("session_id") not in (None, new_session)):
                    self._latest_physics = None  # sample belongs to the previous session
            elif env.message_type == "heartbeat" and status == "new":
                self._latest_heartbeat = (p, now)
            elif env.message_type == "event":
                self.events.append((p, received_utc))
            elif is_physics and status == "new":
                self._latest_physics_status = (p, now)
                run = (env.sender_id, env.sender_run_id)
                if p.get("channels"):
                    held = self._latest_physics
                    if (held is not None and run == self._physics_run
                            and p.get("sample_sequence", 0) < held[0].get("sample_sequence", 0)):
                        self.physics_samples_dropped += 1   # reader sample older than the one held
                        quality.append("physics_sample_older")
                    else:
                        self._latest_physics = (p, now - (p.get("sample_age_s") or 0.0), env.mock)
                        self._physics_run = run
                elif run != self._physics_run and self._latest_physics is not None:
                    self._latest_physics = None  # bridge restarted: the held sample is from its previous run
            self.history.append({"received_at_utc": received_utc, "sequence": env.sequence, "message_type": env.message_type,
                                 "mock": env.mock, "status": status, "payload": p})
        if self.session_logger is not None:
            self.session_logger.log(env.model_dump(mode="json"), received_at_utc=received_utc, quality=quality)
        for listener in self._listeners:
            try:
                listener(env.message_type, p, status, now, received_utc, env.mock)
            except Exception:
                self.listener_errors += 1
                logger.exception("Telemetry listener failed")
        return "accepted" if status == "new" else status

    def _physics_quality(self, p: dict) -> list[str]:
        """Receiver-side quality flags for a physics message (caller holds the lock)."""
        flags = []
        if p.get("invalid"):
            flags.append("physics_invalid_channels")
        tel_session = self._latest[0].get("session_id") if self._latest else None
        if p.get("channels") and tel_session is None:
            flags.append("physics_without_session")
        elif p.get("session_id") and tel_session and p["session_id"] != tel_session:
            flags.append("physics_session_mismatch")
        return flags

    # -- read interface ---------------------------------------------------------
    def latest_telemetry(self) -> Optional[tuple[dict, float]]:
        """``(payload, age_s)`` of the newest valid telemetry, or None."""
        with self._lock:
            if self._latest is None:
                return None
            p, t, _, _ = self._latest
            return p, time.monotonic() - t

    def latest_physics(self) -> Optional[tuple[dict, float, bool]]:
        """``(payload, age_s, mock)`` of the newest physics sample with channels, or None.

        ``age_s`` is measured from when the Windows reader took the sample
        (it includes the sample's age at sending), not from when it arrived."""
        with self._lock:
            if self._latest_physics is None:
                return None
            p, t, mock = self._latest_physics
            return p, time.monotonic() - t, mock

    def latest_physics_status(self) -> Optional[tuple[dict, float]]:
        """Newest physics message of any kind (sample or reader status) and its age."""
        with self._lock:
            if self._latest_physics_status is None:
                return None
            p, t = self._latest_physics_status
            return p, time.monotonic() - t

    def latest_is_mock(self) -> Optional[bool]:
        with self._lock:
            return None if self._latest is None else self._latest[3]

    def latest_heartbeat(self) -> Optional[tuple[dict, float]]:
        with self._lock:
            if self._latest_heartbeat is None:
                return None
            p, t = self._latest_heartbeat
            return p, time.monotonic() - t

    def recent(self, n: int = 50) -> list[dict]:
        with self._lock:
            return list(self.history)[-n:]

    def health_snapshot(self, now: Optional[float] = None) -> HealthSnapshot:
        return self.health.snapshot(now)
