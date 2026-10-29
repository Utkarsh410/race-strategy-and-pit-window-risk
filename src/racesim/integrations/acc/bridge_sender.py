"""Send normalised messages to the Mac receiver inside a signed envelope.

Envelope (JSON, UTF-8, one per UDP datagram)::

    {"message_type": "telemetry" | "heartbeat" | "event",
     "schema_version": "1.0", "sequence": 17, "sent_at_utc": "...",
     "sender_id": "windows-acc-bridge", "sender_run_id": "3f2a...",
     "mock": false, "payload": {...}, "mac": "<HMAC-SHA256 hex>"}

The sequence counter is shared by all message types and starts at 1 for each
sender run. Messages larger than the configured maximum are not sent.
"""

from __future__ import annotations

import json
import logging
import socket
import uuid
from typing import Any, Optional

from racesim.integrations.acc import auth
from racesim.integrations.acc.models import MAX_DATAGRAM_BYTES, SCHEMA_VERSION, utc_now

logger = logging.getLogger(__name__)


class BridgeSender:
    def __init__(self, target_host: str, target_port: int, *, secret: Optional[bytes], sender_id: str,
                 mock: bool = False, max_bytes: int = MAX_DATAGRAM_BYTES):
        self.target = (target_host, target_port)
        self.secret = secret
        self.sender_id = sender_id
        self.run_id = uuid.uuid4().hex[:16]
        self.mock = mock
        self.max_bytes = max_bytes
        self.sequence = 0
        self.sent = 0
        self.errors = 0
        self.last_error: Optional[str] = None
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def build(self, message_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.sequence += 1
        env = {"message_type": message_type, "schema_version": SCHEMA_VERSION, "sequence": self.sequence,
               "sent_at_utc": utc_now().isoformat(), "sender_id": self.sender_id, "sender_run_id": self.run_id,
               "mock": self.mock, "payload": payload}
        if self.secret is not None:
            env["mac"] = auth.sign(env, self.secret)
        return env

    def send(self, message_type: str, payload: dict[str, Any]) -> bool:
        """Serialise, sign and send one message. Returns False (and records the error) on failure."""
        env = self.build(message_type, payload)
        data = json.dumps(env, separators=(",", ":"), ensure_ascii=False, default=str).encode("utf-8")
        if len(data) > self.max_bytes:
            self.errors += 1
            self.last_error = f"message {self.sequence} too large ({len(data)} > {self.max_bytes} bytes); not sent"
            logger.warning(self.last_error)
            return False
        try:
            self._sock.sendto(data, self.target)
        except OSError as exc:  # network unreachable, Wi-Fi drop, etc.: never fatal
            self.errors += 1
            self.last_error = f"send failed: {exc}"
            logger.warning("Forwarding error to %s:%d: %s", *self.target, exc)
            return False
        self.sent += 1
        return True

    def close(self) -> None:
        self._sock.close()
