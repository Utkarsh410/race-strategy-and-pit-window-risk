"""Bridge-to-receiver message authentication (HMAC-SHA256).

The shared secret is never transmitted or logged; each envelope carries an
HMAC of its canonical JSON. This authenticates the sender on a **trusted local
network**. It does not encrypt the telemetry and is not a complete security
solution for networks you do not control - never expose the receiver to the
internet.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from typing import Any, Optional


def canonical_bytes(envelope: dict[str, Any]) -> bytes:
    """Deterministic JSON of the envelope without its ``mac`` field."""
    body = {k: v for k, v in envelope.items() if k != "mac"}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode("utf-8")


def sign(envelope: dict[str, Any], secret: bytes) -> str:
    return hmac.new(secret, canonical_bytes(envelope), hashlib.sha256).hexdigest()


def verify(envelope: dict[str, Any], secret: bytes) -> bool:
    mac = envelope.get("mac")
    if not isinstance(mac, str):
        return False
    return hmac.compare_digest(mac, sign(envelope, secret))


def load_secret(env_var: Optional[str], *, required: bool) -> Optional[bytes]:
    """Read a shared secret from an environment variable (never from the YAML file)."""
    value = os.environ.get(env_var) if env_var else None
    if not value:
        if required:
            raise RuntimeError(f"shared secret environment variable {env_var!r} is not set "
                               "(or pass --insecure-no-auth for a local loopback test)")
        return None
    if len(value) < 12:
        raise RuntimeError(f"shared secret in {env_var!r} is too short (use at least 12 characters)")
    return value.encode("utf-8")
