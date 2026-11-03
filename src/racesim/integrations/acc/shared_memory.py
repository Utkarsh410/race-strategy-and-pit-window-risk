"""Windows-side ACC shared-memory reader and physics forwarder.

Reader: :class:`PhysicsReader` polls a :class:`SharedMemorySource` at
``sample_hz`` in its own thread and keeps only the newest sample in memory.
It never touches the network or disk, so forwarding or logging can never
delay a read.

Forwarder: :class:`PhysicsForwarder` runs in a second thread and sends the
newest sample (message type ``physics``) through the existing
:class:`BridgeSender` at ``forward_hz``. When there is no new sample it
sends a channel-less status message every ``status_interval_s`` so the Mac
can tell "ACC not running" from "bridge down".

Source: :class:`PyAccSharedMemorySource` wraps ``pyaccsharedmemory`` 1.0.0
(``accSharedMemory().read_shared_memory()`` / ``close()``), Windows only.
Two library behaviours verified in its source are handled here:

* ``read_shared_memory`` returns None when the new physics frame has the same
  ``suspension_travel`` as the previous one (e.g. a stationary car), so the
  reader clears ``physics_old`` before every read and de-duplicates on ACC's
  physics ``packetId`` instead;
* unknown enum values make the library raise ``ValueError`` for the whole
  read; that is caught and counted as a read error.
"""

from __future__ import annotations

import logging
import math
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Protocol

from racesim.integrations.acc.models import PhysicsPayload, utc_now
from racesim.integrations.acc.physics_channels import AXES, CHANNELS, Channel

logger = logging.getLogger(__name__)

PAGES = {"physics": "Physics", "graphics": "Graphics", "static": "Static"}
WHEEL_ATTRS = ("front_left", "front_right", "rear_left", "rear_right")
STATUS_MAP = {"ACC_LIVE": "LIVE", "ACC_PAUSE": "PAUSED", "ACC_REPLAY": "REPLAY", "ACC_OFF": "WAITING_FOR_ACC"}


class SharedMemoryUnavailable(RuntimeError):
    """The shared-memory source cannot be used on this machine."""


class SharedMemorySource(Protocol):
    source_name: str

    def read(self) -> Any: ...
    def close(self) -> None: ...


class PyAccSharedMemorySource:
    """``pyaccsharedmemory`` adapter (Windows only)."""

    source_name = "acc_shared_memory"

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise SharedMemoryUnavailable("ACC shared memory is only available on Windows (named memory maps)")
        try:
            from pyaccsharedmemory import accSharedMemory  # type: ignore[import-not-found]
        except ImportError as exc:
            raise SharedMemoryUnavailable("pyaccsharedmemory is not installed: pip install pyaccsharedmemory==1.0.0") from exc
        self._asm = accSharedMemory()

    def read(self) -> Any:
        if hasattr(self._asm, "physics_old"):
            self._asm.physics_old = None  # disable the library's suspension-equality de-duplication
        return self._asm.read_shared_memory()

    def close(self) -> None:
        self._asm.close()


def _num(v: Any) -> Optional[float]:
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _value(ch: Channel, raw: Any) -> Any:
    if ch.kind == "wheels":
        vals = [_num(getattr(raw, a, None)) for a in WHEEL_ATTRS]
        return vals if any(v is not None for v in vals) else None
    if ch.kind == "vector":
        vals = [_num(getattr(raw, a, None)) for a in AXES]
        return vals if any(v is not None for v in vals) else None
    if ch.kind == "enum":
        name = raw.name if isinstance(raw, Enum) else getattr(raw, "name", None)
        return str(name if name is not None else raw)[:64]
    if ch.kind == "text":
        text = str(raw).replace("\x00", "").strip() if raw is not None else ""
        return text[:64] or None
    if ch.kind == "bool":
        return bool(raw) if raw is not None else None
    v = _num(raw)
    if v is None:
        return None
    return int(v) if isinstance(raw, int) else v


@dataclass
class ExtractedSample:
    channels: dict[str, Any]
    available: list[str]
    missing: list[str]
    physics_packet_id: Optional[int]
    graphics_packet_id: Optional[int]
    reader_status: str


def extract_channels(snapshot: Any, selected: Optional[set[str]] = None) -> ExtractedSample:
    """Map a pyaccsharedmemory-like snapshot to normalised channels.

    A channel is *missing* when its page or attribute is absent or its value
    is non-numeric / non-finite; missing channels are listed, never filled in.
    """
    channels, available, missing = {}, [], []
    for ch in CHANNELS:
        if selected is not None and ch.name not in selected:
            continue
        page = getattr(snapshot, PAGES[ch.page], None)
        raw = getattr(page, ch.attr, None) if page is not None else None
        value = _value(ch, raw) if raw is not None else None
        if value is None:
            missing.append(ch.name)
        else:
            channels[ch.name] = value
            available.append(ch.name)
    phys = getattr(snapshot, "Physics", None)
    gfx = getattr(snapshot, "Graphics", None)
    status = getattr(gfx, "status", None)
    status_name = status.name if isinstance(status, Enum) else (str(status) if status is not None else None)
    return ExtractedSample(channels, available, missing, getattr(phys, "packed_id", None), getattr(gfx, "packed_id", None),
                           STATUS_MAP.get(status_name or "", "LIVE" if status_name is None else "WAITING_FOR_ACC"))


@dataclass
class ReaderState:
    status: str = "NOT_AVAILABLE"
    samples_read: int = 0
    new_samples: int = 0
    read_errors: int = 0
    last_error: Optional[str] = None
    last_new_sample_t: Optional[float] = None
    sequence: int = 0


class PhysicsReader:
    """Polls a source in a background thread; holds only the newest sample."""

    def __init__(self, source_factory: Callable[[], SharedMemorySource], *, sample_hz: float = 20.0,
                 stale_after_s: float = 2.0, retry_interval_s: float = 5.0, selected: Optional[set[str]] = None):
        self.source_factory = source_factory
        self.period = 1.0 / sample_hz
        self.stale_after_s = stale_after_s
        self.retry_interval_s = retry_interval_s
        self.selected = selected
        self.state = ReaderState()
        self.source_name = "acc_shared_memory"
        self._latest: Optional[tuple[ExtractedSample, float, Any, int]] = None  # sample, monotonic t, utc, seq
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._source: Optional[SharedMemorySource] = None
        self._last_packet: Optional[int] = None

    def start(self) -> "PhysicsReader":
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="acc-physics-reader", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
        self._close_source()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _close_source(self) -> None:
        if self._source is not None:
            try:
                self._source.close()
            except Exception:  # closing a dead map must not raise during shutdown
                pass
            self._source = None

    def _open(self) -> bool:
        try:
            self._source = self.source_factory()
            self.source_name = getattr(self._source, "source_name", "acc_shared_memory")
            self.state.status, self.state.last_error = "WAITING_FOR_ACC", None
            logger.info("Shared memory source opened (%s)", self.source_name)
            return True
        except SharedMemoryUnavailable as exc:
            self.state.status, self.state.last_error = "NOT_AVAILABLE", str(exc)
        except Exception as exc:
            self.state.status, self.state.last_error = "ERROR", f"opening shared memory failed: {exc}"[:300]
        logger.warning("Shared memory unavailable: %s", self.state.last_error)
        return False

    def poll_once(self, now: Optional[float] = None) -> bool:
        """One read. Returns True if a new sample was stored (public for tests and the diagnostic)."""
        now = time.monotonic() if now is None else now
        if self._source is None and not self._open():
            return False
        try:
            snap = self._source.read()
        except Exception as exc:  # library raises ValueError on unknown enum values
            self.state.read_errors += 1
            self.state.last_error = f"read failed: {exc.__class__.__name__}: {exc}"[:300]
            return False
        self.state.samples_read += 1
        if snap is not None:
            sample = extract_channels(snap, self.selected)
            if sample.physics_packet_id is not None and sample.physics_packet_id == self._last_packet:
                snap = None
            else:
                self._last_packet = sample.physics_packet_id
                with self._lock:
                    self.state.sequence += 1
                    self._latest = (sample, now, utc_now(), self.state.sequence)
                self.state.new_samples += 1
                self.state.last_new_sample_t = now
                self.state.status = "MOCK" if self.source_name == "mock" and sample.reader_status == "LIVE" else sample.reader_status
                return True
        age = None if self.state.last_new_sample_t is None else now - self.state.last_new_sample_t
        if age is None or age > self.stale_after_s:
            self.state.status = "WAITING_FOR_ACC"  # packet id not advancing: menu, paused or ACC closed
        return False

    def _run(self) -> None:
        while not self._stop.is_set():
            if self._source is None and not self._open():
                self._stop.wait(self.retry_interval_s)
                continue
            self.poll_once()
            self._stop.wait(self.period)
        self._close_source()

    def latest(self) -> Optional[tuple[ExtractedSample, float, Any, int]]:
        with self._lock:
            return self._latest

    def build_payload(self, *, session_id: Optional[str] = None, include_sample: bool = True,
                      now: Optional[float] = None) -> PhysicsPayload:
        """Physics payload for the newest sample (or a channel-less status message)."""
        now = time.monotonic() if now is None else now
        latest = self.latest() if include_sample else None
        src = "mock" if self.source_name == "mock" else "acc_shared_memory"
        common = dict(source=src, reader_status=self.state.status, session_id=session_id, samples_read=self.state.samples_read,
                      read_errors=self.state.read_errors, last_error=self.state.last_error)
        if latest is None:
            return PhysicsPayload(timestamp_utc=utc_now(), sample_sequence=self.state.sequence, **common)
        sample, t, utc, seq = latest
        return PhysicsPayload(timestamp_utc=utc, sample_sequence=seq, physics_packet_id=sample.physics_packet_id,
                              graphics_packet_id=sample.graphics_packet_id, channels=sample.channels,
                              available=sample.available, missing=sample.missing, sample_age_s=max(0.0, now - t), **common)


class PhysicsForwarder:
    """Second thread: forwards the newest physics sample at ``forward_hz``."""

    def __init__(self, reader: PhysicsReader, sender, *, forward_hz: float = 5.0, status_interval_s: float = 1.0,
                 session_id_fn: Optional[Callable[[], Optional[str]]] = None):
        self.reader, self.sender = reader, sender
        self.period = 1.0 / forward_hz
        self.status_interval_s = status_interval_s
        self.session_id_fn = session_id_fn or (lambda: None)
        self.sent_samples = 0
        self.sent_status = 0
        self._last_seq = 0
        self._last_status_t = float("-inf")  # first reader status is sent immediately
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "PhysicsForwarder":
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="acc-physics-forwarder", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def tick(self, now: Optional[float] = None) -> Optional[str]:
        """Send at most one message; returns 'sample', 'status' or None."""
        now = time.monotonic() if now is None else now
        latest = self.reader.latest()
        if latest is not None and latest[3] != self._last_seq:
            payload = self.reader.build_payload(session_id=self.session_id_fn(), now=now)
            if self.sender.send("physics", payload.model_dump(mode="json")):
                self._last_seq = latest[3]
                self.sent_samples += 1
                self._last_status_t = now
                return "sample"
            return None
        if now - self._last_status_t >= self.status_interval_s:
            payload = self.reader.build_payload(session_id=self.session_id_fn(), include_sample=False, now=now)
            self._last_status_t = now
            if self.sender.send("physics", payload.model_dump(mode="json")):
                self.sent_status += 1
                return "status"
        return None

    def _run(self) -> None:
        while not self._stop.is_set():
            self.tick()
            self._stop.wait(self.period)
