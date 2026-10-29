"""Read-only ACC broadcasting client (runs on the Windows PC next to ACC).

Lifecycle:

1. Open a UDP socket and send REGISTER_COMMAND_APPLICATION to ACC's
   broadcasting port with the connection password from ``broadcasting.json``
   and an **empty command password** (no control rights are requested).
2. Wait (bounded) for REGISTRATION_RESULT. On failure the ACC error message is
   recorded (e.g. wrong password) and registration is retried after
   ``retry_interval_s``.
3. On success request the entry list and (optionally) track data, then receive
   realtime messages. The entry list is re-requested (throttled) when an
   unknown car appears.
4. If no packet arrives for ``connection_timeout_s`` (game in menus, ACC
   closed), re-register.
5. On stop, send UNREGISTER and close the socket.

Only REGISTER, UNREGISTER, REQUEST_ENTRY_LIST and REQUEST_TRACK_DATA are ever
sent; HUD/camera/replay control commands are never built.

ACC only answers while a session is loaded (driving or spectating); in the
main menu no data is sent.
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from racesim.integrations.acc.parser import NativeMessage, RegistrationResult, parse_native_message
from racesim.integrations.acc.protocol import (
    CONTROL_COMMAND_TYPES, ProtocolError, UnsupportedMessageType, build_register, build_request_entry_list,
    build_request_track_data, build_unregister,
)

logger = logging.getLogger(__name__)
RECV_BUFFER = 65535


@dataclass
class ClientStatus:
    state: str = "NOT_REGISTERED"   # NOT_REGISTERED | REGISTERING | REGISTERED | NO_DATA | AUTH_FAILED | ERROR
    connection_id: Optional[int] = None
    registrations: int = 0
    reconnect_attempts: int = 0
    messages_received: int = 0
    messages_invalid: int = 0
    unsupported_types: dict[int, int] = field(default_factory=dict)
    last_packet_t: Optional[float] = None
    last_error: Optional[str] = None
    started_t: float = field(default_factory=time.monotonic)

    def last_packet_age(self, now: Optional[float] = None) -> Optional[float]:
        now = time.monotonic() if now is None else now
        return None if self.last_packet_t is None else max(0.0, now - self.last_packet_t)


class AccBroadcastClient:
    def __init__(self, acc_host: str, acc_port: int, *, connection_password: str, display_name: str = "RaceSim bridge",
                 update_interval_ms: int = 250, registration_timeout_s: float = 5.0, connection_timeout_s: float = 10.0,
                 retry_interval_s: float = 3.0, request_track_data: bool = True, socket_timeout_s: float = 0.5,
                 on_message: Optional[Callable[[NativeMessage], None]] = None, on_idle: Optional[Callable[[], None]] = None):
        self.target = (acc_host, acc_port)
        self._password = connection_password
        self.display_name = display_name
        self.update_interval_ms = update_interval_ms
        self.registration_timeout_s = registration_timeout_s
        self.connection_timeout_s = connection_timeout_s
        self.retry_interval_s = retry_interval_s
        self.request_track_data = request_track_data
        self.socket_timeout_s = socket_timeout_s
        self.on_message = on_message
        self.on_idle = on_idle
        self.status = ClientStatus()
        self._sock: Optional[socket.socket] = None
        self._stop = threading.Event()
        self._last_entry_request = 0.0
        self.sent_types: list[int] = []

    # -- socket helpers -------------------------------------------------------
    def _open(self) -> None:
        if self._sock is None:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.settimeout(self.socket_timeout_s)

    def _close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def _send(self, data: bytes) -> None:
        if data[0] in CONTROL_COMMAND_TYPES:  # defence in depth: the bridge is read-only
            raise RuntimeError("refusing to send an ACC control command")
        self._open()
        self._sock.sendto(data, self.target)
        self.sent_types.append(data[0])

    def _recv(self) -> Optional[bytes]:
        try:
            data, _ = self._sock.recvfrom(RECV_BUFFER)
            return data
        except socket.timeout:
            return None
        except ConnectionResetError as exc:  # Windows: ICMP port unreachable when ACC is not listening
            self.status.last_error = f"ACC not reachable on {self.target[0]}:{self.target[1]} ({exc.__class__.__name__})"
            return None

    # -- protocol ---------------------------------------------------------------
    def register(self) -> bool:
        """Send REGISTER and wait (bounded) for the result. Returns True on success."""
        self._open()
        self.status.state = "REGISTERING"
        self.status.registrations += 1
        logger.info("Registering with ACC at %s:%d as %r (attempt %d)", *self.target, self.display_name, self.status.registrations)
        try:
            self._send(build_register(self.display_name, self._password, self.update_interval_ms, ""))
        except OSError as exc:
            self.status.state, self.status.last_error = "ERROR", f"register send failed: {exc}"
            logger.warning(self.status.last_error)
            return False
        deadline = time.monotonic() + self.registration_timeout_s
        while time.monotonic() < deadline and not self._stop.is_set():
            data = self._recv()
            if self.on_idle:
                self.on_idle()
            if data is None:
                continue
            msg = self._parse(data)
            if isinstance(msg, RegistrationResult):
                if msg.success:
                    self.status.state, self.status.connection_id, self.status.last_error = "REGISTERED", msg.connection_id, None
                    logger.info("Registered with ACC (connection id %d)", msg.connection_id)
                    self._request_initial_data()
                    return True
                self.status.state = "AUTH_FAILED"
                self.status.last_error = f"ACC rejected registration: {msg.error_message or 'no reason given'}"
                logger.error("%s (check connectionPassword in broadcasting.json)", self.status.last_error)
                return False
            if msg is not None and self.on_message:
                self.on_message(msg)
        if self.status.state == "REGISTERING":
            self.status.state = "NOT_REGISTERED"
            self.status.last_error = self.status.last_error or (
                "no registration result from ACC (is a session loaded, and is updListenerPort correct?)")
            logger.warning(self.status.last_error)
        return False

    def _request_initial_data(self) -> None:
        cid = self.status.connection_id
        self._send(build_request_entry_list(cid))
        self._last_entry_request = time.monotonic()
        if self.request_track_data:
            self._send(build_request_track_data(cid))

    def request_entry_list(self, min_interval_s: float = 2.0) -> None:
        if self.status.connection_id is None or time.monotonic() - self._last_entry_request < min_interval_s:
            return
        self._send(build_request_entry_list(self.status.connection_id))
        self._last_entry_request = time.monotonic()

    def _parse(self, data: bytes) -> Optional[NativeMessage]:
        self.status.messages_received += 1
        self.status.last_packet_t = time.monotonic()
        try:
            res = parse_native_message(data)
        except UnsupportedMessageType as exc:
            self.status.unsupported_types[exc.type_id] = self.status.unsupported_types.get(exc.type_id, 0) + 1
            logger.debug("Ignoring unsupported ACC message type %d", exc.type_id)
            return None
        except ProtocolError as exc:
            self.status.messages_invalid += 1
            logger.warning("Could not decode ACC message (%d bytes): %s", len(data), exc)
            return None
        if res.trailing_bytes:
            logger.debug("%s had %d trailing byte(s)", res.message.native_type, res.trailing_bytes)
        return res.message

    def unregister(self) -> None:
        if self.status.connection_id is not None and self._sock is not None:
            try:
                self._send(build_unregister(self.status.connection_id))
                logger.info("Unregistered from ACC")
            except OSError:
                pass
            self.status.state = "NOT_REGISTERED"
        # Keep AUTH_FAILED / ERROR so the final heartbeat still reports why ACC was not connected.
        self.status.connection_id = None

    # -- main loop --------------------------------------------------------------
    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        """Blocking loop until :meth:`stop`. Every wait is bounded by the socket timeout."""
        self._stop.clear()
        try:
            while not self._stop.is_set():
                if self.status.state != "REGISTERED":
                    if not self.register():
                        self.status.reconnect_attempts += 1
                        self._sleep(self.retry_interval_s)
                        continue
                data = self._recv()
                if self.on_idle:
                    self.on_idle()
                if data is not None:
                    msg = self._parse(data)
                    if msg is not None and self.on_message:
                        self.on_message(msg)
                    continue
                age = self.status.last_packet_age()
                if age is not None and age > self.connection_timeout_s:
                    logger.warning("No ACC data for %.1f s; re-registering", age)
                    self.status.state = "NO_DATA"
                    self.status.reconnect_attempts += 1
                    self.status.last_packet_t = None
        finally:
            self.unregister()
            self._close()

    def _sleep(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while not self._stop.is_set() and time.monotonic() < end:
            if self.on_idle:
                self.on_idle()
            self._stop.wait(min(0.25, max(0.0, end - time.monotonic())))
