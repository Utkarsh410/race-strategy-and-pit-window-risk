"""Windows bridge: ACC broadcasting client -> normalised telemetry -> Mac receiver.

Forwarding policy (kept simple and reliable):

* ``telemetry`` - one snapshot of the target car (focused car unless
  ``car_index`` is set) each time ACC sends a realtime car update for it,
  rate-limited to ``forward_min_interval_s``.
* ``event`` - every BROADCASTING_EVENT (lap completed, penalties, ...).
* ``heartbeat`` - every ``heartbeat_interval_s`` with the bridge's own view of
  its ACC connection, so the Mac can separate bridge health from ACC health.

Everything runs on one thread inside the client's receive loop; every wait is
bounded by the socket timeout.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional

from racesim.integrations.acc.bridge_sender import BridgeSender
from racesim.integrations.acc.client import AccBroadcastClient
from racesim.integrations.acc.config import AccBridgeConfig
from racesim.integrations.acc.models import HeartbeatPayload, utc_now
from racesim.integrations.acc.parser import BroadcastingEvent, NativeMessage, RealtimeCarUpdate, RealtimeUpdate, TrackData
from racesim.integrations.acc.protocol import SessionPhase, SessionType, enum_name
from racesim.integrations.acc.state_builder import AccStateBuilder

logger = logging.getLogger(__name__)


class AccBridge:
    def __init__(self, cfg: AccBridgeConfig, *, connection_password: str, secret: Optional[bytes],
                 client_factory: Callable[..., AccBroadcastClient] = AccBroadcastClient):
        if cfg.target_host is None:
            raise ValueError("acc_bridge.target_host (the Mac's Wi-Fi IPv4 address) must be configured")
        self.cfg = cfg
        self.builder = AccStateBuilder(include_driver_names=cfg.include_driver_names, car_index=cfg.car_index)
        self.sender = BridgeSender(cfg.target_host, cfg.target_port, secret=secret, sender_id=cfg.sender_id, mock=False)
        self.client = client_factory(
            cfg.acc_host, cfg.native_broadcast_port, connection_password=connection_password, display_name=cfg.display_name,
            update_interval_ms=cfg.realtime_update_interval_ms, registration_timeout_s=cfg.registration_timeout_s,
            connection_timeout_s=cfg.connection_timeout_s, retry_interval_s=cfg.retry_interval_s,
            request_track_data=cfg.request_track_data, on_message=self.on_message, on_idle=self.on_idle)
        self.started = time.monotonic()
        self._last_forward = 0.0
        self._last_heartbeat = 0.0
        self._last_status = time.monotonic()
        self._session_label: Optional[tuple] = None
        self.telemetry_forwarded = 0
        self.events_forwarded = 0

    # -- callbacks from the client loop -------------------------------------
    def on_message(self, msg: NativeMessage) -> None:
        notes = self.builder.update(msg)
        if "session_changed" in notes:
            logger.info("Session change detected: new session id %s", self.builder.session_id)
        if "unknown_car" in notes:
            self.client.request_entry_list()
        if isinstance(msg, TrackData):
            logger.info("Track: %s (%d m)", msg.track_name, msg.track_meters)
        elif isinstance(msg, RealtimeUpdate):
            label = (enum_name(SessionType, msg.session_type), enum_name(SessionPhase, msg.session_phase))
            if label != self._session_label:
                logger.info("Session %s / phase %s", *label)
                self._session_label = label
        elif isinstance(msg, RealtimeCarUpdate) and msg.car_index == self.builder.target_car():
            now = time.monotonic()
            if now - self._last_forward >= self.cfg.forward_min_interval_s:
                payload = self.builder.telemetry_payload(self.sender.sequence + 1,
                                                         last_packet_age_s=self.client.status.last_packet_age())
                if payload is not None and self.sender.send("telemetry", payload.model_dump(mode="json")):
                    self.telemetry_forwarded += 1
                    self._last_forward = now
        elif isinstance(msg, BroadcastingEvent):
            if self.sender.send("event", self.builder.event_payload(msg).model_dump(mode="json")):
                self.events_forwarded += 1

    def on_idle(self) -> None:
        now = time.monotonic()
        if now - self._last_heartbeat >= self.cfg.heartbeat_interval_s:
            self.send_heartbeat()
            self._last_heartbeat = now
        if now - self._last_status >= self.cfg.status_interval_s:
            logger.info("Status: %s", self.status_summary())
            self._last_status = now

    def heartbeat_payload(self) -> HeartbeatPayload:
        st = self.client.status
        session = self.builder.session_info()[0]
        return HeartbeatPayload(
            source="acc_udp", timestamp_utc=utc_now(), acc_status=st.state, acc_last_packet_age_s=st.last_packet_age(),
            native_messages_received=st.messages_received, native_messages_invalid=st.messages_invalid,
            messages_forwarded=self.sender.sent, session_id=self.builder.session_id, track_name=session.track_name,
            session_type=session.session_type, last_error=(st.last_error or self.sender.last_error),
            uptime_s=time.monotonic() - self.started)

    def send_heartbeat(self) -> bool:
        return self.sender.send("heartbeat", self.heartbeat_payload().model_dump(mode="json"))

    def status_summary(self) -> dict:
        st = self.client.status
        session = self.builder.session_info()[0]
        age = st.last_packet_age()
        return {"acc": st.state, "since_last_acc_packet_s": None if age is None else round(age, 2),
                "acc_messages": st.messages_received, "invalid": st.messages_invalid, "forwarded": self.sender.sent,
                "send_errors": self.sender.errors, "track": session.track_name, "session": session.session_type,
                "last_error": st.last_error or self.sender.last_error}

    # -- lifecycle ------------------------------------------------------------
    def run(self) -> None:
        logger.info("Forwarding to %s:%d as %s (read-only bridge)", *self.sender.target, self.cfg.sender_id)
        try:
            self.client.run()
        finally:
            self.send_heartbeat()
            self.sender.close()
            logger.info("Bridge stopped. %s", self.status_summary())

    def stop(self) -> None:
        self.client.stop()
