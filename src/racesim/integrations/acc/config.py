"""Configuration for the ACC bridge, Mac receiver, health thresholds and logging.

Secrets (ACC connection password, bridge shared secret) are referenced by
environment-variable *name* only; they never appear in YAML files or logs.

Two different ports are involved:

* ``acc_bridge.native_broadcast_port`` - ACC's own broadcasting listener on the
  Windows PC (``updListenerPort`` in ACC's ``broadcasting.json``). Only the
  bridge talks to it, normally on 127.0.0.1.
* ``mac_receiver.port`` / ``acc_bridge.target_port`` - the separate
  RaceSim forwarding port on which the Mac receiver listens.
"""

from __future__ import annotations

import ipaddress
from pathlib import Path
from typing import Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from racesim.integrations.acc.models import MAX_DATAGRAM_BYTES


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _is_ip(v: str) -> bool:
    try:
        ipaddress.ip_address(v)
        return True
    except ValueError:
        return False


class AccBridgeConfig(_Cfg):
    acc_host: str = "127.0.0.1"
    native_broadcast_port: int = Field(default=9000, ge=1, le=65535)
    connection_password_env: str = "ACC_CONNECTION_PASSWORD"
    display_name: str = Field(default="RaceSim bridge", min_length=1, max_length=64)
    realtime_update_interval_ms: int = Field(default=250, ge=50, le=10_000)
    registration_timeout_s: float = Field(default=5.0, gt=0, le=60)
    connection_timeout_s: float = Field(default=10.0, gt=0, le=600)
    retry_interval_s: float = Field(default=3.0, gt=0, le=600)
    request_track_data: bool = True
    target_host: Optional[str] = None
    target_port: int = Field(default=9100, ge=1, le=65535)
    shared_secret_env: str = "RACESIM_BRIDGE_SECRET"
    sender_id: str = Field(default="windows-acc-bridge", min_length=1, max_length=64)
    forward_min_interval_s: float = Field(default=0.25, ge=0, le=60)
    heartbeat_interval_s: float = Field(default=1.0, gt=0, le=60)
    status_interval_s: float = Field(default=10.0, gt=0, le=3600)
    car_index: Optional[int] = Field(default=None, ge=0, le=65535)
    include_driver_names: bool = False

    @field_validator("acc_host", "target_host")
    @classmethod
    def _ip(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not _is_ip(v):
            raise ValueError(f"{v!r} is not an IP address (use the numeric IPv4 address)")
        return v


class MacReceiverConfig(_Cfg):
    host: str = "127.0.0.1"  # loopback unless explicitly configured (e.g. 0.0.0.0 or the Mac's Wi-Fi IP)
    port: int = Field(default=9100, ge=1, le=65535)
    shared_secret_env: str = "RACESIM_BRIDGE_SECRET"
    require_authentication: bool = True
    max_datagram_bytes: int = Field(default=MAX_DATAGRAM_BYTES, ge=512, le=65507)
    allowed_sender_ips: list[str] = Field(default_factory=list)
    history_size: int = Field(default=500, ge=10, le=100_000)

    @field_validator("host")
    @classmethod
    def _host(cls, v: str) -> str:
        if not _is_ip(v):
            raise ValueError(f"{v!r} is not an IP address")
        return v

    @field_validator("allowed_sender_ips")
    @classmethod
    def _ips(cls, v: list[str]) -> list[str]:
        bad = [x for x in v if not _is_ip(x)]
        if bad:
            raise ValueError(f"invalid sender IP(s): {bad}")
        return v


class HealthConfig(_Cfg):
    stale_after_seconds: float = Field(default=3.0, gt=0)
    disconnect_after_seconds: float = Field(default=10.0, gt=0)

    @model_validator(mode="after")
    def _order(self) -> "HealthConfig":
        if self.disconnect_after_seconds <= self.stale_after_seconds:
            raise ValueError("disconnect_after_seconds must be greater than stale_after_seconds")
        return self


class LoggingConfig(_Cfg):
    level: str = "INFO"
    output_directory: str = "outputs/acc_sessions"
    session_logging: bool = True
    flush_every_records: int = Field(default=20, ge=1, le=10_000)
    flush_interval_seconds: float = Field(default=2.0, gt=0, le=600)

    @field_validator("level")
    @classmethod
    def _level(cls, v: str) -> str:
        v = v.upper()
        if v not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"invalid logging level {v!r}")
        return v


class PhysicsConfig(_Cfg):
    """Windows shared-memory physics forwarding (off by default: broadcast-only setups are unchanged)."""

    enabled: bool = False
    sample_hz: float = Field(default=20.0, gt=0, le=200)
    forward_hz: float = Field(default=5.0, gt=0, le=60)
    status_interval_s: float = Field(default=1.0, gt=0, le=60)
    reader_stale_after_s: float = Field(default=2.0, gt=0, le=60)
    channels: Optional[list[str]] = None   # None = every defined channel

    @field_validator("channels")
    @classmethod
    def _known(cls, v: Optional[list[str]]) -> Optional[list[str]]:
        if v is None:
            return v
        from racesim.integrations.acc.physics_channels import BY_NAME

        bad = [c for c in v if c not in BY_NAME]
        if bad:
            raise ValueError(f"unknown physics channel(s): {bad}")
        return v

    @model_validator(mode="after")
    def _rates(self) -> "PhysicsConfig":
        if self.forward_hz > self.sample_hz:
            raise ValueError("physics.forward_hz must not exceed physics.sample_hz")
        return self


class AccIntegrationConfig(_Cfg):
    acc_bridge: AccBridgeConfig = Field(default_factory=AccBridgeConfig)
    mac_receiver: MacReceiverConfig = Field(default_factory=MacReceiverConfig)
    health: HealthConfig = Field(default_factory=HealthConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    physics: PhysicsConfig = Field(default_factory=PhysicsConfig)

    def bridge_port_conflicts(self) -> list[str]:
        """Port conflicts the bridge would hit (forwarding into ACC's own port on the same machine)."""
        b = self.acc_bridge
        out = []
        if b.target_host is not None and b.target_port == b.native_broadcast_port and ipaddress.ip_address(b.target_host).is_loopback:
            out.append("target_port equals ACC's native broadcast port on this machine: forwarded JSON would be sent into ACC")
        return out


def load_acc_config(path: Optional[Union[str, Path]] = None) -> AccIntegrationConfig:
    """Load and validate a YAML file (missing sections take safe defaults)."""
    if path is None:
        return AccIntegrationConfig()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"ACC configuration file not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{p} must contain a mapping")
    for secret_key in ("connection_password", "shared_secret", "password", "secret"):
        for section in data.values():
            if isinstance(section, dict) and secret_key in section:
                raise ValueError(f"{p}: put secrets in environment variables, not in the YAML file ({secret_key!r})")
    return AccIntegrationConfig.model_validate(data)
