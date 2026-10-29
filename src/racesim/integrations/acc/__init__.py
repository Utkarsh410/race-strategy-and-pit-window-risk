"""Optional, read-only Assetto Corsa Competizione (ACC) telemetry integration.

Pipeline: ACC UDP broadcasting API (Windows) -> :mod:`client` + :mod:`parser`
-> :mod:`state_builder` -> :mod:`bridge_sender` (signed JSON over UDP) ->
:mod:`receiver` (Mac) -> :mod:`health` / :mod:`session_logger` -> dashboard.

This package depends only on the standard library, pydantic and PyYAML so the
Windows bridge can run without RaceSim's numerical stack. Nothing in the
simulation engine imports it; RaceSim works without ACC installed.
"""

from racesim.integrations.acc.models import SCHEMA_VERSION

__all__ = ["SCHEMA_VERSION"]
