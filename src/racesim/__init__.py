"""RaceSim - Race Strategy and Pit-Window Risk Simulator.

All internal times are seconds, fuel masses kilograms, laps 1-indexed
integers. See :mod:`racesim.schemas` for the full set of conventions.
"""

import logging

__version__ = "0.1.0"

logging.getLogger(__name__).addHandler(logging.NullHandler())
