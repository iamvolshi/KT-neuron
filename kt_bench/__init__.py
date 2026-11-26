"""KTN bench assembly: KTLinear layers, bench transformer, record store,
consolidation loop, and vanilla twin harness. Seating rule is a labeled
stub (TODO-ARCHE) -- see seating.py.
"""

from .consolidation import Consolidator
from .flip import TraceFlip
from .linear import KTLinear, Linear, row_hash
from .seating import seat_authority
from .store import RecordStore, SnapshotStore
from .transformer import GPT, GPTConfig
from .twin import (build_pair, check_forward_parity, check_init_parity,
                   deterministic_batches)

__all__ = [
    "Consolidator",
    "GPT",
    "GPTConfig",
    "KTLinear",
    "Linear",
    "RecordStore",
    "SnapshotStore",
    "build_pair",
    "check_forward_parity",
    "check_init_parity",
    "deterministic_batches",
    "row_hash",
    "seat_authority",
    "TraceFlip",
]
