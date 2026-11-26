"""KT neuron layer: one neuron whose weights are beliefs in a K-record.

Neuron ONLY -- no layers, no training loop, no seating rule, no controller.
Derived from the toy (kt_neuron.py) with the same record discipline and
the same demonstrated phenomenon, split into focused modules.
"""

from .kid import make_kid
from .neuron import KTNeuron
from .record import KRecord, q10
from .scar import ScarState, scar_shape

__all__ = ["KTNeuron", "KRecord", "ScarState", "make_kid", "q10", "scar_shape"]
