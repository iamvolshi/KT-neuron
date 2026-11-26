"""Linear layers, vanilla and KT.

Vanilla Linear is the control: a standard dense layer, numpy only
(no torch in this environment).

KTLinear wraps the same math, but each output row is a KT neuron:
one KID per row (row weights plus its bias), one K-record per row kept
cold beside the tensors, a gain vector folded in at inference, and a
scar offset carried in the bias. Forward:

    y = (x @ W.T + b - scar) * gain

With gains all 1 and scars all 0 this is exactly the vanilla layer --
multiplication by 1.0 and subtraction of 0.0 are exact in IEEE, so
parity is bitwise, not approximate. That is tested, not assumed.

Per the plan, a row's belief value is the row itself, stored as a hash
plus a pointer to the tensor snapshot: the record's "row" field holds
the SHA-256 of the row weights concatenated with the bias, and the full
row lives in the snapshot store (kt_bench.store). The record never
holds live tensors.

Scope: mechanism only. Gain-from-authority mapping is wired in
consolidation (each seated row's gain becomes its seated authority);
scar dynamics belong to the later injury playbook. The seating policy
itself is the generic placeholder in kt_bench.seating until ARCHE's
feedback-controller seating replaces it.
"""

import hashlib
import itertools

import numpy as np

from kt_neuron.kid import make_kid
from kt_neuron.record import KRecord

_birth_seq = itertools.count()
BIRTH_AUTHORITY = 0.5


def row_hash(w_row, b_value):
    """SHA-256 over one neuron: its row weights plus its bias."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(w_row, dtype=np.float64).tobytes())
    h.update(np.float64(b_value).tobytes())
    return h.hexdigest()


class Linear:
    """A standard dense layer: y = x @ W.T + b."""

    def __init__(self, in_features, out_features, rng):
        self.in_features = in_features
        self.out_features = out_features
        self.W = rng.normal(0.0, 0.02, size=(out_features, in_features))
        self.b = np.zeros(out_features)

    def forward(self, x):
        return x @ self.W.T + self.b

    def params(self):
        return [self.W, self.b]


class KTLinear(Linear):
    """A linear layer whose rows are KT neurons.

    One KID per output row, one K-record per row, born registered: at
    construction each row's record opens a "row" belief holding the hash
    of its initial weights, at birth authority, from source "birth".
    The live tensors are the fast trace; the record is the slow trace
    and is touched only by consolidation (see kt_bench.consolidation).
    """

    def __init__(self, in_features, out_features, rng, name=""):
        super().__init__(in_features, out_features, rng)
        self.name = name
        self.gain = np.ones(out_features)
        self.scar = np.zeros(out_features)
        self.kids = []
        self.records = []
        for i in range(out_features):
            kid = make_kid(birth_tick=0, seq=next(_birth_seq))
            rec = KRecord(kid)
            rec.open_belief("row", row_hash(self.W[i], self.b[i]),
                            BIRTH_AUTHORITY, "birth")
            self.kids.append(kid)
            self.records.append(rec)

    def forward(self, x):
        return (x @ self.W.T + self.b - self.scar) * self.gain

    def row_hashes(self):
        """Current hash per row, in row order."""
        return [row_hash(self.W[i], self.b[i])
                for i in range(self.out_features)]
