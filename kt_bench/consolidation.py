"""Consolidation: the only place k advances.

Live training steps are the fast trace: they mutate the tensors and
touch nothing else. At a consolidation boundary, the current values
are seated as beliefs -- one interval closes, the next opens -- a
snapshot of the full tensor state is taken, and bench k advances by
exactly one. Steps never advance k; consolidation never does anything
except seat, snapshot, and advance.

Seating mechanics: for each row, the current row hash is seated through
the row's own K-record under the record's authority gate, using the
seating function (currently the generic placeholder policy in
kt_bench.seating) and a per-epoch source ("consolidation:<k>").
The per-epoch source is a small, documented choice: under the gate's
own rules an equal-authority claim from a different source re-seats,
and each consolidation epoch genuinely is a distinct seating event --
so successive consolidations seat instead of reaffirming. ARCHE's
feedback-controller seating (TODO-ARCHE) replaces the placeholder
policy when it arrives; it does not change this mechanism. If a future
seating function ever returns a candidate, consolidation fails loudly
rather than silently dropping a belief.

Gains: consolidation sets each row's gain to its newly seated
authority, so seated trust folds into inference immediately. Scar
dynamics still belong to the later injury playbook.
"""

from .linear import row_hash
from .seating import seat_authority
from .store import RecordStore, SnapshotStore


class Consolidator:
    """Owns bench k, the record store, and the snapshot store."""

    def __init__(self, layers, seating_fn=seat_authority):
        """layers: list of (name, KTLinear), in a fixed order."""
        self.layers = list(layers)
        self.k = 0  # bench knowledge time: counts consolidations only
        self.seating_fn = seating_fn
        self.records = RecordStore()
        for name, layer in self.layers:
            self.records.register_layer(name, layer)
        self.snapshots = SnapshotStore()

    def consolidate(self):
        """Seat every row's current values, snapshot, advance k by one.

        Each seated row's gain is set to its newly seated authority.
        """
        source = f"consolidation:{self.k}"
        for name, layer in self.layers:
            for i, rec in enumerate(layer.records):
                status, row = rec.replace(
                    "row", row_hash(layer.W[i], layer.b[i]),
                    self.seating_fn({}), source)
                if status != "seated":
                    raise ValueError(
                        f"consolidation dropped a belief: layer {name} row {i} "
                        f"returned {status!r} -- seating rule must seat (TODO-ARCHE)")
                layer.gain[i] = row["authority"]
        self.k += 1
        self.snapshots.take(self.k, self.layers)
        return self.k

    def verify_all(self):
        return self.records.verify_all()

    def all_ok(self):
        return self.records.all_ok()
