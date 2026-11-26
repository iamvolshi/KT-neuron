"""Record store and snapshot store for the bench.

The record store is the bench-level registry over the per-row K-records
that live on the KTLinear layers: it maps (layer name, row) to
(KID, KRecord) and can verify every chain at once.

The snapshot store holds the plan's §5 discipline: at each
consolidation a snapshot stores the full tensor state (an ordinary
checkpoint, deep-copied), keyed by bench k, while the records store
per-KID value-hashes of that snapshot. Any past belief re-materializes
exactly: snapshot + interval lookup. Snapshots are derived data -- the
ledger is the truth; snapshots are readings of it.

Both stores are cold beside the tensors. Nothing here runs on the
forward path.
"""

import numpy as np


class RecordStore:
    """Registry over every KT row's K-record in the model."""

    def __init__(self):
        self.entries = []  # (layer_name, row, kid, KRecord)

    def register_layer(self, layer_name, kt_linear):
        for i, (kid, rec) in enumerate(zip(kt_linear.kids, kt_linear.records)):
            self.entries.append((layer_name, i, kid, rec))

    def record_ks(self):
        """Current k per record, in registration order."""
        return [rec.k for *_key, rec in self.entries]

    def row_counts(self):
        """Sealed + open rows per record, in registration order."""
        return [len(rec.rows) for *_key, rec in self.entries]

    def verify_all(self):
        """Verify every chain. Returns a list of (layer, row, result)."""
        out = []
        for layer_name, row, _kid, rec in self.entries:
            out.append((layer_name, row, rec.verify()))
        return out

    def all_ok(self):
        return all(result.startswith("OK") for *_key, result in self.verify_all())


class SnapshotStore:
    """Full tensor checkpoints keyed by bench k."""

    def __init__(self):
        self.snapshots = {}  # bench_k -> {layer_name: (W copy, b copy)}

    def take(self, bench_k, layers):
        """Deep-copy every layer's tensors under bench_k."""
        self.snapshots[bench_k] = {
            name: (layer.W.copy(), layer.b.copy()) for name, layer in layers
        }

    def restore(self, bench_k, layers):
        """Copy a snapshot back into the layers, exactly.

        Raises KeyError for an unknown bench k. Restoration is exact
        because snapshots are copies, not recomputations.
        """
        snap = self.snapshots[bench_k]  # KeyError if never taken
        for name, layer in layers:
            w, b = snap[name]
            layer.W[:] = w
            layer.b[:] = b

    def available(self):
        return sorted(self.snapshots)
