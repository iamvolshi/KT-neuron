"""Shared experiment plumbing: gate-honest consolidation, lineage,
firewall, null-run, surgical repair, abstention, transfer.

Gate honesty: phase transitions drive rec.replace directly, NOT
Consolidator.consolidate, because Consolidator raises on candidate
outcomes and a candidate is legitimate gate data (a weaker claim not
seating), not a failure. Statuses are logged per row; gains are set
to standing authority. Chain verification is unchanged.
"""

import time

import numpy as np

from kt_bench.linear import row_hash
from kt_bench.seating import seat_authority

# param-prefix mapping for named_linears -> collect_params prefixes
_LINEAR_SUFFIX = ["attn.qkv", "attn.out", "mlp.fc1", "mlp.fc2"]


def layer_param_prefix(layer_name):
    """'block0.linear2' -> 'block0.mlp.fc1'; 'lm_head' -> 'lm_head'."""
    if layer_name == "lm_head":
        return "lm_head"
    block, linear = layer_name.split(".")
    return f"{block}.{_LINEAR_SUFFIX[int(linear.replace('linear', ''))]}"


class Bench:
    """Harness driver over a KT model: consolidation, snapshots, ledger."""

    def __init__(self, model):
        self.model = model
        self.layers = model.kt_linears()  # [(name, KTLinear)]
        self.k = 0
        self.snapshots = {}       # key -> {layer_name: (W copy, b copy)}
        self.ledger = []          # per-consolidation: {k, source, pool, statuses}
        self.transfer_ledger = []  # per-transfer: {sender_kid, receiver_kid, sources}

    def consolidate(self, pool, source, dock_map=None):
        """Seat every row's current values under the gate.

        pool: declared interval pool (dict for seat_authority).
        dock_map: {(layer_name, row): deficit} (default 0.0).
        Returns (bench_k, status_counts). Gains are set to standing
        authority for every row, seated or not.
        """
        dock_map = dock_map or {}
        claim = seat_authority(pool)
        counts = {"seated": 0, "candidate": 0, "reaffirmed": 0}
        for name, layer in self.layers:
            for i, rec in enumerate(layer.records):
                status, _row = rec.replace(
                    "row", row_hash(layer.W[i], layer.b[i]), claim, source,
                    authority_dock=dock_map.get((name, i), 0.0))
                counts[status] += 1
                layer.gain[i] = rec.standing("row")["authority"]
        self.k += 1
        self.snapshots[self.k] = self._snapshot_state()
        self.ledger.append({"k": self.k, "source": source, "pool": pool,
                            "statuses": dict(counts)})
        return self.k, counts

    def _snapshot_state(self):
        """Full model state (see take_snapshot)."""
        snap = {name: (layer.W.copy(), layer.b.copy())
                for name, layer in self.layers}
        snap["_tok_emb"] = self.model.tok_emb.copy()
        snap["_pos_emb"] = self.model.pos_emb.copy()
        snap["_ln"] = [(b.ln1.w.copy(), b.ln1.b.copy(),
                       b.ln2.w.copy(), b.ln2.b.copy())
                      for b in self.model.blocks]
        snap["_ln_f"] = (self.model.ln_f.w.copy(), self.model.ln_f.b.copy())
        return snap

    def take_snapshot(self, key):
        """Snapshot full model state: KT rows + embeddings + LayerNorms.

        The assembly's SnapshotStore covers KT rows only; the firewall
        demo moves embeddings and norms too, so a restore-exactness
        claim needs the full state. Records are NOT snapshotted: the
        ledger is never rolled back, only the fast trace is.
        """
        self.snapshots[key] = self._snapshot_state()

    def restore_snapshot(self, key):
        snap = self.snapshots[key]
        for name, layer in self.layers:
            w, b = snap[name]
            layer.W[:] = w
            layer.b[:] = b
        self.model.tok_emb[:] = snap["_tok_emb"]
        self.model.pos_emb[:] = snap["_pos_emb"]
        for b, (w1, b1, w2, b2) in zip(self.model.blocks, snap["_ln"]):
            b.ln1.w[:] = w1
            b.ln1.b[:] = b1
            b.ln2.w[:] = w2
            b.ln2.b[:] = b2
        self.model.ln_f.w[:] = snap["_ln_f"][0]
        self.model.ln_f.b[:] = snap["_ln_f"][1]

    def reset_gains_to_standing(self):
        for _name, layer in self.layers:
            for i, rec in enumerate(layer.records):
                layer.gain[i] = rec.standing("row")["authority"]

    def verify_all(self):
        bad = []
        for name, layer in self.layers:
            for i, rec in enumerate(layer.records):
                r = rec.verify()
                if not r.startswith("OK"):
                    bad.append((name, i, r))
        return (len(bad) == 0, bad)

    def standing_authorities(self, layer_name, rows):
        """Seated authority per requested row of one layer."""
        layer = dict(self.layers)[layer_name]
        return {r: layer.records[r].standing("row")["authority"] for r in rows}

    def lineage_query(self, source_name):
        """Ledger entries whose declared pool contains source_name.

        The bench record stores the seating-event source per interval;
        the interval pool lives in this ledger. The join is stated
        wherever lineage results are reported.
        """
        out = []
        for entry in self.ledger:
            sources = (entry["pool"] or {}).get("sources", {})
            if source_name in sources:
                out.append(entry)
        for entry in self.transfer_ledger:
            if source_name in entry["sources"]:
                out.append(entry)
        return out


class FirewallPot:
    """Per-source damage pot with bleed (playbook step 4, harness form).

    Fills 1.0 per tainted batch, threshold 3.0, bleeds 0.5 per clean
    batch, releases below 1.0. While engaged, the lineage's gains are
    held at null by the caller.
    """

    def __init__(self, threshold=3.0, bleed=0.5, release=1.0):
        self.pot = 0.0
        self.threshold, self.bleed, self.release = threshold, bleed, release

    def tainted_batch(self):
        self.pot += 1.0
        return self.engaged

    def clean_batch(self):
        self.pot = max(0.0, self.pot - self.bleed)
        return self.engaged

    @property
    def engaged(self):
        return self.pot >= self.threshold

    @property
    def released(self):
        return self.pot < self.release


def null_rows(model, rows):
    """Null-run: set gains of the given (layer_name, row) rows to 0."""
    layers = dict(model.kt_linears())
    for name, row in rows:
        layers[name].gain[row] = 0.0


def unnull_rows(bench, rows):
    """Restore gains of the given rows to standing authority."""
    layers = dict(bench.layers)
    for name, row in rows:
        layers[name].gain[row] = layers[name].records[row].standing("row")["authority"]


def capture_weights(model):
    """{layer_name: (W copy, b copy)} for every linear layer."""
    return {name: (layer.W.copy(), layer.b.copy())
            for name, layer in model.all_linears()}


def affected_upstream(pre, post, top_n=20):
    """Rows that moved most during an interval, excluding lm_head.

    Returns [(layer_name, row)] ranked by per-row movement
    (||dW_row||_2 + |db|). APPROXIMATE attribution, labeled as such.
    """
    moves = []
    for name in pre:
        if name == "lm_head":
            continue
        w0, b0 = pre[name]
        w1, b1 = post[name]
        dw = np.linalg.norm(w1 - w0, axis=1) + np.abs(b1 - b0)
        for row, m in enumerate(dw):
            moves.append((float(m), name, row))
    moves.sort(reverse=True)
    return [(name, row) for _m, name, row in moves[:top_n]]


def build_repair_mask(model, allowed_rows):
    """Gradient masks aligned with collect_params order.

    allowed_rows: set of (layer_name, row). Only those rows' W/b
    gradients survive; everything else (including embeddings and
    LayerNorms) is frozen at zero.
    """
    from experiments.trainer import collect_params
    names, arrays = collect_params(model)
    allowed = set(allowed_rows)
    prefix_of = {}
    for layer_name, _layer in model.all_linears():
        prefix_of[layer_name] = layer_param_prefix(layer_name)
    masks = []
    for name, arr in zip(names, arrays):
        mask = np.zeros_like(arr)
        for layer_name, prefix in prefix_of.items():
            if name == f"{prefix}.W":
                rows = [r for (ln, r) in allowed if ln == layer_name]
                if rows:
                    mask[rows, :] = 1.0
            elif name == f"{prefix}.b":
                rows = [r for (ln, r) in allowed if ln == layer_name]
                if rows:
                    mask[rows] = 1.0
        masks.append(mask)
    return names, masks


def abstain_predict(model, prefix, quarantined_lm_rows, floor=0.3):
    """Predict with abstention. Returns (answer, confidence, top1).

    answer is DECLINE if the top-1 token's lm_head row is quarantined
    or its gain is under the floor; else the token itself.
    """
    logits = model.forward(prefix)
    last = logits[0, -1]
    e = np.exp(last - last.max())
    probs = e / e.sum()
    top1 = int(np.argmax(probs))
    conf = float(probs[top1])
    gain = float(model.lm_head.gain[top1])
    if top1 in quarantined_lm_rows or gain < floor:
        return ("DECLINE", conf, top1)
    return (top1, conf, top1)


def time_forwards(model, idx, n=200):
    """Mean forward-pass ms over n forwards of the same batch."""
    model.forward(idx)  # warmup
    t0 = time.perf_counter()
    for _ in range(n):
        model.forward(idx)
    t1 = time.perf_counter()
    return (t1 - t0) / n * 1000.0


def copy_nonrow_state(sender, receiver):
    """Copy the trained state that has no KIDs: embeddings + LayerNorms.

    Linear rows are transferred by transfer_rows; without this, a
    receiver's fresh embeddings/norms mismatch the sender's trained
    rows and behavior does not transfer (found in run 1, E5).
    """
    receiver.tok_emb[:] = sender.tok_emb
    receiver.pos_emb[:] = sender.pos_emb
    for rb, sb in zip(receiver.blocks, sender.blocks):
        rb.ln1.w[:] = sb.ln1.w
        rb.ln1.b[:] = sb.ln1.b
        rb.ln2.w[:] = sb.ln2.w
        rb.ln2.b[:] = sb.ln2.b
    receiver.ln_f.w[:] = sender.ln_f.w
    receiver.ln_f.b[:] = sender.ln_f.b


def transfer_rows(sender, receiver, row_names, dock, bench=None,
                  sender_sources=None):
    """Copy rows sender -> receiver (weights) and seat transfer beliefs.

    row_names: [(layer_name, row)] to transfer. For KT receivers the
    transferred belief seats at (sender seated authority - dock) with
    source 'transfer:sender', and the transfer is recorded in the
    receiver bench's transfer ledger (if a bench is given).
    Vanilla receivers (no records) just copy weights.
    Returns the transfer authority used per row.
    """
    s_layers = dict(sender.all_linears())
    r_layers = dict(receiver.all_linears())
    auth_used = {}
    for name, row in row_names:
        sw, sb = s_layers[name].W[row], s_layers[name].b[row]
        r_layers[name].W[row] = sw
        r_layers[name].b[row] = sb
        if hasattr(r_layers[name], "records"):
            rec = r_layers[name].records[row]
            s_auth = None
            if hasattr(s_layers[name], "records"):
                s_auth = s_layers[name].records[row].standing("row")["authority"]
            claim = (s_auth - dock) if s_auth is not None else 0.5
            status, new_row = rec.replace(
                "row", row_hash(r_layers[name].W[row], r_layers[name].b[row]),
                claim, "transfer:sender")
            r_layers[name].gain[row] = rec.standing("row")["authority"]
            auth_used[(name, row)] = (status, rec.standing("row")["authority"])
            if bench is not None and hasattr(s_layers[name], "records"):
                bench.transfer_ledger.append({
                    "sender_kid": s_layers[name].kids[row],
                    "receiver_kid": r_layers[name].kids[row],
                    "sources": list(sender_sources or []),
                    "layer": name, "row": row})
        else:
            auth_used[(name, row)] = ("n/a-vanilla", None)
    return auth_used
