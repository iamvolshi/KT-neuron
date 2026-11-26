"""Flip to the slow trace (plan §6, step 4).

The moment poison is found (firewall engages), the system stops running
on the fast trace and operates from the slow trace -- the record.
Live learning halts; behavior runs from the last clean consolidated
state and its seated beliefs. It does NOT stay on fast under a gain
mask, because poison would keep accumulating in live tensors behind
the mask. Fast resumes only after repair and release.

Definitions, as built:

- Fast trace = the live KTLinear tensors (W, b), mutated per step.
- Slow trace = the record: consolidated snapshots (SnapshotStore)
  plus seated beliefs (per-row K-records) plus the gains they seat.
- "Last clean consolidated state" = the snapshot from the last
  consolidation before the poison interval was seated. Resolution:
  an explicit clean_k wins; else an explicit poison_k resolves to
  the largest snapshot below it; else the default assumes the most
  recent consolidation IS the poison interval, so clean is the
  snapshot before it. Callers whose poison interval is older than
  the latest consolidation must pass clean_k or poison_k explicitly
  -- the default cannot know which older interval was poisoned, and
  guessing silently would be worse than asking.
- Engagement wiring: the caller owns the firewall pot (the bench
  mechanism deliberately does not import the round-1 experiment
  harness). When the pot engages, the caller calls engage() -- or
  engage_if_pot(pot), which flips only while pot.engaged is true.
  The pot object is duck-typed: anything with .engaged / .released
  attributes works, including the experiment FirewallPot.

What engage() does, exactly:

- Restores W/b from the clean snapshot (SnapshotStore.restore --
  exact copies, so weights are bit-identical to the clean state).
- Restores gains (and scars) for that state. Two paths, in order:
  (1) a gain/scar note taken by note_clean(k) after that
  consolidation -- this is the exact path, and the only one that
  restores scars, because SnapshotStore holds tensors only;
  (2) a fallback that derives each row's clean gain from its own
  record: the clean snapshot's row hash is matched against the
  row's sealed intervals, and the matching interval's authority
  becomes the gain. Consolidation seats the current row hash, so
  a consolidated row always has a matching interval. Scar fallback
  is "leave scars as they are" -- scars are all zero in the current
  bench (scar dynamics belong to the later injury playbook), so
  this is exact today and labeled here for when scars move.
- The record is NEVER rolled back: intervals stay sealed, the
  ledger keeps the poison interval in history. Only the fast trace
  is rewound. A later consolidation after release seats new beliefs
  on top of the full history, poison interval included.

Halting, honestly stated:

- Numpy has no hook on tensor mutation, so no module can stop a
  caller from writing layer.W directly. Halting is therefore
  enforced at the step gate: training steps go through
  apply_training_step(fn), which does NOT call fn while flipped
  (it returns False; fn has no side effects at all). Direct tensor
  writes bypass the gate -- that is a caller-discipline limit, not
  a claim otherwise. Tests prove the gate, not direct writes.
- While flipped, consolidation of the fast trace is meaningless
  (there is no new learning to seat) and is not offered here; the
  Consolidator is untouched and still works after release.

Repair and release (the only path back to fast):

- repair(update_fn, allowed_rows): while flipped only. Runs
  update_fn, then restores every non-allowed row bit-exactly, so
  only the named (layer_name, row) rows can move -- targeted
  repair, other rows frozen, whatever update_fn tried to do.
- reseat_repaired(...): seats ONLY the allowed rows through their
  own records under the authority gate (targeted re-consolidation
  from clean sources, with an optional per-row dock), and sets
  those rows' gains to their new standing authority. Other rows
  are untouched -- no record, no gain, no weight.
- release(pot=None): returns True and re-enables the step gate
  only if repair has been performed since engagement AND, when a
  pot is supplied, the pot has bled below its release line
  (pot.released). Otherwise it returns False and stays flipped.
  A release with no pot attests the release line by the caller.
  After release, apply_training_step calls fn again.

Deterministic: no randomness, no wall clock, no I/O in this module.
"""

import numpy as np

from .linear import row_hash


class TraceFlip:
    """Flip-to-slow controller over a Consolidator's layers/snapshots.

    layers: list of (name, KTLinear), in a fixed order -- the same
    list the Consolidator owns. snapshots: its SnapshotStore.
    current_k: optional callable returning the current bench k
    (e.g. ``lambda: consolidator.k``); used only to resolve the
    default clean snapshot when neither clean_k nor poison_k is
    given. Without it, the latest snapshot is taken as the current
    (poison) consolidation.
    """

    def __init__(self, layers, snapshots, current_k=None):
        self.layers = list(layers)
        self._by_name = dict(self.layers)
        self.snapshots = snapshots
        self._current_k = current_k
        self.flipped = False
        self.clean_k = None      # snapshot restored by the last flip
        self.last_clean_k = None  # audit: survives release
        self._repaired = False
        self._notes = {}  # bench_k -> {layer_name: (gain copy, scar copy)}

    # -- clean-state notes -------------------------------------------------

    def note_clean(self, bench_k=None):
        """Record current gains/scars under bench_k (exact restore path).

        Call after a clean consolidation. bench_k defaults to the
        current bench k, else the latest snapshot. Raises KeyError
        for a bench k with no snapshot.
        """
        if bench_k is None:
            bench_k = self._current_bench_k()
        if bench_k not in self.snapshots.snapshots:
            raise KeyError(f"no snapshot at bench k={bench_k}")
        self._notes[bench_k] = {
            name: (layer.gain.copy(), layer.scar.copy())
            for name, layer in self.layers
        }
        return bench_k

    def _current_bench_k(self):
        if self._current_k is not None:
            return self._current_k()
        available = self.snapshots.available()
        if not available:
            raise ValueError("no snapshots exist yet")
        return available[-1]

    # -- engagement --------------------------------------------------------

    def resolve_clean_k(self, clean_k=None, poison_k=None):
        """Work out which snapshot is the last clean state."""
        available = self.snapshots.available()
        if not available:
            raise ValueError("no snapshots exist yet -- nothing clean to flip to")
        if clean_k is not None:
            if clean_k not in self.snapshots.snapshots:
                raise KeyError(f"no snapshot at bench k={clean_k}")
            return clean_k
        if poison_k is not None:
            earlier = [k for k in available if k < poison_k]
            if not earlier:
                raise ValueError(
                    f"no snapshot before poison k={poison_k} -- no clean state")
            return earlier[-1]
        current = self._current_bench_k()
        earlier = [k for k in available if k < current]
        if earlier:
            return earlier[-1]
        if current in self.snapshots.snapshots:
            # Only one snapshot and it is current: there is no earlier
            # clean state to fall back to. Refuse rather than flip to
            # the poison state itself.
            raise ValueError(
                "only one snapshot exists and it is the current one -- "
                "no earlier clean state to flip to")
        return available[-1]

    def engage(self, clean_k=None, poison_k=None):
        """Flip to the slow trace. Returns the clean bench k restored.

        Raises RuntimeError if already flipped (double engagement is
        a caller error; flip once, repair, release, then flip again
        if new poison is found).
        """
        if self.flipped:
            raise RuntimeError("already flipped -- repair and release first")
        target = self.resolve_clean_k(clean_k, poison_k)
        self.snapshots.restore(target, self.layers)
        self._restore_gains_scars(target)
        self.flipped = True
        self.clean_k = target
        self.last_clean_k = target
        self._repaired = False
        return target

    def engage_if_pot(self, pot, clean_k=None, poison_k=None):
        """Flip only while the firewall pot is engaged.

        pot is duck-typed (.engaged). Returns the clean k restored,
        or None when the pot is not engaged (no flip, no error).
        """
        if not getattr(pot, "engaged", False):
            return None
        return self.engage(clean_k, poison_k)

    def _restore_gains_scars(self, bench_k):
        snap = self.snapshots.snapshots[bench_k]
        note = self._notes.get(bench_k)
        for name, layer in self.layers:
            if note is not None:
                gain, scar = note[name]
                layer.gain[:] = gain
                layer.scar[:] = scar
                continue
            # Fallback: clean gain = authority of the sealed interval
            # whose value-hash matches this snapshot's row hash.
            w, _b = snap[name]
            for i, rec in enumerate(layer.records):
                # Row hash from the snapshot pair, not the live bias:
                # restore() already copied the snapshot bias in, but
                # computing from the snapshot keeps this independent
                # of restore order.
                h = row_hash(w[i], snap[name][1][i])
                for row in rec.rows:
                    if row["field"] == "row" and row["value"] == h:
                        layer.gain[i] = row["authority"]
                        break

    # -- the step gate ------------------------------------------------------

    def apply_training_step(self, fn):
        """Run one training step unless flipped.

        Returns True and calls fn() when on fast; returns False and
        never calls fn() when flipped, so weights cannot move.
        """
        if self.flipped:
            return False
        fn()
        return True

    # -- repair and release -------------------------------------------------

    def repair(self, update_fn, allowed_rows):
        """Targeted repair while flipped: only allowed rows may move.

        allowed_rows: set of (layer_name, row). Runs update_fn, then
        restores every other row bit-exactly. Raises RuntimeError
        when not flipped and KeyError for an unknown layer name.
        Returns the sorted list of allowed rows that actually moved.
        """
        if not self.flipped:
            raise RuntimeError("repair runs while flipped -- engage first")
        allowed = set(allowed_rows)
        for name, _row in allowed:
            if name not in self._by_name:
                raise KeyError(f"unknown layer {name!r}")
        before = {n: (l.W.copy(), l.b.copy()) for n, l in self.layers}
        update_fn()
        moved = []
        for name, layer in self.layers:
            rows = {r for (ln, r) in allowed if ln == name}
            for i in range(layer.out_features):
                if i not in rows:
                    layer.W[i] = before[name][0][i]
                    layer.b[i] = before[name][1][i]
                elif not (np.array_equal(layer.W[i], before[name][0][i])
                          and layer.b[i] == before[name][1][i]):
                    moved.append((name, i))
        self._repaired = True
        return sorted(moved)

    def reseat_repaired(self, allowed_rows, seating_fn, source, dock_map=None):
        """Targeted re-consolidation of repaired rows only.

        Seats each allowed row's current values through its own
        record under the authority gate, with an optional per-row
        dock (repaired beliefs re-seat docked, per plan §6 step 7),
        and sets that row's gain to its new standing authority.
        Other rows: no record write, no gain change. Raises
        RuntimeError when not flipped. Returns status counts.
        """
        if not self.flipped:
            raise RuntimeError("reseat runs while flipped -- engage first")
        dock_map = dock_map or {}
        counts = {"seated": 0, "candidate": 0, "reaffirmed": 0}
        for name, row_idx in sorted(set(allowed_rows)):
            layer = self._by_name[name]  # KeyError for unknown layer
            rec = layer.records[row_idx]
            status, _row = rec.replace(
                "row", row_hash(layer.W[row_idx], layer.b[row_idx]),
                seating_fn({}), source,
                authority_dock=dock_map.get((name, row_idx), 0.0))
            counts[status] += 1
            layer.gain[row_idx] = rec.standing("row")["authority"]
        return counts

    def release(self, pot=None):
        """Resume fast-trace learning, only after repair and release.

        Returns True and re-enables the step gate iff repair has
        been performed since engagement and (when a pot is given)
        pot.released is true. Otherwise returns False and stays
        flipped. Releasing when not flipped returns False.
        """
        if not self.flipped:
            return False
        if not self._repaired:
            return False
        if pot is not None and not getattr(pot, "released", False):
            return False
        self.flipped = False
        self.clean_k = None
        self._repaired = False
        return True

    def status(self):
        """Audit snapshot: flipped, clean k, repair done, snapshots."""
        return {
            "flipped": self.flipped,
            "clean_k": self.clean_k,
            "last_clean_k": self.last_clean_k,
            "repaired": self._repaired,
            "snapshots": self.snapshots.available(),
        }
