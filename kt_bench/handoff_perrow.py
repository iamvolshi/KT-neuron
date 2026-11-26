"""Per-row fast->slow handoff switch + three-number tracking (3-arm v3).

Frozen contract: experiments/preregistration-3arm-v3-2026-10-28.md.
THE ONE CHANGE vs v2: the fast trace is PER ROW, not global.

Three numbers per KT row, NEVER merged:
  F_row = per-row fast trace = F_global * s_row (see below)
  B     = claim authority at the row's last consolidation
          (harness-set, as in v2)
  S     = standing authority, read as layer.gain[row]
          (consolidation sets gains to standing authority by design)
  depth = B - S, COMPUTED at report time from B and S, never stored
  as a merged value. Differences F_row-B, B-S, F_row-S are likewise
  computed from the three numbers when reported, never stored as
  substitutes. F_global (the scheduled driver scar) is reported
  alongside F_row for reference; it is never substituted for it.

Per-row fast trace (frozen rule):
  g_row = ||dW_row||_2 + |db_row| on each arm-3 driver step's batch
          (the row's own gradient magnitude; same movement metric
          v2's affected-row ranking uses, now read per step).
  C_row += g_row every driver step (poison and restore phases only;
          the clean phase never touches this tracker). C is NEVER
          reset across phases.
  s_row = C_row / mean(C over rows in the same layer) when the layer
          mean is > 0, else 1.0. Dimensionless, mean exactly 1.0 over
          a layer whenever any gradient has accumulated; no tuned
          constant anywhere. Per-layer normalization because gradient
          scales differ across layers.
  F_row = F_global * s_row, with F_global = PressureDriver.scar.
  Reduction property: identical gradient histories -> s_row = 1
  everywhere -> v3 reduces EXACTLY to v2's global rule.

Handoff rule (per row, arm 3 only):
  gate: S > 0.5 at crossing time ("strong enough" = above birth
    authority, a genuinely seated belief, not a candidate)
  crossing: upward, of the row's OWN trace:
    prev_F_row < S <= new_F_row, checked after each driver step
    (driver advance + scar application), with new_F_row computed
    from the CURRENT global F and CURRENT s_row (after this step's
    gradient accumulation).
  latch: permanent once crossed — no un-latching, no re-arming.
  No retroactive latch: a consolidation changing S does not latch
  a row whose stored trace already sits above its new S; only an
  upward crossing observed at a later check latches.
  slow governs learning: a latched row's Adam step size is multiplied
    by its CURRENT standing authority S (see lr_scale_list). Gradients
    are NOT scaled: Adam normalizes per-element gradients, so a
    per-row gradient scale cancels out of the update; damping that
    survives Adam must act on the step itself.

This module reads gains and gradients and writes only its own latch
set, baseline record, exposure accumulators, trace state, and event
log. It never writes K-record beliefs.
"""

import numpy as np

AUTHORITY_GATE = 0.5
BIRTH_AUTHORITY = 0.5


class PerRowHandoffTracker:
    """Per-row handoff state for one KT model."""

    def __init__(self, kt_layers):
        """kt_layers: [(name, layer)] from model.kt_linears()."""
        self.layers = list(kt_layers)
        self.layer_of = dict(self.layers)
        self.latched = {name: np.zeros(layer.out_features, dtype=bool)
                        for name, layer in self.layers}
        self.baseline = {name: np.full(layer.out_features, BIRTH_AUTHORITY)
                         for name, layer in self.layers}
        self.exposure = {name: np.zeros(layer.out_features)
                         for name, layer in self.layers}
        self.prev_F_row = {name: np.zeros(layer.out_features)
                           for name, layer in self.layers}
        self.events = []
        self.phase = None
        self.step_in_phase = 0

    def set_phase(self, phase):
        """Call before each driver phase; resets the step counter."""
        self.phase = phase
        self.step_in_phase = 0

    def set_baseline(self, claim):
        """Call after each consolidation: B = claim for every row."""
        for name in self.baseline:
            self.baseline[name][:] = claim

    def share(self, name):
        """Current s_row vector for a layer (mean exactly 1.0 when
        any gradient has accumulated; all 1.0 before that)."""
        C = self.exposure[name]
        mean = float(C.mean()) if C.size else 0.0
        if mean > 0.0:
            return C / mean
        return np.ones_like(C)

    def f_row(self, name, F_global):
        """Current per-row fast trace vector for a layer."""
        return float(F_global) * self.share(name)

    def accumulate(self, grads, prefix_of):
        """Add this step's per-row gradient magnitudes to C.

        grads: dict from loss_and_grads, keyed by parameter name.
        prefix_of: {layer_name: collect_params prefix}
        (playbook.layer_param_prefix). Missing keys are skipped
        (defensive; the real loop always supplies all keys).
        """
        for name, layer in self.layers:
            prefix = prefix_of.get(name)
            if prefix is None:
                continue
            dW = grads.get(f"{prefix}.W")
            db = grads.get(f"{prefix}.b")
            if dW is None or db is None:
                continue
            g_row = np.linalg.norm(dW, axis=1) + np.abs(db)
            self.exposure[name] += g_row

    def check(self, prev_F_global, new_F_global):
        """Per-row upward-crossing check after a driver step.

        Returns the newly latched [(layer_name, row)] list. Rows that
        cross latch for the NEXT step (the crossing step itself trains
        unscaled). Every row's prev_F_row is refreshed to its new
        trace value at every check (including restore steps, when F
        falls), so a stale-high trace is washed out and unlatched
        rows get a fresh crossing window when F rises again. The
        step counter advances once per call.
        """
        newly = []
        for name, layer in self.layers:
            S = layer.gain
            s = self.share(name)
            new_row = float(new_F_global) * s
            prev_row = self.prev_F_row[name]
            cross = ((~self.latched[name]) & (S > AUTHORITY_GATE)
                     & (prev_row < S) & (S <= new_row))
            for row in np.flatnonzero(cross):
                self.latched[name][row] = True
                self.events.append({
                    "phase": self.phase,
                    "step_in_phase": self.step_in_phase,
                    "layer": name, "row": int(row),
                    "F_row_prev": float(prev_row[row]),
                    "F_row_new": float(new_row[row]),
                    "s_row": float(s[row]),
                    "F_global_prev": float(prev_F_global),
                    "F_global_new": float(new_F_global),
                    "S": float(S[row]),
                    "standing_authority": float(S[row]),
                })
                newly.append((name, int(row)))
            self.prev_F_row[name] = new_row
        self.step_in_phase += 1
        return newly

    def latched_total(self):
        return int(sum(int(lat.sum()) for lat in self.latched.values()))

    def total_rows(self):
        return int(sum(layer.out_features for _n, layer in self.layers))

    def mean_scale_latched(self):
        """Mean current step scale over latched rows (None if none)."""
        vals = []
        for name, layer in self.layers:
            lat = self.latched[name]
            if lat.any():
                vals.extend(float(s) for s in layer.gain[lat])
        return float(np.mean(vals)) if vals else None

    def lr_scale_list(self, param_names, prefix_of):
        """Step-size scale list aligned with collect_params order.

        prefix_of: {layer_name: collect_params prefix}
        (playbook.layer_param_prefix). Latched rows step at current S
        (W rows via an (out, 1) broadcast array, b via an (out,)
        array); everything else steps at 1.0. Identical semantics to
        v2's HandoffTracker.lr_scale_list.
        """
        by_prefix = {prefix_of[name]: (name, layer)
                     for name, layer in self.layers}
        out = []
        for pname in param_names:
            matched = False
            for prefix, (lname, layer) in by_prefix.items():
                if pname == f"{prefix}.W":
                    s = np.ones((layer.out_features, 1))
                    lat = self.latched[lname]
                    if lat.any():
                        s[lat, 0] = layer.gain[lat]
                    out.append(s)
                    matched = True
                    break
                if pname == f"{prefix}.b":
                    s = np.ones(layer.out_features)
                    lat = self.latched[lname]
                    if lat.any():
                        s[lat] = layer.gain[lat]
                    out.append(s)
                    matched = True
                    break
            if not matched:
                out.append(1.0)
        return out

    def snapshot(self, F_global, rows):
        """Three-number snapshot for the given [(layer, row)] list.

        Returns per-row F_row/F_global/s_row/B/S/differences plus
        means over the rows. F_row is per row; B/S are per row;
        differences are computed here from the three numbers, never
        stored as merged state.
        """
        per_row = []
        for name, row in rows:
            B = float(self.baseline[name][row])
            S = float(self.layer_of[name].gain[row])
            s = float(self.share(name)[row])
            F_row = float(F_global) * s
            per_row.append({
                "layer": name, "row": int(row),
                "F_row": F_row, "F_global": float(F_global),
                "s_row": s, "B": B, "S": S,
                "F_row_minus_B": F_row - B,
                "B_minus_S": B - S,
                "F_row_minus_S": F_row - S,
            })
        if per_row:
            means = {k: float(np.mean([r[k] for r in per_row]))
                     for k in ("F_row", "F_global", "s_row", "B", "S",
                               "F_row_minus_B", "B_minus_S",
                               "F_row_minus_S")}
        else:
            means = None
        return {"per_row": per_row, "means": means}

    def latch_stats(self):
        """Latch-time distribution over all events + never-latched.

        Per-phase step_in_phase stats (min / quartiles / median /
        max) plus totals. step_in_phase resets per phase, so stats
        are per phase; the overall distinct-step count is the sum of
        per-phase distinct counts, reported as such.
        """
        by_phase = {}
        for e in self.events:
            by_phase.setdefault(e["phase"], []).append(e["step_in_phase"])
        phases = {}
        distinct_total = 0
        for phase, steps in by_phase.items():
            arr = np.array(steps, dtype=float)
            distinct_total += int(len(set(steps)))
            phases[phase] = {
                "count": int(len(steps)),
                "distinct_steps": int(len(set(steps))),
                "min_step": int(arr.min()),
                "q1_step": float(np.percentile(arr, 25)),
                "median_step": float(np.median(arr)),
                "q3_step": float(np.percentile(arr, 75)),
                "max_step": int(arr.max()),
            }
        total = self.total_rows()
        latched = self.latched_total()
        return {
            "total_rows": total,
            "latched_total": latched,
            "never_latched": total - latched,
            "distinct_steps_total": distinct_total,
            "by_phase": phases,
        }

    def latch_step_of(self, layer, row):
        """(phase, step_in_phase) of a row's latch, or None."""
        for e in self.events:
            if e["layer"] == layer and e["row"] == row:
                return (e["phase"], e["step_in_phase"])
        return None
