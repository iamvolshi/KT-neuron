"""Fast->slow handoff switch + three-number tracking (3-arm v2).

Frozen contract: experiments/preregistration-3arm-v2-2026-10-28.md.

Three numbers per KT row, NEVER merged:
  F = global impact scalar (PressureDriver.scar at the current step)
  B = claim authority at the row's last consolidation (harness-set)
  S = standing authority, read as layer.gain[row] (consolidation sets
      gains to standing authority by design)
  depth = B - S, COMPUTED at report time from B and S, never stored
  as a merged value. Differences F-B, B-S, F-S are likewise computed
  from the three numbers when reported, never stored as substitutes.

Handoff rule (per row, arm 3 only):
  gate: S > 0.5 at crossing time ("strong enough" = above birth
    authority, a genuinely seated belief, not a candidate)
  crossing: upward, prev_F < S <= new_F, checked after each driver
    step (driver advance + scar application)
  latch: permanent once crossed — no un-latching, no re-arming
  slow governs learning: a latched row's Adam step size is multiplied
    by its CURRENT standing authority S (see lr_scale_list). Gradients
    are NOT scaled: Adam normalizes per-element gradients, so a
    per-row gradient scale cancels out of the update; damping that
    survives Adam must act on the step itself.

This module reads gains and writes only its own latch set, baseline
record, and event log. It never writes K-record beliefs.
"""

import numpy as np

AUTHORITY_GATE = 0.5
BIRTH_AUTHORITY = 0.5


class HandoffTracker:
    """Per-row handoff state for one KT model."""

    def __init__(self, kt_layers):
        """kt_layers: [(name, layer)] from model.kt_linears()."""
        self.layers = list(kt_layers)
        self.layer_of = dict(self.layers)
        self.latched = {name: np.zeros(layer.out_features, dtype=bool)
                        for name, layer in self.layers}
        self.baseline = {name: np.full(layer.out_features, BIRTH_AUTHORITY)
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

    def check(self, prev_F, new_F):
        """Upward-crossing check after a driver step.

        Returns the newly latched [(layer_name, row)] list. Rows that
        cross latch for the NEXT step (the crossing step itself trains
        unscaled). The step counter advances once per call.
        """
        newly = []
        for name, layer in self.layers:
            S = layer.gain
            cross = ((~self.latched[name]) & (S > AUTHORITY_GATE)
                     & (prev_F < S) & (S <= new_F))
            for row in np.flatnonzero(cross):
                self.latched[name][row] = True
                self.events.append({
                    "phase": self.phase,
                    "step_in_phase": self.step_in_phase,
                    "layer": name, "row": int(row),
                    "F_prev": float(prev_F), "F_new": float(new_F),
                    "S": float(S[row]),
                    "standing_authority": float(S[row]),
                })
                newly.append((name, int(row)))
        self.step_in_phase += 1
        return newly

    def latched_total(self):
        return int(sum(int(lat.sum()) for lat in self.latched.values()))

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
        array); everything else steps at 1.0.
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

    def snapshot(self, F, rows):
        """Three-number snapshot for the given [(layer, row)] list.

        Returns per-row F/B/S/differences plus means over the rows.
        F is global; B/S are per row. Differences are computed here
        from the three numbers, never stored as merged state.
        """
        per_row = []
        for name, row in rows:
            B = float(self.baseline[name][row])
            S = float(self.layer_of[name].gain[row])
            per_row.append({
                "layer": name, "row": int(row),
                "F": float(F), "B": B, "S": S,
                "F_minus_B": float(F) - B,
                "B_minus_S": B - S,
                "F_minus_S": float(F) - S,
            })
        if per_row:
            means = {k: float(np.mean([r[k] for r in per_row]))
                     for k in ("F", "B", "S", "F_minus_B", "B_minus_S",
                               "F_minus_S")}
        else:
            means = None
        return {"per_row": per_row, "means": means}
