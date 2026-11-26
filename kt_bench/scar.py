"""Scar modulation for the scar-only ablation arm (3-arm experiment).

scar_shape is the toy's rendering verbatim (kt_neuron.py): zero
at/below 0.30, peak at 0.75. The PressureDriver is the experiment's
scheduled injury driver (frozen in
experiments/preregistration-3arm-2026-10-28.md): pressure rises
during poison, falls at half rate during restore, capped at the
peak 0.75. The over-injury declining limb (p > 0.75) is NOT
exercised in this experiment — stated limitation, not a hidden
choice. Half-rate healing honors the toy's asymmetry (injure 0.18
vs heal 0.09 per event).

ScarLinear: a plain Linear with a scar vector, no record, no
authority, no gain. Forward = x @ W.T + b - scar. The trainer
handles it via getattr (same path as KTLinear's scar), so no
training code changes are needed. The scar is one global scalar
applied to every row ("global injury state"); per-row scar
variation is later work, not this experiment.
"""

import numpy as np

from .linear import Linear

SCAR_SCALE = 1.2
PEAK_PRESSURE = 0.75
POISON_RATE = 0.75 / 200  # per poison step
RESTORE_RATE = 0.375 / 200  # per restore step (half-rate healing)


def scar_shape(decay_pressure):
    """Toy rendering of the v1.8 landmarks: zero at/below 0.30, peak
    at 0.75."""
    d = float(decay_pressure)
    if d <= 0.30:
        return 0.0
    if d <= 0.75:
        return (d - 0.30) / 0.45
    return max(0.0, 1.0 - (d - 0.75) / 0.25 * 0.5)


class PressureDriver:
    """Scheduled injury driver: rises on poison steps, falls on
    restore steps. Mirrors the toy's injure()/heal() calls, which
    are also scheduled, not measured from model state."""

    def __init__(self):
        self.pressure = 0.0

    @property
    def scar(self):
        return round(SCAR_SCALE * scar_shape(self.pressure), 6)

    def poison_step(self):
        self.pressure = min(PEAK_PRESSURE, self.pressure + POISON_RATE)

    def restore_step(self):
        self.pressure = max(0.0, self.pressure - RESTORE_RATE)

    def apply_to(self, model):
        """Set every scar-carrying layer's scar vector to the
        current scalar. Returns the scalar."""
        s = self.scar
        for _name, layer in model.all_linears():
            if hasattr(layer, "scar"):
                layer.scar[:] = s
        return s


class ScarLinear(Linear):
    """Plain linear + scar vector. No record, no authority, no gain.

    This is the scar-only ablation arm's layer: the ONLY difference
    from Linear is the bias offset. If this arm matches the full
    KTN arm, the record + selection added nothing.
    """

    def __init__(self, in_features, out_features, rng, name=""):
        super().__init__(in_features, out_features, rng)
        self.name = name
        self.scar = np.zeros(out_features)

    def forward(self, x):
        return x @ self.W.T + self.b - self.scar
