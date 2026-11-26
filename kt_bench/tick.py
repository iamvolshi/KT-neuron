"""Clean-phase tick schedules for R3-E3 (plan §2: tick mutability).

The claim (Darren, Oct 26): scale is in the mechanism -- the tick
unit is mutable. k's granularity adapts to the deployment; the law
doesn't change. A tick is one clean consolidation interval: steps
per clean consolidation. This module is pure schedule arithmetic:
it says WHEN consolidations happen during the clean phase. It
trains nothing, touches no weights, and makes no claim about
seating or ARCHE's feedback controller -- both out of scope.

Frozen schedules (preregistration-round3-2026-10-26.md, R3-E3):
- FINE:     400 clean steps, consolidate every 100 steps (4 ticks).
- BASELINE: 400 clean steps, 1 consolidation after 400 steps
  (round-1 structure).
- COARSE:   800 clean steps, 1 consolidation after 800 steps
  (clean length adjusted so each variant holds >= 1 full tick).

Decay is NOT scheduled here. Decay consolidations in the R3-E3 run
are counted by their own index n (0..30) and priced by
kt_bench.tail.SLOW_TAIL (0.01 per decay consolidation) -- per-TICK,
not per-step. That separation is the whole scale-invariance check:
the decay counter never sees the clean-phase step count, so a
variant that trained twice as many clean steps (COARSE, 800) must
still heal in the same number of decay ticks as the others. The
run script records the applied deficit series per variant as the
evidence; the tests below pin the arithmetic that makes it so.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TickSchedule:
    """One deployment's clean-phase tick size.

    clean_steps: total clean training steps in the variant.
    consol_every: steps per clean consolidation (the tick size).
    Invariant: clean_steps is an exact multiple of consol_every,
    so every variant holds >= 1 full tick and no partial tick
    is ever consolidated.
    """

    name: str
    clean_steps: int
    consol_every: int

    def __post_init__(self):
        if self.clean_steps <= 0:
            raise ValueError(
                f"clean_steps must be > 0, got {self.clean_steps}")
        if self.consol_every <= 0:
            raise ValueError(
                f"consol_every must be > 0, got {self.consol_every}")
        if self.clean_steps % self.consol_every != 0:
            raise ValueError(
                f"clean_steps ({self.clean_steps}) must be an exact "
                f"multiple of consol_every ({self.consol_every}): "
                f"a partial final tick is not a tick")

    @property
    def n_ticks(self) -> int:
        """Number of clean consolidations in the variant."""
        return self.clean_steps // self.consol_every

    def boundaries(self) -> list:
        """Clean-step counts at which consolidations happen.

        E.g. FINE -> [100, 200, 300, 400]; BASELINE -> [400].
        """
        return [self.consol_every * (i + 1)
                for i in range(self.n_ticks)]

    def chunk_bounds(self) -> list:
        """[(start_step, end_step)] training chunks between ticks.

        Chunks partition [0, clean_steps) with no gaps or overlaps,
        so a run can carry one Adam across chunks (R3-E1 pattern):
        chunking must not reset optimizer state, or the variant
        would train differently for a bookkeeping reason.
        """
        bounds = []
        prev = 0
        for b in self.boundaries():
            bounds.append((prev, b))
            prev = b
        return bounds


# The three schedules frozen for R3-E3 (rates per consolidation).
FINE_TICK = TickSchedule(name="FINE", clean_steps=400, consol_every=100)
BASELINE_TICK = TickSchedule(name="BASELINE", clean_steps=400,
                             consol_every=400)
COARSE_TICK = TickSchedule(name="COARSE", clean_steps=800,
                           consol_every=800)
