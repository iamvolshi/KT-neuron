"""Tail deployments for scar healing (R3-E2, plan §6 step 7).

The law gives the shape (deficit decays with consolidations); the
deployment sets the tail's parameters (decay rate, what counts as
healed). This module is pure arithmetic: it computes the deficit
schedule for a deployment. It trains nothing, touches no weights,
and makes no claim about ARCHE's feedback controller or seating --
both out of scope for this round.

Definitions (frozen in preregistration-round3-2026-10-26.md):
- deficit(n) = max(0.0, initial_deficit - decay_per_consolidation * n)
  where n counts decay consolidations since repair.
- Healed: deficit(n) <= healed_threshold (default 0.05).
- time_to_healed: the first n with deficit(n) <= threshold, or None
  when the tail never heals (e.g. decay rate 0.0) within the cap.

Note on the authority grid: the bench record quantizes seated
authority to the 0.1 grid (q10), so standing authority (0.9 minus
deficit, quantized) is NOT the deficit. The deficit reported by an
experiment is this module's dock parameter, exact; standing
authority is reported alongside it, never substituted for it.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TailDeployment:
    """One deployment's tail parameters."""

    name: str
    decay_per_consolidation: float
    initial_deficit: float = 0.30
    healed_threshold: float = 0.05

    def __post_init__(self):
        if self.decay_per_consolidation < 0.0:
            raise ValueError(
                f"decay_per_consolidation must be >= 0, got "
                f"{self.decay_per_consolidation}")
        if self.initial_deficit < 0.0:
            raise ValueError(
                f"initial_deficit must be >= 0, got {self.initial_deficit}")
        if self.healed_threshold < 0.0:
            raise ValueError(
                f"healed_threshold must be >= 0, got {self.healed_threshold}")

    def deficit_at(self, n: int) -> float:
        """Deficit after n decay consolidations (n >= 0)."""
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")
        return max(0.0, self.initial_deficit
                   - self.decay_per_consolidation * n)

    def is_healed(self, n: int) -> bool:
        """Healed definition: deficit(n) <= healed_threshold."""
        return self.deficit_at(n) <= self.healed_threshold

    def deficit_series(self, n_max: int) -> list:
        """[deficit(0), ..., deficit(n_max)]."""
        return [self.deficit_at(n) for n in range(n_max + 1)]

    def time_to_healed(self, n_cap: int = 10000):
        """First n <= n_cap with deficit(n) <= threshold, else None.

        A zero-decay tail with initial deficit above threshold never
        heals: returns None (it does not raise, and it is not capped
        into a fake finite time).
        """
        for n in range(n_cap + 1):
            if self.is_healed(n):
                return n
        return None


# The three deployments frozen for R3-E2 (rates per consolidation).
FAST_TAIL = TailDeployment(name="FAST", decay_per_consolidation=0.05)
SLOW_TAIL = TailDeployment(name="SLOW", decay_per_consolidation=0.01)
PERMANENT_TAIL = TailDeployment(name="PERMANENT",
                                decay_per_consolidation=0.00)
