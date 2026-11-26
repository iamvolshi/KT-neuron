"""Scar and deficit: the neuron's fast trace and slow trace.

The scar is carried in the bias. It follows the v1.8 hump landmarks:
zero at or below DECAY 0.30, rising to a peak at 0.75, tapering above.
This is the toy's rendering of those landmarks; the true closed form
lives in the v1.8 proof packet, not here.

Two traces, kept separate on purpose:

- Fast trace (decay_pressure -> scar): rises with injury, bleeds off
  with clean time. Behavioral, and it moves the output now.
- Slow trace (deficit): the consolidated-authority deficit. It grows
  with injury and crawls back on a much slower timescale. It docks
  whatever authority a repaired belief may re-seat at -- which is why
  behavior can return to baseline while seated authority measurably
  lags. The deficit is the slow trace's memory.
"""

# Toy rendering constants (same values as the toy).
SCAR_GAIN = 1.2        # scar magnitude multiplier on the hump shape
INJURY_PRESSURE = 0.18  # decay_pressure added per injury event
HEAL_PRESSURE = 0.09    # decay_pressure bled per clean-time step
DEFICIT_STEP_UP = 0.06  # slow deficit added per injury event
DEFICIT_STEP_DOWN = 0.01  # slow deficit healed per clean-time step
DEFICIT_CAP = 0.4


def scar_shape(decay_pressure: float) -> float:
    """Hump curve over decay pressure: 0 at/below 0.30, peak 1.0 at 0.75."""
    d = decay_pressure
    if d <= 0.30:
        return 0.0
    if d <= 0.75:
        return (d - 0.30) / 0.45
    return max(0.0, 1.0 - (d - 0.75) / 0.25 * 0.5)


class ScarState:
    """Injury state for one neuron: fast scar plus slow deficit."""

    def __init__(self):
        self.decay_pressure = 0.0  # fast trace driver
        self.scar = 0.0            # fast trace: behavioral scar (sits in the bias)
        self.deficit = 0.0         # slow trace: consolidated-authority deficit

    def _refresh_scar(self):
        self.scar = round(SCAR_GAIN * scar_shape(self.decay_pressure), 3)

    def injure(self, amount=INJURY_PRESSURE):
        """Harm arrives: fast trace rises, slow trace takes a deficit."""
        self.decay_pressure = min(1.0, self.decay_pressure + amount)
        self._refresh_scar()
        self.deficit = min(DEFICIT_CAP, round(self.deficit + DEFICIT_STEP_UP, 2))

    def heal(self, amount=HEAL_PRESSURE):
        """Clean time: the fast trace bleeds off; the slow deficit crawls."""
        self.decay_pressure = max(0.0, self.decay_pressure - amount)
        self._refresh_scar()
        self.deficit = max(0.0, round(self.deficit - DEFICIT_STEP_DOWN, 2))
