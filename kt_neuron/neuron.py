"""The KT neuron: one neuron whose weights and bias are beliefs.

    y = sigmoid( a * (W . x) + (b - scar) )

where a is the signal authority -- the minimum authority over the weight
beliefs, so the signal speaks at its weakest authority -- and scar is
the fast-trace injury offset carried in the bias.

The neuron holds no weight variables. Every weight and the bias are
beliefs in its K-record: born registered (initial beliefs from its
maker at authority 0.5), moved only by closing one interval and opening
the next, under the record's authority gate. Learning, injury, and
repair all go through the record; there is no side channel.

Scope note: this module is the neuron ONLY. No layers, no training
loop, no seating rule, no controller. The teach() step below is the
neuron's own local update (one plain step toward an asserted output);
anything that decides what the neuron should be taught lives outside.
"""

import itertools
import math

from .kid import make_kid
from .record import KRecord
from .scar import ScarState

# Process-wide birth counter: guarantees distinct sequence numbers for
# neurons born in the same second, so KIDs do not collide in-process.
_birth_seq = itertools.count()

BIRTH_AUTHORITY = 0.5
AGREE_TOLERANCE = 0.05  # |error| below this: nothing to learn


class KTNeuron:
    """A single KT neuron with n_inputs weights plus a bias."""

    def __init__(self, n_inputs: int = 2, birth_tick: int = 0):
        if n_inputs < 1:
            raise ValueError("a neuron needs at least one input")
        # KID first: no KID, no read, no write.
        self.kid = make_kid(birth_tick=birth_tick, seq=next(_birth_seq))
        self.rec = KRecord(self.kid)
        self.scar_state = ScarState()
        self.n_inputs = n_inputs
        self.weight_fields = [f"w{i}" for i in range(n_inputs)]
        # Born registered: initial beliefs about its own weights, from its maker.
        for field in self.weight_fields + ["b"]:
            self.rec.open_belief(field, 0.0, BIRTH_AUTHORITY, "birth")

    @property
    def k(self) -> int:
        """Knowledge time: advances only when a belief opens."""
        return self.rec.k

    @property
    def scar(self) -> float:
        return self.scar_state.scar

    @property
    def deficit(self) -> float:
        return self.scar_state.deficit

    def standing(self, field):
        """The currently-open belief row for a field, or None."""
        return self.rec.standing(field)

    def _w(self, field):
        row = self.rec.standing(field)
        return row["value"], row["authority"]

    def fire(self, x):
        """Forward pass. Returns (output, signal_authority)."""
        if len(x) != self.n_inputs:
            raise ValueError(f"expected {self.n_inputs} inputs, got {len(x)}")
        weights = [self._w(f) for f in self.weight_fields]
        b, _ = self._w("b")
        a = min(auth for _, auth in weights)  # weakest authority speaks
        z = a * sum(w * xi for (w, _), xi in zip(weights, x)) + (b - self.scar)
        return 1.0 / (1.0 + math.exp(-z)), a

    def teach(self, x, target, source, source_authority, lr=2.0):
        """A source asserts an output. One plain step toward it, taken only
        through the record: beliefs move by closing and opening intervals.
        The slow deficit docks whatever authority a re-seated belief carries.
        Returns a per-field status string (e.g. "w0:seated w1:candidate b:seated").
        """
        y, _ = self.fire(x)
        err = target - y
        if abs(err) < AGREE_TOLERANCE:
            return "agrees -- nothing to learn"
        current = {f: self._w(f)[0] for f in self.weight_fields + ["b"]}
        new = {f: round(current[f] + lr * err * x[i], 3)
               for i, f in enumerate(self.weight_fields)}
        new["b"] = round(current["b"] + lr * err * 0.25, 3)
        moved = []
        for f, v in new.items():
            status, _row = self.rec.replace(f, v, source_authority, source,
                                            authority_dock=self.deficit)
            moved.append(f"{f}:{status}")
        return " ".join(moved)

    def injure(self):
        """Harm arrives: fast scar rises, slow deficit grows."""
        self.scar_state.injure()

    def heal(self):
        """Clean time: fast scar bleeds off, slow deficit crawls back."""
        self.scar_state.heal()

    def verify(self):
        """Verify the neuron's whole belief record (hash chain + intervals)."""
        return self.rec.verify()
