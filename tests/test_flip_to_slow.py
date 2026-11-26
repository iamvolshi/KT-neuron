"""Tests for flip-to-slow (plan §6 step 4, stage 1).

Proves, with real runs:
1. Firewall engagement halts fast-trace weight changes (bit-identical
   across attempted steps while flipped).
2. Forward behavior while flipped matches the last clean consolidated
   state (with seated gains), not the poisoned live tensors.
3. Repair + release resumes fast: training steps change weights again.
4. The pre-existing suite still passes (run the full suite, not just
   this file).

The step gate is the enforcement point: apply_training_step does not
call fn() while flipped. Direct numpy writes bypass any gate by
construction -- that limit is stated in kt_bench/flip.py, not hidden.

Run from ~/workspace/kt-neuron:

    python3 -m unittest discover -s tests -v
"""

import unittest

import numpy as np

from kt_bench.consolidation import Consolidator
from kt_bench.flip import TraceFlip
from kt_bench.linear import KTLinear
from kt_bench.transformer import GPT, GPTConfig
from kt_bench.twin import build_pair, deterministic_batches


def _rng(seed=0):
    return np.random.default_rng(seed)


class FakePot:
    """Duck-typed firewall pot: .engaged / .released, set by the test."""

    def __init__(self, engaged=False, released=False):
        self.engaged = engaged
        self.released = released


def _poisoned_fixture(note_clean=True, move_scar=True):
    """Layer with k=1 (gain 0.6), k=2 clean (gain 0.7, scar 0.25),
    k=3 poison (gain 0.9). Returns (layer, con, flip, clean, x, holder).

    clean = (W copy, b copy, gain copy, scar copy, forward copy) at k=2.
    """
    layer = KTLinear(6, 4, _rng(7))
    holder = {"v": 0.6}
    con = Consolidator([("fc", layer)], seating_fn=lambda pool: holder["v"])
    flip = TraceFlip(con.layers, con.snapshots, current_k=lambda: con.k)
    con.consolidate()  # k=1, gains 0.6
    if note_clean:
        flip.note_clean(1)
    layer.W += _rng(8).normal(0.0, 0.01, size=layer.W.shape)
    holder["v"] = 0.7
    if move_scar:
        layer.scar[:] = 0.25
    con.consolidate()  # k=2, gains 0.7 -- the last clean state
    if note_clean:
        flip.note_clean(2)
    x = _rng(9).normal(size=(2, 6))
    clean = (layer.W.copy(), layer.b.copy(), layer.gain.copy(),
             layer.scar.copy(), layer.forward(x))
    # Poison interval: move weights, change scar, seat at 0.9.
    layer.W += _rng(10).normal(0.0, 0.05, size=layer.W.shape)
    layer.b += _rng(11).normal(0.0, 0.05, size=layer.b.shape)
    if move_scar:
        layer.scar[:] = 0.9
    holder["v"] = 0.9
    con.consolidate()  # k=3, gains 0.9 -- poison seated
    return layer, con, flip, clean, x, holder


class TestFlipHaltsFastTrace(unittest.TestCase):
    def test_weights_bit_identical_across_attempted_steps_while_flipped(self):
        layer, _con, flip, clean, _x, _h = _poisoned_fixture()
        self.assertEqual(flip.engage(), 2)
        self.assertTrue(flip.flipped)
        w_clean, b_clean = clean[0], clean[1]
        self.assertTrue(np.array_equal(layer.W, w_clean))
        self.assertTrue(np.array_equal(layer.b, b_clean))
        for _ in range(3):
            stepped = flip.apply_training_step(
                lambda: (layer.W.__iadd__(0.1), layer.b.__iadd__(0.1)))
            self.assertFalse(stepped)
            self.assertTrue(np.array_equal(layer.W, w_clean))
            self.assertTrue(np.array_equal(layer.b, b_clean))

    def test_step_gate_runs_steps_when_not_flipped(self):
        layer, _con, flip, clean, _x, _h = _poisoned_fixture()
        before = layer.W.copy()
        self.assertTrue(flip.apply_training_step(
            lambda: layer.W.__iadd__(0.1)))
        self.assertFalse(np.array_equal(layer.W, before))

    def test_double_engage_raises(self):
        _layer, _con, flip, _clean, _x, _h = _poisoned_fixture()
        flip.engage()
        with self.assertRaises(RuntimeError):
            flip.engage()


class TestFlipForwardMatchesClean(unittest.TestCase):
    def test_forward_and_gains_match_last_clean_state_not_poison(self):
        layer, _con, flip, clean, x, _h = _poisoned_fixture()
        poison_forward = layer.forward(x)
        self.assertFalse(np.array_equal(poison_forward, clean[4]))
        self.assertTrue(np.all(layer.gain == 0.9))
        flip.engage()
        self.assertTrue(np.array_equal(layer.forward(x), clean[4]))
        self.assertTrue(np.array_equal(layer.gain, clean[2]))
        self.assertTrue(np.all(layer.gain == 0.7))
        self.assertTrue(np.array_equal(layer.scar, clean[3]))
        self.assertTrue(np.all(layer.scar == 0.25))

    def test_fallback_gain_restore_without_notes(self):
        # No note_clean calls: gains come from the sealed record --
        # the clean snapshot's row hash matched to its interval.
        layer, _con, flip, clean, x, _h = _poisoned_fixture(note_clean=False, move_scar=False)
        flip.engage()
        self.assertTrue(np.array_equal(layer.forward(x), clean[4]))
        self.assertTrue(np.all(layer.gain == 0.7))

    def test_explicit_clean_k_and_poison_k(self):
        layer, con, flip, clean, x, holder = _poisoned_fixture()
        holder["v"] = 0.9
        con.consolidate()  # k=4, after the poison interval
        # Poison was seated at k=3, so clean is k=2 even now.
        self.assertEqual(flip.engage(poison_k=3), 2)
        self.assertTrue(np.array_equal(layer.forward(x), clean[4]))
        # Fresh fixture: explicit clean_k=1 restores the k=1 state.
        layer2, con2, flip2, _c2, _x2, _h2 = _poisoned_fixture()
        self.assertEqual(flip2.engage(clean_k=1), 1)
        self.assertTrue(np.all(layer2.gain == 0.6))

    def test_no_earlier_clean_state_refuses(self):
        layer = KTLinear(6, 4, _rng(7))
        con = Consolidator([("fc", layer)])
        flip = TraceFlip(con.layers, con.snapshots, current_k=lambda: con.k)
        con.consolidate()  # only snapshot is k=1, itself current
        with self.assertRaises(ValueError):
            flip.engage()

    def test_gpt_forward_matches_clean_after_flip(self):
        cfg = GPTConfig()
        model, _ = build_pair(cfg, 21)
        con = Consolidator(model.kt_linears())
        flip = TraceFlip(con.layers, con.snapshots, current_k=lambda: con.k)
        con.consolidate()  # k=1
        model.perturb(_rng(22), scale=0.01)
        con.consolidate()  # k=2 clean
        flip.note_clean(2)
        idx = deterministic_batches(5, 1, 2, 8, cfg.vocab_size)[0]
        clean_forward = model.forward(idx)
        model.perturb(_rng(23), scale=0.05)  # poison moves KT rows only
        con.consolidate()  # k=3 poison seated
        self.assertFalse(np.array_equal(model.forward(idx), clean_forward))
        self.assertEqual(flip.engage(), 2)
        self.assertTrue(np.array_equal(model.forward(idx), clean_forward))


class TestFirewallWiring(unittest.TestCase):
    def test_engage_if_pot_flips_only_when_engaged(self):
        _layer, _con, flip, clean, x, _h = _poisoned_fixture()
        pot = FakePot(engaged=False)
        self.assertIsNone(flip.engage_if_pot(pot))
        self.assertFalse(flip.flipped)
        pot.engaged = True
        self.assertEqual(flip.engage_if_pot(pot), 2)
        self.assertTrue(flip.flipped)
        self.assertTrue(np.array_equal(_layer.forward(x), clean[4]))


class TestRepairAndRelease(unittest.TestCase):
    def test_repair_freezes_unlisted_rows_and_release_resumes_fast(self):
        layer, _con, flip, clean, _x, holder = _poisoned_fixture()
        flip.engage()
        # Release before repair is refused; still flipped, still frozen.
        self.assertFalse(flip.release())
        self.assertTrue(flip.flipped)
        self.assertFalse(flip.apply_training_step(
            lambda: layer.W.__iadd__(1.0)))
        # Targeted repair: update tries to move every row; only the
        # two named rows may move, the rest stay bit-identical.
        allowed = {("fc", 0), ("fc", 2)}
        moved = flip.repair(
            lambda: (layer.W.__iadd__(0.05), layer.b.__iadd__(0.05)),
            allowed)
        self.assertEqual(moved, [("fc", 0), ("fc", 2)])
        for i in range(4):
            if i in (0, 2):
                self.assertFalse(np.array_equal(layer.W[i], clean[0][i]))
            else:
                self.assertTrue(np.array_equal(layer.W[i], clean[0][i]))
                self.assertEqual(layer.b[i], clean[1][i])
        # Targeted re-consolidation: allowed rows re-seat docked,
        # other rows keep their clean gains.
        holder["v"] = 0.9
        counts = flip.reseat_repaired(
            allowed, lambda pool: holder["v"], "repair:clean",
            dock_map={("fc", 0): 0.3, ("fc", 2): 0.3})
        self.assertEqual(counts, {"seated": 2, "candidate": 0,
                                  "reaffirmed": 0})
        self.assertTrue(np.all(layer.gain[[0, 2]] == 0.6))
        self.assertTrue(np.all(layer.gain[[1, 3]] == 0.7))
        # Release needs the pot below its release line.
        pot = FakePot(engaged=True, released=False)
        self.assertFalse(flip.release(pot))
        self.assertTrue(flip.flipped)
        pot.released = True
        self.assertTrue(flip.release(pot))
        self.assertFalse(flip.flipped)
        # Fast resumes: steps change weights again.
        before = layer.W.copy()
        self.assertTrue(flip.apply_training_step(
            lambda: layer.W.__iadd__(0.1)))
        self.assertFalse(np.array_equal(layer.W, before))

    def test_repair_and_reseat_require_flipped_state(self):
        _layer, _con, flip, _clean, _x, _h = _poisoned_fixture()
        with self.assertRaises(RuntimeError):
            flip.repair(lambda: None, {("fc", 0)})
        with self.assertRaises(RuntimeError):
            flip.reseat_repaired({("fc", 0)}, lambda pool: 0.9, "repair:x")
        self.assertFalse(flip.release())  # not flipped: nothing to release


if __name__ == "__main__":
    unittest.main()
