"""Tests for kt_bench.handoff (3-arm v2) and the Adam lr_scale path.

Run from ~/workspace/kt-neuron:
    python3 -m unittest discover -s tests -v

Frozen contract: experiments/preregistration-3arm-v2-2026-10-28.md.
The handoff rule: gate S > 0.5, upward crossing prev_F < S <= new_F,
latch permanent, latched rows' Adam step size x current S. Gradients
are NOT scaled (Adam normalizes them); the step is.
"""

import unittest

import numpy as np

from kt_bench.handoff import AUTHORITY_GATE, HandoffTracker
from kt_bench.transformer import GPT, GPTConfig
from kt_neuron.kid import make_kid
from kt_neuron.record import KRecord
from experiments.playbook import layer_param_prefix
from experiments.trainer import (Adam, collect_params, loss_and_grads)
from experiments.run_all import train_steps


class StubLayer:
    """Minimal layer stand-in: only .gain and .out_features are read
    by HandoffTracker."""

    def __init__(self, out_features, gain_value):
        self.out_features = out_features
        self.gain = np.full(out_features, gain_value)


class TestCrossing(unittest.TestCase):
    def test_upward_crossing_latches_with_gate_pass(self):
        tr = HandoffTracker([("lm_head", StubLayer(4, 0.8))])
        tr.set_phase("poison1")
        newly = tr.check(0.79, 0.8)
        self.assertEqual(newly, [("lm_head", 0), ("lm_head", 1),
                                 ("lm_head", 2), ("lm_head", 3)])
        self.assertEqual(tr.latched_total(), 4)
        self.assertEqual(len(tr.events), 4)
        self.assertEqual(tr.events[0]["phase"], "poison1")
        self.assertEqual(tr.events[0]["step_in_phase"], 0)
        self.assertAlmostEqual(tr.events[0]["S"], 0.8)

    def test_no_crossing_below_line(self):
        tr = HandoffTracker([("lm_head", StubLayer(4, 0.8))])
        tr.set_phase("poison1")
        self.assertEqual(tr.check(0.5, 0.79), [])
        self.assertEqual(tr.latched_total(), 0)

    def test_gate_blocks_weak_authority(self):
        # S = 0.5 is NOT > 0.5: a birth/candidate row never latches,
        # even when F crosses its level.
        tr = HandoffTracker([("lm_head", StubLayer(4, 0.5))])
        tr.set_phase("poison1")
        self.assertEqual(tr.check(0.4, 0.6), [])
        self.assertEqual(tr.latched_total(), 0)
        self.assertEqual(AUTHORITY_GATE, 0.5)

    def test_downward_move_does_not_latch(self):
        tr = HandoffTracker([("lm_head", StubLayer(4, 0.8))])
        tr.set_phase("restore1")
        # F falling from 1.2 to 0.2 passes THROUGH 0.8 downward: not an
        # upward crossing, so nothing latches.
        self.assertEqual(tr.check(1.2, 0.2), [])
        self.assertEqual(tr.latched_total(), 0)

    def test_latch_permanent_when_F_falls(self):
        tr = HandoffTracker([("lm_head", StubLayer(4, 0.8))])
        tr.set_phase("poison1")
        tr.check(0.79, 0.8)
        tr.set_phase("restore1")
        tr.check(1.2, 0.2)
        self.assertEqual(tr.latched_total(), 4)

    def test_partial_crossing_only_some_rows(self):
        # Two layers with different S: only the lower line is crossed.
        tr = HandoffTracker([("a", StubLayer(2, 0.6)),
                             ("b", StubLayer(2, 0.9))])
        tr.set_phase("poison1")
        newly = tr.check(0.5, 0.7)
        self.assertEqual(newly, [("a", 0), ("a", 1)])
        self.assertEqual(tr.latched_total(), 2)


class TestStepScale(unittest.TestCase):
    def test_latched_row_scales_by_S_unlatched_by_one(self):
        tr = HandoffTracker([("lm_head", StubLayer(4, 0.6))])
        tr.set_phase("poison1")
        tr.check(0.5, 0.7)  # all rows latch (S=0.6, gate passes)
        scales = tr.lr_scale_list(["tok_emb", "lm_head.W", "lm_head.b"],
                                  {"lm_head": "lm_head"})
        self.assertEqual(scales[0], 1.0)
        np.testing.assert_array_equal(scales[1],
                                      np.full((4, 1), 0.6))
        np.testing.assert_array_equal(scales[2], np.full(4, 0.6))

    def test_unlatched_model_all_ones(self):
        tr = HandoffTracker([("lm_head", StubLayer(4, 0.8))])
        scales = tr.lr_scale_list(["tok_emb", "lm_head.W", "lm_head.b"],
                                  {"lm_head": "lm_head"})
        self.assertEqual(scales[0], 1.0)
        np.testing.assert_array_equal(scales[1], np.ones((4, 1)))
        np.testing.assert_array_equal(scales[2], np.ones(4))

    def test_scale_tracks_current_S_not_latch_time_S(self):
        # S changes at consolidation (0.8 -> 0.6 dock); the step scale
        # must follow the CURRENT S, not the value at latch time.
        layer = StubLayer(4, 0.8)
        tr = HandoffTracker([("lm_head", layer)])
        tr.set_phase("poison1")
        tr.check(0.79, 0.8)
        layer.gain[:] = 0.6  # restore-1 consolidation + dock
        scales = tr.lr_scale_list(["lm_head.W", "lm_head.b"],
                                  {"lm_head": "lm_head"})
        np.testing.assert_array_equal(scales[0], np.full((4, 1), 0.6))
        np.testing.assert_array_equal(scales[1], np.full(4, 0.6))


class TestThreeNumbers(unittest.TestCase):
    def test_snapshot_reports_numbers_separately(self):
        layer = StubLayer(4, 0.6)
        tr = HandoffTracker([("lm_head", layer)])
        tr.set_baseline(0.9)  # repair claim; S=0.6 -> depth 0.3
        snap = tr.snapshot(0.2, [("lm_head", 1)])
        row = snap["per_row"][0]
        self.assertEqual((row["F"], row["B"], row["S"]), (0.2, 0.9, 0.6))
        # Differences recompute from the three numbers, never stored
        # as substitutes for them.
        self.assertAlmostEqual(row["F_minus_B"], 0.2 - 0.9)
        self.assertAlmostEqual(row["B_minus_S"], 0.9 - 0.6)
        self.assertAlmostEqual(row["F_minus_S"], 0.2 - 0.6)
        self.assertAlmostEqual(snap["means"]["B_minus_S"], 0.3)


class TestAdamStepScale(unittest.TestCase):
    def _tiny_model(self):
        cfg = GPTConfig(vocab_size=8, block_size=4, n_embd=8, n_head=2,
                        n_layer=1)
        return GPT(cfg, np.random.default_rng(0),
                   __import__("kt_bench.linear", fromlist=["Linear"]).Linear)

    def test_all_ones_scale_reproduces_train_steps_bit_identical(self):
        batch = np.random.default_rng(5).integers(0, 8, size=(2, 4))
        m1 = self._tiny_model()
        m2 = self._tiny_model()
        names1, params1 = collect_params(m1)
        adam1 = Adam(params1, lr=0.01)
        train_steps(m1, [batch], adam=adam1)
        names2, params2 = collect_params(m2)
        adam2 = Adam(params2, lr=0.01)
        loss, grads = loss_and_grads(m2, batch)
        adam2.step([grads[n] for n in names2],
                   lr_scale=[1.0] * len(names2))
        for p1, p2 in zip(params1, params2):
            self.assertTrue(np.array_equal(p1, p2))

    def test_half_scale_halves_exactly_that_rows_update(self):
        batch = np.random.default_rng(5).integers(0, 8, size=(2, 4))
        m1 = self._tiny_model()
        m2 = self._tiny_model()
        names1, params1 = collect_params(m1)
        w1_before = params1[names1.index("lm_head.W")].copy()
        b1_before = params1[names1.index("lm_head.b")].copy()
        adam1 = Adam(params1, lr=0.01)
        train_steps(m1, [batch], adam=adam1)
        names2, params2 = collect_params(m2)
        w2_before = params2[names2.index("lm_head.W")].copy()
        b2_before = params2[names2.index("lm_head.b")].copy()
        adam2 = Adam(params2, lr=0.01)
        loss, grads = loss_and_grads(m2, batch)
        scales = []
        for n in names2:
            if n == "lm_head.W":
                s = np.ones_like(params2[names2.index("lm_head.W")])
                s[3, :] = 0.5
                scales.append(s)
            elif n == "lm_head.b":
                s = np.ones_like(params2[names2.index("lm_head.b")])
                s[3] = 0.5
                scales.append(s)
            else:
                scales.append(1.0)
        adam2.step([grads[n] for n in names2], lr_scale=scales)
        d1_w = params1[names1.index("lm_head.W")] - w1_before
        d2_w = params2[names2.index("lm_head.W")] - w2_before
        d1_b = params1[names1.index("lm_head.b")] - b1_before
        d2_b = params2[names2.index("lm_head.b")] - b2_before
        # Scaled row: exactly half the update (up to float rounding).
        np.testing.assert_allclose(d2_w[3], 0.5 * d1_w[3],
                                   rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(d2_b[3], 0.5 * d1_b[3],
                                   rtol=1e-9, atol=1e-12)
        # All other rows: bit-identical.
        self.assertTrue(np.array_equal(np.delete(d2_w, 3, axis=0),
                                       np.delete(d1_w, 3, axis=0)))
        self.assertTrue(np.array_equal(np.delete(d2_b, 3),
                                       np.delete(d1_b, 3)))


class TestFactBirthRow(unittest.TestCase):
    def test_birth_plus_three_seatings_is_four_rows_chain_ok(self):
        # V2 construction fix: birth belief at construction, then the
        # same three seating replaces as v1 -> 4 rows, chain sealed.
        rec = KRecord(make_kid(birth_tick=0, seq=999001))
        rec.open_belief("fact", 3, 0.5, "birth")
        rec.replace("fact", 3, 0.8, "teacher")
        rec.replace("fact", 3, 0.9, "teacher", authority_dock=0.3)
        rec.replace("fact", 3, 0.9, "teacher", authority_dock=0.3)
        self.assertEqual(len(rec.rows), 4)
        self.assertTrue(rec.verify().startswith("OK"))
        self.assertEqual(rec.standing("fact")["authority"], 0.6)


if __name__ == "__main__":
    unittest.main()
