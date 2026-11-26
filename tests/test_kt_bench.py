"""Tests for the KTN bench assembly (Phase 0 + Phase 1).

Covers: layer parity with vanilla linear, KIDs per neuron, gain/scar
mechanics, the consolidation clock (k advances at consolidation only),
snapshot/restore round-trips, and the vanilla twin harness.

Run from ~/workspace/kt-neuron:

    python3 -m unittest discover -s tests -v
"""

import unittest

import numpy as np

from kt_bench.consolidation import Consolidator
from kt_bench.linear import KTLinear, Linear, row_hash
from kt_bench.seating import seat_authority
from kt_bench.transformer import GPT, GPTConfig
from kt_bench.twin import (build_pair, check_forward_parity,
                           check_init_parity, deterministic_batches)


def _rng(seed=0):
    return np.random.default_rng(seed)


class TestKTLinearParity(unittest.TestCase):
    def test_matches_vanilla_linear_exactly(self):
        rng = _rng(1)
        kt = KTLinear(16, 8, rng)
        vn = Linear(16, 8, rng)  # noqa: different rng stream, weights reset below
        vn.W[:] = kt.W
        vn.b[:] = kt.b
        x = _rng(2).normal(size=(4, 16))
        self.assertTrue(np.array_equal(kt.forward(x), vn.forward(x)))

    def test_gain_scales_rows_and_scar_shifts_bias(self):
        rng = _rng(3)
        kt = KTLinear(4, 3, rng)
        x = _rng(4).normal(size=(2, 4))
        base = kt.forward(x)
        kt.gain[:] = 0.5
        self.assertTrue(np.array_equal(kt.forward(x), base * 0.5))
        kt.gain[:] = 1.0
        kt.scar[:] = 0.25
        self.assertTrue(np.allclose(kt.forward(x), base - 0.25))


class TestKIDs(unittest.TestCase):
    def test_one_unique_kid_per_neuron(self):
        kt = KTLinear(8, 8, _rng(5))
        self.assertEqual(len(kt.kids), 8)
        self.assertEqual(len(set(kt.kids)), 8)
        self.assertTrue(all(k for k in kt.kids))
        self.assertTrue(all(rec.verify().startswith("OK")
                            for rec in kt.records))

    def test_model_wide_kid_count(self):
        cfg = GPTConfig()  # per block: 192+64+256+64 = 576 rows; lm_head 128
        model, _ = build_pair(cfg, 42)
        kids = [kid for _n, layer in model.kt_linears() for kid in layer.kids]
        self.assertEqual(len(kids), 2 * 576 + 128)
        self.assertEqual(len(set(kids)), len(kids))


class TestConsolidationClock(unittest.TestCase):
    def setUp(self):
        self.layer = KTLinear(6, 4, _rng(7))
        self.con = Consolidator([("fc", self.layer)])

    def test_steps_do_not_touch_records_or_k(self):
        ks_before = self.con.records.record_ks()
        rows_before = self.con.records.row_counts()
        self.layer.W += 0.01  # fast trace: live step, unrecorded
        self.assertEqual(self.con.k, 0)
        self.assertEqual(self.con.records.record_ks(), ks_before)
        self.assertEqual(self.con.records.row_counts(), rows_before)

    def test_consolidate_advances_k_by_one_and_only_k(self):
        self.layer.W += 0.01
        ks_before = self.con.records.record_ks()
        rows_before = self.con.records.row_counts()
        new_k = self.con.consolidate()
        self.assertEqual(new_k, 1)
        self.assertEqual(self.con.k, 1)
        self.assertEqual(self.con.records.record_ks(),
                         [k + 1 for k in ks_before])
        self.assertEqual(self.con.records.row_counts(),
                         [r + 1 for r in rows_before])
        # Old interval closed, new standing row holds the current hash.
        for i, rec in enumerate(self.layer.records):
            self.assertIsNotNone(rec.rows[-2]["k_end"])
            self.assertIsNone(rec.rows[-1]["k_end"])
            self.assertEqual(rec.rows[-1]["value"],
                             row_hash(self.layer.W[i], self.layer.b[i]))
        self.assertTrue(self.con.all_ok())

    def test_second_consolidation_seats_again(self):
        self.con.consolidate()
        self.layer.W += 0.02
        self.con.consolidate()
        self.assertEqual(self.con.k, 2)
        self.assertTrue(self.con.all_ok())
        self.assertEqual(self.con.snapshots.available(), [1, 2])


class TestSnapshotRestore(unittest.TestCase):
    def test_round_trip_reproduces_past_belief_exactly(self):
        cfg = GPTConfig()
        model, _ = build_pair(cfg, 11)
        con = Consolidator(model.kt_linears())
        con.consolidate()  # bench k = 1
        saved = {n: (l.W.copy(), l.b.copy()) for n, l in model.kt_linears()}
        model.perturb(_rng(12), scale=0.05)
        con.consolidate()  # bench k = 2
        con.snapshots.restore(1, model.kt_linears())
        for n, layer in model.kt_linears():
            w, b = saved[n]
            self.assertTrue(np.array_equal(layer.W, w))
            self.assertTrue(np.array_equal(layer.b, b))

    def test_restore_unknown_k_raises(self):
        cfg = GPTConfig()
        model, _ = build_pair(cfg, 13)
        con = Consolidator(model.kt_linears())
        with self.assertRaises(KeyError):
            con.snapshots.restore(99, model.kt_linears())


class TestTwinHarness(unittest.TestCase):
    def test_same_seed_identical_init_and_forward(self):
        cfg = GPTConfig()
        kt, vn = build_pair(cfg, 42)
        self.assertTrue(check_init_parity(kt, vn))
        batches = deterministic_batches(99, 2, 2, 8, cfg.vocab_size)
        for idx in batches:
            self.assertTrue(check_forward_parity(kt, vn, idx))

    def test_different_seed_differs(self):
        cfg = GPTConfig()
        kt, vn = build_pair(cfg, 42)
        _kt2, vn2 = build_pair(cfg, 43)
        self.assertFalse(check_init_parity(kt, vn2))
        self.assertFalse(np.array_equal(kt.tok_emb, vn2.tok_emb))


class TestTransformerRuns(unittest.TestCase):
    def test_forward_shape_and_determinism(self):
        cfg = GPTConfig()
        model, _ = build_pair(cfg, 21)
        idx = deterministic_batches(5, 1, 2, 8, cfg.vocab_size)[0]
        out = model.forward(idx)
        self.assertEqual(out.shape, (2, 8, cfg.vocab_size))
        self.assertTrue(np.array_equal(out, model.forward(idx)))

    def test_param_count_default_config(self):
        # Hand-computed: tok 8192 + pos 2048 + 2 blocks x 49984
        # + ln_f 128 + lm_head 8320 = 118656.
        cfg = GPTConfig()
        model, _ = build_pair(cfg, 21)
        self.assertEqual(model.param_count(), 118656)


class TestGenericSeatingPolicy(unittest.TestCase):
    """The placeholder policy (Darren, Oct 26): basic, generic, labeled.

    Mean of declared source authorities; +0.1 for multi-source
    agreement; -0.3 for a known-bad source in the pool; cap 0.9,
    floor 0.0, 0.1 grid. ARCHE's feedback-controller seating replaces
    this module when it arrives.
    """

    def test_empty_pool_seats_at_birth_authority(self):
        self.assertEqual(seat_authority({}), 0.5)

    def test_single_source_seats_at_its_own_authority(self):
        self.assertEqual(seat_authority({"sources": {"textbook": 0.8}}), 0.8)

    def test_multi_source_agreement_adds_bonus(self):
        self.assertEqual(
            seat_authority({"sources": {"a": 0.8, "b": 0.6}}), 0.8)

    def test_agreement_bonus_capped(self):
        self.assertEqual(
            seat_authority({"sources": {"a": 0.9, "b": 0.9}}), 0.9)

    def test_known_bad_source_docks(self):
        self.assertEqual(
            seat_authority({"sources": {"a": 0.8}, "bad": ["a"]}), 0.5)

    def test_bad_source_docks_multi_source_pool(self):
        self.assertEqual(
            seat_authority(
                {"sources": {"a": 0.8, "b": 0.6}, "bad": ["b"]}), 0.5)

    def test_dock_floors_at_zero(self):
        self.assertEqual(
            seat_authority({"sources": {"a": 0.2}, "bad": ["a"]}), 0.0)

    def test_authority_quantized_to_grid(self):
        self.assertEqual(seat_authority({"sources": {"a": 0.83}}), 0.8)

    def test_consolidation_sets_gains_to_seated_authority(self):
        layer = KTLinear(6, 4, _rng(7))
        con = Consolidator([("fc", layer)],
                           seating_fn=lambda pool: 0.8)
        con.consolidate()
        self.assertTrue(np.all(layer.gain == 0.8))

    def test_consolidation_with_placeholder_policy_sets_birth_gains(self):
        layer = KTLinear(6, 4, _rng(7))
        con = Consolidator([("fc", layer)])
        con.consolidate()
        self.assertTrue(np.all(layer.gain == 0.5))


if __name__ == "__main__":
    unittest.main()
