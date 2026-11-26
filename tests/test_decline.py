"""Tests for the support-aware DECLINE rule (stage 4).

Rule (pre-registered in experiments/preregistration-decline-
2026-10-26.md): DECLINE iff the tainted set is non-empty AND
(fraction_all > 0.5 OR the pre-null winner's lm_head row is
tainted) AND the post-null top-1 confidence <= 0.5. Reason:
"quarantined-support" / "floor-breach" / "both".

These unit tests use a real small KT GPT whose lm_head weights
are zeroed, so logits are (bias - scar) * gain and the winner /
confidence are controlled exactly through biases and gains.
They test the RULE's logic, not bench training physics; the
trained-probe behavior is measured by experiments/run_decline.py.

Run from ~/workspace/kt-neuron:

    python3 -m unittest discover -s tests -v
"""

import unittest

import numpy as np

from kt_bench.decline import (AUTHORITY_FLOOR, FALLBACK_CONF,
                              SUPPORT_THRESHOLD, infer_with_decline,
                              tainted_sets)
from kt_bench.linear import KTLinear, Linear, row_hash
from kt_bench.transformer import GPT, GPTConfig

PREFIX = [1, 2, 3, 4, 0, 1]


def _kt_model():
    cfg = GPTConfig(vocab_size=8, block_size=16, n_embd=8, n_head=2, n_layer=1)
    model = GPT(cfg, np.random.default_rng(0), KTLinear)
    # Zero lm_head weights: logits become bias * gain, exactly.
    model.lm_head.W[:] = 0.0
    model.lm_head.scar[:] = 0.0
    return model


def _set_biases(model, biases):
    model.lm_head.b[:] = np.asarray(biases, dtype=float)


def _seat_low(model, row):
    """Gate-honest re-seat below the floor: claim 0.5, dock 0.3 -> 0.2.

    Birth authority is 0.5 from source "birth"; an equal claim from
    a different source re-seats, and the dock subtracts after.
    """
    layer = model.lm_head
    rec = layer.records[row]
    status, _r = rec.replace("row", row_hash(layer.W[row], layer.b[row]),
                             0.5, "test-low", authority_dock=0.3)
    assert status == "seated", status
    assert rec.standing("row")["authority"] == 0.2


class TestDeclineConstants(unittest.TestCase):
    def test_frozen_numbers(self):
        self.assertEqual(AUTHORITY_FLOOR, 0.3)
        self.assertEqual(SUPPORT_THRESHOLD, 0.5)
        self.assertEqual(FALLBACK_CONF, 0.5)


class TestDeclineRule(unittest.TestCase):
    def test_direct_quarantine_declines_with_reason(self):
        model = _kt_model()
        _set_biases(model, [0, 0, 0, 5, 0, 0, 0, 0])
        model.lm_head.gain[:] = 1.0
        model.lm_head.gain[3] = 0.0  # incoming post-null state
        res = infer_with_decline(model, PREFIX, {("lm_head", 3)})
        self.assertEqual(res.status, "DECLINE")
        self.assertIsNone(res.answer)  # decline returns NO token
        self.assertEqual(res.reason, "quarantined-support")
        self.assertEqual(res.top1_pre, 3)
        self.assertGreater(res.support_fraction, 0.5)
        self.assertLessEqual(res.confidence_post, 0.5)

    def test_floor_breach_declines(self):
        model = _kt_model()
        _set_biases(model, [0, 0, 0, 5, 0, 0, 0, 0])
        _seat_low(model, 3)
        model.lm_head.gain[:] = 1.0
        model.lm_head.gain[3] = 0.2  # standing gain, weak
        res = infer_with_decline(model, PREFIX, set())
        self.assertEqual(res.status, "DECLINE")
        self.assertIsNone(res.answer)
        self.assertEqual(res.reason, "floor-breach")
        self.assertGreater(res.support_fraction, 0.5)

    def test_both_reason_when_quarantined_and_under_floor(self):
        model = _kt_model()
        _set_biases(model, [0, 0, 0, 5, 0, 0, 0, 0])
        _seat_low(model, 3)
        model.lm_head.gain[:] = 1.0
        model.lm_head.gain[3] = 0.0
        res = infer_with_decline(model, PREFIX, {("lm_head", 3)})
        self.assertEqual(res.status, "DECLINE")
        self.assertEqual(res.reason, "both")

    def test_clean_fallback_answers_not_decline(self):
        # Pre-null winner is quarantined (bias 10), but once it is
        # nulled a clean answer (bias 4) emerges at high confidence.
        # Mixed support with a clean fallback must ANSWER, not DECLINE.
        model = _kt_model()
        _set_biases(model, [0, 0, 0, 10, 4, 0, 0, 0])
        model.lm_head.gain[:] = 1.0
        model.lm_head.gain[3] = 0.0
        res = infer_with_decline(model, PREFIX, {("lm_head", 3)})
        self.assertEqual(res.status, "ANSWER")
        self.assertEqual(res.answer, 4)
        self.assertEqual(res.top1_pre, 3)
        self.assertGreater(res.support_fraction, 0.5)
        self.assertGreater(res.confidence_post, 0.5)
        self.assertIsNone(res.reason)

    def test_empty_tainted_set_answers(self):
        model = _kt_model()
        _set_biases(model, [0, 0, 0, 5, 0, 0, 0, 0])
        model.lm_head.gain[:] = 1.0
        res = infer_with_decline(model, PREFIX, set())
        self.assertEqual(res.status, "ANSWER")
        self.assertEqual(res.answer, 3)
        self.assertEqual(res.support_fraction, 0.0)
        self.assertIsNone(res.reason)

    def test_hedge_is_answer_not_decline(self):
        # Low confidence with NO tainted support is a hedge (an
        # ANSWER with low confidence), never a DECLINE.
        model = _kt_model()
        _set_biases(model, [0, 0, 0, 0, 0, 0, 0, 0])
        model.lm_head.gain[:] = 1.0
        res = infer_with_decline(model, PREFIX, set())
        self.assertEqual(res.status, "ANSWER")
        self.assertIsNotNone(res.answer)
        self.assertLessEqual(res.confidence_post, 0.5)

    def test_vanilla_never_declines(self):
        cfg = GPTConfig(vocab_size=8, block_size=16, n_embd=8, n_head=2, n_layer=1)
        model = GPT(cfg, np.random.default_rng(0), Linear)
        model.lm_head.W[:] = 0.0
        model.lm_head.b[:] = np.asarray([0, 0, 0, 5, 0, 0, 0, 0], dtype=float)
        res = infer_with_decline(model, PREFIX, {("lm_head", 3)})
        self.assertEqual(res.status, "ANSWER")
        self.assertEqual(res.answer, 3)
        self.assertEqual(res.support_fraction, 0.0)
        self.assertIsNone(res.reason)
        self.assertIn("no-records", res.note)

    def test_gains_restored_bit_exactly(self):
        model = _kt_model()
        _set_biases(model, [0, 0, 0, 5, 0, 0, 0, 0])
        model.lm_head.gain[:] = 1.0
        model.lm_head.gain[3] = 0.0
        before = {n: l.gain.copy() for n, l in model.kt_linears()}
        infer_with_decline(model, PREFIX, {("lm_head", 3)})
        for n, layer in model.kt_linears():
            self.assertTrue(np.array_equal(layer.gain, before[n]))

    def test_tainted_sets_ignore_nonexistent_rows(self):
        model = _kt_model()
        tainted, quarantined, floor_rows = tainted_sets(
            model, {("lm_head", 3), ("nope", 99)})
        self.assertEqual(quarantined, [("lm_head", 3)])
        self.assertEqual(floor_rows, [])
        self.assertEqual(tainted, [("lm_head", 3)])


if __name__ == "__main__":
    unittest.main()
