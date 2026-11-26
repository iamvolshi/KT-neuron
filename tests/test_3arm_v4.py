"""Tests for the v4 meter (3-arm v4).

Run from ~/workspace/kt-neuron:
    python3 -m unittest discover -s tests -v

Frozen contract: experiments/preregistration-3arm-v4-2026-10-28.md
(including its dated pre-pilot amendments). These tests cover
only the meter's new pieces: the FACTS_V4 corpus extension
(additive, defaults unchanged), the fast-trace-baseline fall
math, the latched-fact-row count, and the trajectory-probe
recording step rule. They do NOT re-test the v2 mechanism
(handoff, drivers, seating), which test_3arm_v2.py covers.
"""

import unittest

import numpy as np

from kt_bench.linear import Linear
from kt_bench.transformer import GPT, GPTConfig
from experiments.corpus import (FACTS, FACTS_V4, LIAR_ONLY,
                                gen_clean_batch, gen_poison_batch,
                                probe_all_facts, summarize_probes)
from experiments.run_3arm_v4 import (CFG, count_latched_fact_rows,
                                     fasttrace_falls, run_phase_with_traj)


class TestCorpusV4(unittest.TestCase):
    def test_facts_untouched(self):
        self.assertEqual(FACTS, [(20, 21, 3, 9), (22, 23, 4, 10),
                                 (24, 25, 5, 11), (26, 27, 6, 12)])
        self.assertEqual(LIAR_ONLY, (28, 29, 13))

    def test_facts_v4_additive_and_disjoint(self):
        self.assertEqual(FACTS_V4[:4], FACTS)
        self.assertEqual(len(FACTS_V4), 7)
        contexts = []
        tokens = []
        for (a, b, t, li) in FACTS_V4:
            contexts.extend([a, b])
            tokens.extend([t, li])
        # All context pairs disjoint; no token collides with any
        # context token or any other fact's token.
        self.assertEqual(len(set(contexts)), 14)
        self.assertEqual(len(set(tokens)), 14)
        self.assertFalse(set(contexts) & set(tokens))
        self.assertNotIn(LIAR_ONLY[2], tokens)

    def test_defaults_unchanged(self):
        # Default gen_* calls equal explicit FACTS calls (same rng
        # seed -> identical batches): old callers are bit-identical.
        rng1 = np.random.default_rng(7)
        rng2 = np.random.default_rng(7)
        self.assertTrue(np.array_equal(
            gen_clean_batch(rng1), gen_clean_batch(rng2, facts=FACTS)))
        rng1 = np.random.default_rng(7)
        rng2 = np.random.default_rng(7)
        self.assertTrue(np.array_equal(
            gen_poison_batch(rng1), gen_poison_batch(rng2, facts=FACTS)))

    def test_v4_batches_same_shape(self):
        rng = np.random.default_rng(7)
        b = gen_clean_batch(rng, facts=FACTS_V4)
        self.assertEqual(b.shape, (8, 16))
        self.assertTrue(((b >= 0) & (b < 32)).all())
        rng = np.random.default_rng(7)
        b = gen_poison_batch(rng, facts=FACTS_V4)
        self.assertEqual(b.shape, (8, 16))

    def test_probe_v4_seven_facts(self):
        model = GPT(CFG, np.random.default_rng(0), Linear)
        probes = probe_all_facts(model, FACTS_V4)
        self.assertEqual(len(probes), 7)
        s = summarize_probes(probes, FACTS_V4)
        self.assertTrue(0.0 <= s["mean_p_true"] <= 1.0)
        self.assertTrue(0 <= s["top1_true_count"] <= 7)
        self.assertTrue(0 <= s["top1_liar_count"] <= 7)


class TestFasttraceFalls(unittest.TestCase):
    def test_math_and_gap_convention(self):
        base = {"arm1_conventional": 0.99, "arm2_scar_only": 0.98,
                "arm3_full_ktn": 0.97}
        p2 = {"arm1_conventional": 0.30, "arm2_scar_only": 0.35,
              "arm3_full_ktn": 0.45}
        falls, gap, gap2 = fasttrace_falls(base, p2)
        self.assertAlmostEqual(falls["arm1_conventional"], 0.69)
        self.assertAlmostEqual(falls["arm3_full_ktn"], 0.52)
        self.assertAlmostEqual(gap, 0.17)
        self.assertAlmostEqual(gap2, 0.06)

    def test_higher_baseline_gives_larger_falls(self):
        # Same poison-2 trajectory means; a higher (fast-trace)
        # baseline than the recovery window gives larger falls
        # for every arm by exactly the baseline difference.
        p2 = {"arm1_conventional": 0.30, "arm2_scar_only": 0.35,
              "arm3_full_ktn": 0.45}
        window = {"arm1_conventional": 0.90, "arm2_scar_only": 0.89,
                  "arm3_full_ktn": 0.88}
        base = {"arm1_conventional": 0.99, "arm2_scar_only": 0.98,
                "arm3_full_ktn": 0.97}
        falls_ft, _, _ = fasttrace_falls(base, p2)
        falls_w, _, _ = fasttrace_falls(window, p2)
        for arm in p2:
            self.assertAlmostEqual(falls_ft[arm] - falls_w[arm],
                                   base[arm] - window[arm])


class TestLatchedFactRows(unittest.TestCase):
    def test_counts_only_fact_rows(self):
        latched = {"lm_head": np.array([False] * 32)}
        latched["lm_head"][[3, 9, 14]] = True
        rows = [("lm_head", 3), ("lm_head", 4), ("lm_head", 14)]
        self.assertEqual(count_latched_fact_rows(latched, rows), 2)


class TestTrajProbeSteps(unittest.TestCase):
    def test_probes_every_10_steps_arm1_only(self):
        # Arm-1 path needs no driver/tracker; 25 steps -> probes
        # at steps 10 and 20 only, each a 7-fact summary.
        model = GPT(CFG, np.random.default_rng(0), Linear)
        rng = np.random.default_rng(5)
        batches = [gen_clean_batch(rng, facts=FACTS_V4)
                   for _ in range(25)]
        traj = run_phase_with_traj({"arm1_conventional": model},
                                   batches, "restore", {}, None, {})
        pts = traj["arm1_conventional"]
        self.assertEqual([p["step"] for p in pts], [10, 20])
        for p in pts:
            self.assertTrue(0.0 <= p["mean_p_true"] <= 1.0)
            self.assertTrue(0 <= p["top1_true_count"] <= 7)
            self.assertTrue(0 <= p["top1_liar_count"] <= 7)


if __name__ == "__main__":
    unittest.main()
