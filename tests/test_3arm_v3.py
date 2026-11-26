"""Tests for kt_bench.handoff_perrow (3-arm v3).

Run from ~/workspace/kt-neuron:
    python3 -m unittest discover -s tests -v

Frozen contract: experiments/preregistration-3arm-v3-2026-10-28.md.
THE ONE CHANGE vs v2: the fast trace is PER ROW:
F_row = F_global * s_row, where s_row is the row's cumulative
gradient-exposure share within its layer (its own signal, never a
global clock). Crossing is per row: gate S > 0.5 + upward crossing
of the row's OWN trace prev_F_row < S <= new_F_row. Latch permanent,
latched rows' Adam step size x current S, gradients NOT scaled.
"""

import unittest

import numpy as np

from kt_bench.handoff_perrow import AUTHORITY_GATE, PerRowHandoffTracker


class StubLayer:
    """Minimal layer stand-in: only .gain and .out_features are read
    by PerRowHandoffTracker (accumulate reads grads, not weights)."""

    def __init__(self, out_features, gain_value):
        self.out_features = out_features
        self.gain = np.full(out_features, gain_value)


def grads_for(prefix, dW_rows, db):
    """Build a grads dict for one layer: dW_rows is a list of row
    vectors (norms set the exposure), db a vector."""
    return {f"{prefix}.W": np.array(dW_rows, dtype=float),
            f"{prefix}.b": np.array(db, dtype=float)}


class TestShare(unittest.TestCase):
    def test_share_all_one_before_any_gradient(self):
        tr = PerRowHandoffTracker([("lm_head", StubLayer(4, 0.8))])
        np.testing.assert_array_equal(tr.share("lm_head"), np.ones(4))

    def test_share_mean_is_one_and_tracks_exposure(self):
        tr = PerRowHandoffTracker([("lm_head", StubLayer(4, 0.8))])
        # Row norms 4, 2, 1, 1 (db 0): mean 2 -> shares 2, 1, .5, .5.
        g = grads_for("lm_head", [[4, 0], [2, 0], [1, 0], [1, 0]],
                      [0, 0, 0, 0])
        tr.accumulate(g, {"lm_head": "lm_head"})
        s = tr.share("lm_head")
        np.testing.assert_allclose(s, [2.0, 1.0, 0.5, 0.5])
        self.assertAlmostEqual(float(s.mean()), 1.0)

    def test_share_normalized_per_layer_not_globally(self):
        # Two layers with very different gradient scales: each
        # layer's shares still mean 1.0 within that layer.
        tr = PerRowHandoffTracker([("a", StubLayer(2, 0.8)),
                                   ("b", StubLayer(2, 0.8))])
        g = {"a.W": np.array([[100.0, 0], [0.0, 0]]),
             "a.b": np.array([0.0, 0.0]),
             "b.W": np.array([[0.01, 0], [0.03, 0]]),
             "b.b": np.array([0.0, 0.0])}
        tr.accumulate(g, {"a": "a", "b": "b"})
        np.testing.assert_allclose(tr.share("a"), [2.0, 0.0])
        np.testing.assert_allclose(tr.share("b"), [0.5, 1.5])

    def test_accumulate_skips_missing_keys_without_crashing(self):
        tr = PerRowHandoffTracker([("lm_head", StubLayer(2, 0.8))])
        tr.accumulate({"lm_head.W": np.ones((2, 3))},
                      {"lm_head": "lm_head"})
        np.testing.assert_array_equal(tr.share("lm_head"), np.ones(2))

    def test_exposure_never_resets_across_phases(self):
        tr = PerRowHandoffTracker([("lm_head", StubLayer(2, 0.8))])
        g = grads_for("lm_head", [[4, 0], [0, 0]], [0, 0])
        tr.accumulate(g, {"lm_head": "lm_head"})
        before = tr.share("lm_head").copy()
        tr.set_phase("restore1")  # only the step counter resets
        np.testing.assert_array_equal(tr.share("lm_head"), before)
        self.assertEqual(tr.step_in_phase, 0)


class TestPerRowCrossing(unittest.TestCase):
    def test_uniform_gradients_reduce_exactly_to_v2(self):
        # Identical gradient histories -> s = 1 everywhere -> the
        # v2 global rule: all rows cross together at F = S.
        tr = PerRowHandoffTracker([("lm_head", StubLayer(4, 0.8))])
        tr.set_phase("poison1")
        g = grads_for("lm_head", [[1, 0]] * 4, [0, 0, 0, 0])
        tr.accumulate(g, {"lm_head": "lm_head"})
        self.assertEqual(tr.check(0.5, 0.79), [])
        newly = tr.check(0.79, 0.8)
        self.assertEqual(newly, [("lm_head", 0), ("lm_head", 1),
                                 ("lm_head", 2), ("lm_head", 3)])
        self.assertEqual(tr.latched_total(), 4)

    def test_differentiated_rows_cross_at_different_steps(self):
        # C = [8, 6, 2, 2], mean 4.5 -> s = [1.778, 1.333, .444, .444].
        # S = 0.8: row 0 crosses when F >= 0.45, row 1 when F >= 0.6,
        # rows 2-3 never (peak F_row 1.2 * .444 = 0.533 < 0.8).
        tr = PerRowHandoffTracker([("lm_head", StubLayer(4, 0.8))])
        tr.set_phase("poison1")
        g = grads_for("lm_head", [[8, 0], [6, 0], [2, 0], [2, 0]],
                      [0, 0, 0, 0])
        tr.accumulate(g, {"lm_head": "lm_head"})
        self.assertEqual(tr.check(0.0, 0.4), [])  # row0: 0.711 < 0.8
        newly = tr.check(0.4, 0.5)  # row0: 0.889 >= 0.8
        self.assertEqual(newly, [("lm_head", 0)])
        newly = tr.check(0.5, 0.7)  # row1: 0.933 >= 0.8
        self.assertEqual(newly, [("lm_head", 1)])
        self.assertEqual(tr.check(0.7, 1.2), [])  # rows 2-3: 0.533 max
        self.assertEqual(tr.latched_total(), 2)
        # Distinct latch steps: row 0 at step 1, row 1 at step 2.
        steps = {(e["layer"], e["row"]): e["step_in_phase"]
                 for e in tr.events}
        self.assertEqual(steps[("lm_head", 0)], 1)
        self.assertEqual(steps[("lm_head", 1)], 2)

    def test_gate_blocks_weak_authority(self):
        # S = 0.5 is NOT > 0.5: a birth/candidate row never latches,
        # even when its own trace crosses its level.
        tr = PerRowHandoffTracker([("lm_head", StubLayer(2, 0.5))])
        tr.set_phase("poison1")
        g = grads_for("lm_head", [[5, 0], [5, 0]], [0, 0])
        tr.accumulate(g, {"lm_head": "lm_head"})
        self.assertEqual(tr.check(0.4, 0.6), [])
        self.assertEqual(tr.latched_total(), 0)
        self.assertEqual(AUTHORITY_GATE, 0.5)

    def test_downward_move_does_not_latch(self):
        tr = PerRowHandoffTracker([("lm_head", StubLayer(2, 0.8))])
        tr.set_phase("restore1")
        g = grads_for("lm_head", [[1, 0], [1, 0]], [0, 0])
        tr.accumulate(g, {"lm_head": "lm_head"})
        # F falling from 1.2 to 0.2 passes THROUGH 0.8 downward: not
        # an upward crossing of the row's own trace.
        self.assertEqual(tr.check(1.2, 0.2), [])
        self.assertEqual(tr.latched_total(), 0)

    def test_latch_permanent_when_F_falls(self):
        tr = PerRowHandoffTracker([("lm_head", StubLayer(2, 0.8))])
        tr.set_phase("poison1")
        g = grads_for("lm_head", [[1, 0], [1, 0]], [0, 0])
        tr.accumulate(g, {"lm_head": "lm_head"})
        tr.check(0.79, 0.8)
        tr.set_phase("restore1")
        tr.check(1.2, 0.2)
        self.assertEqual(tr.latched_total(), 2)

    def test_no_retroactive_latch_on_S_drop_then_fresh_window(self):
        # Row trace 0.85 sits below S = 0.9 (no latch). Consolidation
        # drops S to 0.6: the row must NOT latch retroactively, even
        # though its stored trace (0.85) now exceeds S. After the
        # falling restore check washes prev down to 0.5, a fresh
        # upward crossing in the next rise DOES latch it.
        layer = StubLayer(1, 0.9)
        tr = PerRowHandoffTracker([("lm_head", layer)])
        tr.set_phase("poison1")
        g = grads_for("lm_head", [[1, 0]], [0])
        tr.accumulate(g, {"lm_head": "lm_head"})
        self.assertEqual(tr.check(0.0, 0.85), [])
        layer.gain[:] = 0.6  # restore-1 consolidation + dock
        tr.set_phase("restore1")
        self.assertEqual(tr.check(0.85, 0.5), [])  # no retro latch
        self.assertEqual(tr.latched_total(), 0)
        tr.set_phase("poison2")
        newly = tr.check(0.5, 0.7)  # prev 0.5 < 0.6 <= 0.7
        self.assertEqual(newly, [("lm_head", 0)])

    def test_event_carries_per_row_and_global_values(self):
        tr = PerRowHandoffTracker([("lm_head", StubLayer(1, 0.8))])
        tr.set_phase("poison1")
        g = grads_for("lm_head", [[2, 0]], [0])
        tr.accumulate(g, {"lm_head": "lm_head"})
        # Single row: s = 1, so F_row == F_global here; the fields
        # are still recorded separately. A prior check first sets the
        # stored per-row prev to 0.79 (prev_F_row is stored state,
        # not the passed global prev).
        tr.check(0.0, 0.79)
        tr.check(0.79, 0.8)
        e = tr.events[0]
        self.assertEqual((e["F_row_prev"], e["F_row_new"]), (0.79, 0.8))
        self.assertEqual((e["F_global_prev"], e["F_global_new"]),
                         (0.79, 0.8))
        self.assertAlmostEqual(e["s_row"], 1.0)
        self.assertAlmostEqual(e["S"], 0.8)


class TestStepScale(unittest.TestCase):
    def test_latched_row_scales_by_current_S_unlatched_by_one(self):
        tr = PerRowHandoffTracker([("lm_head", StubLayer(4, 0.6))])
        tr.set_phase("poison1")
        g = grads_for("lm_head", [[1, 0]] * 4, [0, 0, 0, 0])
        tr.accumulate(g, {"lm_head": "lm_head"})
        tr.check(0.5, 0.7)  # all rows latch (S=0.6, gate passes)
        scales = tr.lr_scale_list(["tok_emb", "lm_head.W", "lm_head.b"],
                                  {"lm_head": "lm_head"})
        self.assertEqual(scales[0], 1.0)
        np.testing.assert_array_equal(scales[1],
                                      np.full((4, 1), 0.6))
        np.testing.assert_array_equal(scales[2], np.full(4, 0.6))

    def test_partial_latch_scales_only_latched_rows(self):
        tr = PerRowHandoffTracker([("lm_head", StubLayer(4, 0.8))])
        tr.set_phase("poison1")
        g = grads_for("lm_head", [[8, 0], [6, 0], [2, 0], [2, 0]],
                      [0, 0, 0, 0])
        tr.accumulate(g, {"lm_head": "lm_head"})
        tr.check(0.0, 0.4)
        tr.check(0.4, 0.5)  # only row 0 latched
        scales = tr.lr_scale_list(["lm_head.W", "lm_head.b"],
                                  {"lm_head": "lm_head"})
        np.testing.assert_array_equal(scales[0],
                                      np.array([[0.8], [1.0], [1.0],
                                                [1.0]]))
        np.testing.assert_array_equal(scales[1],
                                      np.array([0.8, 1.0, 1.0, 1.0]))


class TestThreeNumbersAndStats(unittest.TestCase):
    def test_snapshot_reports_per_row_trace_separately(self):
        layer = StubLayer(2, 0.6)
        tr = PerRowHandoffTracker([("lm_head", layer)])
        tr.set_baseline(0.9)  # repair claim; S=0.6 -> depth 0.3
        # Shares [1.5, 0.5]: F_row at F_global 0.2 -> [0.3, 0.1].
        g = grads_for("lm_head", [[3, 0], [1, 0]], [0, 0])
        tr.accumulate(g, {"lm_head": "lm_head"})
        snap = tr.snapshot(0.2, [("lm_head", 0), ("lm_head", 1)])
        r0, r1 = snap["per_row"]
        self.assertAlmostEqual(r0["F_row"], 0.3)
        self.assertAlmostEqual(r1["F_row"], 0.1)
        self.assertEqual((r0["F_global"], r0["B"], r0["S"]),
                         (0.2, 0.9, 0.6))
        self.assertAlmostEqual(r0["F_row_minus_B"], 0.3 - 0.9)
        self.assertAlmostEqual(r0["B_minus_S"], 0.9 - 0.6)
        self.assertAlmostEqual(r0["F_row_minus_S"], 0.3 - 0.6)
        self.assertAlmostEqual(snap["means"]["B_minus_S"], 0.3)
        self.assertAlmostEqual(snap["means"]["F_row"], 0.2)

    def test_latch_stats_counts_never_latched_and_phases(self):
        tr = PerRowHandoffTracker([("lm_head", StubLayer(4, 0.8))])
        tr.set_phase("poison1")
        g = grads_for("lm_head", [[8, 0], [6, 0], [2, 0], [2, 0]],
                      [0, 0, 0, 0])
        tr.accumulate(g, {"lm_head": "lm_head"})
        tr.check(0.0, 0.4)
        tr.check(0.4, 0.5)
        tr.check(0.5, 0.7)
        stats = tr.latch_stats()
        self.assertEqual(stats["total_rows"], 4)
        self.assertEqual(stats["latched_total"], 2)
        self.assertEqual(stats["never_latched"], 2)
        self.assertEqual(stats["by_phase"]["poison1"]["count"], 2)
        self.assertEqual(stats["by_phase"]["poison1"]["distinct_steps"], 2)
        self.assertEqual(stats["by_phase"]["poison1"]["min_step"], 1)
        self.assertEqual(stats["by_phase"]["poison1"]["max_step"], 2)

    def test_latch_step_of_returns_none_when_unlatched(self):
        tr = PerRowHandoffTracker([("lm_head", StubLayer(2, 0.8))])
        self.assertIsNone(tr.latch_step_of("lm_head", 0))
        tr.set_phase("poison1")
        g = grads_for("lm_head", [[1, 0], [1, 0]], [0, 0])
        tr.accumulate(g, {"lm_head": "lm_head"})
        tr.check(0.79, 0.8)
        self.assertEqual(tr.latch_step_of("lm_head", 1),
                         ("poison1", 0))


if __name__ == "__main__":
    unittest.main()
