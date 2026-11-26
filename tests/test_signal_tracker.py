"""Tests for kt_bench.signal (R3-E1, one-signal regulation).

All tests are synthetic-number tests: they feed probe values to the
tracker directly, so they run in milliseconds and do not train.
The live run (experiments/run_r3_e1.py) is the integration proof.

Run from ~/workspace/kt-neuron:
    python3 -m unittest discover -s tests -v
"""

import unittest

from kt_bench.signal import SignalTracker

# Pilot clean-phase trajectory (single-Adam run, seeds init 0 / data 1):
# fast at boundaries 1..4. Used as a fixture, not as a live result.
PILOT_FAST = [0.398654, 0.952743, 0.983077, 0.989586]


class TestSlowMovesOnlyAtConsolidation(unittest.TestCase):
    def test_current_signal_never_moves_slow(self):
        tr = SignalTracker()
        tr.consolidate(0.4, 0.4)
        slow_after = tr.slow
        for live in (0.9, 0.1, 0.99, 0.0):
            tr.current_signal(live)
        self.assertEqual(tr.slow, slow_after)
        self.assertEqual(len(tr.history), 1)

    def test_slow_starts_below_fast(self):
        tr = SignalTracker()
        self.assertEqual(tr.slow, 0.0)
        self.assertLess(tr.slow, PILOT_FAST[0])


class TestAccumulationShape(unittest.TestCase):
    def test_ratchet_never_moves_down(self):
        tr = SignalTracker()
        tr.consolidate(0.9, 0.9)
        slow_high = tr.slow
        tr.consolidate(0.2, 0.2)  # seated value dips: slow must not fall
        self.assertEqual(tr.slow, slow_high)

    def test_non_decreasing_on_pilot_sequence(self):
        tr = SignalTracker()
        slows = [tr.consolidate(f, f)["slow"] for f in PILOT_FAST]
        for a, b in zip(slows, slows[1:]):
            self.assertLessEqual(a, b)

    def test_formula_verbatim(self):
        # slow(k) = slow(k-1) + 0.8 * max(0, seated - slow(k-1))
        tr = SignalTracker(alpha=0.8)
        r1 = tr.consolidate(0.398654, 0.398654)
        self.assertAlmostEqual(r1["slow"], 0.8 * 0.398654, places=6)
        r2 = tr.consolidate(0.952743, 0.952743)
        expect = r1["slow"] + 0.8 * (0.952743 - r1["slow"])
        self.assertAlmostEqual(r2["slow"], expect, places=6)


class TestCrossingAndSwitch(unittest.TestCase):
    def _pilot_tracker(self, **kw):
        tr = SignalTracker(**kw)
        for f in PILOT_FAST:
            tr.consolidate(f, f)
        return tr

    def test_crossing_at_k3_with_frozen_alpha(self):
        tr = self._pilot_tracker()
        self.assertEqual(tr.k_star, 3)
        slows = [r["slow"] for r in tr.history]
        self.assertAlmostEqual(slows[0], 0.318923, places=5)
        self.assertAlmostEqual(slows[1], 0.825979, places=5)
        self.assertAlmostEqual(slows[2], 0.951657, places=5)
        self.assertAlmostEqual(slows[3], 0.982000, places=5)

    def test_single_switch_no_flip_flop(self):
        tr = self._pilot_tracker()
        owners = [r["owner"] for r in tr.history]
        self.assertEqual(owners, ["fast", "fast", "slow", "slow"])
        v = tr.verdicts()
        self.assertTrue(v["b_single_switch"])

    def test_continuity_at_switch_within_noise_band(self):
        tr = self._pilot_tracker()
        v = tr.verdicts()
        self.assertTrue(v["c_pass"])
        self.assertLessEqual(v["c_continuity_at_switch"], 0.05)

    def test_post_handoff_signal_equals_slow(self):
        tr = self._pilot_tracker()
        v = tr.verdicts()
        self.assertTrue(v["d_pass"])
        self.assertEqual(v["d_post_handoff_max_signal_minus_slow"], 0.0)

    def test_all_verdict_legs_pass_on_pilot_sequence(self):
        tr = self._pilot_tracker()
        v = tr.verdicts()
        self.assertTrue(v["a_crossing_in_clean_phase"])
        self.assertTrue(v["b_single_switch"])
        self.assertTrue(v["c_pass"])
        self.assertTrue(v["d_pass"])

    def test_alpha_sensitivity_slow_alpha_fails_to_cross(self):
        # Honest sensitivity note: a textbook slow integrator
        # (alpha 0.5) does NOT cross inside the 4-boundary window
        # on this trajectory -- it ends 0.105 below fast. That is a
        # property of the window, reported not hidden.
        tr = self._pilot_tracker(alpha=0.5)
        self.assertIsNone(tr.k_star)
        self.assertFalse(tr.verdicts()["a_crossing_in_clean_phase"])
        self.assertAlmostEqual(tr.slow, 0.8845, places=3)

    def test_no_crossing_when_fast_keeps_rising(self):
        # Steady large jumps keep the gap above the noise band: the
        # slow trace only catches a fast trace that levels off.
        tr = SignalTracker()
        for f in (0.26, 0.52, 0.78, 0.99):
            tr.consolidate(f, f)
        self.assertIsNone(tr.k_star)
        self.assertEqual([r["owner"] for r in tr.history], ["fast"] * 4)


class TestFlipCorollary(unittest.TestCase):
    def test_frozen_signal_while_fast_moves(self):
        # Post-handoff: signal frozen at slow, fast probe collapses.
        res = SignalTracker.flip_corollary(
            signal_before=0.982000, signal_after=0.982000,
            fast_before=0.989586, fast_after=0.544069)
        self.assertTrue(res["pass"])
        self.assertEqual(res["signal_change"], 0.0)
        self.assertGreaterEqual(res["fast_move"], 0.10)

    def test_fast_move_too_small_fails_corollary(self):
        res = SignalTracker.flip_corollary(
            signal_before=0.98, signal_after=0.98,
            fast_before=0.99, fast_after=0.95)
        self.assertFalse(res["pass"])
        self.assertFalse(res["fast_moved"])

    def test_signal_moving_fails_corollary(self):
        res = SignalTracker.flip_corollary(
            signal_before=0.98, signal_after=0.80,
            fast_before=0.99, fast_after=0.50)
        self.assertFalse(res["pass"])
        self.assertFalse(res["signal_still"])

    def test_post_handoff_current_signal_ignores_live(self):
        tr = SignalTracker()
        for f in PILOT_FAST:
            tr.consolidate(f, f)
        self.assertTrue(tr.handed_off)
        self.assertEqual(tr.current_signal(0.544069), tr.slow)
        self.assertEqual(tr.current_signal(0.10), tr.slow)

    def test_pre_handoff_current_signal_is_live(self):
        tr = SignalTracker()
        tr.consolidate(PILOT_FAST[0], PILOT_FAST[0])
        self.assertFalse(tr.handed_off)
        self.assertEqual(tr.current_signal(0.77), 0.77)


class TestValidation(unittest.TestCase):
    def test_bad_alpha_rejected(self):
        for bad in (0.0, -0.5, 1.5):
            with self.assertRaises(ValueError):
                SignalTracker(alpha=bad)

    def test_negative_noise_band_rejected(self):
        with self.assertRaises(ValueError):
            SignalTracker(noise_band=-0.01)


if __name__ == "__main__":
    unittest.main()
