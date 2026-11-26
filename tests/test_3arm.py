"""Tests for kt_bench.scar (3-arm experiment) and kt_bench.mock_rk.

Run from ~/workspace/kt-neuron:
    python3 -m unittest discover -s tests -v
"""

import unittest

import numpy as np

from kt_bench.linear import Linear
from kt_bench.mock_rk import MockRK
from kt_bench.scar import (PressureDriver, ScarLinear, scar_shape)


class TestScarShape(unittest.TestCase):
    def test_landmarks(self):
        self.assertEqual(scar_shape(0.0), 0.0)
        self.assertEqual(scar_shape(0.30), 0.0)
        self.assertEqual(scar_shape(0.29), 0.0)
        self.assertAlmostEqual(scar_shape(0.75), 1.0)
        self.assertAlmostEqual(scar_shape(0.525), 0.5)
        self.assertAlmostEqual(scar_shape(1.0), 0.5)

    def test_rising_limb_monotone(self):
        xs = [0.31, 0.40, 0.50, 0.60, 0.75]
        vals = [scar_shape(x) for x in xs]
        self.assertEqual(vals, sorted(vals))


class TestPressureDriver(unittest.TestCase):
    def test_frozen_trajectory(self):
        d = PressureDriver()
        self.assertEqual(d.scar, 0.0)
        for _ in range(200):
            d.poison_step()
        self.assertAlmostEqual(d.pressure, 0.75)
        self.assertAlmostEqual(d.scar, 1.2)
        for _ in range(200):
            d.restore_step()
        self.assertAlmostEqual(d.pressure, 0.375)
        self.assertAlmostEqual(d.scar, 0.2)
        for _ in range(200):
            d.poison_step()
        self.assertAlmostEqual(d.pressure, 0.75)
        self.assertAlmostEqual(d.scar, 1.2)
        for _ in range(200):
            d.restore_step()
        self.assertAlmostEqual(d.pressure, 0.375)
        self.assertAlmostEqual(d.scar, 0.2)

    def test_caps(self):
        d = PressureDriver()
        for _ in range(1000):
            d.poison_step()
        self.assertEqual(d.pressure, 0.75)
        for _ in range(1000):
            d.restore_step()
        self.assertEqual(d.pressure, 0.0)


class TestScarLinear(unittest.TestCase):
    def test_forward_matches_linear_at_zero_scar(self):
        rng = np.random.default_rng(0)
        plain = Linear(4, 3, rng)
        rng = np.random.default_rng(0)
        scar = ScarLinear(4, 3, rng)
        x = rng.normal(size=(2, 4))
        self.assertTrue(np.array_equal(plain.forward(x), scar.forward(x)))

    def test_scar_shifts_bias(self):
        rng = np.random.default_rng(1)
        layer = ScarLinear(4, 3, rng)
        layer.scar[:] = 0.5
        x = rng.normal(size=(2, 4))
        base = x @ layer.W.T + layer.b
        self.assertTrue(np.allclose(layer.forward(x), base - 0.5))


class TestMockRK(unittest.TestCase):
    def test_pre_poison_no_offer(self):
        rk = MockRK()
        e = rk.select("fact-0", 3, 0.8, 3)
        self.assertEqual(e["selected_token"], 3)
        self.assertEqual(e["winner"], "teacher")
        self.assertFalse(e["overridden"])
        self.assertTrue(e["MOCK"] and e["DERIVED"])
        self.assertEqual(len(e["candidates"]), 1)

    def test_unflagged_liar_beats_standing(self):
        rk = MockRK()
        rk.activate_liar()
        e = rk.select("fact-0", 3, 0.8, 9, liar_token=9)
        self.assertEqual(e["selected_token"], 9)
        self.assertEqual(e["winner"], "liar")
        self.assertEqual(e["candidates"][0]["status"], "candidate")
        self.assertEqual(e["candidates"][1]["status"], "selected")

    def test_flagged_liar_ties_incumbent_wins(self):
        rk = MockRK()
        rk.activate_liar()
        rk.flag_liar_known_bad()
        e = rk.select("fact-0", 3, 0.6, 9, liar_token=9)
        self.assertEqual(e["selected_token"], 3)
        self.assertEqual(e["winner"], "teacher")
        self.assertEqual(e["candidates"][1]["authority"], 0.6)
        self.assertEqual(e["candidates"][1]["status"], "candidate")

    def test_weaker_liar_loses(self):
        rk = MockRK()
        rk.activate_liar()
        rk.flag_liar_known_bad()
        e = rk.select("fact-0", 3, 0.8, 3, liar_token=9)
        self.assertEqual(e["selected_token"], 3)

    def test_override_flag_and_counts(self):
        rk = MockRK()
        rk.activate_liar()
        rk.select("fact-0", 3, 0.8, 9, liar_token=9)  # liar wins, raw was liar: no override
        rk.flag_liar_known_bad()
        e = rk.select("fact-1", 4, 0.6, 10, liar_token=10)  # teacher wins, raw was liar: override
        self.assertTrue(e["overridden"])
        t, li = rk.selected_counts()
        self.assertEqual((t, li), (1, 1))


if __name__ == "__main__":
    unittest.main()
