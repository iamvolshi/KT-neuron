"""Tests for kt_bench.tail (R3-E2 tail deployments). Pure arithmetic:
no training, no model, no seating -- so no gradient gate is owed
by this module, and the run script's gate covers its training path.
"""

import unittest

from kt_bench.tail import (FAST_TAIL, PERMANENT_TAIL, SLOW_TAIL,
                           TailDeployment)


class TestTailDeployments(unittest.TestCase):
    def test_fast_trajectory_to_plus6(self):
        self.assertAlmostEqual(FAST_TAIL.deficit_at(0), 0.30)
        self.assertAlmostEqual(FAST_TAIL.deficit_at(1), 0.25)
        self.assertAlmostEqual(FAST_TAIL.deficit_at(5), 0.05)
        self.assertAlmostEqual(FAST_TAIL.deficit_at(6), 0.00)

    def test_slow_trajectory_to_plus6(self):
        self.assertAlmostEqual(SLOW_TAIL.deficit_at(0), 0.30)
        self.assertAlmostEqual(SLOW_TAIL.deficit_at(6), 0.24)
        self.assertAlmostEqual(SLOW_TAIL.deficit_at(30), 0.00)

    def test_permanent_never_moves(self):
        for n in (0, 1, 6, 30, 1000):
            self.assertAlmostEqual(PERMANENT_TAIL.deficit_at(n), 0.30)

    def test_healed_definition_boundary(self):
        # Healed is <= 0.05: FAST is healed at +5 (0.05 exactly),
        # not at +4 (0.10).
        self.assertFalse(FAST_TAIL.is_healed(4))
        self.assertTrue(FAST_TAIL.is_healed(5))
        self.assertTrue(FAST_TAIL.is_healed(6))
        self.assertFalse(SLOW_TAIL.is_healed(6))
        self.assertTrue(SLOW_TAIL.is_healed(25))
        self.assertFalse(SLOW_TAIL.is_healed(24))
        self.assertFalse(PERMANENT_TAIL.is_healed(30))

    def test_time_to_healed_ordering(self):
        fast = FAST_TAIL.time_to_healed()
        slow = SLOW_TAIL.time_to_healed()
        perm = PERMANENT_TAIL.time_to_healed()
        self.assertEqual(fast, 5)
        self.assertEqual(slow, 25)
        self.assertIsNone(perm)
        self.assertLess(fast, slow)

    def test_deficit_never_negative(self):
        self.assertEqual(FAST_TAIL.deficit_at(100), 0.0)
        self.assertEqual(SLOW_TAIL.deficit_at(100), 0.0)

    def test_series_monotone_nonincreasing(self):
        for tail in (FAST_TAIL, SLOW_TAIL, PERMANENT_TAIL):
            series = tail.deficit_series(30)
            self.assertEqual(len(series), 31)
            for a, b in zip(series, series[1:]):
                self.assertGreaterEqual(a, b)

    def test_negative_decay_rejected(self):
        with self.assertRaises(ValueError):
            TailDeployment(name="BAD", decay_per_consolidation=-0.01)

    def test_negative_n_rejected(self):
        with self.assertRaises(ValueError):
            FAST_TAIL.deficit_at(-1)

    def test_zero_initial_healed_immediately(self):
        tail = TailDeployment(name="NONE", decay_per_consolidation=0.0,
                              initial_deficit=0.0)
        self.assertTrue(tail.is_healed(0))
        self.assertEqual(tail.time_to_healed(), 0)

    def test_time_to_healed_respects_cap(self):
        # SLOW needs 25; a cap below that yields None, not a wrong time.
        self.assertIsNone(SLOW_TAIL.time_to_healed(n_cap=10))
        self.assertEqual(SLOW_TAIL.time_to_healed(n_cap=25), 25)


if __name__ == "__main__":
    unittest.main()
