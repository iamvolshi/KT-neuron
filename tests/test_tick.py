"""Tests for kt_bench.tick (R3-E3 tick schedules). Pure schedule
arithmetic: no training, no model, no seating -- so no gradient
gate is owed by this module, and the run script's gate covers its
training path (which reuses experiments/run_all.train_steps).
"""

import unittest

from kt_bench.tail import SLOW_TAIL
from kt_bench.tick import (BASELINE_TICK, COARSE_TICK, FINE_TICK,
                           TickSchedule)


class TestTickSchedules(unittest.TestCase):
    def test_frozen_schedules(self):
        self.assertEqual((FINE_TICK.clean_steps, FINE_TICK.consol_every,
                          FINE_TICK.n_ticks), (400, 100, 4))
        self.assertEqual((BASELINE_TICK.clean_steps,
                          BASELINE_TICK.consol_every,
                          BASELINE_TICK.n_ticks), (400, 400, 1))
        self.assertEqual((COARSE_TICK.clean_steps, COARSE_TICK.consol_every,
                          COARSE_TICK.n_ticks), (800, 800, 1))

    def test_boundaries(self):
        self.assertEqual(FINE_TICK.boundaries(), [100, 200, 300, 400])
        self.assertEqual(BASELINE_TICK.boundaries(), [400])
        self.assertEqual(COARSE_TICK.boundaries(), [800])

    def test_chunk_bounds_partition_clean_steps(self):
        for sched in (FINE_TICK, BASELINE_TICK, COARSE_TICK):
            chunks = sched.chunk_bounds()
            self.assertEqual(len(chunks), sched.n_ticks)
            self.assertEqual(chunks[0][0], 0)
            self.assertEqual(chunks[-1][1], sched.clean_steps)
            for (s0, e0), (s1, e1) in zip(chunks, chunks[1:]):
                self.assertEqual(e0, s1)  # no gaps, no overlaps
            for s, e in chunks:
                self.assertEqual(e - s, sched.consol_every)

    def test_partial_tick_rejected(self):
        with self.assertRaises(ValueError):
            TickSchedule(name="BAD", clean_steps=450, consol_every=100)

    def test_nonpositive_rejected(self):
        with self.assertRaises(ValueError):
            TickSchedule(name="BAD", clean_steps=0, consol_every=100)
        with self.assertRaises(ValueError):
            TickSchedule(name="BAD", clean_steps=400, consol_every=0)
        with self.assertRaises(ValueError):
            TickSchedule(name="BAD", clean_steps=-400, consol_every=100)

    def test_every_variant_holds_at_least_one_full_tick(self):
        for sched in (FINE_TICK, BASELINE_TICK, COARSE_TICK):
            self.assertGreaterEqual(sched.n_ticks, 1)
            self.assertEqual(sched.clean_steps % sched.consol_every, 0)

    def test_decay_tick_count_independent_of_clean_schedule(self):
        # The scale-invariance wiring, pinned at the arithmetic
        # level: the SLOW tail's time-to-healed is a function of
        # decay consolidations only. It cannot see clean_steps,
        # so FINE (400 clean steps), BASELINE (400) and COARSE
        # (800) must all heal in the same number of decay ticks.
        # The run script verifies the same fact on applied
        # trajectories; this test pins the schedule/tail contract.
        for sched in (FINE_TICK, BASELINE_TICK, COARSE_TICK):
            self.assertEqual(SLOW_TAIL.time_to_healed(n_cap=30), 25)
            self.assertEqual(SLOW_TAIL.deficit_series(30),
                             SLOW_TAIL.deficit_series(30))
        self.assertAlmostEqual(SLOW_TAIL.deficit_at(25), 0.05)
        self.assertFalse(SLOW_TAIL.is_healed(24))
        self.assertTrue(SLOW_TAIL.is_healed(25))

    def test_decay_series_nonincreasing(self):
        series = SLOW_TAIL.deficit_series(30)
        for a, b in zip(series, series[1:]):
            self.assertGreaterEqual(a, b)


if __name__ == "__main__":
    unittest.main()
