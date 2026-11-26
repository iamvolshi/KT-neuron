"""Smoke test for the §11 side-by-side C2C demo.

Runs demo.run_demo() end to end and asserts the differential holds
live: vanilla inherits exactly with no records; KT names the transfer
lineage, quarantines back to clean behavior, repairs, and carries an
authority lag, with the chain ok. Run from ~/workspace/kt-neuron:

    python3 -m unittest discover -s tests
"""

import unittest

from demo.run_demo import run_demo


class TestDemoEndToEnd(unittest.TestCase):
    def test_demo_runs_and_shows_differential(self):
        out = run_demo()
        # Vanilla: inherits the poison EXACTLY, no lineage possible.
        self.assertFalse(out["vanilla_has_records"])
        self.assertAlmostEqual(
            out["vanilla_receiver_post_transfer"]["mean_p_liar"],
            out["vanilla_sender_poison"]["mean_p_liar"], places=9)
        self.assertAlmostEqual(
            out["vanilla_receiver_post_transfer"]["mean_p_true"],
            out["vanilla_sender_poison"]["mean_p_true"], places=9)
        self.assertGreater(
            out["vanilla_receiver_post_transfer"]["mean_p_liar"],
            out["vanilla_receiver_post_transfer"]["mean_p_true"])
        # KT: lineage names the transfer, null restores clean behavior.
        self.assertEqual(out["kt_lineage_liar_entries"], 1184)
        self.assertGreaterEqual(
            out["kt_receiver_post_null"]["mean_p_true"],
            out["kt_sender_clean"]["mean_p_true"] - 0.10)
        self.assertEqual(out["kt_receiver_post_null"]["top1_true_count"], 4)
        # KT: repair restores ranking with a visible authority lag.
        self.assertEqual(out["kt_receiver_post_repair"]["top1_true_count"], 4)
        self.assertGreaterEqual(out["kt_authority_lag"], 0.1)
        self.assertTrue(out["kt_chain_ok"])


if __name__ == "__main__":
    unittest.main()
