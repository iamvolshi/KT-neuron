"""Tests for the KT neuron layer (neuron ONLY).

Run from ~/workspace/kt-neuron:
    python3 -m unittest discover -s tests -v

Every test uses only the standard library. The headline test,
TestPhenomenon, replays the toy's exact OR-teach / injury / restore
script and asserts the law in miniature: behavior returns to baseline
while seated authority measurably lags (0.9 -> 0.6).
"""

import copy
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kt_neuron import KTNeuron, KRecord, ScarState, make_kid, q10, scar_shape

OR_PATTERNS = [((1, 1), 1), ((1, 0), 1), ((0, 1), 1), ((0, 0), 0)]
LIAR_PATTERNS = [((1, 1), 0), ((1, 0), 0), ((0, 1), 1), ((0, 0), 0)]


def probe(n):
    return {x: n.fire(x)[0] for x, _ in OR_PATTERNS}


class TestBirth(unittest.TestCase):
    def test_kid_assigned_at_birth(self):
        n = KTNeuron()
        self.assertTrue(n.kid)
        self.assertEqual(n.rec.kid, n.kid)

    def test_two_neurons_have_different_kids(self):
        self.assertNotEqual(KTNeuron().kid, KTNeuron().kid)

    def test_born_registered(self):
        n = KTNeuron()
        for field in ("w0", "w1", "b"):
            row = n.standing(field)
            self.assertIsNotNone(row, f"no birth belief for {field}")
            self.assertEqual(row["value"], 0.0)
            self.assertEqual(row["authority"], 0.5)
            self.assertEqual(row["source"], "birth")

    def test_no_kid_no_record(self):
        with self.assertRaises(ValueError):
            KRecord("")
        with self.assertRaises(ValueError):
            KRecord(None)

    def test_kid_format(self):
        kid = make_kid(0, 0)
        self.assertEqual(len(kid), 32)
        self.assertTrue(all(c.isalnum() or c in "-_" for c in kid))


class TestRecordDiscipline(unittest.TestCase):
    def test_k_advances_only_on_open(self):
        n = KTNeuron()
        self.assertEqual(n.k, 3)  # three birth beliefs opened
        row = n.standing("w0")
        n.rec.close_belief(row)
        self.assertEqual(n.k, 3)  # close does not advance k
        n.rec.open_belief("w0", 1.0, 0.9, "test")
        self.assertEqual(n.k, 4)

    def test_close_is_the_only_mutation(self):
        n = KTNeuron()
        row = n.standing("w0")
        n.rec.close_belief(row)
        self.assertIsNotNone(row["k_end"])
        with self.assertRaises(AssertionError):
            n.rec.close_belief(row)  # history does not reopen
        self.assertTrue(n.verify().startswith("OK"))

    def test_closed_interval_never_rewritten(self):
        n = KTNeuron()
        n.rec.replace("w0", 1.5, 0.9, "teacher")
        closed = n.rec.rows[0]
        self.assertEqual(closed["field"], "w0")
        self.assertIsNotNone(closed["k_end"])
        self.assertEqual(closed["value"], 0.0)  # birth value sealed
        self.assertTrue(n.verify().startswith("OK"))

    def test_intervals_contiguous(self):
        n = KTNeuron()
        for i in range(5):
            # distinct source each time, or the gate would reaffirm
            n.rec.replace("w0", float(i), 0.9, f"teacher{i}")
        seen = {}
        for row in n.rec.rows:
            if row["field"] != "w0":
                continue
            if row["field"] in seen:
                self.assertEqual(row["k_start"], seen[row["field"]])
            if row["k_end"] is not None:
                seen[row["field"]] = row["k_end"]
        self.assertTrue(n.verify().startswith("OK"))

    def test_tampered_value_detected_and_named(self):
        n = KTNeuron()
        n.teach((1, 1), 1, "teacher", 0.9)
        tampered = copy.deepcopy(n.rec)
        victim = tampered.rows[0]  # birth w0 row: closed, sealed
        self.assertIsNotNone(victim["k_end"])
        victim["value"] = 999.0
        result = tampered.verify()
        self.assertIn("FAIL", result)
        self.assertIn(f"seq={victim['seq']}", result)
        self.assertIn(f"field={victim['field']}", result)

    def test_broken_link_detected(self):
        n = KTNeuron()
        n.teach((1, 1), 1, "teacher", 0.9)
        tampered = copy.deepcopy(n.rec)
        tampered.rows[-1]["prev_hash"] = "0" * 64
        self.assertIn("FAIL", tampered.verify())


class TestAuthorityGate(unittest.TestCase):
    def test_weaker_claim_is_candidate_only(self):
        n = KTNeuron()
        n.rec.replace("w0", 2.0, 0.9, "teacher")
        before = n.standing("w0")["value"]
        status, _ = n.rec.replace("w0", 9.0, 0.3, "driveby")
        self.assertEqual(status, "candidate")
        self.assertEqual(n.standing("w0")["value"], before)

    def test_equal_same_source_reaffirms(self):
        n = KTNeuron()
        n.rec.replace("w0", 2.0, 0.9, "teacher")
        status, _ = n.rec.replace("w0", 2.5, 0.9, "teacher")
        self.assertEqual(status, "reaffirmed")
        self.assertEqual(n.standing("w0")["value"], 2.0)

    def test_stronger_different_source_reseats(self):
        n = KTNeuron()
        status, row = n.rec.replace("w0", 2.0, 0.9, "teacher")
        self.assertEqual(status, "seated")
        self.assertEqual(row["authority"], 0.9)

    def test_authority_quantized_to_grid(self):
        self.assertEqual(q10(0.93), 0.9)
        self.assertEqual(q10(0.96), 1.0)
        self.assertEqual(q10(-0.2), 0.0)
        n = KTNeuron()
        row = n.rec.open_belief("w0", 1.0, 0.93, "test")
        self.assertEqual(row["authority"], 0.9)

    def test_dock_subtracts_from_reseated_authority(self):
        n = KTNeuron()
        _, row = n.rec.replace("w0", 2.0, 0.9, "teacher", authority_dock=0.3)
        self.assertEqual(row["authority"], 0.6)


class TestScar(unittest.TestCase):
    def test_hump_landmarks(self):
        self.assertEqual(scar_shape(0.30), 0.0)
        self.assertEqual(scar_shape(0.29), 0.0)
        self.assertEqual(scar_shape(0.75), 1.0)
        self.assertAlmostEqual(scar_shape(0.525), 0.5)
        self.assertAlmostEqual(scar_shape(0.875), 0.75)
        self.assertAlmostEqual(scar_shape(1.0), 0.5)

    def test_pressure_below_threshold_no_scar(self):
        s = ScarState()
        s.injure()  # pressure 0.18, still below 0.30
        self.assertEqual(s.scar, 0.0)

    def test_injure_raises_scar_and_deficit_heal_lowers_both(self):
        s = ScarState()
        for _ in range(4):
            s.injure()
        scar_after, deficit_after = s.scar, s.deficit
        self.assertGreater(scar_after, 0.0)
        self.assertGreater(deficit_after, 0.0)
        s.heal()
        self.assertLess(s.scar, scar_after)
        self.assertLess(s.deficit, deficit_after)

    def test_deficit_capped(self):
        s = ScarState()
        for _ in range(50):
            s.injure()
        self.assertLessEqual(s.deficit, 0.4)

    def test_scar_suppresses_output_through_bias(self):
        n = KTNeuron()
        n.rec.replace("w0", 2.0, 0.9, "t")
        n.rec.replace("w1", 2.0, 0.9, "t")
        n.rec.replace("b", 0.0, 0.9, "t")
        clean, _ = n.fire((1, 1))
        n.scar_state.scar = 0.6
        scarred, _ = n.fire((1, 1))
        self.assertLess(scarred, clean)


class TestFire(unittest.TestCase):
    def test_signal_speaks_at_weakest_authority(self):
        n = KTNeuron()
        n.rec.replace("w0", 1.0, 0.9, "a")
        n.rec.replace("w1", 1.0, 0.5, "b")
        _, a = n.fire((1, 1))
        self.assertEqual(a, 0.5)

    def test_wrong_input_count_rejected(self):
        n = KTNeuron()
        with self.assertRaises(ValueError):
            n.fire((1, 1, 1))


class TestPhenomenon(unittest.TestCase):
    """The law in miniature, replaying the toy's exact script."""

    def test_behavior_restored_authority_lags(self):
        n = KTNeuron()
        for _ in range(6):
            for x, t in OR_PATTERNS:
                n.teach(x, t, source="teacher", source_authority=0.9)
        pre = probe(n)
        pre_auth = n.standing("w0")["authority"]
        self.assertEqual(pre_auth, 0.9)

        # A weak source attacks: the gate holds.
        status = n.teach((1, 1), 0, source="driveby", source_authority=0.3)
        self.assertIn("w0:candidate", status)
        self.assertIn("w1:candidate", status)
        self.assertIn("b:candidate", status)

        # Injury: a strong source lies, repeatedly.
        for _ in range(4):
            for x, t in LIAR_PATTERNS:
                n.teach(x, t, source="liar", source_authority=0.9)
                n.injure()
        injured = probe(n)
        self.assertEqual(n.scar, 0.6)
        self.assertLess(injured[(1, 1)], 0.3)  # behavior bent

        # Restore: the trusted teacher returns.
        for _ in range(6):
            for x, t in OR_PATTERNS:
                n.teach(x, t, source="teacher", source_authority=0.9)
            n.heal()
        restored = probe(n)
        restored_auth = n.standing("w0")["authority"]

        # Headline: behavior back to baseline, authority measurably lags.
        self.assertEqual(restored_auth, 0.6)
        self.assertLess(restored_auth, pre_auth)
        for x, _ in OR_PATTERNS:
            self.assertLessEqual(abs(restored[x] - pre[x]), 0.20)
        for x, t in OR_PATTERNS:
            if t == 1:
                self.assertGreaterEqual(abs(restored[x] - injured[x]), 0.70)
        self.assertTrue(n.verify().startswith("OK"))
