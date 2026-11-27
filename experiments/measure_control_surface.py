"""Measure the slow-trace volume as a control surface.

Preregistration: experiments/preregistration-control-surface.md
Run from ~/workspace/kt-neuron-repo:
    python3 experiments/measure_control_surface.py

Darren's claim (Nov 27): the scar volume is predictable, so it is a
CONTROL SURFACE. Every step down from 1.0 is a different number, a
different string of weights, a new thread. The KT neuron itself is
the machine. PART 1 walks the steps and counts threads and weights.
PART 2 maps a graph from the threads to the weight states it can
point to.

Deterministic. No RNG anywhere. The chop methods are the Nov 26
M5 (sub-interval) and M6 (digit) chops. The KRecord seating reuses
the Nov 26 M3 pattern through the real replace path with the
record's own grid (gap-12 fix). Writes
experiments/results-control-surface.json.
"""

import hashlib
import json
import math
import os
import struct
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from kt_neuron.kid import make_kid
from kt_neuron.record import KRecord

OUT = os.path.join(ROOT, "experiments", "results-control-surface.json")

GRANULARITIES = [0.1, 0.01, 0.001, 0.0001]
DIGIT_PLACES = [1, 2, 3, 6, 10, 15]
CHOP_FINENESS = [0.1, 0.01, 0.001, 1e-4, 1e-5, 1e-6]
CONTROL_STEP = 0.7


def fake_row_hash(value):
    """Stand-in row value: a 64-hex hash, the shape a real row belief holds."""
    return hashlib.sha256(f"grid:{value}".encode()).hexdigest()


def steps(g):
    """Grid values from 1.0 down to 0.0: the walk, top-down."""
    n = int(round(1.0 / g))
    return [(n - i) / n for i in range(n + 1)]


def digit_string(v, d):
    """The step number chopped into digits: d+1 numbers, a weight string."""
    written = f"{v:.{d}f}"
    return [int(ch) for ch in written if ch.isdigit()]


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


def structural_chain_check(rec):
    """Independent re-check of hash, linkage and contiguity.

    Same checks as KRecord.verify() minus the grid assertion, so a
    record's integrity is measured twice, once by the bench and once
    here, rather than assumed. (Same checks as Nov 26 M3,
    re-implemented locally so this script needs no model-build
    imports.)
    """
    prev = "GENESIS"
    last_close = {}
    for row in rec.rows:
        if row["hash"] != rec._hash(row):
            return (False, f"row seq={row['seq']} hash mismatch")
        if row["prev_hash"] != prev:
            return (False, f"row seq={row['seq']} chain link broken")
        f = row["field"]
        if f in last_close and row["k_start"] != last_close[f]:
            return (False, f"row seq={row['seq']} field={f} interval gap")
        if row["k_end"] is not None:
            if row["k_end"] <= row["k_start"]:
                return (False, f"row seq={row['seq']} empty interval")
            last_close[f] = row["k_end"]
        prev = row["hash"]
    return (True, f"{len(rec.rows)} rows, hashes/linkage/contiguity OK")


def w2_seat_walk(g):
    """Seat every grid value in [0, 1] into one real KRecord.

    Seated bottom-up (0.0 -> 1.0) because the authority gate only
    seats equal-or-stronger claims; the walk reads the same set
    top-down. Returns the seated authorities in walk order for W6.
    """
    n = int(round(1.0 / g))
    rec = KRecord(make_kid(birth_tick=0, seq=910000), grid=g)
    for i in range(n + 1):
        v = i / n
        status, row = rec.replace("row", fake_row_hash(v), v, f"grid:{v}")
        assert status == "seated", (g, v, status)
        assert row["authority"] == v, (g, v, row["authority"])
    verify = rec.verify()
    struct_ok, struct_note = structural_chain_check(rec)
    blob = json.dumps({"kid": rec.kid, "rows": rec.rows}).encode()
    seated = [row["authority"] for row in rec.rows]
    return {"granularity": g, "grid_values": n + 1,
            "seated_rows": len(seated), "chain": verify,
            "chain_ok": verify.startswith("OK"),
            "structural_chain_ok": struct_ok,
            "structural_chain_note": struct_note,
            "record_json_bytes": len(blob),
            "bytes_per_row": len(blob) / len(seated),
            "seated_authorities_walk_order": list(reversed(seated))}


def w3_digit_strings(g, d):
    """Digit strings for every step at d places; count distinct threads."""
    strs = [digit_string(v, d) for v in steps(g)]
    distinct = len({tuple(s) for s in strs})
    return {"granularity": g, "decimal_places": d,
            "steps": len(strs), "distinct_strings": distinct,
            "all_steps_distinct": distinct == len(strs),
            "weights_per_thread": d + 1,
            "total_digit_weights": len(strs) * (d + 1)}


def w4_subinterval_chop(g, c):
    """Chop every cell [v, v+g) into pieces of width c; count distinct numbers.

    All cells are measured at once (one vectorized array); the total
    is the distinct count over the union, not an assumed multiple.
    """
    n_cells = int(round(1.0 / g))
    n_pieces = int(round(g / c))
    starts = np.arange(n_cells) * g
    mids = starts[:, None] + (np.arange(n_pieces) + 0.5) * c
    distinct = int(np.unique(mids).size)
    return {"granularity": g, "chop_fineness": c,
            "cells": n_cells, "pieces_per_cell": n_pieces,
            "distinct_midpoints_total": distinct,
            "weights_per_cell": n_pieces,
            "total_subinterval_weights": distinct,
            "expected_total": int(round(1.0 / c)),
            "match": distinct == int(round(1.0 / c))}


def g1_tables(g, d):
    """One weight table per thread: the digit string as a weight vector."""
    tables = [tuple(digit_string(v, d)) for v in steps(g)]
    return {"granularity": g, "decimal_places": d,
            "nodes": len(tables),
            "distinct_tables": len(set(tables)),
            "all_tables_distinct": len(set(tables)) == len(tables)}


def g2_outputs(g, d):
    """Distinct outputs on input (1, 1): 2-weight, full-sum, linear readouts.

    The linear readout is the dot product of the table's full weight
    vector with fixed positional weights, powers of 10: w_i =
    10^(n-1-i) for position i in 0..n-1, n = d + 1. It reconstructs
    the digit string's numeric value scaled by 10^d (the integer
    whose decimal digits are the table's digits), computed with
    exact integer arithmetic. Its output IS that reconstructed
    number (no sigmoid squash, which would saturate); distinct
    outputs are distinct reconstructed values.
    """
    tables = [digit_string(v, d) for v in steps(g)]
    z2 = sorted({t[0] + t[1] for t in tables})
    zsum = sorted({sum(t) for t in tables})
    out2 = sorted({round(sigmoid(z), 6) for z in z2})
    outsum = sorted({round(sigmoid(z), 6) for z in zsum})
    n = d + 1
    weights = [10 ** (n - 1 - i) for i in range(n)]
    zlin = sorted({sum(w * x for w, x in zip(weights, t)) for t in tables})
    return {"granularity": g, "decimal_places": d,
            "distinct_z_2weight": len(z2),
            "distinct_outputs_2weight": len(out2),
            "distinct_z_fullsum": len(zsum),
            "distinct_outputs_fullsum": len(outsum),
            "distinct_z_linear": len(zlin),
            "distinct_outputs_linear": len(zlin),
            "outputs_agree_with_z": (len(out2) == len(z2)
                                     and len(outsum) == len(zsum))}


def g3_held_configurations(n_steps):
    """Pairs of steps the neuron can hold at once (2 states per tick)."""
    return math.comb(2 + n_steps, 2)


def g4_float64_ceiling():
    """Exact count of representable float64 values in [0, 1]."""
    def bits(x):
        return struct.unpack("<Q", struct.pack("<d", x))[0]

    count = bits(1.0) - bits(0.0) + 1
    formula = 1023 * 2 ** 52 + 1  # 1022 normal binades + subnormals + 0 + 1.0
    return {"interval": [0.0, 1.0], "count_by_bits": count,
            "count_by_binade_formula": formula,
            "match": count == formula,
            "max_threads": count,
            "max_addressable_weight_states": count,
            "max_held_configurations": math.comb(2 + count, 2)}


def pure_measurements():
    """Everything that needs no record: run twice for determinism (W5)."""
    return {
        "w1_steps": [{"granularity": g, "steps": len(steps(g)),
                      "first_step": steps(g)[0], "last_step": steps(g)[-1],
                      "expected_steps": int(round(1.0 / g)) + 1,
                      "match": len(steps(g)) == int(round(1.0 / g)) + 1}
                     for g in GRANULARITIES],
        "w3_digit_strings": [w3_digit_strings(g, d) for g in GRANULARITIES
                             for d in DIGIT_PLACES],
        "w4_subinterval_chop": [w4_subinterval_chop(g, c)
                                for g in GRANULARITIES
                                for c in CHOP_FINENESS if c <= g],
        "g1_tables": [g1_tables(g, d) for g in GRANULARITIES
                      for d in DIGIT_PLACES],
        "g2_outputs": [g2_outputs(g, 15) for g in GRANULARITIES],
        "g3_held_configurations": [{"granularity": g,
                                    "steps": int(round(1.0 / g)) + 1,
                                    "held_configurations": g3_held_configurations(
                                        int(round(1.0 / g)) + 1)}
                                   for g in GRANULARITIES],
        "g4_float64_ceiling": g4_float64_ceiling(),
    }


def w6_predictability(seatings):
    """Arithmetic strings vs strings from record-seated authorities."""
    out = []
    for g, seated in seatings.items():
        arith = {v: digit_string(v, 15) for v in steps(g)}
        seated_strs = {v: digit_string(v, 15) for v in seated}
        out.append({"granularity": g,
                    "steps_compared": len(arith),
                    "all_match": arith == seated_strs,
                    "mismatches": [v for v in arith
                                   if arith[v] != seated_strs.get(v)]})
    return out


def w7_control(pure_a, pure_b):
    """Step 0.7 vs its neighbours, and stable across the two runs."""
    out = []
    for g in GRANULARITIES:
        above, below = CONTROL_STEP + g, CONTROL_STEP - g
        row = {"granularity": g, "control_step": CONTROL_STEP}
        for label, v in (("control", CONTROL_STEP),
                         ("above", above), ("below", below)):
            s = digit_string(v, 15)
            row[label] = {"digit_string": s,
                          "table": list(s),
                          "output_2weight": round(sigmoid(s[0] + s[1]), 6),
                          "output_fullsum": round(sigmoid(sum(s)), 6)}
        row["differs_from_above"] = (
            row["control"]["digit_string"] != row["above"]["digit_string"])
        row["differs_from_below"] = (
            row["control"]["digit_string"] != row["below"]["digit_string"])
        row["stable_across_runs"] = (
            pure_a["w3_digit_strings"] == pure_b["w3_digit_strings"]
            and pure_a["g1_tables"] == pure_b["g1_tables"])
        out.append(row)
    return out


def main():
    pure_a = pure_measurements()
    pure_b = pure_measurements()
    deterministic = (json.dumps(pure_a, sort_keys=True)
                     == json.dumps(pure_b, sort_keys=True))
    seatings = {}
    w2 = []
    for g in GRANULARITIES:
        r = w2_seat_walk(g)
        seatings[g] = r["seated_authorities_walk_order"]
        w2.append({k: v for k, v in r.items()
                   if k != "seated_authorities_walk_order"})
    results = {
        "preregistration": "experiments/preregistration-control-surface.md",
        "command": "python3 experiments/measure_control_surface.py",
        "seeds": "none (no RNG anywhere)",
        "note": ("No parameter-count race was run (Darren correction, "
                 "Nov 26). No paper changes. Bench only."),
        "w1_steps": pure_a["w1_steps"],
        "w2_record_seating": w2,
        "w3_digit_strings": pure_a["w3_digit_strings"],
        "w4_subinterval_chop": pure_a["w4_subinterval_chop"],
        "w5_determinism": {"two_in_process_runs_identical": deterministic},
        "w6_predictability": w6_predictability(seatings),
        "w7_control_step_0.7": w7_control(pure_a, pure_b),
        "g1_tables": pure_a["g1_tables"],
        "g2_outputs": pure_a["g2_outputs"],
        "g3_held_configurations": pure_a["g3_held_configurations"],
        "g4_float64_ceiling": pure_a["g4_float64_ceiling"],
    }
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"wrote {OUT}")
    for r in results["w1_steps"]:
        print(f"W1 g={r['granularity']}: steps={r['steps']} "
              f"match={r['match']}")
    for r in results["w2_record_seating"]:
        print(f"W2 g={r['granularity']}: rows={r['seated_rows']} "
              f"chain_ok={r['chain_ok']} "
              f"structural_ok={r['structural_chain_ok']} "
              f"bytes/row={r['bytes_per_row']:.1f}")
    for r in results["w3_digit_strings"]:
        if r["decimal_places"] == 15:
            print(f"W3 g={r['granularity']} d=15: distinct="
                  f"{r['distinct_strings']}/{r['steps']} "
                  f"weights/thread={r['weights_per_thread']} "
                  f"total={r['total_digit_weights']}")
    for r in results["w4_subinterval_chop"]:
        if r["chop_fineness"] == 1e-6:
            print(f"W4 g={r['granularity']} c=1e-6: per_cell="
                  f"{r['weights_per_cell']} total={r['total_subinterval_weights']} "
                  f"match={r['match']}")
    print(f"W5 deterministic: {deterministic}")
    for r in results["w6_predictability"]:
        print(f"W6 g={r['granularity']}: all_match={r['all_match']}")
    for r in results["w7_control_step_0.7"]:
        print(f"W7 g={r['granularity']}: differs_above="
              f"{r['differs_from_above']} differs_below="
              f"{r['differs_from_below']} stable={r['stable_across_runs']}")
    for r in results["g1_tables"]:
        if r["decimal_places"] == 15:
            print(f"G1 g={r['granularity']} d=15: distinct_tables="
                  f"{r['distinct_tables']}/{r['nodes']}")
    for r in results["g2_outputs"]:
        print(f"G2 g={r['granularity']}: distinct_2weight="
              f"{r['distinct_outputs_2weight']} distinct_fullsum="
              f"{r['distinct_outputs_fullsum']} distinct_linear="
              f"{r['distinct_outputs_linear']}")
    for r in results["g3_held_configurations"]:
        print(f"G3 g={r['granularity']}: held={r['held_configurations']}")
    c = results["g4_float64_ceiling"]
    print(f"G4 ceiling: threads={c['max_threads']} match={c['match']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
