"""Measure the scar's volume as a function of granularity.

Preregistration: experiments/preregistration-memory-equal-accuracy.md
Run from ~/workspace/kt-neuron:
    python3 experiments/measure_scar_volume.py

Deterministic. The only RNG is the fixed model-build seed (0) in M2.
The grid seatings (M3), chops (M5, M6) and ceiling (M7) consume no RNG.
Writes experiments/results-memory-equal-accuracy.json.
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

from experiments.playbook import Bench
from experiments.run_all import CFG, CLEAN_POOL, REPAIR_DOCK
from kt_bench.seating import seat_authority
from kt_bench.transformer import GPTConfig
from kt_bench.twin import build_pair
from kt_neuron.kid import make_kid
from kt_neuron.record import KRecord, q, q10

OUT = os.path.join(ROOT, "experiments", "results-memory-equal-accuracy.json")

GRANULARITIES = [0.1, 0.01, 0.001, 0.0001]
POINTS_SCALING = [1, 2, 5, 10, 11]
CHOP_FINENESS = [0.1, 0.01, 0.001, 1e-4, 1e-5, 1e-6]
DIGIT_PLACES = [1, 2, 3, 6, 10, 15]
VOLUME_WIDTH = 0.1          # v1 volume per point (the 0.1 grid step)
VOLUME_START = 0.5          # the volume [0.5, 0.6], the birth point's cell


def fake_row_hash(value):
    """Stand-in row value: a 64-hex hash, the shape a real row belief holds."""
    return hashlib.sha256(f"grid:{value}".encode()).hexdigest()


def m1_five_points():
    """The five v1 authority points, from the code path (no RNG)."""
    clean = seat_authority(CLEAN_POOL)
    poison = seat_authority({"sources": {"liar": 0.9}})
    repair = q10(0.9 - REPAIR_DOCK)
    deficit6 = max(0.0, REPAIR_DOCK - 0.01 * 6)
    decay6 = q10(0.9 - deficit6)
    birth = 0.5  # BIRTH_AUTHORITY, kt_bench/linear.py + seating.py
    points = sorted({birth, clean, poison, repair, decay6})
    return {"birth": birth, "clean": clean, "poison": poison,
            "repair": repair, "decay_plus6": decay6,
            "five_points": points,
            "grid_levels_v1": [q10(i / 10) for i in range(11)],
            "expected": [0.5, 0.6, 0.7, 0.8, 0.9],
            "match": points == [0.5, 0.6, 0.7, 0.8, 0.9]}


def m2_structural_counts():
    """v1-size model bookkeeping: params, KIDs, record rows (seed 0)."""
    kt, _vn = build_pair(CFG, 0)
    bench = Bench(kt)
    bench.consolidate(CLEAN_POOL, "consolidation:0")
    n_kids = sum(layer.out_features for _n, layer in kt.kt_linears())
    n_rows = sum(len(rec.rows) for _n, layer in kt.kt_linears()
                 for rec in layer.records)
    params = kt.param_count()
    return {"param_count": params, "n_kids": n_kids,
            "record_rows_after_1_consolidation": n_rows,
            "snapshot_bytes_each": params * 8,
            "expected": {"param_count": 105248, "n_kids": 1184,
                         "record_rows_after_1_consolidation": 2368,
                         "snapshot_bytes_each": 841984},
            "match": (params == 105248 and n_kids == 1184
                      and n_rows == 2368)}


def structural_chain_check(rec):
    """Independent re-check of hash, linkage and contiguity.

    Same checks as KRecord.verify() minus the grid assertion, so a
    record's integrity is measured twice, once by the bench and once
    here, rather than assumed.
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


def m3_grid_seating(step):
    """Seat every grid value in [0, 1] into one real KRecord.

    The record carries its grid (KRecord grid parameter, fixed Nov 26,
    gap 12); the bench's own verify() checks the record's grid, so no
    monkeypatching is needed. Values are seated in increasing order
    from distinct sources, so each seats under the gate. Chain
    verified after every seating.
    """
    n = int(round(1.0 / step))
    # The five v1 pool values must be grid-stable at this step.
    pool_seats = {"clean": seat_authority(CLEAN_POOL, grid=step),
                  "poison": seat_authority({"sources": {"liar": 0.9}}, grid=step),
                  "repair": q(0.9 - REPAIR_DOCK, step)}
    rec = KRecord(make_kid(birth_tick=0, seq=900000), grid=step)
    seated = []
    for i in range(n + 1):
        v = i / n
        status, row = rec.replace("row", fake_row_hash(v), v, f"grid:{v}")
        assert status == "seated", (step, v, status)
        assert row["authority"] == v, (step, v, row["authority"])
        seated.append(v)
    verify = rec.verify()
    struct_ok, struct_note = structural_chain_check(rec)
    blob = json.dumps({"kid": rec.kid, "rows": rec.rows}).encode()
    return {"granularity": step, "grid_values": n + 1,
            "seated_rows": len(seated), "chain": verify,
            "chain_ok": verify.startswith("OK"),
            "verify_failure_cause": (None if verify.startswith("OK")
                else verify),
            "structural_chain_ok": struct_ok,
            "structural_chain_note": struct_note,
            "record_json_bytes": len(blob),
            "bytes_per_row": len(blob) / len(seated),
            "pool_seats_stable": pool_seats,
            "max_configurations": math.comb(3 + n, 2)}


def m4_points_scaling():
    """Volume and configurations vs number of points, g fixed at 0.1."""
    out = []
    for n_points in POINTS_SCALING:
        total_points = 2 + n_points  # 2 states + n slow-trace points
        out.append({"n_points": n_points,
                    "volume_per_point": VOLUME_WIDTH,
                    "total_volume": n_points * VOLUME_WIDTH,
                    "total_points": total_points,
                    "configurations": math.comb(total_points, 2)})
    return out


def m5_subinterval_chop():
    """Chop one 0.1 volume into pieces of width c; count distinct values."""
    out = []
    for c in CHOP_FINENESS:
        n_pieces = int(round(VOLUME_WIDTH / c))
        mids = VOLUME_START + (np.arange(n_pieces) + 0.5) * c
        distinct = int(np.unique(mids).size)
        out.append({"chop_fineness": c, "pieces": n_pieces,
                    "distinct_float64_midpoints": distinct,
                    "weights_per_volume": distinct,
                    "weights_from_5_points": 5 * distinct,
                    "expected_weights_per_volume": int(round(VOLUME_WIDTH / c)),
                    "match": distinct == n_pieces})
    return out


def m6_digit_chop():
    """Chop the volumetric number itself into digits; count them."""
    out = []
    for x in (0.1, 0.8):
        for d in DIGIT_PLACES:
            written = f"{x:.{d}f}"
            digits = [ch for ch in written if ch.isdigit()]
            out.append({"volumetric_number": x, "decimal_places": d,
                        "written": written, "digits": digits,
                        "weights_per_volume": len(digits),
                        "expected_weights_per_volume": d + 1,
                        "match": len(digits) == d + 1})
    return out


def m7_float64_ceiling():
    """Exact count of representable float64 values in [0.5, 0.6]."""
    def bits(x):
        return struct.unpack("<Q", struct.pack("<d", x))[0]

    lo, hi = 0.5, 0.6
    count = bits(hi) - bits(lo) + 1  # positive floats: bits order with value
    spacing = float(np.spacing(0.5))
    return {"volume": [lo, hi], "spacing_at_0.5": spacing,
            "spacing_is_2_to_minus_53": spacing == 2.0 ** -53,
            "distinct_float64_in_volume": count,
            "max_weights_per_volume": count,
            "max_weights_from_5_points": 5 * count}


def main():
    results = {
        "preregistration": "experiments/preregistration-memory-equal-accuracy.md",
        "command": "python3 experiments/measure_scar_volume.py",
        "seeds": {"model_build": 0, "other_measurements": "none (fixed grid values, no RNG)"},
        "note": "No parameter-count race was run (Darren correction, Nov 26).",
        "m1_five_v1_points": m1_five_points(),
        "m2_structural_counts": m2_structural_counts(),
        "m3_grid_seating": [m3_grid_seating(g) for g in GRANULARITIES],
        "m4_points_scaling_g_0.1": m4_points_scaling(),
        "m5_subinterval_chop_g_0.1": m5_subinterval_chop(),
        "m6_digit_chop": m6_digit_chop(),
        "m7_float64_ceiling": m7_float64_ceiling(),
    }
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"wrote {OUT}")
    print("M1 five points:", results["m1_five_v1_points"]["five_points"],
          "match:", results["m1_five_v1_points"]["match"])
    print("M2:", {k: results["m2_structural_counts"][k] for k in
                  ("param_count", "n_kids", "record_rows_after_1_consolidation")},
          "match:", results["m2_structural_counts"]["match"])
    for r in results["m3_grid_seating"]:
        print(f"M3 g={r['granularity']}: rows={r['seated_rows']} "
              f"bytes={r['record_json_bytes']} chain_ok={r['chain_ok']} "
              f"structural_ok={r['structural_chain_ok']} "
              f"max_configurations={r['max_configurations']}")
    for r in results["m4_points_scaling_g_0.1"]:
        print(f"M4 n={r['n_points']}: volume={r['total_volume']} "
              f"configurations={r['configurations']}")
    for r in results["m5_subinterval_chop_g_0.1"]:
        print(f"M5 c={r['chop_fineness']}: weights/volume="
              f"{r['weights_per_volume']} match={r['match']}")
    for r in results["m6_digit_chop"]:
        if r["volumetric_number"] == 0.1:
            print(f"M6 0.1 d={r['decimal_places']}: "
                  f"weights/volume={r['weights_per_volume']} "
                  f"match={r['match']}")
    c = results["m7_float64_ceiling"]
    print(f"M7 ceiling: spacing={c['spacing_at_0.5']} "
          f"distinct_in_volume={c['distinct_float64_in_volume']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
