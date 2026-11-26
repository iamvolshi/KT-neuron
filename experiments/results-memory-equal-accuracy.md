# Results: the scar's volume as a function of granularity

Date: 2026-11-26. Preregistration:
`experiments/preregistration-memory-equal-accuracy.md`. Script:
`experiments/measure_scar_volume.py` (237 lines). Machine-readable
results: `experiments/results-memory-equal-accuracy.json`.

Command (from `~/workspace/kt-neuron`):

    python3 experiments/measure_scar_volume.py

Seeds: model build seed 0 (M2, the same seed `run_all.py` E1 uses).
Every other measurement uses fixed grid values and consumes no RNG.
All measurements deterministic.

No parameter-count race at equal accuracy was run. Darren ordered
it out on Nov 26: that comparison does not matter. The memory is
the slow-trace record, and the measurement is its volume.

## What the scar stores (code facts)

- The scar (`KTLinear.scar`, `kt_bench/scar.py`) is a per-row
  float64 offset, set by `PressureDriver` to ONE global scalar F in
  [0, 1.2]. Phase-boundary values in the v4 graded run: 0.0, 1.2,
  0.2, 1.2, 0.2. The scar is live fast-trace state: `row_hash`
  covers W and b only, so no scar value ever enters a K-record row.
- The slow trace (the record) is the memory. One `KRecord` per KT
  row, one KID per row. Authority lives on the 0.1 grid (`q10`):
  11 levels, 0.0 to 1.0.
- The five authority points as run in v1 (M1, verified from the
  code path, not assumed): 0.5 birth, 0.8 clean consolidation,
  0.9 poison seating, 0.6 repair seating (0.9 - dock 0.3), 0.7
  decay +6 (E3 trajectory). Five distinct grid values, each with a
  known distinct numerical difference.

## M1: the five v1 points

Measured: [0.5, 0.6, 0.7, 0.8, 0.9]. Match with the expected set: yes.

## M2: structural counts (v1-size model, seed 0)

| quantity | measured | expected (paper / E1) | match |
|---|---:|---:|---|
| parameters | 105248 | 105248 | yes |
| KIDs | 1184 | 1184 | yes |
| record rows after 1 consolidation | 2368 | 2368 | yes |
| snapshot bytes each (params x 8) | 841984 | 841984 | yes |

## M3: grid seating at finer granularities

One fresh `KRecord` per granularity, every grid value in [0, 1]
seated in increasing order through the real `replace` path (the
quantizer parameterized in the measurement script only; bench files
untouched). Pool seatings (clean 0.8, poison 0.9, repair 0.6) are
stable at every granularity: the five v1 points do not move when
the grid refines.

| granularity g | grid values | seated rows | record JSON bytes | bytes/row | max configurations C(3+1/g, 2) |
|---:|---:|---:|---:|---:|---:|
| 0.1 | 11 | 11 | 4083 | 371.2 | 78 |
| 0.01 | 101 | 101 | 37926 | 375.5 | 5253 |
| 0.001 | 1001 | 1001 | 380829 | 380.4 | 502503 |
| 0.0001 | 10001 | 10001 | 3854832 | 385.4 | 50025003 |

Scaling: max configurations = C(3 + 1/g, 2), which grows as
~1/(2g^2). Each factor of 10 in granularity multiplies the
configuration count by ~100. Record bytes grow ~linearly in rows
(~371-385 bytes per row).

Finding (measured, not assumed): `KRecord.verify()` FAILED at every
granularity finer than 0.1, with "FAIL: row seq=1 -- authority off
the 0.1 grid". The 0.1 grid was hardcoded as an invariant inside
`verify()` (`kt_neuron/record.py`), so the verifier itself could not
check a finer-grained record. The chain underneath was intact: the
structural re-check (hash recomputation, prev_hash linkage,
interval contiguity, same checks minus the grid assertion) passed
at all four granularities, including 10001 rows at g = 0.0001.
Fixed November 26, 2026: the grid is now a parameter the record
carries (`KRecord(kid, grid=...)`, default 0.1; `verify()` checks
the record's grid; `seat_authority` takes the same grid).
Re-running this script against the fixed bench, `verify()` passes
at all four granularities, and every other number in this file
reproduces exactly.

## M4: volume and configurations vs number of points (g = 0.1)

| points on slow trace | volume per point | total volume | total points (2 states + n) | configurations C(2+n, 2) |
|---:|---:|---:|---:|---:|
| 1 | 0.1 | 0.1 | 3 | 3 |
| 2 | 0.1 | 0.2 | 4 | 6 |
| 5 (as run) | 0.1 | 0.5 | 7 | 21 |
| 10 | 0.1 | 1.0 | 12 | 66 |
| 11 (grid cap at g = 0.1) | 0.1 | 1.1 | 13 | 78 |

The 21 as-run count and the 66 with-10-points count from paper
section 5.8 reproduce exactly. At g = 0.1 at most 11 distinct
points fit (11 grid values); a 12th distinct point requires a
finer grid.

## M5: sub-interval chop, one 0.1 volume (law level 1)

One volume [0.5, 0.6] chopped into pieces of width c. Each piece
is a number assignable to a weight. Piece midpoints enumerated in
float64 and counted distinct; every count matched g/c exactly.

| chop fineness c | weights per volume | weights from the 5 v1 points |
|---:|---:|---:|
| 0.1 | 1 | 5 |
| 0.01 | 10 | 50 |
| 0.001 | 100 | 500 |
| 0.0001 | 1000 | 5000 |
| 0.00001 | 10000 | 50000 |
| 0.000001 | 100000 | 500000 |

Scaling: weights per volume = 0.1 / c. Each factor of 10 in chop
fineness multiplies the weights per volume by 10.

## M6: digit chop of the volumetric number itself (law level 2)

The volumetric numbers 0.1 and 0.8, written to d decimal places.
Every digit is a number assignable to a weight. Both numbers give
the same counts (1 integer digit + d fractional digits).

| decimal places d | weights per volume |
|---:|---:|
| 1 | 2 |
| 2 | 3 |
| 3 | 4 |
| 6 | 7 |
| 10 | 11 |
| 15 | 16 |

Scaling: weights per volume = d + 1, linear in the chop fineness,
with no end to how fine the chop can go.

## M7: the float64 ceiling (the implementation's bound on the law)

The law is unbounded as the chop refines without end. The bench
stores numbers as float64, and float64 caps the chop. Measured:

- float64 spacing at 0.5: 1.1102230246251565e-16 (= 2^-53, exact).
- Distinct float64 values in one 0.1 volume [0.5, 0.6], counted
  exactly by bit arithmetic: 900719925474100.
- Max weights from one volume in this implementation:
  900719925474100. Max weights from the 5 v1 points:
  4503599627370500.

So: in the reals the chop has no end (infinite weights); in this
float64 bench one volume yields at most 900719925474100 distinct
weights, and the 5 v1 points at most 4503599627370500. The ceiling
is a property of the implementation, not of the law.

## Verdict, in one paragraph

The volume of the slow-trace record was measured, not asserted.
At the v1 granularity (0.1) one point is a volume of 0.1 holding
exactly 1 number, the 5 points as run hold 0.5 of volume and give
the paper's 21 configurations, and the grid caps distinct points
at 11 (78 configurations). Refining the granularity multiplies the
volume: max configurations grow as ~1/(2g^2), reaching 50025003
at g = 0.0001 with the record's hash chain intact at 10001 rows.
Chopping one volume gives weights without end in the law's terms:
0.1/c weights per volume by sub-interval chop (100000 at
c = 0.000001) and d + 1 by digit chop (16 at 15 places), both
measured exactly. The float64 bench caps one volume at
900719925474100 distinct weights. One measured limitation: the
bench's own `verify()` hardcoded the 0.1 grid and failed any
finer-grained record on that assertion alone; the chain underneath
verified fine. That limitation is now fixed in the bench (grid
parameterized, November 26, 2026), and the re-run passes `verify()`
at all four granularities. No parameter-count comparison was run,
per Darren's correction.
