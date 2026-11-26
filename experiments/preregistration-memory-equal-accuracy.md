# Preregistration: the scar's volume as a function of granularity

Date: 2026-11-26. Author: bench run for the KTN v1.1 paper.
Status: this file REPLACES the original brief for this filename. The
original brief ordered a parameter-count race at equal accuracy
against a plain baseline. Darren corrected it on Nov 26: that
comparison does not matter and is NOT run here. Nothing in this file
orders it, and no parameter-count comparison appears in the results.

## The correction (Darren, Nov 26)

The measured memory result comes from the volume of the scar, not
from a parameter-count race. The slow-trace record is the memory.
You can put an infinite amount in there if the granularity is good
enough.

## The law (Darren, Nov 26, his words)

Level 1: "because it is a law, you find a number on point one. Any
number, when the volume is point one, can be assigned to a weight.
Then you create another point, and it is in that group of numbers
that gets assigned to another weight."

Level 2: "you can actually just use the volumetric number itself. If
you chop it up, you can literally take the first number, assign that
to a weight, take the second number, assign that to a weight, take
the third number, assign that to a weight, ad nauseam. We are talking
infinite weights."

So one volume does not give one weight. The volumetric number itself
chops into numbers, and each number in the chop assigns to a weight,
with no end to how fine the chop can go. This file records that law
as the thing being measured.

## Code facts: what the scar stores

Read from the bench code (`kt_bench/`, `kt_neuron/`), not assumed:

1. The scar (`KTLinear.scar`, `ScarLinear.scar`, `kt_bench/scar.py`).
   A per-row float64 offset vector. Forward subtracts it:
   `y = (x @ W.T + b - scar) * gain` (KT) and
   `y = x @ W.T + b - scar` (scar-only). `PressureDriver.apply_to`
   sets every scar-carrying layer's scar vector to ONE global scalar
   F. F = round(1.2 * scar_shape(pressure), 6), pressure in
   [0, 0.75], so F is in [0, 1.2]. Phase-boundary values in the v4
   graded run (seeds 21-25): 0.0, 1.2, 0.2, 1.2, 0.2. The scar is
   live fast-trace state: `row_hash` covers W and b only, so no scar
   value ever enters a K-record row. Per-row scar variation is later
   work, not in this bench.
2. The slow trace (the record). One `KRecord` per KT row, one KID
   per row. The v1-size model (vocab 32, block 16, n_embd 64,
   n_head 4, n_layer 2) carries 1,184 KIDs and 105,248 parameters
   (re-verified from `experiments/results.json` E1 and from live
   code in measurement M2). Each record row holds: seq, kid, field,
   value (the row hash), authority, source, k_start, k_end,
   prev_hash, hash. After 1 consolidation each record holds 2 rows
   (birth + seated): 2,368 rows in the model. Snapshot bytes each
   (E1 definition, param_count * 8): 841,984.
3. The authority grid (`kt_neuron/record.py`, `q10`). Authority
   lives on the 0.1 grid: 11 levels, 0.0 to 1.0. `seat_authority`
   (`kt_bench/seating.py`): mean of declared source authorities,
   +0.1 agreement bonus for 2+ sources, -0.3 for a known-bad source,
   hard cap 0.9, quantized to the grid.
4. The five authority points as run (v1 run, `experiments/results.json`
   + `run_all.py`). Each is a distinct grid value with a known
   distinct numerical difference:
   - 0.5 birth (`BIRTH_AUTHORITY`)
   - 0.8 clean consolidation (`CLEAN_POOL` textbook 0.9 + web 0.5
     -> mean 0.7 + 0.1 bonus; also the E5 transfer mean)
   - 0.9 poison seating (`POISON_POOL` liar 0.9)
   - 0.6 repair seating (0.9 - dock 0.3; E2 post-repair lm_head
     authorities all 0.6)
   - 0.7 decay +6 (E3 trajectory: deficit 0.3 - 0.01*6 = 0.24,
     0.9 - 0.24 = 0.66 -> q10 0.7)
   That is the five the paper's section 5.8 counts. Two states
   (fast, slow) + 5 points = 7 points; 7 choose 2 = 21 distinct
   configurations as run. With 10 points on the slow trace it would
   be 12 choose 2 = 66, not claimed in v1.

## Definitions (the measurement)

- Granularity g: the authority grid step. v1: g = 0.1 (`q10`).
- Point: one distinct authority value held on the slow trace.
- Volume per point V_point = g, the width of one authority cell.
  At v1: 0.1. This is the "volume is point one" of the law.
- Numbers per volume at reading resolution r: N = g / r. At the
  record's own resolution (r = g) each volume holds exactly 1
  number, and that number assigns to 1 weight.
- Total volume V_total = n_points * g. At v1 (5 points): 0.5.
- Configurations (paper section 5.8, kept as the as-run count):
  C(2 + n_points, 2). At v1: C(7,2) = 21. At 10 points: C(12,2) = 66.
- Max distinct points at granularity g: 1/g + 1 grid values in
  [0, 1]. Max configurations at granularity g: C(3 + 1/g, 2).
- Weights per volume, sub-interval chop: a volume of width g
  chopped into pieces of width c yields W = g / c numbers, each
  assignable to a weight. Finer chop c -> more weights.
- Weights per volume, digit chop: the volumetric number in [0, 1)
  written to d decimal places yields d + 1 digits (1 integer digit
  + d fractional digits); each digit is a number assignable to a
  weight. W_digit = d + 1. Finer chop d -> more weights.
- Implementation ceiling: the bench stores numbers as float64.
  The law is unbounded as the chop refines without end; the
  float64 implementation caps the chop at the float64 spacing in
  the volume. Both are measured and reported (M7). The ceiling is
  a property of the implementation, not of the law.

## Measurements

All deterministic. No RNG is consumed except the fixed model-build
seed in M2 (seed 0, the same seed `run_all.py` E1 uses).

- M1: verify the five v1 points from the code. Call
  `seat_authority` with the run's pools and docks, and read the
  birth authority and the E3 decay value from the code path.
  Expect {0.5, 0.6, 0.7, 0.8, 0.9}.
- M2: structural counts. Build the v1-size pair with
  `build_pair(CFG, 0)`, consolidate once with `CLEAN_POOL`, count
  param_count, KIDs, record rows after 1 consolidation. Expect
  105,248 / 1,184 / 2,368. Snapshot bytes = param_count * 8.
- M3: grid seating at finer granularities. For g in
  {0.1, 0.01, 0.001, 0.0001}: parameterize the record quantizer
  (monkeypatch `kt_neuron.record.q10` and `kt_bench.seating.q10`
  in the measurement script only; bench files untouched), seat
  every grid value in [0, 1] into one fresh `KRecord` in increasing
  order from distinct sources, count seated rows, verify the chain
  with `rec.verify()`, measure the record's JSON bytes. Expect
  rows = 1/g + 1 = {11, 101, 1001, 10001}. Max configurations =
  C(3 + 1/g, 2) = {78, 5253, 502503, 50025003}.
  Amendment (found on the first run, 2026-11-26): `KRecord.verify()`
  hardcodes the 0.1 grid as an invariant, so at any finer g it fails
  on the grid assertion ("authority off the 0.1 grid") and never
  reaches the later rows. The `verify()` string is reported as
  measured, AND the hash/linkage/contiguity checks are re-run
  directly (structural check, same checks minus the grid assertion)
  so a finer-grained record's integrity is still measured rather
  than assumed broken.
- M4: points scaling at fixed g = 0.1, n_points in {1, 2, 5, 10, 11}.
  V_total = n * 0.1; configurations = C(2 + n, 2) =
  {3, 6, 21, 66, 78}. n = 11 is the grid cap at g = 0.1 (11 grid
  values); holding a 12th distinct point requires a finer grid.
- M5: sub-interval chop at g = 0.1, chop fineness c in
  {0.1, 0.01, 0.001, 1e-4, 1e-5, 1e-6}. Enumerate piece midpoints
  in a 0.1 volume with numpy float64 and count distinct values.
  Expect weights per volume = g / c = {1, 10, 100, 1000, 10000,
  100000}. Total weights from the 5 v1 points = 5 * g / c.
- M6: digit chop on the volumetric numbers 0.1 and 0.8, at d in
  {1, 2, 3, 6, 10, 15} decimal places. Count digits of the written
  number. Expect weights per volume = d + 1 = {2, 3, 4, 7, 11, 16}.
- M7: float64 ceiling. `np.spacing(0.5)` and the exact count of
  representable float64 values in [0.5, 0.6] by bit arithmetic on
  the float64 patterns. This is the largest chop the implementation
  can distinguish in one v1 volume.

## What is NOT done here

- No parameter-count race at equal accuracy against a plain
  baseline. Ordered out by Darren, Nov 26.
- No edit to the paper draft, the PDF, or the repo README.
- No publishing anywhere (no Zenodo, no alphaXiv).

## Commands

Run from `~/workspace/kt-neuron`:

    python3 experiments/measure_scar_volume.py

Writes `experiments/results-memory-equal-accuracy.json`. The
human-readable report is `experiments/results-memory-equal-accuracy.md`.
Script: `experiments/measure_scar_volume.py` (one focused module,
under 500 lines). Seeds: model build seed 0 (M2); everything else
is fixed grid values with no RNG.
