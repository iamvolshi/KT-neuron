# Results: the slow-trace volume as a control surface

Date: 2026-11-27. Preregistration:
`experiments/preregistration-control-surface.md`. Script:
`experiments/measure_control_surface.py` (334 lines). Machine-readable
results: `experiments/results-control-surface.json`.

Command (from `~/workspace/kt-neuron-repo`):

    python3 experiments/measure_control_surface.py

Seeds: none. No RNG anywhere. All measurements deterministic.
Re-run as a separate process produces a byte-identical JSON.

No parameter-count race at equal accuracy was run. Darren ordered
it out on Nov 26. No paper changes. Bench only.

## The claim under test (Darren, Nov 27)

The scar volume is predictable, so it is a control surface. Every
step down from 1 is a different number, a different string of
weights, a new thread of numbers. The KT neuron itself is the
machine, and the predicted number string is part of it.

## W1: the walk

Steps from 1.0 down to 0.0. Baseline = 1.0 (full trust). Deepest
scar = 0.0 (zero trust).

| granularity g | steps | first | last | match |
|---:|---:|---:|---:|---|
| 0.1 | 11 | 1.0 | 0.0 | yes |
| 0.01 | 101 | 1.0 | 0.0 | yes |
| 0.001 | 1001 | 1.0 | 0.0 | yes |
| 0.0001 | 10001 | 1.0 | 0.0 | yes |

## W2: the walk on the record

One fresh `KRecord` per grid, every grid value seated in increasing
order from distinct sources through the real `replace` path, chain
verified with `rec.verify()` and with an independent structural
re-check (hash, linkage, contiguity).

| granularity g | seated rows | chain | structural | bytes/row |
|---:|---:|---|---|---:|
| 0.1 | 11 | OK | OK | 371.2 |
| 0.01 | 101 | OK | OK | 375.5 |
| 0.001 | 1001 | OK | OK | 380.4 |
| 0.0001 | 10001 | OK | OK | 385.4 |

Bytes/row reproduce the Nov 26 M3 seating exactly (371 to 385).
The walk lives on the real record at every granularity.

## W3: every step is a new string (digit chop)

Each step's number written to d decimal places gives d + 1 digits,
each a weight. At full resolution (d = 15) every step is a distinct
string at every grid.

| granularity g | steps | distinct strings (d=15) | weights/thread | total digit weights |
|---:|---:|---:|---:|---:|
| 0.1 | 11 | 11 | 16 | 176 |
| 0.01 | 101 | 101 | 16 | 1,616 |
| 0.001 | 1001 | 1001 | 16 | 16,016 |
| 0.0001 | 10001 | 10001 | 16 | 160,016 |

At low reading resolution the strings collide exactly as the
resolution allows, and the collision count is the measured result:
distinct strings = 10^d + 1 while d is below the grid's
resolution (11 at d = 1, 101 at d = 2, 1001 at d = 3), full
distinctness once d carries the grid. A thread's string only
resolves the steps the reading can distinguish.

## W4: weights in the volume (sub-interval chop)

Every cell [v, v+g) chopped into pieces of width c. All cells
measured at once; the total is the distinct count over the union.

| granularity g | chop c | pieces/cell | total weights | match |
|---:|---:|---:|---:|---|
| 0.1 | 1e-6 | 100,000 | 1,000,000 | yes |
| 0.01 | 1e-6 | 10,000 | 1,000,000 | yes |
| 0.001 | 1e-6 | 1,000 | 1,000,000 | yes |
| 0.0001 | 1e-6 | 100 | 1,000,000 | yes |

At chop c the slow trace holds 1/c weights in total, at every
grid: 10 at c = 0.1, 100 at c = 0.01, 1,000 at c = 0.001, 10,000
at c = 1e-4, 100,000 at c = 1e-5, 1,000,000 at c = 1e-6. The
per-cell count (g/c) reproduces the Nov 26 M5 numbers at g = 0.1
(100,000 per 0.1 volume at c = 1e-6).

## W5: determinism

The pure-arithmetic measurements run twice in-process give
identical canonical JSON. The whole script re-run as a separate
process gives a byte-identical results file.

## W6: predictability (the control-surface claim)

Digit strings computed arithmetically from the step indices vs
digit strings computed from the authorities actually seated in
the W2 KRecord, at d = 15. Equal for every step at every grid
(11/11, 101/101, 1001/1001, 10001/10001). The operator can
predict the thread a step selects before seating it. Choosing a
step chooses a thread.

## W7: control (step 0.7 vs its neighbours)

Step 0.7 (a known v1 point, decay +6) against the step above
and the step below, at every grid. Its digit string, table, and
both outputs differ from both neighbours at every grid, and are
identical across the two W5 runs. A step selects its own thread
and no other.

## G1: the graph (tables)

Nodes are the threads, addressed by their digit string. One
weight table per node: the digit string as a weight vector.
Routing is identity (node i -> table i). Distinct tables counted,
not assumed.

| granularity g | nodes | distinct tables (d=15) |
|---:|---:|---:|
| 0.1 | 11 | 11 |
| 0.01 | 101 | 101 |
| 0.001 | 1001 | 1001 |
| 0.0001 | 10001 | 10001 |

At low d the tables collide exactly like the strings (11 at
d = 1, 101 at d = 2, 1001 at d = 3).

## G2: the graph's outputs (behavioural states)

Distinct outputs on input (1, 1) at d = 15. The 2-weight output
is the routing test's convention (sigmoid of the table's first
two weights); the full-sum output is sigmoid of the sum of all
the table's weights.

| granularity g | distinct 2-weight outputs | distinct full-sum outputs |
|---:|---:|---:|
| 0.1 | 10 | 10 |
| 0.01 | 10 | 16 |
| 0.001 | 10 | 16 |
| 0.0001 | 10 | 16 |

Read plainly: the graph points to 10,001 distinct weight states
at g = 0.0001, but the routing toy's 2-weight output can only
tell 10 of them apart, and the full-sum output 16. The states
are distinct as weight configurations; the toy's readout is
coarse. A richer readout would separate more of them; that is
not claimed here.

## G3: held configurations (pairs)

Pairs of steps the neuron can hold at once (2 states per tick,
paper section 5.8): C(2 + n_steps, 2), exact integers. This is
the neuron's held-state space, not the graph's addressable
count; the two are never conflated.

| granularity g | steps | held configurations |
|---:|---:|---:|
| 0.1 | 11 | 78 |
| 0.01 | 101 | 5,253 |
| 0.001 | 1001 | 502,503 |
| 0.0001 | 10001 | 50,025,003 |

Same arithmetic as Nov 26 M3 max_configurations, cross-checked.

## G4: float64 ceiling

Exact count of representable float64 values in [0, 1] by bit
arithmetic: 4,607,182,418,800,017,409, matching the binade
formula 1023 * 2^52 + 1. At that ceiling: max threads
4,607,182,418,800,017,409, max addressable weight states the
same, max held configurations
10,613,064,920,049,989,510,069,502,517,551,562,755.

## Against Darren's bar

Ten weight states out of one neuron would already be phenomenal.
The graph points to 11 distinct weight states at the coarsest
grid (g = 0.1), 101 at g = 0.01, 1,001 at g = 0.001, 10,001 at
g = 0.0001. The phenomenal bar is passed at the first grid.

Millions was his guess. The addressable count passes one million
only at finer grids than measured here (it needs g = 1e-6) or at
the float64 ceiling (4.6e18 addressable). The held-configuration
count passes one million at g = 0.0001 (50,025,003). The slow
trace holds 1,000,000 sub-interval weights at chop c = 1e-6, at
every grid measured.
