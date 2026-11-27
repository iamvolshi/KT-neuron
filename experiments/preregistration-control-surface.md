# Preregistration: the slow-trace volume as a control surface

Date: 2026-11-27. Author: bench run for the KTN work.
Status: preregistered BEFORE the measurement script ran.

## Darren's idea (Nov 27, 1:42-1:46 AM, his words, settled)

The scar volume is predictable, so it is a CONTROL SURFACE, to be
exploited. At full trust (authority 1.0) is the baseline, at zero
trust (0.0) the deepest scar. Every step down from 1 is a different
number, a different string of weights, a new thread of numbers. The
KT neuron itself is the machine, and the predicted number string is
part of it.

"Take the next number down from one, is it the same number as one,
no it's a different number, that's a different string of weights.
Go down one step again, is it the same number as the one above it,
no it's not, that means it's new, that means every step is a new
string, a new thread of numbers."

Then: map a graph to the weights it can control. How many different
weight states can it pick, could it point to? Darren's guess is "a
lot"; ten weight states out of one neuron would already be
phenomenal. The run decides the number; nothing here is steered
toward either guess.

## What this run measures

PART 1, THE WALK: walk the authority steps from 1.0 down to 0.0.
At each step, take the number and show it is a new string of
weights, a new thread. Count the threads and the weights the slow
trace can hold. Run at g = 0.1 first (11 steps), then at finer
grids 0.01, 0.001, 0.0001, the same granularities as the Nov 26
scar-volume run.

PART 2, THE GRAPH MAP: once the thread count is known, map a graph
to the weights it can control. Nodes are the threads (number strings
as addresses, generalizing the Nov 26 address-routing test, where 5
addresses routed to 3 weight tables). Each thread points to weight
states. Count how many DIFFERENT weight states the graph can pick /
point to.

## Definitions (fixed before measuring)

- Step: one grid value v = 1.0 - i*g, i = 0..1/g. Steps at grid g:
  1/g + 1 = {11, 101, 1001, 10001}. Baseline = step 1.0 (full
  trust). Deepest scar = step 0.0 (zero trust). The walk is defined
  top-down (1.0 -> 0.0); the record seating runs bottom-up (0.0 ->
  1.0) because the authority gate only seats equal-or-stronger
  claims, so a decreasing seating would record candidates, not
  seats. The thread set is the same either way; the walk order is
  a reading of it, not a seating order.
- Thread: one step. Thread address = the step number. Every step
  is a thread; every thread is a new string (that is the claim
  under test, W3/W6).
- Digit string (primary string): the step number written to d
  decimal places; its digits are d + 1 numbers (1 integer digit +
  d fractional digits). Each digit is a number assignable to a
  weight, the Nov 26 M6 digit chop. Weights per thread (digit) =
  d + 1. Total digit weights at grid g = (1/g + 1) * (d + 1).
- Cell: the half-open interval [v, v+g) anchored at grid value v
  for v in {0, g, ..., 1-g}. There are 1/g cells; they tile [0, 1].
  Sub-interval chop of a cell at fineness c gives g/c distinct
  midpoints, the Nov 26 M5 chop. Weights per cell = g/c. Total
  sub-interval weights across all cells = (1/g) * (g/c) = 1/c.
  (The top step 1.0 anchors no cell; it is the ceiling. The digit
  string applies to every step including 1.0 and 0.0.)
- Deterministic: the walk run twice gives identical output.
- Predictable (the control-surface claim): the thread string for
  step i is a pure function of i (and g, c, d), computable in
  advance without running the record. Choosing a step chooses a
  thread. Verified by computing the strings arithmetically from
  the step indices and comparing against strings computed from
  authorities actually seated in a real KRecord (W6). If they
  agree everywhere, the operator can predict the thread a step
  selects before seating it.
- Control: step i selects thread i and no other. Shown by taking
  a step and showing its string, table, and outputs differ from
  its neighbours above and below, and are stable across runs (W7).
- Graph: nodes = threads (steps, addressed by their digit string).
  Table per node = the node's digit string as a weight vector
  (length d + 1). Routing: node i -> table i (identity; this
  generalizes the routing test's index -> table map, where the
  tables were fixed and few. Here the tables come from the
  threads themselves, so no sharing is assumed and none is
  hidden).
- Weight state: a distinct weight-table configuration addressable
  by a thread, i.e. a distinct table tuple. Distinct weight
  states = the count of distinct table tuples actually produced
  (measured, not assumed from the thread count; two threads could
  in principle produce the same table).
- Behavioural state: a distinct output on the routing test's
  input (1, 1). Two outputs are measured: the 2-weight output,
  sigmoid of the table's first two weights (the routing test's
  convention), and the full-sum output, sigmoid of the sum of all
  the table's weights. Distinct tables can share an output; both
  counts are reported, and the table count is the answer to "how
  many weight states can it point to".
- Held configuration: a pair of steps the neuron can hold at once.
  The paper's section 5.8 counts configurations as C(2 states +
  n points, 2) because the neuron holds 2 states in each tick.
  Held configurations at grid g = C(2 + n_steps, 2), exact integer
  arithmetic. This is the neuron's held-state space, NOT the
  graph's addressable count; the two are reported separately and
  never conflated.
- Float64 ceiling: the implementation stores numbers as float64.
  The law is unbounded as the chop refines without end; the
  float64 implementation caps it. The ceiling is measured by bit
  arithmetic on the float64 patterns in [0, 1] (same method as
  Nov 26 M7), cross-checked against the binade formula.

## Measurements

All deterministic. No RNG is consumed anywhere.

- W1: steps per grid. Expect {11, 101, 1001, 10001}. List the
  first and last steps; the walk is 1.0 down to 0.0.
- W2: KRecord seating at each grid. One fresh KRecord per grid
  (grid parameter, Nov 26 gap-12 fix), every grid value seated in
  increasing order from distinct sources through the real
  `replace` path, chain verified with `rec.verify()` and with an
  independent structural re-check (hash, linkage, contiguity;
  same checks as Nov 26 M3, re-implemented locally so this script
  needs no model-build imports). Expect rows = 1/g + 1, verify
  OK, structural OK. Measure record JSON bytes and bytes/row.
- W3: digit strings at d in {1, 2, 3, 6, 10, 15} (the Nov 26 M6
  places). For each grid and d: distinct digit strings (expect
  all steps distinct when d carries the grid's resolution; at low
  d, rounding collides, and the collision count is the measured
  result, not a failure), weights per thread = d + 1, total digit
  weights = steps * (d + 1).
- W4: sub-interval chop per cell at c in {0.1, 0.01, 0.001, 1e-4,
  1e-5, 1e-6}, only where c <= g. Distinct midpoints per cell
  (expect g/c, matching the Nov 26 M5 numbers at g = 0.1),
  weights per cell = g/c, total = 1/c.
- W5: determinism. Run the pure-arithmetic measurements
  (W1, W3, W4, G1, G2, G3) twice in-process and compare canonical
  JSON; expect identical. The whole script is also re-run as a
  separate process and the JSON diffed; expect identical.
- W6: predictability. Digit strings computed arithmetically from
  step indices vs digit strings computed from the authorities
  actually seated in the W2 KRecord at each grid (d = 15).
  Expect equal for every step at every grid.
- W7: control. Take step 0.7 at g = 0.1 (a known v1 point, decay
  +6) and the same-valued step at each finer grid. Show its digit
  string, table, and both outputs differ from the step above
  (0.8) and the step below (0.6), and are identical across the
  two W5 runs. Expect all distinct from neighbours, stable
  across runs.
- G1: graph tables. At each grid and each d: build the table per
  node (digit string as weight vector), count distinct table
  tuples. Expect == steps at d = 15 (full resolution); at low d
  the collision count is measured.
- G2: graph outputs. At each grid, d = 15: count distinct
  2-weight outputs and distinct full-sum outputs on input (1, 1).
  Expect measured values (the 2-weight output is coarse and will
  collide; the full-sum output distinguishes more; neither count
  is assumed).
- G3: held configurations. C(2 + n_steps, 2) per grid, exact
  integers. Expect {78, 5253, 502503, 50025003} (same arithmetic
  as Nov 26 M3 max_configurations, cross-checked).
- G4: float64 ceiling in [0, 1]. Exact count by bit arithmetic
  (bits(1.0) - bits(0.0) + 1), cross-checked against the binade
  formula 1023 * 2^52 + 1. Max threads, max addressable weight
  states (identity routing), and max held configurations
  C(2 + N, 2) at that ceiling, exact integers.

## What is NOT done here

- No parameter-count race at equal accuracy. Ordered out Nov 26.
- No edit to the paper draft, the PDF, the repo README, or any
  published record. Bench only.
- No publishing anywhere (no Zenodo, no alphaXiv).
- No steering: the graph count is measured under the definitions
  above. Ten states is Darren's phenomenal bar; millions is his
  guess. The run reports what it finds against both.

## Commands

Run from `~/workspace/kt-neuron-repo` (the git clone of
iamvolshi/KT-neuron; `~/workspace/kt-neuron` is a non-git working
copy with identical bench code):

    python3 experiments/measure_control_surface.py

Writes `experiments/results-control-surface.json`. The
human-readable report is `experiments/results-control-surface.md`.
Script: `experiments/measure_control_surface.py` (one focused
module, under 500 lines). Seeds: none. No RNG anywhere.

## Addendum (Nov 27): richer readout for G2

The G2 result has one caveat: the graph points to 10,001 distinct
weight states at g = 0.0001, but the routing toy's 2-weight output
can only tell 10 apart and the full-sum output 16. The states are
distinct as weight configurations; the readout is coarse.

This addendum adds a third, richer readout to G2, measured the
same way (d = 15, each grid). Nothing else changes.

- Definition (fixed before measuring): the linear readout is the
  dot product of the table's full weight vector (the digit string,
  length n = d + 1) with fixed positional weights, powers of 10:
  w_i = 10^(n-1-i) for position i in 0..n-1. This reconstructs
  the digit string's numeric value scaled by 10^d: the integer
  whose decimal digits are the table's digits. Computed with
  exact integer arithmetic (no float rounding), so the readout
  value is a pure function of the digit string. The readout's
  output IS that reconstructed number (no sigmoid squash, which
  would saturate at these magnitudes); distinct outputs are
  distinct reconstructed values.
- Measurement: count distinct linear-readout outputs across the
  tables at each grid, d = 15. Expectation, not an assumption:
  all tables separate, because the tables are distinct strings
  and the readout is injective on digit strings. The count is
  MEASURED, and if it comes out lower than the distinct-table
  count, that is reported honestly and the readout is not
  adjusted to force it.
- The new readout joins the 2-weight and full-sum outputs in G2.
  Determinism (W5) covers it automatically, since g2_outputs is
  in the pure measurements run twice.
