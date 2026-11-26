# KTN bench assembly (Phase 0 + Phase 1)

The bench model, put together from the finished KT neuron layer -- with
intention, per the plan's discipline. Assembly only: no experiments, no
controller, no demo.

## Files

- `kt_bench/linear.py` -- vanilla `Linear` (the control) and `KTLinear`:
  each output row is a KT neuron (one KID per row, one K-record per
  row, cold beside the tensors). Forward is
  `y = (x @ W.T + b - scar) * gain`; at gains 1 / scars 0 it matches
  the vanilla layer bitwise. A row's belief value is the SHA-256 of its
  row weights plus its bias; the full row lives in snapshots.
- `kt_bench/transformer.py` -- small nanoGPT-class transformer (pre-norm
  blocks, causal multi-head attention, GELU MLP, final norm, untied
  lm_head), parameterized by linear class so the twin shares every
  other line of code. `perturb()` is the fast trace: it moves live
  tensors and touches no record.
- `kt_bench/store.py` -- `RecordStore` (bench registry over every row's
  K-record, verifies all chains) and `SnapshotStore` (full tensor
  checkpoints keyed by bench k; restore is exact because snapshots are
  copies).
- `kt_bench/consolidation.py` -- `Consolidator`: the ONLY place k
  advances. One call seats every row's current values, snapshots the
  full state, and advances bench k by exactly one. Steps never advance
  k; consolidation never does anything except seat, snapshot, advance.
- `kt_bench/seating.py` -- generic placeholder seating policy (see below).
- `kt_bench/twin.py` -- `build_pair` (same seed -> identical initial
  weights), deterministic batches, init/forward parity checks.
- `tests/test_kt_bench.py` -- 13 tests, numpy + stdlib only.

## Bench config (default)

`GPTConfig`: vocab 128, block 32, n_embd 64, n_head 4, n_layer 2.
**118,656 parameters** (asserted exactly by test). One KID per neuron:
1,280 KIDs per model (576 per block x 2 + 128 lm_head). Full test suite
(37 tests: 13 bench + 24 neuron) runs in under a second on CPU.

## Run the tests

From `~/workspace/kt-neuron`:

    python3 -m unittest discover -s tests -v

## Seating: generic placeholder policy (Darren, Oct 26)

- **Seating rule**: `seat_authority(pool)` is a basic generic policy,
  not ARCHE's rule: mean of declared source authorities; +0.1 for
  multi-source agreement; -0.3 for a known-bad source in the pool;
  cap 0.9, floor 0.0, 0.1 grid; empty pool seats at birth authority
  0.5. Per Darren's Oct 26 direction it will be REPLACED by ARCHE's
  feedback-controller seating -- that rule is his to formulate, and
  this policy designs no part of it. Per-row source attribution
  (which sources taught which rows) is later work; pools here are
  caller-declared, not measured.
- **Per-epoch consolidation source**: consolidation seats with source
  `"consolidation:<k>"`. Under the record's own gate, an
  equal-authority claim from a different source re-seats, and each
  consolidation epoch genuinely is a distinct seating event -- so
  successive consolidations seat instead of reaffirming. If a future
  seating function ever returns a candidate, consolidation fails loudly
  rather than silently dropping a belief.
- **Gain-from-authority mapping**: consolidation sets each seated
  row's gain to its seated authority. Gains are wired into forward
  and tested for effect.
- **Scar dynamics**: the scar vector is mechanism only (settable,
  tested). Injury/heal dynamics belong to the later injury playbook.

## Left for later (not this job)

Phase 3 experiments (poison/liar source, repair cost, abstention
probes), full twin training, the attention engine, the feedback
controller, and the side-by-side KT-vs-normal demo (which comes when
the differential is obvious, after experiments).

## Small choices made

- numpy only: no torch in this environment. The bench is forward-only
  assembly; no autograd is needed for it.
- Untied lm_head (it needs its own KIDs; tying would share neurons).
- No dropout: the bench must be deterministic.
