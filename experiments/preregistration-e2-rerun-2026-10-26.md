# E2 RERUN — Pre-registration (frozen BEFORE any rerun run, 2026-10-26)

A NEW round, not a rescue of round 1. Round-1 verdicts STAND and are
never rewritten: E2 FAILED on the strict step-cost leg (repair 10 vs
full-retrain 10, TIE at 5-step check granularity). This rerun asks a
different, better-measured cost question: WORK TOUCHED, not steps
alone. Whatever this rerun finds, round-1's FAIL stands.

This file is frozen before running. No threshold in it may be changed
after a run. Any post-run correction goes in a dated amendment that
does not alter a frozen threshold, and a rerun under changed rules
would be a further new round, not an edit of this one.

## Flip-status declaration (required, frozen)

**The stage-1 flip-to-slow mechanism (`kt_bench/flip.py`) is NOT
engaged during this rerun.** Round-1 behavior is preserved:
firewall nulls gains, the fast trace continues. Reason: setup
comparability with round 1 is the point of this rerun; engaging the
flip would change the repair start state (weights rewound to the
last clean snapshot) and the comparison would no longer be
round-1-like. Flip engagement is a later stage's question, not
smuggled into this rerun. No mid-run switch: if the flip were
engaged it would have to be declared here, before running, and it
is not.

## Setup carried over unchanged from round-1 E2

- Config: GPTConfig(vocab_size=32, block_size=16, n_embd=64,
  n_head=4, n_layer=2). Parameter count, KID count, and row counts
  are computed and reported by the harness, not hand-asserted here.
- Model init seed 0 (build_pair). Data seeds: clean batches rng 1,
  poison batches rng 2, repair batches rng 3, retrain batches rng 4
  — the same batch lists as round 1, in the same order, so the
  clean/poison phases reproduce round 1 exactly.
- Batch 8 sequences x 16 tokens. Adam lr 0.01, betas (0.9, 0.999),
  eps 1e-8. Loss: cross-entropy on all positions.
- Schedule: clean phase 400 steps; poison phase 200 steps;
  repair up to 200 steps; full retrain up to 400 steps.
- Corpus, facts, and liar exactly as round 1: counting grammar
  tokens 0..15; facts F1 (20,21)->3/9, F2 (22,23)->4/10,
  F3 (24,25)->5/11, F4 (26,27)->6/12; liar-only LX (28,29)->13.
  Sources (placeholder policy, caller-declared): textbook 0.9,
  web 0.5, liar 0.9 (presents as textbook-grade; that is why it
  is dangerous). Consolidation pools: clean {"textbook": 0.9,
  "web": 0.5} -> seated 0.8; poison {"liar": 0.9} -> 0.9.
- Poison protocol: same liar-mixed batches for both models,
  lockstep, then one KT consolidation with the poison pool
  ("consolidation:1").
- Repair protocol: retrain ONLY affected rows — lm_head rows for
  fact tokens (exact by construction: true tokens 3,4,5,6 and
  liar tokens 9,10,11,12) plus top-20 upstream rows by |dW|
  during the poison interval (labeled APPROXIMATE, as round 1) —
  all other rows frozen by zeroed gradients. Re-seat repaired
  rows with authority dock 0.3 (seated 0.6 expected, as round 1).
  Full retrain (vanilla): whole model on clean data.
- Recovery criterion (unchanged): mean P(true) on the four fact
  probes >= that model's own clean-baseline mean P(true) - 0.05.
  Repair is judged against the KT baseline; retrain against the
  vanilla baseline, exactly as round 1. Hitting the cap without
  recovery is a FAIL on that leg, reported as such, not a silent
  extension.
- Gradient-check gate (unchanged): the trainer's numerical
  gradient check must PASS (amended A6 criterion: rel err < 1e-4
  OR abs diff < 1e-9 per entry, forward diff < 1e-12) before any
  training run, else all training results are VOID.

## The ONE deliberate measurement change (pre-registered, not post-hoc)

Round 1 checked recovery every 5 steps (plus step 1). This rerun
checks recovery EVERY step (1-step granularity) for both repair
and retrain. Reason: the round-1 tie lived exactly at the 5-step
granularity — a step-count tie at coarse granularity says nothing
about work. Finer checking can only move each step count down to
its true first-recovery step; it cannot manufacture a work saving,
because the work leg below multiplies counted params by the
measured steps. Step counts from this rerun are therefore NOT
directly comparable to round-1's 10/10 as step numbers; the work
comparison is the rerun's verdict leg, and steps are reported
alongside for transparency.

## NEW cost leg — WORK TOUCHED (frozen definitions)

Work touched is OPTIMIZER-APPLIED work: parameters the optimizer
actually writes. Definitions, counted by the harness, not estimated:

- `rows_updated (design)`: repair = number of (layer, row) pairs
  in the repair mask; retrain = all linear rows (all KIDs).
- `rows_moved (actual)`: rows whose W row or bias value differs
  bit-wise between pre-training-loop and post-training-loop
  weights (measured BEFORE re-seating, since re-seating changes
  gains/records, not weights). Frozen rows must show zero moved
  rows outside the mask; any leak is reported as a rig failure.
- `params_updated_per_step`: repair = number of ones in the
  gradient mask (counted, not asserted); retrain = total model
  parameter count (counted by the harness).
- `params_moved (actual)`: parameter entries differing bit-wise
  pre vs post training loop. Reported as a cross-check on the
  design counts; the verdict uses the design per-step counts,
  because a zero-gradient entry on one step is still work the
  optimizer was asked to do — the mask, not the accident of a
  zero gradient, is the surgical boundary.
- `total_param_updates = params_updated_per_step x steps_to_recovery`,
  with steps_to_recovery measured at 1-step check granularity
  above. This is the work-comparison quantity for both sides.

### FLOPs — why they are NOT counted here (stated before running)

Exact FLOP savings are NOT claimed in this rerun, for a harness
reason, not a hand-wave: this numpy harness computes FULL
gradients for every parameter on every step (`loss_and_grads`)
and THEN zeroes the frozen rows' gradients with the mask. So a
repair step in this harness pays essentially the same backward
cost as a retrain step; the saving is in optimizer writes (which
rows/params move), not in backward FLOPs. Claiming a FLOP saving
from this harness would be false. A FLOP saving would require a
sparse/frozen-gradient implementation that skips frozen rows'
backward work entirely — that implementation does not exist here
and is not measured here. It is left for later work, named as
such, not smuggled in as a number.

### Frozen PASS/FAIL threshold for the work comparison

**PASS (work leg) requires:
`total_param_updates(repair) < 0.10 x total_param_updates(retrain)`.**

Justification in one line: surgical repair must cost an order of
magnitude less optimizer work than retraining the whole model,
because the injury touches a small fraction of rows, not the
model — a 10x margin is the honest form of "proportional to the
injury, not the model."

## Behavior legs (comparability with round 1, same thresholds)

These legs are reported so the rerun is comparable to round 1;
they do not change round-1's verdict whatever they show here:

1. Vanilla flip: post-poison mean P(liar) > mean P(true) AND
   top-1 = liar on >= 3/4 facts.
2. KT lineage: the lineage query returns exactly the poison
   intervals (no missing, no extras, at interval granularity).
3. Firewall: pot trajectory from tainted batches reaches the 3.0
   engagement threshold; while engaged, liar-lineage gains are
   held at 0 (null); quarantine probes and post-release probes
   reported.
4. Quarantine: nulling liar lm_head rows moves fact probes back
   toward clean behavior; unnulling restores standing gains
   without weight changes (weights bit-identical before/after
   the null/unnull round-trip).
5. Repair seating: repaired rows re-seat (count seated of total
   attempted reported); post-repair lm_head authorities equal
   0.6 (0.9 - dock 0.3); chain verification ok, no bad rows.
6. Firewall restore exactness: weights (and gains, after reset
   to standing) after the firewall-demo restore are bit-identical
   to the pre-demo post-poison state — checked by direct tensor
   comparison, NOT by the round-1 probe-based flag (which was a
   check-design flaw: it compared restored probes against a
   pre-consolidation baseline with different gains).

## What this rerun does NOT do

- Does not overwrite any round-1 file (preregistration.md,
  results.md, results.json, results_run1.json, results_run2.json).
- Does not modify kt_neuron/ modules, kt_neuron.py,
  KT-LLM-PLAN.md, MEMORY.md, or kt_bench/seating.py. No seating
  redesign, no ARCHE feedback controller, no decline rule, no
  demo. New code lives only under experiments/ for this rerun.
- Does not engage the flip (see declaration above).
- Does not change any threshold after a run. If this rerun's
  work leg fails, it fails, and that failure is reported plainly.

## Outputs (new files only)

- This file: experiments/preregistration-e2-rerun-2026-10-26.md
- Harness: experiments/run_e2_rerun.py (new; reuses trainer.py,
  corpus.py, playbook.py unchanged)
- Results: experiments/results-e2-rerun.md and
  experiments/results-e2-rerun.json
