# E3 RERUN — Pre-registration (frozen BEFORE any rerun run, 2026-10-26)

A NEW round, not a rescue of round 1. Round-1 verdicts STAND and are
never rewritten: E3 FAILED on its frozen confidence threshold
(post-repair mean P(true) 0.820 < baseline − 0.10 = 0.878), even
though ranking restored (top-1 true 4/4), authority lagged
(0.8 → 0.6, lag 0.2 ≥ 0.1), and authority decayed to 0.7 by +6
consolidations (deficit 0.30 → 0.24).

## Why this rerun exists (the round-1 diagnosis, stated, not assumed)

Round-1's E3 threshold conflated two different things. The dock
itself softens confidence: repaired rows' gains are 0.6 < 1, which
scales fact logits down, so behavior restores in RANKING (top-1
4/4) while CONFIDENCE tracks authority as it heals (0.820 → 0.881
as seated authority 0.6 → 0.7 across +6 clean consolidations).
Ranking and confidence are therefore measured here as SEPARATE
bars with SEPARATE verdicts. Ranking can pass while confidence is
still healing, or vice versa, and each verdict is reported on its
own. There is NO single combined threshold in this rerun: any
combined pass/fail claim would re-conflate them and is forbidden
by this file.

This file is frozen before running. No threshold in it may be
changed after a run. Any post-run correction goes in a dated
amendment that does not alter a frozen threshold, and a rerun
under changed rules would be a further new round, not an edit of
this one.

## Flip-status declaration (required, frozen)

**The stage-1 flip-to-slow mechanism (`kt_bench/flip.py`) is NOT
engaged during this rerun.** Round-1 behavior is preserved:
firewall nulls gains, the fast trace continues, repair re-seats
from the poisoned-then-repaired state. Reason: setup
comparability with round 1 is the point of this rerun; engaging
the flip would rewind weights to the last clean snapshot and the
repair start state — and therefore the confidence trajectory —
would no longer be round-1-like. Flip engagement is a later
stage's question, not smuggled into this rerun. No mid-run
switch: if the flip were engaged it would have to be declared
here, before running, and it is not.

## Setup carried over unchanged from round-1 E3

- Continues E2's KT run (same run, no fresh model): clean phase,
  poison phase, surgical repair, then tracked consolidations.
  The E2 mainline is reproduced exactly as round 1 (and as the
  E2 rerun, stage 2): clean consolidation ("consolidation:0",
  pool {"textbook": 0.9, "web": 0.5} → seated 0.8), poison
  consolidation ("consolidation:1", pool {"liar": 0.9} → seated
  0.9), firewall demo (branch, full-state snapshot restored
  afterwards, gains reset to standing — as round-1 run 2 / E2
  rerun), surgical repair of the affected rows only.
- Config: GPTConfig(vocab_size=32, block_size=16, n_embd=64,
  n_head=4, n_layer=2). Parameter count, KID count, and row
  counts are computed and reported by the harness, not
  hand-asserted here.
- Model init seed 0 (build_pair). Data seeds: clean batches rng 1,
  poison batches rng 2, repair batches rng 3 — the same batch
  lists as round 1, in the same order, so the clean/poison/repair
  phases reproduce round 1 exactly.
- Batch 8 sequences x 16 tokens. Adam lr 0.01, betas (0.9, 0.999),
  eps 1e-8. Loss: cross-entropy on all positions.
- Schedule: clean phase 400 steps; poison phase 200 steps;
  repair up to 200 steps with recovery checked EVERY step
  (1-step granularity, as the E2 rerun; round 1 checked at step
  1 and every 5 steps, and stopped at step 10 either way, so
  this does not change the repair state — it only measures the
  first-recovery step more precisely).
- Recovery criterion (unchanged): mean P(true) on the four fact
  probes >= KT clean-baseline mean P(true) − 0.05, judged during
  the repair loop BEFORE re-seating (re-seating then drops
  confidence by the dock — that drop is exactly what this rerun
  measures, not a contradiction).
- Corpus, facts, and liar exactly as round 1: counting grammar
  tokens 0..15; facts F1 (20,21)→3/9, F2 (22,23)→4/10,
  F3 (24,25)→5/11, F4 (26,27)→6/12; liar-only LX (28,29)→13
  (not probed here). Sources (placeholder policy,
  caller-declared): textbook 0.9, web 0.5, liar 0.9 (presents as
  textbook-grade; that is why it is dangerous). Consolidation
  pools: clean {"textbook": 0.9, "web": 0.5} → seated 0.8;
  poison {"liar": 0.9} → 0.9.
- Surgical repair (as round 1 / E2 rerun): retrain ONLY affected
  rows — lm_head rows for fact tokens (exact by construction:
  true tokens 3,4,5,6 and liar tokens 9,10,11,12) plus top-20
  upstream rows by |dW| during the poison interval (labeled
  APPROXIMATE, as round 1) — all other rows frozen by zeroed
  gradients. Re-seat repaired rows with authority dock 0.3
  (claim 0.9 − dock 0.3 = 0.6 seated expected, as round 1).
- Decay (as round 1, amendment A4): after repair, re-seat the
  affected rows at each of 6 further clean consolidations with
  the deficit reduced by 0.01 per consolidation:
  deficit(n) = 0.30 − 0.01·n, n = 1..6, so the seated claim is
  0.9 − deficit(n), quantized to the record's 0.1 grid
  (q10). No training steps run between decay points: a decay
  point is a re-seating event only, so probe changes across
  decay points are pure gain effects, not new learning. This is
  stated so the trajectory cannot later be misread as retraining.
- Gradient-check gate (unchanged): the trainer's numerical
  gradient check must PASS (amended A6 criterion: rel err < 1e-4
  OR abs diff < 1e-9 per entry, forward diff < 1e-12) before any
  training run, else all training results are VOID.
- Validity criterion (carried over): clean-phase fact top-1 must
  be >= 3/4 facts correct for the comparisons to be labeled
  STRONG evidence. Below that, results are still reported in
  full but labeled WEAK evidence with this caveat.

## Tracked points (all probed identically, same four fact probes)

- baseline (post-clean, pre-poison, KT)
- post-poison (KT)
- post-repair (after re-seating with dock 0.3)
- +1, +2, +6 clean consolidations (after re-seating with the
  reduced deficit for that n)

At each point the harness reports: mean P(true), mean P(liar),
top-1 true count, top-1 liar count, per-probe P(true)/P(liar)/
top-1, seated authority mean over the lm_head fact+liar rows
(true 3,4,5,6 + liar 9,10,11,12), raw deficit, and chain
verification (all_ok / bad rows). No point is skipped and no
point is added after running.

## Frozen bars — each verdictable independently

### Bar R — RANKING (independent verdict)

**PASS requires: post-repair top-1 true count == baseline
top-1 true count AND top-1 true count == baseline count at +1,
+2, and +6 consolidations.**

Baseline count is whatever the clean phase measures (round 1:
4/4). This bar asks only whether the correct token ranks first
on each fact — it is blind to confidence levels. A ranking PASS
with a confidence miss is a real, reportable split, not a
contradiction.

### Bar C — CONFIDENCE TRAJECTORY (independent verdict)

The confidence bar judges the TRAJECTORY as authority heals,
not instant recovery to baseline. Frozen expectation, stated
before running: because repaired gains start docked at 0.6 and
rise only as the deficit decays (and the 0.1 grid masks healing
until the seated value steps), mean P(true) should start
softened at repair, hold roughly flat while the quantized
seated authority is unchanged, and rise by +6 as seated
authority steps up — i.e. confidence TRACKS authority.

**PASS requires ALL of:**
1. mean P(true) is non-decreasing across the tracked post-repair
   points: repair ≤ +1 ≤ +2 ≤ +6 (no backsliding as the
   deficit decays);
2. mean P(true) at +6 is STRICTLY greater than mean P(true) at
   post-repair (confidence actually rises as authority heals);
3. mean P(true) at +6 is closer to baseline mean P(true) than
   post-repair mean P(true) is (the rise is toward baseline,
   not away from it).

If mean P(true) is flat through +6 (grid masking with no step
yet), bar C FAILS on leg 2 even if nothing got worse — a flat
trajectory is not a healing trajectory, and it is reported as
such, not rounded into a pass.

### Bar C-abs — CONFIDENCE ABSOLUTE (comparability line, its own verdict)

Round-1's bar, carried over UNCHANGED so this rerun stays
comparable to round 1, and reported as its own verdict line —
it is NOT folded into bar C or bar R:

**At each tracked point, mean P(true) >= baseline mean P(true)
− 0.10.** (Round-1 numbers: baseline 0.978, so the bar is
0.878 at every point.)

The round-1 diagnosis predicts this bar misses at post-repair
(the dock softens confidence by construction) and may still
miss at +1/+2, closing toward baseline only as authority heals.
Whatever it shows is reported point by point; a miss here does
NOT fail bar R or bar C, and a hit here does NOT pass them.

### Bar A — AUTHORITY LAG (independent verdict)

**PASS requires: post-repair seated authority mean on the
lm_head fact+liar rows <= pre-injury seated authority mean
− 0.1**, where pre-injury = seated authority on those rows
after the clean consolidation (round 1: 0.8).

### Bar D — DECAY (independent verdict)

**PASS requires ALL of:**
1. raw deficit strictly decreases across the decay: deficit(+6)
   < deficit(post-repair) (round-1 schedule: 0.30 → 0.24);
2. seated authority at +2 AND at +6 is still < pre-injury
   seated authority (the deficit decays, it does not snap
   back);
3. seated authority at +6 >= seated authority at post-repair
   (healing direction, allowing grid-flat stretches).

### Validity line — CHAIN (reported at every tracked point, not a bar)

Chain verification (rec.verify over all rows) must read all_ok
at baseline, post-poison, post-repair, +1, +2, and +6. A chain
failure at any point does not by itself flip a bar, but it
flags that point's numbers as untrusted and is reported
prominently. If the chain fails at the post-repair point, the
trajectory verdicts are reported as VOID, because the record
the trajectory is read from would be broken.

## What this rerun does NOT do

- Does not overwrite any round-1 file (preregistration.md,
  results.md, results.json, results_run1.json, results_run2.json)
  or any stage-2 file (preregistration-e2-rerun-2026-10-26.md,
  results-e2-rerun.md, results-e2-rerun.json, run_e2_rerun.py).
- Does not modify kt_neuron/ modules, kt_neuron.py,
  KT-LLM-PLAN.md, MEMORY.md, or kt_bench/seating.py. No seating
  redesign, no ARCHE feedback controller, no decline rule, no
  demo, no flip engagement. New code lives only under
  experiments/ for this rerun.
- Does not engage the flip (see declaration above).
- Does not change any threshold after a run. If any bar fails,
  it fails, and that failure is reported plainly. In
  particular, bar C-abs is expected to miss at post-repair on
  the round-1 diagnosis — if it instead passes, that is
  reported as a surprise, not hidden, and no bar is rewritten
  to accommodate it.

## Outputs (new files only)

- This file: experiments/preregistration-e3-rerun-2026-10-26.md
- Harness: experiments/run_e3_rerun.py (new; reuses trainer.py,
  corpus.py, playbook.py unchanged)
- Results: experiments/results-e3-rerun.md and
  experiments/results-e3-rerun.json
