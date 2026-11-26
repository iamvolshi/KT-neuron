# 3-arm controlled KTN experiment, v2 — results

**Verdict: FAIL (MECHANISMS PASS, ADVANTAGE FAIL)** — 7 of 8 prediction
legs pass; only P2 fails. The headline is a near-miss with a real
effect inside it: **the crossing handoff damped the fall, but not by
the frozen bar.** V1's fall was 0.6324; v2's is 0.5861 — a 0.0463
reduction in the predicted direction, against a 0.05 bar. P2 fails as
written by 0.0063. No threshold moves, ever.

- Frozen contract: `experiments/preregistration-3arm-v2-2026-10-28.md`
  (predictions frozen BEFORE any v2 run; never edited after).
- Machine data: `experiments/results-3arm-v2.json` (phases, handoff
  events, three-number snapshots, verdicts).
- Run: `python3 experiments/run_3arm_v2.py` from `~/workspace/kt-neuron`.
- Gradient gate: PASS (forward max abs diff 0.0, max rel err 0.000599,
  1056 params checked). Init parity: all three arms bit-identical
  weights at seed 0. Suite: 131 tests green (118 baseline + 13 new
  v2 tests) before the run; v2 + v1 test files re-run green after it.
- Bench: GPTConfig vocab 32, block 16, n_embd 64, 4 heads, 2 layers,
  105,248 params, 1,184 KIDs. 1,200 steps per arm, same batch lists
  lockstep as v1. V1's files are untouched by this run (mtimes
  verified: v1 preregistration, harness, and results all predate v2).

## The P2 question, answered first

**Does the crossing handoff produce the gradient damping V1 lacked?
Yes, measurably — but not by the frozen 0.05 bar, so P2 FAILS.**

| | v1 fall (recovery-1 → resistance) | v2 fall | change |
| --- | ---: | ---: | ---: |
| arm 1 conventional | 0.6298 | 0.6298 | 0.0000 (identical: arms 1–2 run v1 code) |
| arm 2 scar-only | 0.7224 | 0.7224 | 0.0000 |
| arm 3 full KTN + handoff | 0.6324 | **0.5861** | **−0.0463** |

V2's P2 bar: arm-3 fall smaller than arm-1 fall by ≥ 0.05.
Actual: 0.6298 − 0.5861 = 0.0437. Short by 0.0063. P2b (arm 2 not
beating arm 1 by > 0.05, magnitude form coded from the start in v2)
passes: arm-2 fall 0.7224 ≥ 0.6298 − 0.05. P2 overall fails via P2a.

Corroborating movement in the predicted direction: arm-3 resistance
mean P(true) rose 0.2891 (v1) → 0.3206 (v2), and resistance raw top-1
went 0/4 true (v1) → 1/4 true (v2). The damping is not a rounding
artifact; it is a real effect that falls short of the bar at this
bench scale. Whether a stronger scale, a longer poison phase, or a
per-row fast trace would clear the bar is a new experiment, not a
reinterpretation of this one.

## The handoff worked exactly as specified (mechanism checks)

- Crossing: all 1,184 rows latched at poison-1 step 159 (0-based;
  the 160th training step), F_prev 0.79 → F_new 0.8, S 0.8, gate
  passed — exactly the preregistration's ≈160 prediction. No rows
  latched in any other phase (latch is permanent; 1,184 events, all
  poison-1). The uniformity note held: one global F, uniform S 0.8
  after the clean consolidation, one shared crossing step.
- Step scales applied: mean scale over latched rows 0.8 at injury
  (all rows at clean seated 0.8); 0.8929 from recovery-1 onward
  (28 affected rows at 0.6, the rest at 0.9 — exactly the overall
  mean gain). During poison 2, affected rows stepped at 0.6×,
  unaffected latched rows at 0.9×.
- Three numbers at every boundary (fact-row means; never merged):

| phase | F | B | S | F−B | B−S (depth) | F−S (handoff signal) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.0 | 0.8 | 0.8 | −0.8 | 0.0 | −0.8 |
| injury | 1.2 | 0.8 | 0.8 | +0.4 | 0.0 | +0.4 |
| recovery-1 | 0.2 | 0.9 | 0.6 | −0.7 | +0.3 | −0.4 |
| resistance | 1.2 | 0.9 | 0.6 | +0.3 | +0.3 | +0.6 |
| recovery-2 | 0.2 | 0.9 | 0.6 | −0.7 | +0.3 | −0.4 |

Read it as Darren specified: F−S crosses zero at the handoff (fast
over the scar line, slow governs from there); B−S is the depth of
lasting damage (0.3 after the dock, holding through both restores);
F−B is the deviation from consolidated knowledge. Affected-row means
at both restores are identical to the fact-row means above
(B 0.9, S 0.6, depth 0.3).
- Fact-row weight movement during poison 2 (mean per-row
  ||dW||₂ + |db|): 0.4552. No v1 comparator was recorded for this
  quantity, so it stands as absolute mechanism data, not a comparison.
- Selection untouched by the handoff (as designed): trajectory
  [4,0,4,4,4] exact; 3 overrides total (all at resistance, where raw
  went 1/3 true and selection held 4/4; at injury raw was already
  0/4, so selection following the liar overrode nothing).

## Phase results (mean P(true), top-1 true/liar, arm 3 selected, scar)

| phase | arm 1 | arm 1 top1 | arm 2 | arm 2 top1 | arm 3 | arm 3 top1 | arm 3 selected | scar 2/3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.9780 | 4/0 | 0.9780 | 4/0 | 0.9445 | 4/0 | 4/0 | 0.0/0.0 |
| injury (poison 1) | 0.4246 | 1/3 | 0.3354 | 1/3 | 0.3160 | 0/4 | 0/4 | 1.2/1.2 |
| recovery 1 | 0.9925 | 4/0 | 0.9919 | 4/0 | 0.9067 | 4/0 | 4/0 | 0.2/0.2 |
| resistance (poison 2) | 0.3626 | 1/3 | 0.2695 | 0/4 | 0.3206 | 1/3 | 4/0 | 1.2/1.2 |
| recovery 2 | 0.9927 | 4/0 | 0.9953 | 4/0 | 0.9894 | 4/0 | 4/0 | 0.2/0.2 |

V2 vs v1 trajectory notes (honest differences, not hidden):

- Baseline is bit-identical to v1 (0.9780 / 0.9780 / 0.9445;
  counting 0.9667 / 0.9667 / 0.9646) — the clean phase runs v1 code,
  and the birth-row fix changed no probe value, as predicted.
- Injury arm 3 is LOWER in v2 (0.3160, raw 0/4) than v1 (0.3512, raw
  1/3). Expected direction of change, not a defect: the handoff
  starts scaling steps mid-poison-1 (from step 160), so arm 3's
  poison-1 trajectory differs from v1's by design. The P2 comparison
  is recovery-1 → resistance within each run, and v1-vs-v2 fall
  comparison — both reported above.
- Recovery-1 arm 3 is slightly lower in v2 (0.9067 vs v1 0.9214):
  restore-1 trained at 0.8× on all rows, slowing recovery a little.
  Still P1-passing (0.0378 below baseline, band 0.10).
- Recovery-2 arm 3 is higher in v2 (0.9894 vs v1 0.9778).
- Affected rows: 28 per poison phase (8 fact lm_head rows + top-20
  moved rows), same as v1.
- Counting-grammar retention (32 fresh sequences): baseline
  0.9667 / 0.9667 / 0.9646 → final 1.0000 / 0.9979 / 0.9958.
  Nothing forgotten anywhere.

## Verdicts vs frozen predictions

| leg | prediction | result | verdict |
| --- | --- | ---: | --- |
| P1 recovery-raw | all arms within 0.10 of baseline after restore 1 | arm1 0.9925 (base 0.9780), arm2 0.9919, arm3 0.9067 (base 0.9445; gap 0.0378) | PASS |
| P2 resistance-raw | arm-3 fall smaller than arm-1's by ≥ 0.05; arm-2 not beating arm-1 by > 0.05 | falls: arm1 0.6298, arm2 0.7224, arm3 0.5861. Arm 3 beats arm 1 by 0.0437, short of 0.05 by 0.0063 | FAIL |
| P3 selection adds | selected trajectory exactly [4,0,4,4,4]; selected > scar raw at resistance; selected ≥ own raw | trajectory [4,0,4,4,4] exact; selected 4 > scar raw 0; selected 4 ≥ own raw 1 | PASS |
| P4 deficit persists | affected-row mean authority 0.6 after restore 2; fact authority 0.6; behavior recovered | affected mean gain 0.6000; fact authorities [0.6, 0.6, 0.6, 0.6]; behavior recovered per P1 | PASS |
| P5 retention | counting accuracy within 0.05 of baseline, all arms | diffs +0.0333 / +0.0313 / +0.0313 | PASS |
| P6 cost | arm-3 forward < 2× arm-1; row rows 4736; fact rows 16 | ratio 0.9999× (21.77 ms vs 21.77 ms); row rows 4736 exact; fact rows 16 exact (birth-row fix) | PASS |
| P7 scar trajectory | scar [0.0, 1.2, 0.2, 1.2, 0.2] both scar arms | exact in both arms at all five boundaries | PASS |
| P8 re-compromise | arm-1 raw liar ≥ 3, arm-2 raw liar ≥ 3, arm-3 selected liar = 0 at resistance | 3, 4, 0 | PASS |

**Overall: FAIL (MECHANISMS PASS, ADVANTAGE FAIL)** — mechanism legs
P1, P4, P5, P6, P7 all pass; advantage legs fail only via P2 (P3 and
P8 pass). Per the v2 verdict rule, that is the labeled FAIL, not a
partial-credit PASS.

## The P6 fix, verified

Fact records now carry 4 rows each (birth + clean + restore-1 +
restore-2 = 16 total), all four chains verify OK
("4 rows, chain sealed, intervals contiguous, k now 4"). Row records
unchanged at 4736. The v1 asymmetry is closed by the disclosed
construction fix, and no probe value moved because of it (baseline
bit-identical to v1).

## Honest scope (from the preregistration, not softened)

- P2's near-miss is a measurement at this bench scale, not a proof
  about larger scales. 200 poison steps, uniform crossing, and a
  0.6/0.9 step scale are the setup; a different setup could clear
  the bar, and that would be a new experiment with its own frozen
  predictions.
- The crossing is uniform in this bench (global F, uniform S after
  the clean consolidation) — the switch is genuinely per-row (latch
  + scale are per row), but the crossing step is shared. Per-row
  fast traces would discriminate crossing steps; that is later work.
- The scar driver is scheduled (a clock), not measured from model
  state. The over-injury declining limb (pressure > 0.75) was not
  exercised. The mock RK is a stand-in for RK (ARCHE); nothing here
  claims anything about RK's real design, which is Darren's to
  formulate.
- Selection's resistance win still depends on quarantine (poison
  never seated) + bad-flagging + tie-to-incumbent, as v1. Flagging
  was an experimenter quarantine decision from lineage, not a system
  discovery.
- The step-scaling path (`Adam.step` `lr_scale`, default None) is
  backward-compatible: all existing callers are bit-identical, proven
  by unit test. The gradient gate is unchanged and ran FIRST.
- Bench scale only. No claim transfers to frontier models without a
  new experiment at that scale. Never claim 120,000 neurons or any
  scale not run: one small GPT (105,248 params, 1,184 KIDs) per arm,
  three arms, same data, same seeds.

## What this means for the critic's question

V1 showed selection saves decisions while weights re-poison. V2
shows the crossing handoff moves the weights' story in the right
direction — the fall shrank by 0.0463 and resistance top-1 improved
0/4 → 1/3 — but at this scale it does not yet constitute weight-level
resistance by the frozen bar. The deficit now gates learning (step
size), not just forward gain and decisions, and the gate is exact
(0.6 affected / 0.9 unaffected, latched at the predicted crossing).
What remains unproven is a configuration where that gate clears a
0.05 damping bar. That is the next experiment's question, not this
one's answer.

## Preserved distinction (Darren, Oct 28, post-run note — interpretation, not a change to any number or verdict)

Uniform latch timing and row-specific learning consequences are two
different claims, and this run establishes only one of them:

- DEMONSTRATED: a common crossing event governing differentiated
  optimizer behavior through standing authority. All 1,184 rows
  latched at the same shared crossing step (poison-1 step 159), and
  from there the optimizer behaved differently per row by standing
  authority: affected rows stepped at 0.6×, unaffected latched rows
  at 0.9× during poison 2. One crossing event, differentiated
  consequences.
- NOT DEMONSTRATED: different rows recognizing injury at different
  times. The crossing is uniform in this bench (global F, uniform S),
  so no row has its own crossing step. Per-row fast traces that would
  discriminate crossing steps are later work — the legitimate
  question for a V3, not a finding of this run.

## Files

- `experiments/preregistration-3arm-v2-2026-10-28.md` — frozen contract
- `experiments/run_3arm_v2.py` — harness
- `experiments/results-3arm-v2.json` — full phase data, handoff
  events, three-number snapshots, verdicts
- `experiments/results-3arm-v2.md` — this report
- `kt_bench/handoff.py` — handoff tracker + three-number snapshots
- `experiments/trainer.py` — `Adam.step` gains optional `lr_scale`
  (default None; existing callers bit-identical)
- `tests/test_3arm_v2.py` — 13 new tests (suite 131 green)
- V1 files untouched: `experiments/preregistration-3arm-2026-10-28.md`,
  `experiments/run_3arm.py`, `experiments/results-3arm.json`,
  `experiments/results-3arm.md`
