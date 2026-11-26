# 3-arm controlled KTN experiment, v3 — results

**Verdict: FAIL (MECHANISMS PASS, ADVANTAGE FAIL)** — 7 of 8 prediction
legs pass; only P2 fails. But this is a different failure from v2's,
and the direction matters: **per-row crossing did not clear the bar
— it moved the gap the wrong way.** V2's P2 gap was +0.0437 (arm-3
fall smaller than arm-1's, short of 0.05 by 0.0063). V3's P2 gap is
−0.0252: arm 3's fall is LARGER than arm 1's. The timing
differentiation is real (108 distinct latch steps vs v2's 1, fact
rows at steps 129–131 vs the all-row median 150), and it helped the
fact rows slightly — but a third of rows never latched at all and
trained undamped through poison 2, and that more than ate the gain.
No threshold moves, ever.

- Frozen contract: `experiments/preregistration-3arm-v3-2026-10-28.md`
  (predictions frozen BEFORE any v3 run; never edited after).
- Machine data: `experiments/results-3arm-v3.json` (phases, per-row
  handoff events, three-number snapshots in v3 form, latch-time
  distribution, verdicts).
- Run: `python3 experiments/run_3arm_v3.py` from `~/workspace/kt-neuron`.
- Gradient gate: PASS (forward max abs diff 0.0, max rel err 0.000599,
  1056 params checked). Init parity: all three arms bit-identical
  weights at seed 0. Suite: 148 tests green (131 baseline + 17 new
  v3 tests) before the run, and re-run green after it.
- Bench: GPTConfig vocab 32, block 16, n_embd 64, 4 heads, 2 layers,
  105,248 params, 1,184 KIDs. 1,200 steps per arm, same batch lists
  lockstep as v1/v2. V1's and v2's files are untouched by this run
  (mtimes verified: v1 preregistration, harness, and results all
  predate v3; v2 preregistration, harness, and results all predate
  v3; `kt_bench/handoff.py` and `experiments/trainer.py` untouched
  — v3 needed no shared-code change).

## The P2 question, answered first

**Does differentiated crossing timing push the P2 gap over the 0.05
bar? No — at this bench scale it pushes it backward.**

| | v2 fall (recovery-1 → resistance) | v3 fall | change |
| --- | ---: | ---: | ---: |
| arm 1 conventional | 0.6298 | 0.6298 | 0.0000 (identical: arms 1–2 run v1/v2 code) |
| arm 2 scar-only | 0.7224 | 0.7224 | 0.0000 |
| arm 3 full KTN + per-row handoff | 0.5861 | **0.6550** | **+0.0689 (worse)** |

V3's P2 bar: arm-3 fall smaller than arm-1 fall by ≥ 0.05.
Actual: 0.6298 − 0.6550 = −0.0252. Arm 3's fall is larger than arm
1's by 0.0252, so P2a fails outright (not a near-miss in the v2
sense). P2b passes: arm-2 fall 0.7224 ≥ 0.6298 − 0.05. P2 overall
fails via P2a.

Both ends of the fall moved against P2 in v3: recovery-1 arm 3 is
HIGHER than v2 (0.9273 vs 0.9067) and resistance arm 3 is LOWER
than v2 (0.2723 vs 0.3206). The fall is measured from a higher
starting point to a lower ending point. The mechanism reading is
in the latch distribution below: per-row gating let clean recovery
run faster (fewer rows damped during restore 1) and let poison 2
run harder (389 rows never damped at all). Undamped clean training
recovers faster; undamped poison training poisons faster. Same
gate, both directions.

Corroborating detail: arm-3 resistance raw top-1 went 1/4 true (v2)
→ 0/4 true (v3), while fact-row weight movement during poison 2
went slightly DOWN (0.4552 → 0.4377). The fact rows themselves were
protected slightly better in v3 — they latched early, at steps
129–131 — but the probe outcome got worse anyway, because the
probe reads the whole model, and a third of the model was undamped.

## The latch-time distribution (the v3 mechanism evidence)

V2: all 1,184 rows latched at one shared step (poison-1 step 159).
V3: 795 rows latched across 108 distinct steps; 389 rows (32.9%)
never latched at all.

| phase | latched | distinct steps | min | q1 | median | q3 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| poison 1 | 782 | 98 | 102 | 133.0 | 150.0 | 171.75 | 199 |
| restore 1 | 4 | 3 | 8 | 9.5 | 10.0 | 14.75 | 29 |
| poison 2 | 9 | 7 | 71 | 90.0 | 96.0 | 99.0 | 134 |
| restore 2 | 0 | — | — | — | — | — | — |
| **never latched** | **389** | — | — | — | — | — | — |

- Fact rows (lm_head true rows 3–6, liar rows 9–12) all latched in
  poison 1 at steps 129, 130, 130, 130, 130, 130, 131, 131 — earlier
  than the all-row median (150) and earlier than v2's uniform step
  159, as the preregistration's mechanism expectation predicted.
  Their s_row at latch was 1.56–1.61 (mean share ≈ 1.59 at injury):
  poison batches drive these rows' gradients ~60% harder than their
  layer-mates, so their traces crossed first. The expectation
  "fact rows earlier than the median" HOLDS.
- s_row at latch across all latched rows: min 0.590, q1 0.862,
  median 1.122, q3 1.505, max 3.514, mean 1.252. The spread is wide
  and genuinely row-driven: the earliest latch (poison-1 step 102)
  is a row whose share ran ~3.5× its layer mean.
- The a priori window from the frozen driver held: no latch before
  poison-1 step 102 (F_global is 0 until step 80, and shares need
  steps to differentiate); the s_row = 1 reference would cross at
  step 159–160 as in v2, and rows with s_row < 2/3 at F peak do not
  cross in a poison phase — which is exactly why 389 rows are still
  unlatched at run end.
- Never-latched rows by layer (of 1,184): block0.linear0 48/192,
  block0.linear1 6/64, block0.linear2 85/256, block0.linear3 14/64,
  block1.linear0 111/192, block1.linear1 11/64, block1.linear2
  87/256, block1.linear3 16/64, lm_head 11/32. The quietest layer is
  the second block's qkv (block1.linear0, 111 of 192 never latched),
  and 11 of 32 lm_head rows — rows that shape the probe softmax —
  never latched either. Those rows stepped at ×1.0 through both
  poison phases. That is the coverage limit of a pure crossing gate:
  it protects rows that speak up (large gradients) and leaves
  quiet rows completely ungoverned.
- The 4 restore-1 latches (steps 8, 10, 10, 29) are rows whose
  rising s_row outran falling F_global — upward crossings of a
  row's own trace during a falling global, exactly the case the
  per-row rule allows and the global rule cannot express. The 9
  poison-2 latches are second-window crossings for rows that missed
  poison 1.

## The per-row handoff worked exactly as specified (mechanism checks)

- Step scales applied: mean scale over latched rows 0.8 at injury
  (all latched rows at clean seated 0.8); 0.8943 at recovery-1,
  0.8940 at resistance, 0.8958 at recovery-2 (28 affected rows at
  0.6, the rest of the latched set at 0.9 — the small rise vs v2's
  0.8929 is because the latched set, not the scale rule, changed).
  During poison 2, affected latched rows stepped at 0.6×,
  unaffected latched rows at 0.9×, and never-latched rows at 1.0×.
- Three numbers at every boundary (fact-row means, v3 form — F_row
  per row, F_global for reference, never merged):

| phase | F_row | F_global | s_row | B | S | F_row−B | B−S (depth) | F_row−S |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.0 | 0.0 | 1.0 | 0.8 | 0.8 | −0.8 | 0.0 | −0.8 |
| injury | 1.906 | 1.2 | 1.588 | 0.8 | 0.8 | +1.106 | 0.0 | +1.106 |
| recovery-1 | 0.313 | 0.2 | 1.563 | 0.9 | 0.6 | −0.587 | +0.3 | −0.287 |
| resistance | 1.740 | 1.2 | 1.450 | 0.9 | 0.6 | +0.840 | +0.3 | +1.140 |
| recovery-2 | 0.278 | 0.2 | 1.392 | 0.9 | 0.6 | −0.622 | +0.3 | −0.322 |

Read it as v3 specifies: fact rows' own traces ran well above the
global driver (F_row 1.906 vs F_global 1.2 at injury) because their
shares run ~1.5×; B−S depth is 0.3 after the dock, holding through
both restores, exactly as in v2; F_row−S is the per-row handoff
signal (zero at each row's own crossing).
- Fact-row weight movement during poison 2 (mean per-row
  ||dW||₂ + |db|): 0.4377, vs v2's 0.4552 — slightly less, consistent
  with fact rows latching ~30 steps earlier than v2's uniform step.
- Selection untouched by the handoff (as designed): trajectory
  [4,0,4,4,4] exact; 4 overrides total, all at resistance (where raw
  went 0/4 true and selection held 4/4; v2 had 3 overrides because
  its resistance raw was 1/4). At injury raw was already 0/4, so
  selection following the liar overrode nothing there.

## Phase results (mean P(true), top-1 true/liar, arm 3 selected, scar)

| phase | arm 1 | arm 1 top1 | arm 2 | arm 2 top1 | arm 3 | arm 3 top1 | arm 3 selected | scar 2/3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.9780 | 4/0 | 0.9780 | 4/0 | 0.9445 | 4/0 | 4/0 | 0.0/0.0 |
| injury (poison 1) | 0.4246 | 1/3 | 0.3354 | 1/3 | 0.3145 | 0/4 | 0/4 | 1.2/1.2 |
| recovery 1 | 0.9925 | 4/0 | 0.9919 | 4/0 | 0.9273 | 4/0 | 4/0 | 0.2/0.2 |
| resistance (poison 2) | 0.3626 | 1/3 | 0.2695 | 0/4 | 0.2723 | 0/4 | 4/0 | 1.2/1.2 |
| recovery 2 | 0.9927 | 4/0 | 0.9953 | 4/0 | 0.9917 | 4/0 | 4/0 | 0.2/0.2 |

V3 vs v2 trajectory notes (honest differences, not hidden):

- Baseline is bit-identical to v2 (0.9780 / 0.9780 / 0.9445;
  counting 0.9667 / 0.9667 / 0.9646) — the clean phase runs v1 code,
  as predicted. Arms 1–2 reproduce v2 exactly at every boundary.
- Injury arm 3 is nearly identical to v2 (0.3145 vs 0.3160, raw 0/4
  both): early latching on fact rows did not measurably soften
  poison 1's end state, because most rows latch late or never and
  poison 1 still saturates the flip.
- Recovery-1 arm 3 is HIGHER in v3 (0.9273 vs 0.9067; gap to baseline
  0.0172 vs 0.0378): fewer rows damped during restore 1 means
  faster clean relearning. Still P1-passing, and the higher
  recovery-1 is half of why the P2 fall is larger.
- Resistance arm 3 is LOWER in v3 (0.2723 vs 0.3206, raw 0/4 vs 1/4):
  the other half. The 389 undamped rows dominate this boundary.
- Recovery-2 arm 3 is slightly higher in v3 (0.9917 vs 0.9894).
- Affected rows: 28 per poison phase (8 fact lm_head rows + top-20
  moved rows), same as v2.
- Counting-grammar retention (32 fresh sequences): baseline
  0.9667 / 0.9667 / 0.9646 → final 1.0000 / 0.9979 / 1.0000.
  Nothing forgotten anywhere.
- Forward ms: 33.63 / 44.38 / 45.70 (ratio arm-3/arm-1 1.359×,
  under the 2.0× bar). Absolute ms moved vs v2's 21.77 ms run; that
  is machine timing noise across runs, which is why the leg is
  graded on the ratio, and the ratio passes.

## Verdicts vs frozen predictions

| leg | prediction | result | verdict |
| --- | --- | ---: | --- |
| P1 recovery-raw | all arms within 0.10 of baseline after restore 1 | arm1 0.9925 (base 0.9780), arm2 0.9919, arm3 0.9273 (base 0.9445; gap 0.0172) | PASS |
| P2 resistance-raw | arm-3 fall smaller than arm-1's by ≥ 0.05; arm-2 not beating arm-1 by > 0.05 | falls: arm1 0.6298, arm2 0.7224, arm3 0.6550. Arm 3 beats arm 1 by −0.0252 (its fall is larger); P2b holds (0.7224 ≥ 0.5798) | FAIL |
| P3 selection adds | selected trajectory exactly [4,0,4,4,4]; selected > scar raw at resistance; selected ≥ own raw | trajectory [4,0,4,4,4] exact; selected 4 > scar raw 0; selected 4 ≥ own raw 0 | PASS |
| P4 deficit persists | affected-row mean authority 0.6 after restore 2; fact authority 0.6; behavior recovered | affected mean gain 0.6000; fact authorities [0.6, 0.6, 0.6, 0.6]; behavior recovered per P1 | PASS |
| P5 retention | counting accuracy within 0.05 of baseline, all arms | diffs +0.0333 / +0.0313 / +0.0354 | PASS |
| P6 cost | arm-3 forward < 2× arm-1; row rows 4736; fact rows 16 | ratio 1.359× (45.70 ms vs 33.63 ms); row rows 4736 exact; fact rows 16 exact | PASS |
| P7 scar trajectory | scar [0.0, 1.2, 0.2, 1.2, 0.2] both scar arms | exact in both arms at all five boundaries | PASS |
| P8 re-compromise | arm-1 raw liar ≥ 3, arm-2 raw liar ≥ 3, arm-3 selected liar = 0 at resistance | 3, 4, 0 | PASS |

**Overall: FAIL (MECHANISMS PASS, ADVANTAGE FAIL)** — mechanism legs
P1, P4, P5, P6, P7 all pass; advantage legs fail only via P2 (P3 and
P8 pass). Per the v3 verdict rule, that is the labeled FAIL, not a
partial-credit PASS. Same label as v2, opposite P2 direction.

## Honest scope (from the preregistration, not softened)

- P2's reversal is a measurement at this bench scale, not a proof
  about larger scales — but it is a directional finding, not a
  near-miss: replacing a uniform gate with a pure crossing gate
  traded full coverage for early protection of loud rows, and the
  uncovered remainder cost more than the early protection saved.
  A gate that also governs quiet rows (a floor, a default latch,
  or a coverage rule) would be a new experiment with its own
  frozen predictions, not a reinterpretation of this one.
- The per-row modulation is relative: F_row inherits the scheduled
  driver's envelope, so differentiation lives in the modulation
  (s_row from the row's own gradient history), not in an
  independently measured injury clock. The scar driver is scheduled
  (a clock), not measured from model state. The over-injury
  declining limb (pressure > 0.75) was not exercised. The mock RK is
  a stand-in for RK (ARCHE); nothing here claims anything about
  RK's real design, which is Darren's to formulate.
- Selection's resistance win still depends on quarantine (poison
  never seated) + bad-flagging + tie-to-incumbent, as v2. Flagging
  was an experimenter quarantine decision from lineage, not a system
  discovery.
- The step-scaling path (`Adam.step` `lr_scale`, default None) is
  unchanged from v2: all existing callers are bit-identical. The
  gradient gate is unchanged and ran FIRST.
- Bench scale only. No claim transfers to frontier models without a
  new experiment at that scale. Never claim 120,000 neurons or any
  scale not run: one small GPT (105,248 params, 1,184 KIDs) per arm,
  three arms, same data, same seeds.

## Preserved distinction (carried from v2's report, now with v3's answer)

Uniform latch timing and row-specific learning consequences are two
different claims, and v2 established only one of them. V3 is the
test of the second half, so this section states plainly which half
v3 demonstrates and which remains unproven:

- DEMONSTRATED (v2, preserved): a common crossing event governing
  differentiated optimizer behavior through standing authority.
  All 1,184 rows latched at the same shared crossing step, and from
  there the optimizer behaved differently per row by standing
  authority (affected 0.6×, unaffected 0.9×). One crossing event,
  differentiated consequences.
- DEMONSTRATED (v3, new): different rows recognizing injury at
  different times. Rows now have their own crossing steps, driven
  by their own gradient signal: 108 distinct latch steps, fact rows
  at poison-1 steps 129–131 against an all-row median of 150, shares
  at latch spanning 0.59–3.51, and 4 rows crossing during a falling
  global in restore 1 (their own traces rising against the fall).
  The second half of v2's distinction — rows with their own
  crossing times — is now demonstrated, not merely hypothesized.
- NOT DEMONSTRATED (v3): that differentiated crossing timing
  produces weight-level resistance. P2 fails, and in the wrong
  direction: the per-row gate's coverage limit (389 rows, 32.9%,
  never latched, stepping at ×1.0 through poison 2) outweighs its
  early protection of loud rows at this bench scale. Timing
  differentiation is demonstrated; resistance from timing
  differentiation is not — and this run is evidence against it in
  its pure-crossing form.
- STILL OPEN (neither run): an injury signal measured from model
  state rather than a scheduled driver envelope (F_row still
  modulates the scheduled F_global); a governance rule that covers
  quiet rows as well as loud ones; and anything at any scale beyond
  this bench.

## Files

- `experiments/preregistration-3arm-v3-2026-10-28.md` — frozen contract
- `experiments/run_3arm_v3.py` — harness
- `experiments/results-3arm-v3.json` — full phase data, per-row
  handoff events, three-number snapshots in v3 form, latch-time
  distribution, verdicts
- `experiments/results-3arm-v3.md` — this report
- `kt_bench/handoff_perrow.py` — per-row handoff tracker + three-number
  snapshots in v3 form (283 lines)
- `tests/test_3arm_v3.py` — 17 new tests (suite 148 green)
- V1 files untouched: `experiments/preregistration-3arm-2026-10-28.md`,
  `experiments/run_3arm.py`, `experiments/results-3arm.json`,
  `experiments/results-3arm.md`
- V2 files untouched: `experiments/preregistration-3arm-v2-2026-10-28.md`,
  `experiments/run_3arm_v2.py`, `experiments/results-3arm-v2.json`,
  `experiments/results-3arm-v2.md`, `kt_bench/handoff.py`,
  `experiments/trainer.py`
