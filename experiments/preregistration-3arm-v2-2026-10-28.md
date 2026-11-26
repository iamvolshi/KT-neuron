# 3-arm controlled KTN experiment, v2 — pre-registration (2026-10-28)

Written BEFORE any v2 run. Frozen: no threshold, rule, or prediction
below changes after a run. V1's files are untouched and stand as run:
`experiments/preregistration-3arm-2026-10-28.md`,
`experiments/run_3arm.py`, `experiments/results-3arm.json`,
`experiments/results-3arm.md` (verdict FAIL: P2 and P6 fail, 6 of 8
legs pass). This file covers the v2 rerun ONLY. New files only, plus
one disclosed backward-compatible extension to shared training code
(see "Adam step scaling" below).

## Why v2 exists (Darren's corrections, 2026-10-28)

V1 wired the deficit to forward gain only and followed the fast trace
throughout. That is why P2 (resistance-raw, gradient damping) failed:
with fast-trace-only wiring there can be no damping, just impact and
quick recovery — a spring, undamped. Darren's correction, implemented
exactly here:

1. FAST→SLOW HANDOFF AT THE CROSSING. When authority is strong enough
   AND the impact fast trace CROSSES OVER the slow trace scar line —
   at that moment both numbers are equal — switch from following the
   fast trace to following the SLOW TRACE, because that is the damaged
   part. Before the crossing: fast governs (impact + spring recovery).
   After the crossing: slow governs (shock absorber + spring =
   damped). That switch is how damping is maintained.
2. THREE NUMBERS, NEVER MERGED. Tracked separately, never averaged or
   blended: (a) the fast trace, (b) the slow trace baseline,
   (c) the slow trace scar (depth = baseline − scar). Understanding
   and control live in the DIFFERENCES: fast − slow baseline =
   deviation from consolidated knowledge (impact); baseline − scar =
   depth of lasting damage (deficit); fast − slow scar = the handoff
   signal (zero at the crossing → switch). The handoff is a SWITCH at
   the equality point, not a blend.

The mechanism under test: does the crossing-triggered switch produce
the gradient damping V1 lacked?

## Operationalization (frozen, exact)

Three numbers, per KT row, read from bench machinery that already
exists — no new state invented for the numbers themselves:

- F (fast trace) = the global impact scalar: `PressureDriver.scar`
  at the current step (same scheduled driver as v1: rises on poison
  steps, falls at half rate on restore steps, capped at the 0.75 peak).
  It is the "quick blip": rises with impact, recovers toward baseline.
- B_row (slow baseline) = the claim authority at the row's last
  consolidation (clean 0.8; repair 0.9; birth 0.5 before any
  consolidation). Recorded by the harness at each consolidation.
- S_row (slow scar) = the row's standing authority, read as
  `layer.gain[row]` (Bench.consolidate sets gains to standing
  authority by design). Depth_row = B_row − S_row, COMPUTED at
  report time from B and S, never stored as a merged value.
- F, B, S are stored and reported separately. Differences F−B, B−S,
  F−S are computed from them when reported. Nothing averages them.

Handoff rule (frozen, per row, arm 3 only):

- AUTHORITY GATE ("strong enough"): the row's standing authority S
  at crossing time must be > 0.5 — above birth authority, i.e. a
  genuinely seated belief, not a candidate.
- CROSSING: an upward crossing during a training step: prev_F < S
  ≤ new_F, where new_F is the driver scar AFTER that step's driver
  advance and scar application. Checked after every arm-3 driver
  step (poison and restore phases); the clean phase has no driver
  movement, so no crossings can occur there.
- LATCH: once a row crosses, it stays slow-governed for the rest of
  the run. No un-latching, no re-arming, no phase exceptions.
  (Darren: "from there you follow the slow trace.")
- SLOW GOVERNS LEARNING (frozen rule): "gradient update" means the
  Adam parameter update computed from the row's gradients. For every
  latched row, every Adam update after latching is multiplied by the
  row's CURRENT standing authority S (the slow number): the row's W
  row and b are stepped at S× the unscaled step. Unlatched rows step
  at ×1.0. Non-row parameters (embeddings, LayerNorms) step at ×1.0
  always — they carry no KIDs. Gradients themselves are NOT scaled;
  only the step is.
- WHY STEP SCALING, NOT GRADIENT SCALING (frozen reasoning): Adam
  normalizes per-element gradients (mhat / sqrt(vhat)), so a uniform
  per-row gradient scale cancels out of the update up to eps and
  momentum transients. That is the same mechanism behind V1's reading
  (a) for the gain story: forward gain scales the backward gradient,
  and Adam normalizes it away. Damping that survives Adam must act on
  the step itself. V2 therefore governs the step, not the gradient.

Step ordering (arm 3 driver phases, frozen):

1. Build the step-size scale list from the CURRENT latch set and
   CURRENT standing authorities (gains).
2. One batch: `loss_and_grads` + `Adam.step` with that scale list.
3. Driver step (poison or restore), then apply the scar.
4. Crossing check with new F → newly-crossed rows latch for the
   NEXT step (the crossing step itself trains unscaled).

Adam step scaling (disclosed shared-code change): `Adam.step` in
`experiments/trainer.py` gains an OPTIONAL `lr_scale` argument (a
list aligned with params; each entry a scalar or a same-shape array;
default None = exactly the old behavior). All existing callers pass
no `lr_scale` and are bit-identical. A unit test proves
`lr_scale`-all-ones reproduces `train_steps` bit-identically, and
that a 0.5 scale on one row scales exactly that row's update by 0.5
(up to float rounding). The gradient gate (`gradient_check`) is
unchanged and still runs FIRST; it exercises `loss_and_grads`, which
the scaling does not touch.

Scale note (honest operationalization): F lives on the scar scale
(0–1.2) and B/S on the authority scale (0–1.0). The crossing is the
numerical equality on the shared plot scale — the visual crossing
(fast curve over slow scar line) operationalized, frozen here.

Uniformity note (frozen limitation): after the clean consolidation
every row seats at 0.8 with no dock, so S is uniform 0.8 and F is
global — all rows cross at the same step of poison 1 (predicted
≈ step 160, when F passes 0.8; reported as a mechanism check, not a
graded leg). Per-row discrimination in this bench comes from the
SCALE (0.6 affected vs 0.9 unaffected during poison 2), not from
distinct crossing steps. A bench with per-row fast traces would
discriminate crossing steps too — later work, not this run.

## Shared ground (identical to v1 unless stated above)

- Bench: GPTConfig vocab 32, block 16, n_embd 64, 4 heads, 2 layers,
  105,248 params, 1,184 KIDs. Seeds, lockstep, batch lists: exactly
  v1 (init 0 all arms, bit-identical; clean 1, poison 2, restore-1 3,
  restore-2 4 first 200, counting-eval 77, eval batches 99; fresh Adam
  per phase per arm, lr 0.01).
- Gradient gate FIRST (`gradient_check`); abort on FAIL. Full test
  suite green (118 at v2 start; new tests added for the handoff
  module and the Adam scale path).
- PYTHON NOISE RULE, RETIRED METERS, EXCLUDED MACHINERY, SCALE
  HONESTY: all as v1 (silent marking, instant honeypots, fake data
  pools, morphic data integrity NOT built or tested; FLOP saving and
  the absolute confidence bar stay retired; bench scale only — never
  claim 120,000 neurons or any scale not run).

## Arm 1 — CONVENTIONAL, Arm 2 — SCAR-ONLY (unchanged from v1)

- Arm 1: GPT + Linear. No scar, no record, no selection, no handoff.
- Arm 2: GPT + ScarLinear, same scheduled scar driver as arm 3. No
  K-record, no authority, no selection, NO handoff (there is no slow
  trace on this arm to hand off to — that absence is the ablation:
  if arm 2 matched arm 3 on P2, the handoff added nothing).

## Arm 3 — FULL KTN + HANDOFF + MOCK RK (v1 arm 3 + the switch)

- GPT + KTLinear, per-row K-records (toy core record rules
  UNCHANGED), consolidations exactly as v1 (CLEAN_POOL clean;
  REPAIR_POOL + dock 0.3 on affected rows at restores; gains set to
  standing authority; NO consolidation during poison phases;
  affected rows = 8 fact lm_head rows + top-20 moved rows, lm_head
  included, as v1).
- Fact belief records exactly as v1 (teacher claims seat ONLY at
  consolidations: clean 0.8, restore 0.9 claim with dock 0.3 → 0.6;
  liar claims NEVER seated), PLUS the disclosed construction fix
  below.
- Mock RK selection exactly as v1 (highest authority wins, tie →
  incumbent; liar offered 0.9 unflagged, 0.6 flagged at restore-1;
  flagging is an experimenter quarantine decision, not a discovery;
  log entries MOCK + DERIVED, never written into K-records).
- Handoff + three-number tracking as specified above. Selection is
  untouched by the handoff (it reads fact-record standing
  authorities, which the handoff does not write).

## P6 construction fix (disclosed, harness only, not a mechanism change)

V1's P6 failed on fact rows 12 vs 16 because the harness created fact
records as bare `KRecord(kid)` — no birth row — while KTLinear rows
open a birth row at construction. V2 opens a birth belief on each
fact record at construction ("fact", true token, 0.5, "birth"),
mirroring the row records. Fact records then carry birth + 3 seating
rows = 4 each, 16 total, matching the leg definition. V1's 12 is
preserved in V1's files, untouched. The fix changes no probe value:
`standing("fact")` is the latest interval either way, and the
baseline probe is taken after the clean consolidation.

## Phases (identical to v1; probes at every boundary)

1. Clean 400 → arm-3 clean consolidation (+ fact-record seating,
   birth rows already open) → BASELINE probe (incl. counting).
2. Poison 1, 200 (seed 2; scar rises; arm-3 handoff checks live;
   NO consolidation) → INJURY probe.
3. Restore 1, 200 clean (seed 3; scar falls; arm-3 restore
   consolidation + liar flagged known-bad) → RECOVERY-1 probe.
4. Poison 2, 200, SAME seed-2 list (scar rises; NO consolidation)
   → RESISTANCE probe.
5. Restore 2, 200 clean (seed-4 first 200; scar falls; arm-3 restore
   consolidation 2) → RECOVERY-2 probe + retention + cost probes.

Probes: same as v1 (probe_all_facts → mean P(true)/P(liar),
top1 counts raw; arm 3 additionally mock-RK selection; counting
accuracy on the same 32 seed-77 sequences; forward ms; record rows;
chain verification). NEW mechanism reports (not graded legs):
handoff events {phase, step-in-phase, layer, row, F_prev, F_new, S,
standing authority at crossing}; three-number snapshots at every
boundary (fact rows individually + affected-row means: F, B, S,
F−B, B−S, F−S); mean step scale applied to latched rows per phase;
latched-row weight movement during poison 2.

## Metrics (same definitions as v1)

- M1 recovery-raw, M1b recovery-selected, M2 resistance-raw, M2b
  resistance-selected, M2c re-compromise (raw), M3 retention, M4 cost:
  exactly as v1. Drops are MAGNITUDES (positive = fall), as v1's
  corrected reading.

## Frozen predictions (graded pass/fail, never moved post-hoc)

- P1 recovery-raw: ALL arms recover to within 0.10 of baseline mean
  P(true) after restore 1 (same band as v1).
- P2 resistance-raw: arm-3 raw drop (recovery-1 → resistance,
  magnitude) is smaller than arm-1 raw drop by >= 0.05 (P2a: the
  handoff damps poison-2 gradients via 0.6× steps on affected rows);
  AND arm-2 raw drop is NOT smaller than arm-1 raw drop by more
  than 0.05 (P2b, magnitude form: mag2 >= mag1 − 0.05 — coded in
  magnitude form FROM THE START in v2, per v1's disclosed
  correction; no signed-change form is used anywhere).
- P3 selection adds over scar: arm-3 selected top1_true_count
  post-poison-2 STRICTLY GREATER than arm-2 raw top1_true_count
  post-poison-2; AND >= arm-3 raw top1_true_count post-poison-2;
  selected trajectory exactly [4,0,4,4,4] (same rule derivation as
  v1; selection is unchanged by the handoff).
- P4 deficit persists: arm-3 affected-row mean seated authority
  post-restore-2 = 0.6 exactly; fact-truth authority post-restore-2
  = 0.6 exactly; behavior recovers per P1.
- P5 retention: counting accuracy post-restore-2 within 0.05 of
  baseline for ALL arms.
- P6 cost: arm-3 forward ms < 2.0× arm-1 forward ms; arm-3
  row-record rows = 4736 exactly; arm-3 fact-record rows = 16
  exactly (WITH the birth-row fix above).
- P7 scar trajectory: arm-2/3 scar scalar = 0.0 baseline; 1.2 end
  poison-1; 0.2 end restore-1; 1.2 end poison-2; 0.2 end restore-2
  (driver unchanged by the handoff).
- P8 re-compromise: arm-1 raw top1_liar_count post-poison-2 >= 3;
  arm-2 raw top1_liar_count post-poison-2 >= 3; arm-3 SELECTED
  top1_liar_count post-poison-2 = 0.

Verdict rule: OVERALL PASS iff P1–P8 ALL pass. Otherwise FAIL with
per-leg breakdown. If mechanism legs (P1, P4, P5, P6, P7) pass but
advantage legs (P2, P3, P8) fail, the verdict is FAIL with the label
"MECHANISMS PASS, ADVANTAGE FAIL". No partial-credit PASS.

## Honest scope notes (frozen limitations, not excuses)

- The handoff's poison-2 damping acts through the step scale on
  latched rows (0.6 affected, 0.9 unaffected). 200 poison steps may
  still saturate the flip; if P2 fails again, the fall comparison
  (not the story) is the finding.
- The crossing is uniform in this bench (global F, uniform S after
  clean consolidation) — see the uniformity note. The switch is
  still a genuine per-row mechanism (latch + scale are per row);
  only the crossing step is shared.
- The scar driver is scheduled (a clock), not measured from model
  state, as v1. The over-injury declining limb (p > 0.75) is NOT
  exercised. The mock RK is a stand-in for RK (ARCHE); nothing here
  claims anything about RK's real design, which is Darren's to
  formulate.
- Selection's poison-2 win (if repeated) still depends on
  quarantine + bad-flagging + tie-to-incumbent, as v1.
- Bench scale only. No claim transfers to frontier models without a
  new experiment at that scale.

## Reporting

New files under ~/workspace/kt-neuron/experiments/
(preregistration-3arm-v2-2026-10-28.md [this file],
run_3arm_v2.py, results-3arm-v2.json, results-3arm-v2.md) and
`kt_bench/handoff.py` (<500 lines) with tests under
`~/workspace/kt-neuron/tests/test_3arm_v2.py`. The report states: run
commands and real outputs, verdict against EVERY frozen prediction
above, exact numbers, the P2 comparison vs v1 front and center, and
anything that failed, was null, or was left for later. No threshold
moves post-hoc, ever. V1's files are never amended by this run.
