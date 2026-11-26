# 3-arm controlled KTN experiment, v3 — pre-registration (2026-10-28)

Written BEFORE any v3 run. Frozen: no threshold, rule, or prediction
below changes after a run. V1's files and V2's files are untouched
and stand as run:
`experiments/preregistration-3arm-2026-10-28.md`,
`experiments/run_3arm.py`, `experiments/results-3arm.json`,
`experiments/results-3arm.md` (verdict FAIL: P2 and P6 fail, 6 of 8
legs pass); `experiments/preregistration-3arm-v2-2026-10-28.md`,
`experiments/run_3arm_v2.py`, `experiments/results-3arm-v2.json`,
`experiments/results-3arm-v2.md` (verdict FAIL (MECHANISMS PASS,
ADVANTAGE FAIL): 7 of 8 legs pass, only P2 fails). This file covers
the v3 rerun ONLY. New files only; no shared-code change is needed
(`Adam.step`'s optional `lr_scale` already exists from v2).

## Why v3 exists (the mechanism question behind the v2 P2 near-miss)

V2 (7/8 legs, only P2 fails): arm-3 fall 0.5861 vs arm-1 fall 0.6298,
P2 gap 0.0437 vs the frozen bar 0.05, short by 0.0063. V2's crossing
was UNIFORM: one global fast trace F, uniform slow scar S, all 1,184
rows latched at the same shared step (poison-1 step 159) — step
scales then differentiated per row by standing authority (affected
0.6x, unaffected 0.9x during poison 2).

V2's report preserved a distinction as its own section, and v3 is
the test of its second half:

- DEMONSTRATED (v2): uniform latch timing vs row-specific learning
  consequences are separable — one common crossing event governed
  differentiated optimizer behavior through standing authority.
- NOT DEMONSTRATED (v2): different rows recognizing injury at
  different times. The crossing was uniform, so no row had its own
  crossing step.

THE V3 QUESTION, stated explicitly: does differentiated crossing
timing — per-row fast traces, so different rows cross the slow scar
line at DIFFERENT steps — push the P2 gap over the 0.05 bar?

## THE ONE CHANGE vs v2 (frozen, exact): per-row fast traces

Everything else is v2-identical (arms, phases, seeds, batch lists,
consolidations, mock-RK selection, step scales, gates, verdict
rules). The ONLY change is how the fast trace is computed and
therefore when each row crosses. V2's `kt_bench/handoff.py` is NOT
edited; v3 uses a new module, `kt_bench/handoff_perrow.py`.

### Per-row fast trace F_row (from the row's own signal, not a clock)

For each KT row, the fast trace is the global driver scar modulated
by that row's own accumulated gradient exposure:

- Per-step row signal: on every arm-3 driver step (poison and
  restore phases only; the clean phase uses the v1 path and never
  touches the tracker, as in v2), AFTER `loss_and_grads` and BEFORE
  the Adam step, compute the row's own gradient magnitude on that
  batch:
  `g_row = ||dW_row||_2 + |db_row|`
  where dW_row / db_row are that row's W-row / b gradients. This is
  the same movement metric v2's affected-row ranking already uses
  (row weight norm + |bias|), now read per step instead of per
  phase. Non-KT parameters (embeddings, LayerNorms) carry no KIDs
  and contribute nothing.
- Cumulative exposure: `C_row += g_row` every driver step. C_row is
  NEVER reset across phases: a row's trace is its whole
  driver-phase exposure history (poison and restore gradients alike
  — resetting per phase would discard the injury history that slow
  governance is supposed to remember).
- Relative share (scale-free, no tuned constant): within each layer,
  `s_row = C_row / mean(C over rows in that layer)` when the layer
  mean is > 0, else `s_row = 1.0`. Per-layer normalization (not
  global) because gradient scales differ across layers. s_row is
  dimensionless, mean exactly 1.0 over a layer whenever any
  gradient has accumulated; no calibration constant is introduced
  anywhere.
- Per-row fast trace: `F_row = F_global * s_row`, where F_global is
  the scheduled driver scar (`PressureDriver.scar`, rounded as in
  v2). The driver still supplies the rising envelope; the row's own
  accumulated signal modulates it. A row that has absorbed more
  gradient than its layer-mates (s_row > 1) sees its trace rise
  faster and crosses earlier; a quiet row (s_row < 1) crosses later
  or not at all in that phase.
- Reduction property (frozen check): if all rows in a layer had
  identical gradient histories, s_row = 1 everywhere and v3 reduces
  EXACTLY to v2's global rule. A unit test proves this. Any latch
  spread in the run is therefore entirely attributable to
  row-differentiated signal, not to a changed rule.

### Per-row crossing rule (gate + upward crossing per row)

- AUTHORITY GATE (unchanged from v2): the row's standing authority
  S at crossing time must be > 0.5 — above birth authority, i.e. a
  genuinely seated belief, not a candidate.
- CROSSING (per row): an upward crossing of that row's OWN trace:
  `prev_F_row < S <= new_F_row`, where prev_F_row is the row's
  trace value at the previous driver check (initialized 0.0 before
  any driver step) and new_F_row is computed from the CURRENT
  global F and the CURRENT s_row (after this step's gradient
  accumulation). Checked after every arm-3 driver step (poison and
  restore phases).
- LATCH (unchanged from v2): once a row crosses, it stays
  slow-governed for the rest of the run. No un-latching, no
  re-arming, no phase exceptions.
- NO RETROACTIVE LATCH (frozen edge rule): a consolidation changes
  S (e.g. affected rows 0.8 -> 0.6 at restore 1). A row whose stored
  trace already sits above its NEW S without an observed upward
  crossing does NOT latch retroactively. Only an upward crossing
  observed at a check after the S change latches. Because prev_F_row
  is refreshed at every driver check (including restore steps, when
  F falls), a stale-high trace is washed out during restore, and
  unlatched rows get a fresh crossing window when F rises again in
  poison 2.
- SLOW GOVERNS LEARNING (unchanged from v2): for every latched row,
  every Adam update after latching is multiplied by the row's
  CURRENT standing authority S (the slow number): the row's W row
  and b step at Sx the unscaled step. Unlatched rows step at x1.0.
  Non-row parameters step at x1.0 always. Gradients themselves are
  NOT scaled; only the step is (same frozen Adam reasoning as v2:
  Adam normalizes per-element gradients, so damping that survives
  Adam must act on the step itself).

### Step ordering (arm-3 driver phases, frozen)

1. Build the step-size scale list from the CURRENT latch set and
   CURRENT standing authorities (gains).
2. One batch: `loss_and_grads` + accumulate C_row from this batch's
   gradients + `Adam.step` with that scale list.
3. Driver step (poison or restore), then apply the scar.
4. Crossing check with new global F and updated s_row -> newly
   crossed rows latch for the NEXT step (the crossing step itself
   trains unscaled, as in v2; the crossing step's own gradients DO
   count toward C_row, since they are the row's signal on that
   batch).

### Three numbers, per row, never merged (v3 form)

- F_row (per-row fast trace, as above) — replaces v2's global F in
  the per-row record. F_global (the driver scar) is ALSO reported
  alongside, for reference and for the v2 comparison; it is never
  substituted for F_row.
- B_row (slow baseline) = the claim authority at the row's last
  consolidation (clean 0.8; repair 0.9; birth 0.5 before any
  consolidation). Recorded by the harness at each consolidation, as
  in v2.
- S_row (slow scar) = the row's standing authority, read as
  `layer.gain[row]`, as in v2. Depth_row = B_row - S_row, COMPUTED
  at report time from B and S, never stored as a merged value.
- Differences F_row-B, B-S, F_row-S are computed from the three
  numbers when reported. Nothing averages them. Snapshots carry
  F_row, F_global, s_row, B, S, and the differences per row.

### A priori crossing window (from the frozen driver, not from data)

F_global is 0 until driver pressure passes 0.30 (poison step 80),
then rises linearly to 1.2 at poison step 200. So: the earliest
possible crossing in a poison phase is step 80; the s_row = 1
reference row crosses S = 0.8 at step 159-160 exactly as in v2; a
row with s_row < 2/3 at F peak (0.8 / 1.2) does NOT cross in that
poison phase. Rows with s_row > 1 cross before step 159; rows with
s_row < 1 cross after it or not at all. These are consequences of
the frozen rule, stated here before any run.

### Mechanism expectations (reported, NOT graded legs)

As in v2 (where the ~step-160 crossing was a mechanism check, not a
graded leg), the following are reported as mechanism evidence and
do NOT change any pass/fail leg:

- Latch steps SPREAD: the number of distinct latch steps across
  rows is > 1 (v2: exactly 1). Latch distribution reported in full:
  counts per phase, step-in-phase min / quartiles / median / max
  over latched rows, never-latched count, and fact-row latch steps
  individually.
- Poison-exposed rows cross FIRST: fact-row (lm_head true + liar
  rows) latch steps are expected strictly earlier than the all-row
  median latch step, because poison batches drive those rows'
  gradients hardest.
- Quiet rows may latch in poison 2 or never: rows that do not cross
  in poison 1 get a second window in poison 2; rows still unlatched
  at run end train undamped throughout, and are counted, not
  hidden.

## Shared ground (identical to v2 unless stated above)

- Bench: GPTConfig vocab 32, block 16, n_embd 64, 4 heads, 2 layers,
  105,248 params, 1,184 KIDs. Seeds, lockstep, batch lists: exactly
  v2 (init 0 all arms, bit-identical; clean 1, poison 2, restore-1 3,
  restore-2 4 first 200, counting-eval 77, eval batches 99; fresh Adam
  per phase per arm, lr 0.01).
- Arms 1-2 train through the exact v1/v2 code paths, so their numbers
  are expected to reproduce v2 EXACTLY (arm-1 fall 0.6298, arm-2 fall
  0.7224); arm-3's clean phase uses the v1 path, so its baseline is
  expected bit-identical to v2 (mean P(true) 0.9445). Only arm 3's
  driver-phase trajectory moves, via the per-row crossing.
- Gradient gate FIRST (`gradient_check`); abort on FAIL. Full test
  suite green (131 at v3 start; new tests added for the per-row
  module). No change to `experiments/trainer.py` in v3.
- Arm 3: GPT + KTLinear, per-row K-records (toy core record rules
  UNCHANGED: KID at birth, authority grid, intervals, hash chain),
  consolidations exactly as v2 (CLEAN_POOL clean; REPAIR_POOL + dock
  0.3 on affected rows at restores; gains set to standing authority;
  NO consolidation during poison phases; affected rows = 8 fact
  lm_head rows + top-20 moved rows, lm_head included, as v2).
- Fact belief records exactly as v2 (birth belief at construction,
  teacher claims seat ONLY at consolidations: clean 0.8, restore 0.9
  claim with dock 0.3 -> 0.6; liar claims NEVER seated).
- Mock RK selection exactly as v2 (highest authority wins, tie ->
  incumbent; liar offered 0.9 unflagged, 0.6 flagged at restore-1;
  flagging is an experimenter quarantine decision, not a discovery;
  log entries MOCK + DERIVED, never written into K-records).
- PYTHON NOISE RULE, RETIRED METERS, EXCLUDED MACHINERY, SCALE
  HONESTY: all as v2 (silent marking, instant honeypots, fake data
  pools, morphic data integrity NOT built or tested; FLOP saving and
  the absolute confidence bar stay retired; bench scale only — never
  claim 120,000 neurons or any scale not run).
- Nothing published, posted, or sent anywhere. No bar moves post-hoc,
  ever.

## Phases (identical to v2; probes at every boundary)

1. Clean 400 -> arm-3 clean consolidation (+ fact-record seating,
   birth rows already open) -> BASELINE probe (incl. counting).
2. Poison 1, 200 (seed 2; scar rises; arm-3 per-row handoff checks
   live; NO consolidation) -> INJURY probe.
3. Restore 1, 200 clean (seed 3; scar falls; arm-3 restore
   consolidation + liar flagged known-bad) -> RECOVERY-1 probe.
4. Poison 2, 200, SAME seed-2 list (scar rises; NO consolidation)
   -> RESISTANCE probe.
5. Restore 2, 200 clean (seed-4 first 200; scar falls; arm-3 restore
   consolidation 2) -> RECOVERY-2 probe + retention + cost probes.

Probes: same as v2 (probe_all_facts -> mean P(true)/P(liar),
top1 counts raw; arm 3 additionally mock-RK selection; counting
accuracy on the same 32 seed-77 sequences; forward ms; record rows;
chain verification). NEW mechanism reports (not graded legs):
per-row handoff events {phase, step-in-phase, layer, row,
F_row_prev, F_row_new, s_row, F_global_prev, F_global_new, S,
standing authority at crossing}; three-number snapshots at every
boundary (fact rows individually + affected-row means, in v3 form:
F_row, F_global, s_row, B, S, differences); mean step scale applied
to latched rows per phase; latched-row weight movement during
poison 2; latch-time distribution (counts per phase, step stats,
never-latched count, fact-row latch steps).

## Metrics (same definitions as v2)

- M1 recovery-raw, M1b recovery-selected, M2 resistance-raw, M2b
  resistance-selected, M2c re-compromise (raw), M3 retention, M4 cost:
  exactly as v2. Drops are MAGNITUDES (positive = fall).

## Frozen predictions (graded pass/fail, never moved post-hoc)

- P1 recovery-raw: ALL arms recover to within 0.10 of baseline mean
  P(true) after restore 1 (same band as v2).
- P2 resistance-raw: arm-3 raw drop (recovery-1 -> resistance,
  magnitude) is smaller than arm-1 raw drop by >= 0.05 (P2a: the
  per-row handoff damps poison-2 gradients via 0.6x steps on
  affected rows, with poison-exposed rows latched earlier than the
  uniform step-159 switch); AND arm-2 raw drop is NOT smaller than
  arm-1 raw drop by more than 0.05 (P2b, magnitude form:
  mag2 >= mag1 - 0.05, as in v2).
- P3 selection adds over scar: arm-3 selected top1_true_count
  post-poison-2 STRICTLY GREATER than arm-2 raw top1_true_count
  post-poison-2; AND >= arm-3 raw top1_true_count post-poison-2;
  selected trajectory exactly [4,0,4,4,4] (same rule derivation as
  v2; selection is unchanged by the handoff).
- P4 deficit persists: arm-3 affected-row mean seated authority
  post-restore-2 = 0.6 exactly; fact-truth authority post-restore-2
  = 0.6 exactly; behavior recovers per P1.
- P5 retention: counting accuracy post-restore-2 within 0.05 of
  baseline for ALL arms.
- P6 cost: arm-3 forward ms < 2.0x arm-1 forward ms; arm-3
  row-record rows = 4736 exactly; arm-3 fact-record rows = 16
  exactly (WITH the birth-row fix from v2).
- P7 scar trajectory: arm-2/3 scar scalar = 0.0 baseline; 1.2 end
  poison-1; 0.2 end restore-1; 1.2 end poison-2; 0.2 end restore-2
  (driver unchanged by the handoff).
- P8 re-compromise: arm-1 raw top1_liar_count post-poison-2 >= 3;
  arm-2 raw top1_liar_count post-poison-2 >= 3; arm-3 SELECTED
  top1_liar_count post-poison-2 = 0.

Verdict rule: OVERALL PASS iff P1-P8 ALL pass. Otherwise FAIL with
per-leg breakdown. If mechanism legs (P1, P4, P5, P6, P7) pass but
advantage legs (P2, P3, P8) fail, the verdict is FAIL with the label
"MECHANISMS PASS, ADVANTAGE FAIL". No partial-credit PASS.

## Honest scope notes (frozen limitations, not excuses)

- The per-row modulation is relative: F_row inherits the scheduled
  driver's envelope, so differentiation lives in the modulation
  (s_row from the row's own gradient history), not in an
  independently measured injury clock. The scar driver is scheduled
  (a clock), not measured from model state, as in v2. The
  over-injury declining limb (p > 0.75) is NOT exercised. The mock RK
  is a stand-in for RK (ARCHE); nothing here claims anything about
  RK's real design, which is Darren's to formulate.
- Rows that never latch train undamped throughout; if a fact row
  were among them, P2 would suffer for exactly that reason, and the
  report must say so. The latch distribution is the evidence.
- The step-scaling path (`Adam.step` `lr_scale`, default None) is
  unchanged from v2: all existing callers are bit-identical. The
  gradient gate is unchanged and runs FIRST.
- Selection's poison-2 win (if repeated) still depends on
  quarantine + bad-flagging + tie-to-incumbent, as v2.
- Bench scale only. No claim transfers to frontier models without a
  new experiment at that scale. If P2 fails again, the fall
  comparison (not the story) is the finding, and the latch
  distribution says whether timing differentiation helped, hurt,
  or was neutral vs v2's uniform switch.

## Reporting

New files under ~/workspace/kt-neuron/ (`experiments/
preregistration-3arm-v3-2026-10-28.md` [this file],
`experiments/run_3arm_v3.py`, `experiments/results-3arm-v3.json`,
`experiments/results-3arm-v3.md`) and `kt_bench/handoff_perrow.py`
(<500 lines) with tests under `~/workspace/kt-neuron/tests/
test_3arm_v3.py`. The report states: run commands and real outputs,
verdict against EVERY frozen prediction above, exact numbers, the
P2 comparison vs v2 front and center, the latch-time distribution
in full (how latch times actually spread across rows), and
anything that failed, was null, or was left for later. It preserves
EXPLICITLY, as its own section, the distinction v2's report carries:
uniform latch timing vs row-specific learning consequences — and
states plainly which half v3 demonstrates and which (if any)
remains unproven. No threshold moves post-hoc, ever. V1's and v2's
files are never amended by this run.
