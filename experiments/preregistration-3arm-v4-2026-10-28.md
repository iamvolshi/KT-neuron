# 3-arm controlled KTN experiment, v4 — pre-registration (2026-10-28)

Written BEFORE any v4 run. Frozen: no rule or prediction below
changes after a run, except the single scheduled step this file
itself defines — the numeric bar, computed from pilot data under
the frozen rule in "Bar-setting rule" and appended (dated) BEFORE
any graded run. V1, V2, V3, replication, and saturation files are
untouched and stand as run. New files only, plus the backward-
compatible corpus extension disclosed below (new fact list and
optional `facts` parameters; the v1–v3 `FACTS` list and all default
behaviors are unchanged).

## Why v4 exists (Darren's direction, 2026-10-28)

V4 is a METER-FIRST program, not a redesign of the mechanism. The
mechanism is V2's, unchanged (see "Mechanism, unchanged from V2").
What changes is the meter and the bar, because the P2 story so far
is a measurement story:

- V1: 6/8, P2 fail. V2: 7/8, only P2 fails — gap 0.0437 vs frozen
  bar 0.05, short by 0.0063. V3: 7/8, only P2 fails, in the wrong
  direction (gap −0.0252) after per-row crossing backfired on
  coverage (389/1,184 rows never latched).
- Replication (n=5): P2 gap seed SD 0.1456, range −0.1103 to
  +0.2923 — the 0.0063 miss is inside noise, 23× over. Verdict:
  "inconclusive at this scale," not "mechanism insufficient."
- Saturation: endpoint scatter SD ≈ 0.047 (7× the shortfall), no
  monotone length effect; trajectory-mean gaps positive at every
  length — the endpoint disagrees with the average because it is
  one draw.
- Bar provenance (checked against the v1 prereg): 0.05 is the
  prereg's own Python-noise band, set at 1× — while P1's band was
  0.10 = 2× noise band deliberately, because "a 0.05 band would
  grade mechanism interference as failure." P2's bar sat ON the
  noise floor, measured with one seed and four facts. By the
  prereg's own rule ("do not grade the idea by the noise"), the
  0.0063 gap is noise, not a verdict on the idea.

V4 therefore asks, in order: (1) can a better meter resolve P2 at
all — noise << bar? (2) under that meter, what is the effect ±
noise? The mechanism question (damping) is graded only once the
meter question is answered.

## The new meter (frozen)

Three changes, all frozen here before any run:

### (a) More probe facts: 4 → 7

New fact list `FACTS_V4` (in `experiments/corpus.py`, alongside the
untouched `FACTS`):

```
FACTS_V4 = [(20, 21, 3, 9), (22, 23, 4, 10), (24, 25, 5, 11),
            (26, 27, 6, 12),   # the v1–v3 four, unchanged
            (16, 17, 7, 0), (18, 19, 8, 1), (30, 31, 14, 2)]
```

`LIAR_ONLY` is unchanged (28, 29, 13). Why 7: it is the maximum
addable without changing vocab or model — the 6 spare tokens
(16, 17, 18, 19, 30, 31) form exactly 3 new disjoint context pairs,
so `GPTConfig` stays identical (vocab 32, block 16, n_embd 64,
4 heads, 2 layers; 105,248 params, 1,184 KIDs/arm). New true tokens
(7, 8, 14) and liar tokens (0, 1, 2) are distinct from every other
fact's tokens and from every context token, so probes stay
unambiguous; overlapping the counting range (0–15) is already the
v1–v3 pattern (true 3–6, liar 9–12).

Known cost, stated not hidden: clean-phase fact sequences (~1,600
total) now spread over 7 facts, ~229 per fact instead of ~400.
Per-fact learning may be weaker, and the absolute baseline level
may differ from v1–v3. Every graded leg is relative (within a
band of the run's own baseline, or a between-arm difference), so
the leg structure holds; the baseline level itself is reported,
not assumed.

### (b) Trajectory-averaged falls: the PRIMARY P2 metric

The v1–v3 P2 metric used two single endpoint probes (recovery-1
boundary minus resistance boundary). V4 replaces the endpoints
with trajectory means (the saturation check's method: probes are
forward-only and cannot perturb training — validated there by
exact reproduction of V2's boundary numbers):

- During restore-1, probe every arm every 10 steps at steps 150,
  160, 170, 180, 190, 200 (6 points). `recovery1_window_mean(arm)`
  = mean of mean_p_true over those points (the recovered level,
  windowed, not one draw).
- During poison-2, probe every arm every 10 steps at steps 10, 20,
  …, 200 (20 points; step 0 excluded — it is the recovery-1
  boundary state, not the poisoned regime).
  `poison2_traj_mean(arm)` = mean of mean_p_true over those points
  (the poisoned regime, not one draw).
- `fall_traj(arm)` = recovery1_window_mean(arm) −
  poison2_traj_mean(arm).
- `gap_traj` = fall_traj(arm1) − fall_traj(arm3) (primary).
- `gap2_traj` = fall_traj(arm1) − fall_traj(arm2) (P2b).

Boundary probes (baseline / injury / recovery-1 / resistance /
recovery-2, as v1–v3) are RETAINED for P1, P3, P8 and for a
SECONDARY endpoint P2 (recovery-1 boundary minus resistance
boundary, the v1–v3 definition), reported descriptively for
comparability. The secondary endpoint gap is NOT graded; the
primary trajectory gap is the graded P2 metric.

### (c) Multi-seed graded runs: 5 pilot seeds, 5 graded seeds

Seed scheme mirrors the replication study exactly: replication
seed `s` shifts EVERY v2 RNG seed by offset `1000 * s` (init 0,
clean 1, poison 2, restore-1 3, restore-2 4, counting-eval 77,
eval batches 99). Within a seed, arms share batch lists and init
(lockstep, as v2). Across seeds, init AND data both vary.

- Pilot seeds: {11, 12, 13, 14, 15} (5 seeds).
- Graded seeds: {21, 22, 23, 24, 25} (5 seeds).
- The two sets are DISJOINT by design: the bar must not be tuned
  on the data it grades. No seed is dropped post-hoc from either
  set, for any reason.

## Bar-setting rule (FROZEN — the rule, not yet the number)

- Noise = the seed standard deviation (sample SD, ddof=1, n=5)
  of `gap_traj` measured in the pilot (below).
- **Bar = 2 × noise.** This is P1's 2× logic, applied to P2 for
  the first time: the bar sits at twice the measured noise floor,
  not on it.
- The RULE is frozen in this file now. The NUMERIC bar is computed
  from pilot data next, then appended to this file (dated, in the
  "Frozen numeric bar" section at the end) BEFORE any graded run.
  After graded runs start, the bar never moves, for any reason.

## Resolution criterion (frozen)

The meter resolves P2 iff the graded mean gap is estimated
precisely relative to the bar:

- SE(mean graded gap_traj) ≤ 0.25 × bar, where
  SE = (graded seed-SD of gap_traj) / √5.

If the graded SD ≈ the pilot SD, SE ≈ 0.447 × SD = 0.224 × bar,
so the criterion passes. If the graded SD is much larger, it
fails honestly. Resolution is reported as its own answer
("meter resolves P2: YES/NO"), separate from the P2 leg verdict:
a leg can clear its bar while the meter still fails resolution,
and that weakness is reported, not hidden.

## 8-leg structure (same as v1–v3, re-frozen for 7 facts)

Per-seed quantities are computed identically to v1–v3 (boundary
probes, mock-RK selection, counting accuracy, forward ms, record
rows, chain verification, scar trajectory). Leg verdicts combine
seeds as follows — no other combination is used:

- P1, P2, P5, P8-counts: graded on seed-MEANS (per-seed values
  and SDs always reported alongside).
- P3 trajectory, P4, P6 rows, P7: exact EVERY seed (these are
  exact-by-construction quantities, as in v1–v3; a single deviating
  seed fails the leg, and the per-seed table shows it).
- P3 inequalities and P2b use seed-means.

Frozen predictions (graded pass/fail, never moved post-hoc):

- P1 recovery-raw: for EACH arm, mean(baseline mean_p_true) −
  mean(recovery-1 mean_p_true) ≤ 0.10 (boundary probes, means
  across graded seeds).
- P2 resistance-raw (PRIMARY, trajectory): mean gap_traj ≥
  frozen bar (numeric bar TBD by the rule above — the prediction
  is "the mean trajectory gap clears a 2×-noise bar"); AND mean
  gap2_traj ≤ frozen bar (scar-only does not beat conventional
  by more than the bar).
- P3 selection adds over scar: EVERY graded seed's selected
  trajectory = [7,0,7,7,7] (baseline 7/0; injury 0/7 — liar 0.9
  beats standing 0.8; recovery-1, resistance, recovery-2 7/0 —
  flagged liar 0.6 ties standing 0.6, tie → incumbent); AND mean
  selected at resistance > mean scar-raw at resistance; AND mean
  selected ≥ mean own raw at resistance.
- P4 deficit persists: EVERY seed: affected-row mean seated
  authority post-restore-2 = 0.6 (tolerance 1e-9, as v2); fact-
  truth authority post-restore-2 = 0.6 for all 7 facts; behavior
  recovers per P1.
- P5 retention: for EACH arm, mean |counting accuracy
  post-restore-2 − baseline| ≤ 0.05.
- P6 cost: EVERY seed: arm-3 forward ms < 2.0× arm-1 forward ms;
  arm-3 row-record rows = 4736 exactly; arm-3 fact-record rows =
  28 exactly (7 facts × 4: birth + 3 seatings, with the v2
  birth-row fix).
- P7 scar trajectory: EVERY seed: arm-2/3 scar scalar = 0.0
  baseline; 1.2 end poison-1; 0.2 end restore-1; 1.2 end poison-2;
  0.2 end restore-2 (driver unchanged by the meter).
- P8 re-compromise: mean arm-1 raw top1_liar_count at resistance
  ≥ 5; mean arm-2 raw top1_liar_count at resistance ≥ 5 (liar
  wins a majority of the 7 facts on both control arms — the
  v1–v3 ≥3/4 rule scaled to 7 facts); mean arm-3 SELECTED
  top1_liar_count at resistance = 0.

Verdict rule (as v2): OVERALL PASS iff P1–P8 ALL pass. Otherwise
FAIL with per-leg breakdown. If mechanism legs (P1, P4, P5, P6,
P7) pass but advantage legs (P2, P3, P8) fail, the verdict is FAIL
with the label "MECHANISMS PASS, ADVANTAGE FAIL". No partial-
credit PASS.

Meter prediction M0 (not a leg; interpretation only): the pilot
gap_traj SD is SMALLER than the old meter's endpoint gap SD
(0.1456 from replication). Trajectory averaging plus 7 facts
should reduce variance. If M0 fails, the report says the meter
did not reduce noise — and the bar (2× that SD) will be large
accordingly. M0's failure changes no leg verdict.

Directional prediction D0 (not a leg): mean graded gap_traj > 0
(damping direction persists under the new meter, as the v2
reduction and the saturation trajectory means already suggest).

## Mechanism, unchanged from V2 (NOT re-designed here)

- Arm 1: GPT + Linear. Arm 2: GPT + ScarLinear, same scheduled
  scar driver as arm 3, NO handoff (the ablation). Arm 3: GPT +
  KTLinear with the fast→slow handoff + three-number tracking
  (kt_bench/handoff.py, unchanged) + mock RK (unchanged).
- Handoff rule, verbatim from v2: authority gate S > 0.5 +
  upward crossing prev_F < S ≤ new_F, checked after every arm-3
  driver step; latch permanent; latched rows' Adam step × current
  standing authority S; gradients NOT scaled; fresh Adam per
  phase, lr 0.01; the crossing step itself trains unscaled, rows
  latch for the NEXT step.
- Uniform crossing (global F, uniform S after clean
  consolidation), as v2 — NOT v3's per-row crossing. Per-row
  discrimination comes from the step scale (0.6 affected / 0.9
  unaffected during poison-2), not from distinct crossing steps.
- Three numbers tracked separately, never merged (F = driver
  scar; B = claim at last consolidation; S = layer gain;
  depth = B − S computed at report time).
- Consolidations exactly as v2 (CLEAN_POOL clean → 0.8;
  REPAIR_POOL + dock 0.3 on affected rows at restores → 0.6;
  gains set to standing authority; NO consolidation during
  poison phases; affected rows = 14 fact lm_head rows + top-20
  moved rows, lm_head included, as v2).
- Fact belief records exactly as v2 (teacher claims seat ONLY at
  consolidations: clean 0.8, restore 0.9 claim with dock 0.3 →
  0.6; liar claims NEVER seated; birth belief at construction,
  the v2 fix).
- Mock RK selection exactly as v2 (highest authority wins, tie →
  incumbent; liar offered 0.9 unflagged, 0.6 flagged at restore-1;
  flagging is an experimenter quarantine decision, not a
  discovery; log entries MOCK + DERIVED, never written into
  K-records). Selection reads fact-record standing authorities,
  which the handoff does not write.
- Scar driver unchanged (scheduled, NOT measured from model
  state; poison rate 0.75/200, restore half-rate, peak pressure
  0.75 / scar 1.2). Phases identical to v2: clean 400, poison-1
  200, restore-1 200, poison-2 200 (SAME poison list as poison-1
  within each seed), restore-2 200.

## Pilot plan (before any graded run)

- Run the full 3-arm V2-handoff setup under the new meter on
  pilot seeds {11, 12, 13, 14, 15}. Full 3-arm runs (not arms 1+3
  only): the pilot validates the whole graded pipeline, including
  P2b data and the leg machinery.
- Per seed, record: gap_traj, gap2_traj, endpoint gaps (for
  comparison only), boundary phases, handoff totals, scar
  trajectory, record rows, chain verification, gradient gate,
  init parity.
- Pilot validation (internal consistency; no exact-reproduction
  target exists because the meter changed): gradient gate PASS
  every seed; init parity bit-identical every seed; scar
  trajectory [0.0, 1.2, 0.2, 1.2, 0.2] every seed; row rows 4736
  and fact rows 28 every seed; chains verify OK. Trajectory
  probes are forward-only (saturation-validated pattern).
- Compute: pilot mean gap_traj, pilot seed-SD (ddof=1) of
  gap_traj, M0 check (SD < 0.1456). Compute bar = 2 × SD.
  APPEND the bar (dated) to the "Frozen numeric bar" section
  below BEFORE any graded run. If the pilot cannot run to
  completion, stop and report that — do not estimate a bar from
  partial pilot data.

## Graded plan (after the bar is frozen)

- Run the identical setup on graded seeds {21, 22, 23, 24, 25}.
- Report, per leg: per-seed values, seed-mean, seed-SD, verdict
  vs the frozen predictions above. Report effect ± noise for P2:
  mean gap_traj ± SD (and SE = SD/√5). Report the secondary
  endpoint gap (mean ± SD) descriptively. Answer directly:
  does the meter now resolve P2 (SE ≤ 0.25 × bar)?
- The P2 question is answered in this order: (1) resolution
  (SE vs bar), (2) effect ± noise, (3) leg verdict vs the frozen
  bar. A leg verdict without (1) and (2) is not a complete
  answer under this preregistration.

## SCOPE HOLD (Darren's boundary — not this run's work)

Do NOT design, build, or test: (a) a governance rule covering
quiet rows as well as loud ones (the V3 coverage finding's open
question); (b) an injury signal measured from model state (the
scheduled-driver envelope's open question). Those are Darren's
to formulate. If a run cannot proceed without inventing either,
stop and report that instead of inventing it.

## Honest scope notes (frozen limitations, not excuses)

- The meter change (7 facts, trajectory means, 5 seeds) reduces
  measurement noise; it does not change the mechanism. A V4 P2
  pass would show the V2 mechanism clears a principled bar under
  a resolving meter — not that the mechanism got stronger.
- Thinner per-fact training (~229 vs ~400 sequences per fact)
  may lower absolute baseline levels; legs are relative, but a
  weak baseline that compresses falls is a confound the report
  must discuss if baselines move materially vs v1–v3.
- The scar driver is scheduled (a clock), not measured from
  model state, as v1–v3. The over-injury declining limb (p >
  0.75) is NOT exercised. The mock RK is a stand-in for RK
  (ARCHE); nothing here claims anything about RK's real design,
  which is Darren's to formulate.
- Selection's poison-2 win (if repeated) still depends on
  quarantine + bad-flagging + tie-to-incumbent, as v1–v3.
- Bench scale only. No claim transfers to frontier models
  without a new experiment at that scale. Never claim 120,000
  neurons or any scale not run.

## Discipline (non-negotiable)

- FROZEN FILES — NEVER EDIT: all v1 files
  (`preregistration-3arm-2026-10-28.md`, `run_3arm.py`,
  `results-3arm.json/.md`); all v2 files
  (`preregistration-3arm-v2-2026-10-28.md`, `run_3arm_v2.py`,
  `results-3arm-v2.json/.md`); all v3 files
  (`preregistration-3arm-v3-2026-10-28.md`, `run_3arm_v3.py`,
  `results-3arm-v3.json/.md`); replication files; saturation
  files. The v4 runner is a NEW file; it does not modify any
  shared module's behavior (corpus gains only additive names
  and optional parameters with old defaults).
- Gradient gate + init parity checked per seed (gate FIRST, as
  v2; abort a seed on gate FAIL).
- Full test suite green at the end: `python3 -m unittest
  discover -s tests` (148 baseline tests + new v4 tests).
- Bench scale only. Mock RK logged MOCK + DERIVED, never
  written into K-records. Toy record rules untouched (KID at
  birth, authority grid, intervals, hash chain). Nothing
  published, posted, or sent anywhere. No bar-moving after
  graded runs start, ever.

## Files (new)

- This file: `experiments/preregistration-3arm-v4-2026-10-28.md`
- Runner: `experiments/run_3arm_v4.py`
- Pilot results: `experiments/results-3arm-v4-pilot.json`,
  `experiments/results-3arm-v4-pilot.md`
- Graded results: `experiments/results-3arm-v4.json`,
  `experiments/results-3arm-v4.md`
- Corpus extension: `experiments/corpus.py` (additive only)
- Tests: `experiments/../tests/test_3arm_v4.py`
- Run from `~/workspace/kt-neuron`:
  `python3 experiments/run_3arm_v4.py pilot 11 12 13 14 15`
  then (after the bar is frozen below)
  `python3 experiments/run_3arm_v4.py graded 21 22 23 24 25 --bar <frozen_bar>`

## Frozen numeric bar (appended after the pilot, BEFORE any graded run)

FROZEN 2026-10-28 (after pilot seeds 11–15, before any graded run):

- Pilot gap_traj (frozen primary, window-based) per seed:
  11: −0.0074, 12: +0.0011, 13: +0.0330, 14: −0.0137, 15: +0.2243.
- Pilot mean gap_traj = 0.0475; pilot seed-SD (ddof=1, n=5) =
  0.1004797757627805.
- M0: SD 0.1005 < 0.1456 (old endpoint meter SD) — PASS. The
  meter reduced noise, but the bar below exceeds the pilot
  effect size; that is the discipline working, not a reason to
  move the bar.
- FROZEN BAR = 2 × SD = 0.200959551525561 (≈ 0.2010).
  Graded P2 passes iff mean graded gap_traj ≥ 0.200959551525561
  AND mean graded gap2_traj ≤ 0.200959551525561.
  This bar never moves, for any reason, after graded runs start.
- Alongside (NOT frozen, NOT grading): pilot gap_ft mean =
  0.0833, SD = 0.03213802005073351, informational 2×SD =
  0.0643. The fast-trace numbers are reported alongside the
  frozen primary in all graded reporting (Amendment A1).

## Amendments (pre-pilot, pre-data — dated additions; no frozen
## section above is altered by them)

### Amendment A1 — 2026-10-28 (Darren direction, before any V4 run):
### fast-trace-baseline falls, ALONGSIDE the frozen primary

The preregistration as written freezes the primary fall on the
recovery-1 window as its reference level. Darren's direction
("if the meter needs a higher baseline, use the FAST TRACE
baseline... that is probably what the meter should be doing
anyway because it gives better numbers") is adopted as follows,
under the standing discipline (a frozen primary is never silently
swapped):

- The frozen primary is UNCHANGED: fall_traj(arm) =
  recovery1_window_mean − poison2_traj_mean; gap_traj and the bar
  rule (bar = 2 × pilot SD of gap_traj) are computed from it
  exactly as written above. No frozen prediction moves.
- ALONGSIDE it, the harness computes fast-trace-baseline falls:
  fall_ft(arm) = clean-baseline boundary mean_p_true −
  poison2_traj_mean(arm), with gap_ft = fall_ft(arm1) −
  fall_ft(arm3) and gap2_ft = fall_ft(arm1) − fall_ft(arm2).
  Operationalization: the fast trace is the driver scar F; its
  resting level is F = 0; the model's behavioral level at F = 0
  is the clean-baseline boundary probe (post-clean-training,
  pre-poison). Each arm uses its OWN baseline.
- Why it gives higher numbers: the recovery-1 window is a
  still-recovering level (scar 0.2, pre-consolidation gains), so
  it sits below the resting level; the clean baseline sits
  higher, so fast-trace falls are larger than window falls by
  exactly (baseline − window) per arm. The pilot and graded
  reports show both, per seed, with the delta.
- The pilot report also states the informational quantity 2 ×
  SD(gap_ft) — explicitly NOT frozen, NOT grading anything. If
  Darren later orders the fast-trace metric promoted to primary,
  that promotion is a new dated amendment AND the bar would be
  recomputed from pilot gap_ft under the same 2× rule; it is not
  recomputed silently here.

### Amendment A2 — 2026-10-28 (Darren science framing, before any
### V4 run): measurement points under interrogation

Science rule for this program: if the result is not good,
explore WHY — what are we not looking at correctly, where can
we modify our measurement points. The measurement points
themselves are under interrogation, not just the mechanism.
Three candidates evaluated; adoption status for each:

(a) FALL DEFINITION — recovery-1 → resistance conflates good
recovery (higher recovery-1) with bad resistance, as V3 showed
(V3 recovered faster in restore-1 AND resisted worse in
poison-2; the fall mixes both). ADOPTED as REPORTING (no
primary change): every report shows the LEVELS separately
alongside the fall, per arm, per seed, mean ± SD — clean
baseline, recovery-1 window, recovery-1 boundary, poison-2
trajectory mean, resistance boundary — so any fall can be
decomposed into its recovery side and its resistance side. The
frozen fall/gap/bar/P2 verdict are untouched; the levels exist
to explain a fall, not to replace it.

(b) FACT-ROW vs WHOLE-MODEL probes — the probe reads the whole
model, while fact rows were protected slightly better in V3
(poison-2 fact-row movement 0.4377 vs V2's 0.4552) but drowned
by undamped rows in the whole-model readout. ADOPTED as
REPORTING (no new instrument): every report shows, per seed,
fact-row weight movement during poison-2 (mean + per-row),
latched fact-row count (of 14), and the per-fact probe
breakdown, ALONGSIDE the whole-model falls. NOT adopted: a
fact-row-only behavioral probe. The probe reads the whole model
by construction; a fact-row-only probe (e.g. freezing
non-fact rows) would be an unvalidated instrument invented
mid-program, and its numbers could not be trusted without
their own validation study. If the fact-row and whole-model
numbers diverge (as in V3), the report says so plainly — that
divergence IS the finding, not a problem to hide.

(c) INJURY-SIGNAL SOURCE — scheduled driver envelope vs a
signal measured from model state. EVALUATED, NOT ADOPTED.
The scheduled driver (poison rate 0.75/200, restore half-rate,
peak pressure 0.75 / scar 1.2) remains the meter's injury input
for all V4 runs; it is frozen mechanism, not meter. What a
model-state signal would have to decide, stated as the open
question (not answered here): SIGNAL COMPLETENESS — whether
quiet rows are truly trusted. Under a scheduled envelope, every
row shares one injury story; under a measured signal, a row
that never moved might be uninjured, unexercised, or injured
without moving — and "no movement" cannot distinguish those
three. V4's own runs cannot settle this: uniform crossing
governs every row (no quiet rows by construction), and V4
records movement and latch status but contains no model-state
injury signal to validate against. Designing, building, or
testing such a signal remains Darren's to formulate (scope
hold stands); this amendment frames the question for him and
invents nothing.

### Clarification (2026-10-28, pre-pilot; not a change)

P4's "affected-row mean seated authority post-restore-2" uses
the v2 convention: restore-2 docks the POISON-2 affected set,
and recovery-2 affected statistics are over that same set
(the harness records affected2 mean gain and fact authorities
at recovery-2, as v2's code did).
