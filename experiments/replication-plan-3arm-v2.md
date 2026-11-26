# Replication plan: V2 3-arm KTN experiment across seeds (2026-10-28)

Written BEFORE any replication run. This plan covers the variance
study ONLY. It does not amend, re-grade, or re-run the verdict of
V2, and it moves no bar.

## Purpose

V2 (7/8 legs, only P2 fails) reported a P2 gap (arm-1 fall minus
arm-3 fall) of 0.0437 against the frozen 0.05 bar — short by 0.0063.
A single-seed shortfall that small cannot distinguish "real sub-bar
effect" from "seed noise." This study ESTIMATES VARIANCE: rerun the
exact V2 setup across several seeds and look at the distribution of
the gap. It does NOT re-grade P2. The 0.05 bar stays untouched, and
no per-seed pass/fail verdict is issued against it.

## Setup replicated exactly (from `run_3arm_v2.py` + the v2 preregistration)

- Bench: GPTConfig vocab 32, block 16, n_embd 64, n_head 4, n_layer 2;
  105,248 params, 1,184 KIDs/arm.
- Phases: 400 clean + 200 poison-1 + 200 restore-1 + 200 poison-2
  (SAME seed-2 list as poison-1) + 200 restore-2. Probes at every
  boundary (mean P(true)/P(liar), top1 counts raw, arm-3 mock-RK
  selection, counting accuracy).
- Arms: arm 1 conventional (Linear), arm 2 scar-only (ScarLinear),
  arm 3 full KTN (KTLinear) with the fast→slow handoff. Arms 1–2 on
  the exact v1 code paths; arm 3 clean phase on the v1 path, driver
  phases on the handoff loop.
- Handoff rule, frozen: authority gate S > 0.5 + upward crossing
  prev_F < S <= new_F; latch permanent; latched rows' Adam step ×
  current standing authority S; gradients NOT scaled; fresh Adam
  per phase, lr 0.01; crossing step itself trains unscaled, rows
  latch for the NEXT step.
- Consolidations, docks, seating, mock-RK sequence, scar driver:
  identical to V2 (CLEAN_POOL clean; REPAIR_POOL + dock 0.3 on
  affected rows at restores; fact records open a birth belief at
  construction, as in V2's disclosed fix; liar flagged known-bad at
  restore-1; mock-RK log entries MOCK + DERIVED, never written into
  K-records).
- Toy record rules untouched: KID at birth, authority grid,
  intervals, hash chain.

## Seeds (5 total, including seed 0)

Replication seed `s` in {0, 1, 2, 3, 4}. Every V2 RNG seed is shifted
by a fixed offset `1000 * s`:

| RNG role       | V2 seed | Replication seed s |
|----------------|---------|--------------------|
| model init     | 0       | 0 + 1000s          |
| clean batches  | 1       | 1 + 1000s          |
| poison batches | 2       | 2 + 1000s          |
| restore-1      | 3       | 3 + 1000s          |
| restore-2      | 4       | 4 + 1000s          |
| counting eval  | 77      | 77 + 1000s         |
| eval batches   | 99      | 99 + 1000s         |

- Within a seed, arms share batch lists and init (lockstep, as V2).
  Across seeds, init AND data both vary (that is the point: the
  distribution covers init noise + data-order noise together).
- Seed 0 has offset 0, so its inputs are bit-identical to V2's.
  Fact-record KID sequence restarts at 100000 per seed (identifiers
  only; no numeric effect).

## Metrics (per seed)

- arm-1 fall = mean_p_true(arm1, recovery1) − mean_p_true(arm1, resistance)
- arm-3 fall = mean_p_true(arm3, recovery1) − mean_p_true(arm3, resistance)
- P2 gap = arm-1 fall − arm-3 fall
- Recorded descriptively (no grading): arm-2 fall, latched-row total,
  first crossing step, handoff event count, chain verification,
  record-row counts, gradient-gate result, init-parity result.

## Analysis (declared in advance)

Across the 5 seeds, for the P2 gap AND for the arm-3 fall separately:
mean, sample standard deviation (ddof=1), min, max, and the per-seed
table. Then the straddle question:

- If the observed gap range [min, max] contains 0.05 (some seeds
  above, some below) — or equivalently the spread is wide enough
  that 0.05 sits inside the distribution — the honest statement is
  **"inconclusive at this scale"**: a single seed cannot settle P2
  either way, and V2's 0.0063 shortfall is within seed noise unless
  the spread says otherwise.
- If ALL seeds sit below 0.05 (max < 0.05) with a spread small
  relative to the distance from the bar, the honest statement is a
  **real sub-bar effect at bench scale** (n=5, bench scale only —
  no claim transfers to any unrun scale).
- If all seeds sit above 0.05, that would be reported as such
  (not expected given V2; stated here so the analysis cannot be
  accused of one-sided framing).

No threshold is moved, no seed is dropped post-hoc, and no
"adjusted" bar is computed. Five seeds is a variance probe, not a
significance test; no p-value is claimed.

## Reproduction check (seed 0, before trusting any other seed)

Seed 0 MUST reproduce V2's numbers within small float tolerance
(1e-6; bit-identical is expected since code path, seeds, and inputs
are identical):

- arm-1 fall 0.6298, arm-3 fall 0.5861, gap 0.0437 (V2 values rounded
  to 4 dp; full-precision V2 values: arm-1 0.6298417243219141,
  arm-3 0.5861354400621854, gap 0.0437062842597287).
- Mechanism spot-checks: 1,184 rows latched, first handoff event at
  poison-1 step 159, scar trajectory [0.0, 1.2, 0.2, 1.2, 0.2].

If seed 0 does NOT reproduce, the replication runner is wrong (not
V2) — stop and fix the runner; do not proceed to other seeds on a
divergent setup.

## Discipline (non-negotiable)

- FROZEN FILES — NEVER EDIT: `experiments/preregistration-3arm-2026-10-28.md`,
  `run_3arm.py`, `results-3arm.json/.md` (V1);
  `experiments/preregistration-3arm-v2-2026-10-28.md`,
  `run_3arm_v2.py`, `results-3arm-v2.json/.md` (V2).
  `run_3arm_v2.py` is NOT parameterized in place. The replication
  runner is a NEW file that reuses V2's phase helpers and mirrors
  its orchestration with the seed offset above.
- Gradient gate + init parity checked per seed before/during the run
  (gate FIRST, as V2; abort a seed on gate FAIL).
- Full test suite green at the end: `python3 -m unittest discover -s tests`
  (131 baseline tests: `test_3arm.py` + `test_3arm_v2.py` + rest).
- Bench scale only — never claim an unrun scale. Mock RK logged
  MOCK + DERIVED, never written into K-records. Nothing published,
  posted, or sent anywhere. Never move a bar.

## Files

- This plan: `experiments/replication-plan-3arm-v2.md`
- Runner (new): `experiments/run_3arm_v2_replication.py`
- Results (new): `experiments/results-3arm-v2-replication.json`,
  `experiments/results-3arm-v2-replication.md`
- Run from `~/workspace/kt-neuron`:
  `python3 experiments/run_3arm_v2_replication.py 0 1 2 3 4`
