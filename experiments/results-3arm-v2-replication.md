# V2 3-arm KTN experiment — seed replication (variance study, 2026-10-28)

Plan: `experiments/replication-plan-3arm-v2.md` (written BEFORE any
replication run). Runner: `experiments/run_3arm_v2_replication.py`
(NEW file; mirrors `run_3arm_v2.py`'s orchestration exactly,
parameterized only by a seed offset). Machine data:
`experiments/results-3arm-v2-replication.json`.

## What this study is and is not

This study ESTIMATES VARIANCE. It does NOT re-grade P2, and the
0.05 bar stays untouched. No per-seed pass/fail verdict is issued
against the bar, no seed is dropped post-hoc, and no "adjusted" bar
is computed. Five seeds is a variance probe, not a significance
test; no p-value is claimed. Bench scale only — no claim transfers
to any unrun scale.

## Seed scheme

Replication seed `s` in {0, 1, 2, 3, 4}; every V2 RNG seed shifted by
`1000 * s` (init 0, clean 1, poison 2, restore-1 3, restore-2 4,
counting 77, eval 99). Within a seed, arms share batch lists and
init (lockstep, as V2); across seeds, init AND data both vary.
Everything else — GPTConfig, phases (400 + 200 + 200 + 200 + 200),
arms, handoff rule (gate S > 0.5 + upward crossing
prev_F < S <= new_F; Adam step × standing authority S; gradients
NOT scaled), consolidations, docks, seating, mock-RK sequence, scar
driver — is identical to V2.

Run from `~/workspace/kt-neuron`:
`python3 experiments/run_3arm_v2_replication.py 0 1 2 3 4`
(Seed 0 was run first and checked before seeds 1–4 were launched.)

## Reproduction check (seed 0) — BIT-IDENTICAL to V2

Seed 0 (offset 0) reproduced V2 exactly, differences 0.0:

| | V2 (frozen) | Replication seed 0 | Diff |
|---|---|---|---|
| arm-1 fall | 0.6298417243 | 0.6298417243 | 0.0 |
| arm-3 fall | 0.5861354401 | 0.5861354401 | 0.0 |
| P2 gap (arm-1 fall − arm-3 fall) | 0.0437062843 | 0.0437062843 | 0.0 |

Mechanism spot-checks also matched V2 exactly: 1,184 rows latched,
first handoff event at poison-1 step 159 (F 0.79 → 0.80, S 0.8),
scar trajectory [0.0, 1.2, 0.2, 1.2, 0.2], row-record rows 4736,
fact-record rows 16, chains OK, gradient gate PASS, init parity
bit-identical across arms. The runner is faithful; the other seeds'
numbers are therefore genuine seed variation, not runner drift.

## Per-seed results

Falls are magnitudes: mean_p_true(recovery1) − mean_p_true(resistance).
Gap = arm-1 fall − arm-3 fall. Arm-2 fall shown descriptively.

| Seed | arm-1 fall | arm-2 fall | arm-3 fall | P2 gap |
|---|---|---|---|---|
| 0 (V2) | 0.6298 | 0.7224 | 0.5861 | 0.0437 |
| 1 | 0.8029 | 0.5583 | 0.5107 | 0.2923 |
| 2 | 0.5495 | 0.5483 | 0.5032 | 0.0463 |
| 3 | 0.4483 | 0.5202 | 0.5585 | −0.1103 |
| 4 | 0.7190 | 0.8093 | 0.6962 | 0.0228 |

Full precision is in the JSON. Underlying probe values, e.g. seed 3:
arm-1 recovery-1 0.7038 → resistance 0.2555; arm-3 recovery-1
0.9008 → resistance 0.3423 (arm 3 fell further that seed).
Seed 1: arm-1 0.9762 → 0.1732; arm-3 0.9202 → 0.4095.

## Distribution (n = 5)

| Metric | Mean | Sample SD (ddof=1) | Min | Max | Range |
|---|---|---|---|---|---|
| P2 gap | 0.0590 | 0.1456 | −0.1103 (seed 3) | 0.2923 (seed 1) | 0.4025 |
| arm-3 fall | 0.5709 | 0.0779 | 0.5032 (seed 2) | 0.6962 (seed 4) | 0.1929 |
| arm-1 fall | 0.6299 | 0.1390 | 0.4483 (seed 3) | 0.8029 (seed 1) | 0.3547 |

## Finding: inconclusive at this scale

The gap distribution STRADDLES 0.05 decisively: one seed (seed 1)
at 0.2923, far above the bar; one seed (seed 3) NEGATIVE at
−0.1103 (arm 3 fell more than arm 1 that seed); three seeds
clustered below the bar (0.0228–0.0463). The honest statement,
per the pre-declared analysis, is **"inconclusive at this scale."**

Scale of the noise vs the V2 shortfall: V2 fell short of the bar
by 0.0063. The seed SD of the gap is 0.1456 — 23× that shortfall —
and the full range is 0.4025, 64× the shortfall. A single seed
cannot settle P2 either way at this bench scale; V2's 0.0437 is
well inside seed noise. (The mean gap, 0.0590, sits above the bar,
but with SD 0.1456 and a negative seed in n=5, the mean is not
evidence for anything — it is dominated by seed 1.)

Two further honest observations, descriptive only:

- The machinery is seed-INVARIANT: every seed latched exactly
  1,184 rows with the first crossing at poison-1 step 159, the
  same scar trajectory, the same 4736/16 record rows, and clean
  chains. The variance lives in the falls (how far each arm's
  probe probability drops under poison 2), not in the handoff
  mechanism itself.
- The arm-3 fall varies less across seeds (SD 0.0779) than the
  arm-1 fall (SD 0.1390), but with n=5 that is a description of
  these five runs, not a claim about damping — seed 3's negative
  gap is a direct counterexample to any blanket damping story at
  this scale.

## Discipline

- FROZEN FILES UNTOUCHED: V1 (`preregistration-3arm-2026-10-28.md`,
  `run_3arm.py`, `results-3arm.json/.md`) and V2
  (`preregistration-3arm-v2-2026-10-28.md`, `run_3arm_v2.py`,
  `results-3arm-v2.json/.md`) were never edited; the replication
  runner is a new file and `run_3arm_v2.py` was not parameterized
  in place.
- Gradient gate PASS + init parity bit-identical across arms in
  every seed (checked per seed, gate first, as V2).
- Full test suite green at the end, run twice:
  `python3 -m unittest discover -s tests` → **Ran 148 tests, OK**
  (twice). Note: the task cited 131 baseline tests from the V2
  preregistration; the suite as it stands runs 148 tests, all
  green — the count grew since V2, nothing failed or was skipped.
- Bench scale only. Mock RK logged MOCK + DERIVED, never written
  into K-records. Toy record rules untouched (KID at birth,
  authority grid, intervals, hash chain). Nothing published,
  posted, or sent anywhere. No bar moved.
