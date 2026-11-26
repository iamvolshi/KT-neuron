# V4 graded report — 2026-10-28

Frozen contract: `experiments/preregistration-3arm-v4-2026-10-28.md`
(including Amendments A1, A2). Graded seeds {21, 22, 23, 24, 25},
full 3-arm V2-handoff runs under the new meter, graded against
the frozen bar 0.200959551525561 (= 2 × pilot gap_traj SD,
frozen 2026-10-28 before any graded run). Pilot seeds {11–15}
are disjoint; no seed dropped from either set.

Answer order, per the preregistration: (1) resolution,
(2) effect ± noise, (3) leg verdicts.

## (1) Resolution: YES — the meter resolves P2

SE(mean graded gap_traj) = 0.0804 / √5 = 0.0359 ≤ 0.25 × bar =
0.0502. The graded SD (0.0804) came in near the pilot SD
(0.1005), so the resolution criterion passes. The meter's
precision is now finer than a quarter of the bar. What it
resolves is below, and it is not a pass.

## (2) Effect ± noise (frozen primary, window-based)

fall_traj(arm) = recovery-1 window mean − poison-2 trajectory
mean. gap_traj = fall(arm1) − fall(arm3).

| seed | fall arm1 | fall arm2 | fall arm3 | gap_traj | gap2_traj | gap_endpoint | gap_ft |
|------|-----------|-----------|-----------|----------|-----------|--------------|--------|
| 21 | 0.4381 | 0.4554 | 0.5630 | −0.1249 | −0.0173 | −0.1088 | 0.1019 |
| 22 | 0.5209 | 0.5192 | 0.5886 | −0.0676 | 0.0017 | 0.1733 | 0.0446 |
| 23 | 0.6267 | 0.5485 | 0.6507 | −0.0239 | 0.0782 | −0.0690 | 0.0457 |
| 24 | 0.6043 | 0.5370 | 0.5107 | 0.0936 | 0.0673 | 0.1814 | 0.0313 |
| 25 | 0.6299 | 0.6413 | 0.6495 | −0.0196 | −0.0114 | 0.1150 | 0.0453 |
| mean | 0.5640 | 0.5403 | 0.5925 | **−0.0285** | 0.0237 | 0.0584 | 0.0538 |
| SD | — | — | — | **0.0804** | 0.0454 | 0.1376 | 0.0276 |
| SE | — | — | — | **0.0359** | — | — | 0.0123 |

- Effect ± noise: mean gap_traj = **−0.0285 ± 0.0804 (SD)**,
  SE 0.0359. The SE band [−0.0644, +0.0074] sits entirely
  below the bar (0.2010) and mostly below zero. Arm3 fell MORE
  than arm1 on the window-based metric in 4 of 5 seeds.
- Secondary endpoint gap (descriptive only): mean 0.0584 ±
  0.1376 — still noisy, still disagreeing with the trajectory
  mean in sign pattern. One draw, as established; not graded.
- Pilot vs graded (frozen primary): pilot +0.0475 ± 0.1005
  (outlier-driven: seed 15 alone contributed +0.2243),
  graded −0.0285 ± 0.0804. The two multi-seed samples bracket
  zero. There is no window-based damping effect at this scale.

## (3) Leg verdicts vs the frozen bar

| leg | prediction | result | verdict |
|-----|-----------|--------|---------|
| P1 recovery-raw | each arm baseline − rec-1 ≤ 0.10 | arm1 −0.085, arm2 −0.056, arm3 −0.059 (recovery ABOVE baseline — extra training improves facts) | PASS |
| P2 resistance-raw (PRIMARY) | mean gap_traj ≥ 0.2010 AND mean gap2_traj ≤ 0.2010 | gap −0.0285 (fails), gap2 0.0237 (passes) | FAIL |
| P3 selection adds | every seed [7,0,7,7,7]; sel > scar raw; sel ≥ own raw | trajectory exact 5/5; mean sel true 7.0 > scar raw true 3.0; ≥ own raw true 2.0 | PASS |
| P4 deficit persists | every seed aff gain rec-2 = 0.6; fact auths all 0.6; behavior recovers | 0.6000 and all 0.6, 5/5; P1 arm3 passes | PASS |
| P5 retention | each arm mean \|count rec-2 − base\| ≤ 0.05 | worst single diff 0.0354 (seed 22 arm1); means ≤ 0.01 | PASS |
| P6 cost | every seed fwd ratio < 2; rows 4736; fact rows 28 | ratios 0.98–1.06; 4736/28, 5/5 | PASS |
| P7 scar trajectory | every seed [0.0,1.2,0.2,1.2,0.2] | exact, 5/5, both scar arms | PASS |
| P8 re-compromise | mean raw liar arm1 ≥ 5; arm2 ≥ 5; mean sel liar arm3 = 0 | arm1 3.4 (fails), arm2 4.0 (fails), arm3 sel liar 0.0 (passes) | FAIL |

OVERALL: **FAIL (MECHANISMS PASS, ADVANTAGE FAIL)** — the same
label as V2 and V3, now on a meter that resolves the question.

P8 note (meter interrogation, not a rewrite): with 7 facts the
liar wins a smaller share of conventional arms (3.4/7 arm1,
4.0/7 arm2) than the ≥3/4 rate at 4 facts. The ≥5 bar (scaled
from ≥3/4) fails on the liar-win side, not on arm3's side
(arm3 selected liar = 0 every seed — selection holds). The
re-compromise premise itself weakens at 7 facts; whether the
scaled bar or the premise needs revisiting is a meter question
for Darren, not a silent adjustment here.

## Alongside: fast-trace-baseline falls (Amendment A1, NOT grading)

- gap_ft per seed: 0.1019, 0.0446, 0.0457, 0.0313, 0.0453 —
  all five positive. Mean 0.0538, SD 0.0276, SE 0.0123.
  Informational 2×SD = 0.0551; mean 0.0538 MISSES it by 0.0013.
  (Pilot: mean 0.0833, SD 0.0321, would-be bar 0.0643 — cleared.
  Graded does not clear its own informational bar.)
- Decomposition (the pilot's caveat, confirmed): gap_ft =
  (base1 − base3) + (p2traj3 − p2traj1). Graded means:
  base-diff +0.0581, p2-diff −0.0044. The ENTIRE graded
  fast-trace gap is baseline offset; the genuine resistance
  component (arm3 holding higher P(true) DURING poison-2) is
  −0.0044 — slightly negative. Pilot p2-diff was +0.013.
  Across 10 seeds the resistance component is ≈ 0. The
  fast-trace metric stays ALONGSIDE: tight, but its gap is not
  damping evidence.

## Levels separately from falls (Amendment A2a) — the key finding

Mean levels across graded seeds (mean_p_true):

| arm | baseline | rec-1 window | rec-1 boundary | poison-2 traj | resist boundary |
|-----|----------|--------------|----------------|---------------|-----------------|
| arm1 | 0.8450 | 0.9335 | 0.9298 | 0.3695 | 0.4222 |
| arm2 | 0.8450 | 0.9170 | 0.9010 | 0.3767 | 0.4233 |
| arm3 | 0.7869 | 0.9576 | 0.8461 | 0.3651 | 0.3968 |

Decomposition of the negative gap: gap_traj = winDiff +
p2Diff = (−0.0241) + (−0.0044) = −0.0285. Arm3's recovery-1
window (0.9576) sits HIGHER than arm1's (0.9335), while
poison-2 levels are nearly equal (0.3651 vs 0.3695). **The
fall definition penalizes better recovery**: arm3 drops more
because it recovered higher, not because it resisted worse.
Resistance levels are equal; recovery favors arm3. The fall
alone says "arm3 worse"; the levels say "resistance equal,
recovery better." That is exactly the conflation Amendment A2a
was written to expose — and it is now measured, not theorized.

Consolidation shock (pilot finding, present in graded too):
arm3's recovery-1 boundary (0.8461) sits below its window
(0.9576) — the gain reset moves the reading. Boundary probes
after a consolidation are not comparable to pre-consolidation
trajectory levels.

## Fact-row vs whole-model (Amendment A2b)

- Fact-row weight movement during poison-2: 0.4777, 0.3748,
  0.4683, 0.3978, 0.4799 (mean ≈ 0.44) — same range as pilot
  (≈ 0.43), V2 (0.4552), V3 (0.4377). Fact rows moved
  substantially under uniform governance (14/14 latched every
  seed); whole-model poison-2 levels ≈ 0.37. No V3-style
  divergence: no undamped third exists under uniform crossing.
- Per-fact probe detail is in `results-3arm-v4.json`
  (phases → per_fact) for any fact-level follow-up.

## Injury-signal source (Amendment A2c — evaluation only)

Unchanged from the pilot: scheduled driver only; no
model-state signal designed, built, or tested (scope hold
stands). V4's uniform crossing has no quiet rows, so the
completeness question (is an unmoved row truly trusted?)
remains untestable on V4 data and open for Darren's
formulation.

## Internal consistency (graded, every seed)

Gradient gate PASS, init parity True, scar trajectory exact,
rows 4736/28, chains OK, latched 1184/1184, latched fact rows
14/14 — 5/5 seeds. Handoff events by phase recorded in JSON.

## V4 verdict, in one paragraph

The meter now resolves P2 (SE 0.0359 ≤ 0.0502 = 0.25 × bar),
and the resolved answer on the frozen primary is FAIL:
mean gap_traj −0.0285 ± 0.0804 against a frozen bar of 0.2010,
with mechanisms (P1, P4, P5, P6, P7) and selection (P3) all
passing and P8 failing on the liar-win side. The damping
claimed in V2's single-seed gap (+0.0437) does not survive
multi-seed trajectory measurement: pilot +0.0475 ± 0.1005,
graded −0.0285 ± 0.0804, resistance component ≈ 0 across
10 seeds. What the levels DO show is arm3 recovering higher
(window 0.9576 vs 0.9335) while resisting equally (0.3651 vs
0.3695) — a recovery advantage the fall definition reads
backwards. The next question is a meter question (what
reference level makes "resistance" mean resistance?) and a
mechanism question Darren owns (quiet rows, model-state
injury signal) — neither is answered by moving this bar.
