# V4 pilot report — 2026-10-28

Frozen contract: `experiments/preregistration-3arm-v4-2026-10-28.md`
(including Amendments A1, A2). Pilot seeds {11, 12, 13, 14, 15},
full 3-arm V2-handoff runs under the new meter. No graded run
has started; the bar below was frozen from this pilot BEFORE
any graded run.

## Internal consistency (every seed)

- Gradient gate PASS, init parity bit-identical: 5/5 seeds.
- Scar trajectory [0.0, 1.2, 0.2, 1.2, 0.2]: 5/5 seeds.
- Row-record rows 4736, fact-record rows 28, chains verify OK:
  5/5 seeds.
- Handoff: 1184/1184 rows latched every seed; latched fact rows
  14/14 every seed. (Latch-step distribution not re-checked
  here; uniform crossing, same as V2.)
- Selected trajectory [7, 0, 7, 7, 7]: 5/5 seeds (P3 shape holds
  under 7 facts).

## Frozen primary: trajectory-averaged falls (window-based)

fall_traj(arm) = recovery-1 window mean (steps 150–200) −
poison-2 trajectory mean (steps 10–200).

| seed | fall arm1 | fall arm2 | fall arm3 | gap_traj | gap2_traj | gap_endpoint | gap_ft |
|------|-----------|-----------|-----------|----------|-----------|--------------|--------|
| 11 | 0.6168 | — | 0.6242 | −0.0074 | −0.0071 | 0.1828 | 0.0376 |
| 12 | 0.5715 | — | 0.5705 | 0.0011 | −0.0220 | 0.2202 | 0.0955 |
| 13 | 0.5930 | — | 0.5600 | 0.0330 | −0.0265 | 0.5120 | 0.0919 |
| 14 | 0.5757 | — | 0.5894 | −0.0137 | −0.0551 | −0.1208 | 0.1232 |
| 15 | 0.5825 | — | 0.3582 | 0.2243 | 0.0787 | 0.3360 | 0.0681 |
| mean | 0.5879 | 0.5943 | 0.5405 | **0.0475** | −0.0064 | 0.2260 | 0.0833 |
| SD (ddof=1) | — | — | — | **0.1005** | 0.0506 | 0.2324 | 0.0321 |

(gap2_traj = arm1 − arm2; arm2 falls: 0.6239, 0.5935, 0.6195,
0.6308, 0.5038 — mean 0.5943. Full per-seed falls are in
`results-3arm-v4-pilot.json`.)

- M0: pilot gap_traj SD 0.1005 < 0.1456 (old endpoint meter SD)
  — PASS. Trajectory averaging reduced noise ~31% on the
  primary, and the endpoint meter run alongside is far noisier
  (SD 0.2324) — the endpoint is one draw, as the saturation
  study warned.
- **FROZEN BAR = 2 × 0.1004797757627805 =
  0.200959551525561 (≈ 0.2010).** Pilot mean gap_traj (0.0475)
  sits well BELOW the bar. That is the discipline working: the
  meter's noise, measured honestly, sets a bar the pilot effect
  does not clear. Graded P2 on the frozen primary is therefore
  expected to fail unless the graded mean lands far above the
  pilot mean. No bar-moving, for any reason.

## Alongside: fast-trace-baseline falls (Amendment A1)

fall_ft(arm) = clean-baseline boundary mean_p_true (fast trace
at rest, F = 0) − poison-2 trajectory mean.

- gap_ft per seed: 0.0376, 0.0955, 0.0919, 0.1232, 0.0681 —
  ALL FIVE POSITIVE. Mean 0.0833, SD 0.0321, SE 0.0144.
  Informational 2×SD = 0.0643 (NOT frozen, NOT grading).
- The fast-trace reference removes most of the seed noise
  (SD 0.032 vs 0.100). WHY, decomposed per seed:
  - The window reference injects recovery-path variance:
    arm3's recovery-1 window is 0.9845, 0.9714, 0.9501, 0.9834,
    then 0.7407 on seed 15 (a recovery-path event: arm3's
    restore-1 ended low that seed). That single collapse is
    the whole window-based outlier (+0.2243).
  - The baseline reference is stable because clean training is
    the same story every seed.
- HONEST CAVEAT (read before citing gap_ft): the fast-trace
  gap is mostly a resting-level offset, NOT demonstrated
  damping. Decomposition: gap_ft = (base1 − base3) +
  (p2traj3 − p2traj1). base1 − base3 per seed: 0.0404, 0.0803,
  0.0578, 0.0925, 0.0783 (mean 0.070, always positive — arm3's
  baseline sits lower because KT gains < 1 scale its
  activations). p2traj3 − p2traj1 per seed: −0.0027, +0.0154,
  +0.0341, +0.0307, −0.0102 (mean +0.013, mixed sign). The
  genuine resistance component — arm3 holding higher P(true)
  DURING poison-2 — is +0.013, small. The window comparison
  is the fairer damping comparison (equalized training: both
  arms' windows ≈ 0.96 excluding seed 15), and it shows ≈ 0
  gap excluding the seed-15 recovery event. Promoting the
  fast-trace metric to primary on tightness alone would flatter
  arm3 with a baseline offset. It stays ALONGSIDE.

## Levels separately from falls (Amendment A2a)

Mean levels across pilot seeds (mean_p_true):

| arm | baseline | rec-1 window | rec-1 boundary | poison-2 traj | resist boundary |
|-----|----------|--------------|----------------|---------------|-----------------|
| arm1 | 0.9174 | 0.9601 | 0.9470 | 0.3721 | 0.3594 |
| arm2 | 0.9174 | 0.9543 | 0.9469 | 0.3600 | 0.3603 |
| arm3 | 0.8476 | 0.9260 | 0.7291 | 0.3856 | 0.3675 |

Two measurement artifacts the levels expose (not mechanism
findings):

1. CONSOLIDATION SHOCK: arm3's recovery-1 boundary sits far
   below its window (mean 0.7291 vs 0.9260; worst seed 13:
   0.4519 vs 0.9501). The restore-1 consolidation resets gains
   (affected 0.6, unaffected 0.9) after the model adapted to
   0.8 gains — the boundary probe measures the model AFTER a
   gain intervention, not the recovered model. V1–V3 endpoint
   falls mixed this shock into the "recovered level." The
   window avoids the shock; the baseline avoids it too.
2. RECOVERY-PATH VARIANCE: arm3's window collapses on seed 15
   (0.7407 vs 0.95–0.98 elsewhere). A window is a still-moving
   trajectory, not a settled level; one seed's slow recovery
   moves the whole gap.

## Fact-row vs whole-model (Amendment A2b)

- Fact-row weight movement during poison-2 (mean over 14 fact
  rows): 0.3897, 0.4794, 0.4299, 0.4388, 0.4118 (mean ≈ 0.43) —
  in line with V2 (0.4552) and V3 (0.4377). Fact rows moved
  substantially; they were governed (14/14 latched), not
  frozen. Whole-model P(true) falls (≈ 0.54–0.59) exceed
  fact-row movement in scale, but the two are different units
  (behavior vs weight distance) — reported side by side, never
  merged.
- No divergence of the V3 kind appears in the pilot: under
  uniform crossing every row is governed, so there is no
  undamped third to drown the fact rows. The V3 divergence
  remains a per-row-crossing phenomenon, not a V4 one.

## Injury-signal source (Amendment A2c — evaluation only)

- The scheduled driver envelope remains the injury input; no
  model-state signal was designed, built, or tested (scope
  hold stands).
- What V4 data can and cannot say: uniform crossing governs
  every row, so V4 has no quiet rows — the completeness
  question (is an unmoved row truly trusted?) cannot be tested
  on V4 runs, which contain no unmoved-but-possibly-injured
  rows to examine. The question stays open for Darren's
  formulation: a model-state signal would have to distinguish
  uninjured vs unexercised vs injured-without-moving, and
  "no movement" alone cannot do that.

## Pilot verdict

Meter: noise measured (SD 0.1005 primary, 0.0321 fast-trace
alongside); M0 PASS; bar frozen at 0.2010. Effect: pilot mean
gap_traj 0.0475 ± 0.1005 — inside noise, outlier-driven, and
below the bar. The pilot does not resolve P2; it sets the bar
the graded runs must clear. Graded seeds {21–25} run next under
the frozen bar.
