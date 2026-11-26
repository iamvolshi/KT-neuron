# Saturation check on the V2 3-arm KTN experiment — results

**Question:** does the 200-step poison phase saturate the flip and
compress the measurable P2 gap (recovery-1 → resistance fall, arm 1
minus arm 3)?

**Short answer: the flip does saturate early — but not as a smooth
plateau at a floor, and it does NOT produce a monotone compression
curve. The 200-step gap (0.0437) is a noisy endpoint draw, and the
noise (±0.05 or more) dwarfs the 0.0063 shortfall against the frozen
0.05 bar. Saturation is real in the mean; it cannot, at this bench
scale, explain the miss as a clean length effect.**

- This check does NOT re-grade P2 and moves no bar. V2's verdict
  stands: FAIL (MECHANISMS PASS, ADVANTAGE FAIL), P2 short by 0.0063.
  What follows is a measurement of how the gap varies with phase
  length, nothing more.
- Frozen files untouched: `preregistration-3arm-v2-2026-10-28.md`,
  `run_3arm_v2.py`, `results-3arm-v2.json/.md` (and all v1 files).
  New files only: this report, `results-3arm-v2-saturation.json`,
  `saturation-check-3arm-v2.py`. No source module was edited; the
  driver-rate patch for the peak-matched variant is runtime-only
  and restored after each run.
- Bench scale only, same as V2: small GPT (105,248 params, 1,184
  KIDs/arm), clean 400, restores 200, same seeds and batch lists.
  Gradient gate PASS (forward max abs diff 0.0, max rel err
  0.000599, 1,056 params checked) and init parity bit-identical
  across arms in all 8 configs. Full suite green at the end:
  148 tests OK (the task brief's "131 baseline" predates the
  current tree; nothing in this check added or changed tests).

## Method (two variants — the driver rate is itself a confound)

Poison batches for length N are the FIRST N of the seed-2 list
(V2 uses the same list for poison 1 and poison 2), so shorter runs
are strict prefixes of the baseline, not new draws. Clean and
restores are unchanged (400 / 200 / 200). Arms 1 and 3 in every
config; arm 2 only in the 200-step validation config (it is not
needed for the P2 gap).

- **Variant A, "frozen-driver":** driver rates unchanged (poison
  0.75/200 per step, restore 0.375/200). This is the literal "keep
  everything else identical" sweep — but shorter phases then reach
  a LOWER peak pressure/scar and may never cross the handoff gate.
  At N=50/100 the handoff never latches at all (see table), so
  those gaps are NOT handoff measurements; they are reported
  precisely so they are not misread as such.
- **Variant B, "peak-matched":** poison rate scaled to 0.75/N per
  step, so every length reaches the same peak pressure 0.75
  (scar 1.2) and the same restore-1 end state (pressure 0.375,
  scar 0.2) as the 200-step baseline. Only the number of training
  steps varies. This is the clean handoff comparison.
- **Trajectory (part b):** forward-only probes at poison-2 steps
  0, 10, 20, …, N (poison 1 every 25 steps). Probes cannot change
  weights, so the (A, 200) config must — and does — reproduce V2.

## Validation: the (A, 200) replication is exact

| phase | arm 1 mean P(true) | arm 3 mean P(true) | V2 match |
| --- | ---: | ---: | --- |
| baseline | 0.9780 | 0.9445 | exact |
| injury | 0.4246 | 0.3160 | exact |
| recovery 1 | 0.9925 | 0.9067 | exact |
| resistance | 0.3626 | 0.3206 | exact |
| recovery 2 | 0.9927 | 0.9894 | exact |

Falls: arm 1 0.6298, arm 2 0.7224, arm 3 0.5861; gap 0.0437;
latched 1,184; affected rows 28 / 28 — all identical to V2.
Peak-matched 200 is likewise identical (rate 0.75/200 = the
original), as it must be. Arm-1 falls are identical across
variants at each N (arm 1 has no driver), an internal-consistency
check that passed everywhere.

## (a) Gap vs poison-phase length (endpoint falls, P2 definition)

Peak-matched (the clean comparison — same scar peak, latch active
at every length):

| poison N | arm-1 fall | arm-3 fall | P2 gap (arm1 − arm3) | scar p1 / p2 | latched |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 0.5951 | 0.5193 | **+0.0758** | 1.2 / 1.2 | 1,184 |
| 100 | 0.4566 | 0.4684 | **−0.0118** | 1.2 / 1.2 | 1,184 |
| 150 | 0.3928 | 0.2976 | **+0.0953** | 1.2 / 1.2 | 1,184 |
| 200 (V2) | 0.6298 | 0.5861 | **+0.0437** | 1.2 / 1.2 | 1,184 |

Frozen-driver (literal sweep — driver state varies with N):

| poison N | arm-1 fall | arm-3 fall | P2 gap | scar p1 / p2 | latched |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 0.5951 | 0.3838 | +0.2112 | 0.0 / 0.0 | 0 |
| 100 | 0.4566 | 0.4643 | −0.0076 | 0.2 / 0.2 | 0 |
| 150 | 0.3928 | 0.4311 | −0.0383 | 0.7 / 1.2 | 1,184 |
| 200 (V2) | 0.6298 | 0.5861 | +0.0437 | 1.2 / 1.2 | 1,184 |

There is NO monotone trend in either variant. Peak-matched gaps
run +0.076, −0.012, +0.095, +0.044 — mean 0.051, spread (SD) ≈ 0.047,
range 0.107. Two shorter lengths clear the 0.05 bar (50, 150) and
one fails it outright (100). Length alone does not determine
pass/fail at this bench scale. The frozen-driver N=50 gap (+0.2112)
must NOT be quoted as a handoff effect: with scar 0.0 and zero
latched rows there is no handoff in that run at all — it is
gain/scar architecture plus endpoint luck.

What the endpoints DO show is that neither arm's fall deepens
systematically with length: arm-1 endpoint falls across N are
0.595, 0.457, 0.393, 0.630 (non-monotone, spread 0.24), and arm-3
peak-matched falls are 0.519, 0.468, 0.298, 0.586 (same story).
A quantity that bounces by ±0.1 between adjacent lengths cannot
resolve a 0.0063 effect. That is the measurement's own noise
floor, and the 200-step shortfall sits well inside it.

## (b) Plateau evidence: the flip enters a noisy regime early and stays there

The 200-step poison-2 trajectory (mean P(true), every 10 steps)
does not look like a curve approaching a floor. It looks like a
system that flips fast and then bounces:

| step | arm 1 | arm 1 top1 | arm 3 | arm 3 top1 |
| ---: | ---: | ---: | ---: | ---: |
| 0 (recovery-1) | 0.9925 | 4/0 | 0.9067 | 4/0 |
| 10 | 0.1639 | 0/3 | 0.4900 | 3/1 |
| 20 | 0.5704 | 3/1 | 0.4610 | 2/2 |
| 50 | 0.4202 | 2/2 | 0.4250 | 2/2 |
| 100 | 0.6005 | 3/1 | 0.5691 | 3/1 |
| 150 | 0.5505 | 2/2 | 0.6562 | 4/0 |
| 200 (resistance) | 0.3626 | 1/3 | 0.3206 | 1/3 |

Full trajectory statistics, poison 2 at N=200 (steps > 0, n=20
checkpoints): arm 1 mean 0.4192, SD 0.109, range 0.164–0.601,
liar-majority (top-1 liar ≥ 3) at 50% of checkpoints; arm 3 mean
0.4859, SD 0.103, range 0.293–0.688, liar-majority at 25%.
First liar-majority checkpoint: arm 1 at step 10, arm 3 at step 70
(arm 3 is visibly damped early — 0.4900, still true-majority, at
step 10 while arm 1 has already collapsed to 0.1639).

Plateau test, first half vs second half of poison 2 (N=200):
arm 1 mean 0.4291 → 0.4093 (Δ −0.020); arm 3 mean 0.4878 → 0.4841
(Δ −0.004); liar-majority fraction arm 1 0.50 → 0.50, arm 3
0.20 → 0.30. The second half is statistically indistinguishable
from the first: no deepening trend across the phase. Poison 1
shows the same bounce (arm-1 checkpoints 0.28–0.54 with no trend
to a floor).

Saturation across lengths, in trajectory means (poison-2 mean
P(true) over steps > 0 — a far less noisy summary than the single
endpoint):

| N | arm-1 traj mean | arm-3 traj mean, peak-matched | arm-3 traj mean, frozen |
| ---: | ---: | ---: | ---: |
| 50 | 0.4157 | 0.5044 | 0.5367 |
| 100 | 0.5067 | 0.5107 | 0.4248 |
| 150 | 0.4763 | 0.4946 | 0.4551 |
| 200 | 0.4192 | 0.4859 | 0.4859 |

Arm-3's peak-matched trajectory mean moves only 0.511 → 0.486
across 100 → 200 steps (full range 0.486–0.511 over all four
lengths): the average flipped level is flat in length. Arm 1's
mean likewise wanders 0.42–0.51 with no downward trend. In the
mean, the flip is fully established by 50 steps; the remaining
150 steps resample the same noisy regime rather than deepening it.

Trajectory-mean fall gaps (recovery-1 minus poison-2 trajectory
mean, arm 1 minus arm 3 — same construction as P2 but with the
trajectory mean in place of the endpoint) are positive at every
peak-matched length: 0.1902 (50), 0.0411 (100), 0.1433 (150),
0.1525 (200). Damping is present on average at all lengths
including 200. The endpoint gap disagrees with the trajectory
gap at 200 precisely because the endpoint is one draw: arm 3's
step-200 value (0.3206) sits 0.165 below its own trajectory mean
(0.4859) — about 1.6 checkpoint-SDs — while arm 1's endpoint
(0.3626) sits 0.057 below its mean (0.4192).

Decomposition of the 200-step endpoint gap, for clarity:
gap = (recovery-1 arm 1 − recovery-1 arm 3) + (resistance arm 3
− resistance arm 1) = 0.0858 + (−0.0420) = 0.0437. The recovery-1
advantage (+0.0858, arm 3 recovers less because restore trains
damped) is real; the endpoint resistance term (−0.0420, arm 3
caught lower than arm 1 at the final draw) erases half of it.
On trajectory means the same decomposition is 0.0858 + 0.0667
= 0.1525 — arm 3 sits HIGHER on average during poison 2, as the
damping predicts. The endpoint and the average tell different
stories because the endpoint is noisy, not because the damping
vanishes at 200 steps.

## Conclusion — does saturation compress the gap, and by how much?

1. **Yes, the flip saturates — early, and in the mean.** The
   flipped regime is entered within ~10–40 steps (arm 1 at 0.164
   by poison-2 step 10 at N=200) and the average flipped level
   does not deepen with length (arm-3 peak-matched trajectory
   mean 0.49–0.51 at every length from 50 to 200; first-half vs
   second-half means within 0.02 at N=200). There is no smooth
   plateau at a floor to point at — the "plateau" is a noisy
   steady regime (per-checkpoint SD ≈ 0.11 on a 4-fact probe),
   and 200 steps buy nothing the first 50 did not already buy,
   on average.
2. **No, saturation does not compress the gap by a measurable,
   monotone amount at this bench scale.** Peak-matched endpoint
   gaps by length are +0.076, −0.012, +0.095, +0.044. If
   saturation compressed the gap, shorter lengths would show
   systematically larger gaps; they do not — they show the same
   ±0.05 scatter. The honest magnitude statement is a noise
   statement, not a compression statement: endpoint-gap scatter
   across lengths (SD ≈ 0.047, range 0.107) is roughly 7× the
   0.0063 shortfall, so the 200-step miss is consistent with
   endpoint noise alone. Any true compression effect, if present,
   is smaller than the noise floor of this measurement and would
   need a less noisy meter (more probe facts, trajectory-averaged
   falls, and/or multiple seeds) to resolve — a new experiment,
   not a reinterpretation of V2.
3. **Do not use the frozen-driver short lengths as evidence
   either way.** At N=50/100 the driver never reaches the gate
   (scar 0.0/0.2, zero latched rows), so those runs contain no
   handoff to measure; their gaps (+0.2112, −0.0076) reflect
   gain/scar architecture and endpoint luck, not damping.

## Files

- `experiments/saturation-check-3arm-v2.py` — this check's runner
  (imports V2 code paths; parameterizes poison length + driver
  variant; trajectory probes; merge-safe results writing)
- `experiments/results-3arm-v2-saturation.json` — all 8 configs:
  phase summaries, both trajectories, falls, gaps, scar/latch
  state, affected-row counts
- `experiments/results-3arm-v2-saturation.md` — this report

## What was NOT done (gaps, stated plainly)

- No multi-seed replication: every config uses the single V2 seed
  set, so the scatter quoted above is across lengths, not an
  across-seed error bar. A seed sweep would be the direct way to
  put an error bar on the 200-step gap itself.
- No per-config handoff event list is stored in the saturation
  JSON (only totals: 1,184 latched wherever the gate is reached);
  the step-159 poison-1 crossing is V2's recorded event, and
  peak-matched crossing steps follow from the scaled-rate
  arithmetic (F reaches 0.8 at 80% of poison 1) rather than a
  per-config event log.
- Mock-RK selection omitted by design (read-only; cannot affect
  raw falls). No claim here touches selection.
