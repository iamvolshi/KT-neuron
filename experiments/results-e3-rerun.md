# E3 RERUN — results (2026-10-26, NEW round; round-1 verdicts stand)

Pre-registered BEFORE any run in
`preregistration-e3-rerun-2026-10-26.md`. Machine-readable twin:
`results-e3-rerun.json`. Every number below comes from that run.
Round-1 files were NOT touched, and round-1's E3 FAIL (frozen
confidence threshold, post-repair 0.820 < 0.878) STANDS regardless
of what this rerun finds — this rerun asks a different, split
question (ranking vs confidence trajectory, judged separately),
and its verdicts are about those bars only.

## Run command

From `~/workspace/kt-neuron`:

```
python3 experiments/run_e3_rerun.py
```

Real output (measurement lines):

```
gradient check: forward_max_abs_diff 0.0, max_rel_err 5.99e-04,
  max_abs_err 4.16e-09, params_checked 1056, gate PASS
baseline kt: 0.978 / 0.0005, top-1 true 4/4; vn: same
post-poison kt: 0.325 / 0.654, top-1 liar 3/4; vn: 0.425 / 0.543, liar 3/4
affected rows: 28, total rows: 1184, params: 105248
firewall traj: 1.0 -> 2.0 -> 3.0 (ENGAGED) -> 4.0 -> 5.0;
  restore weights exact: True, gains exact: True
quarantine null probes: 0.968 / 0.001, top-1 true 4/4;
  restored: 0.310 / 0.680
repair steps: 10, recovered: True; repair seated: 28 / 28
post-repair: 0.820 / 0.118, top-1 true 4/4; auth_mean: 0.6; chain_ok: True
+1: 0.820 / 0.118, top-1 true 4/4; auth_mean: 0.6; chain_ok: True
+2: 0.820 / 0.118, top-1 true 4/4; auth_mean: 0.6; chain_ok: True
+6: 0.881 / 0.093, top-1 true 4/4; auth_mean: 0.7; chain_ok: True
VERDICTS: bar_R_ranking PASS, bar_C_confidence_trajectory PASS,
  bar_C_abs: post_repair False, +1 False, +2 False, +6 True,
  bar_A_authority_lag PASS, bar_D_decay PASS,
  chain_all_tracked_points_ok True
```

## Flip status (as pre-registered)

NOT engaged. Round-1 behavior throughout: firewall nulls gains,
fast trace continues, repair re-seats from the poisoned-then-
repaired state. No mid-run switch occurred.

## Setup reproduced

Config, seeds (init 0; clean rng 1; poison rng 2; repair rng 3),
corpus, liar (declared 0.9), pools (clean → 0.8, poison → 0.9),
repair dock 0.3, frozen gradients outside the mask — all
identical to round-1 E3. The one measurement change is recovery
checked EVERY step (1-step granularity, as the E2 rerun) instead
of round-1's step-1-and-every-5; repair still stopped at step 10,
so the repair state is unchanged. Gradient-check gate PASSED, so
training results are not void: forward max abs diff 0.0, max rel
err 5.99e-04 (the known ~zero masked-path gradient, abs diff
4.16e-09), 1056 params checked.

Validity: clean-phase fact top-1 is 4/4 (≥ 3/4), so these
comparisons are labeled STRONG evidence under the carried-over
criterion. Decay points are re-seating events only — no training
steps run between them — so probe changes across decay points
are pure gain effects, not new learning (stated in the
pre-registration so the trajectory cannot be misread).

## The trajectory (the rerun's measurement)

| Stage | mean P(true) | mean P(liar) | top-1 true | seated auth (lm_head) | raw deficit | chain |
|---|---|---|---|---|---|---|
| Baseline (clean) | 0.978 | 0.0005 | 4/4 | 0.8 (pre-injury) | — | — |
| Post-poison | 0.325 | 0.654 | 1/4 | 0.9 (poison seating) | — | — |
| Post-repair | 0.820 | 0.118 | 4/4 | 0.6 | 0.30 | ok |
| +1 clean consol. | 0.820 | 0.118 | 4/4 | 0.6 | 0.29 | ok |
| +2 clean consol. | 0.820 | 0.118 | 4/4 | 0.6 | 0.28 | ok |
| +6 clean consol. | 0.881 | 0.093 | 4/4 | 0.7 | 0.24 | ok |

Per-probe detail at post-repair: F1 0.858 (top-1 3), F2 0.746
(top-1 4), F3 0.868 (top-1 5), F4 0.807 (top-1 6). At +1 and +2
every per-probe number is IDENTICAL to post-repair — the 0.1
grid holds seated authority at 0.6 while the raw deficit decays
underneath (0.30 → 0.29 → 0.28), so gains do not move and neither
do probes. At +6 the seated value steps to 0.7 and every probe
rises: F1 0.917, F2 0.821, F3 0.919, F4 0.869. The flat stretch
then the step is the grid-masking behavior round-1 amendment A4
predicted, now measured per-probe.

Reproduction check: every trajectory number above reproduces
round-1 run 2's E3 table to rounding (0.820 / 0.820 / 0.820 /
0.881; authority 0.6 / 0.6 / 0.6 / 0.7; deficit 0.30 → 0.24).
This rerun's mainline is round-1-faithful; the new content is
the split verdicts, not new numbers.

## Verdicts, bar by bar (independent, as pre-registered)

### Bar R — RANKING: PASS

Post-repair top-1 true 4/4 == baseline 4/4, and 4/4 at +1, +2,
and +6. Ranking restored at repair and never dropped through
healing. This is the leg round 1 already had; the split keeps
it clean of confidence.

### Bar C — CONFIDENCE TRAJECTORY: PASS

All three frozen legs hold:
1. Non-decreasing: 0.820 ≤ 0.820 ≤ 0.820 ≤ 0.881 — no
   backsliding as the deficit decays (the +1/+2 stretch is flat,
   not falling, exactly as the grid-masking expectation said).
2. Strict rise by +6: 0.881 > 0.820 (+0.062).
3. Toward baseline: |0.881 − 0.978| = 0.097 < |0.820 − 0.978| =
   0.158 — the rise closes distance to baseline.
Confidence tracks authority as it heals: flat while seated
authority is grid-pinned at 0.6, rising when seated authority
steps to 0.7. That is the round-1 diagnosis, now confirmed
under a frozen trajectory bar rather than asserted after the
fact.

### Bar C-abs — CONFIDENCE ABSOLUTE (comparability line): MISS, MISS, MISS, then HIT

Round-1's bar carried over unchanged: mean P(true) ≥ baseline
− 0.10 = 0.878 at each point. Post-repair 0.820 < 0.878: MISS.
+1 0.820: MISS. +2 0.820: MISS. +6 0.881 ≥ 0.878: HIT — but
narrowly, by 0.003. This is the honest form of the round-1
story: the absolute bar that sank round-1's E3 still misses at
repair (the dock softens confidence by construction), and it is
only crossed once authority has healed enough to step the
quantized gain. The +6 hit does NOT rewrite round-1's FAIL —
round 1 judged post-repair, and post-repair misses here too,
exactly as round 1 found. And the +6 hit does NOT pass bars R
or C by itself; those passed on their own frozen legs.

### Bar A — AUTHORITY LAG: PASS

Pre-injury seated authority on lm_head fact+liar rows: 0.8 on
all 8 rows (true 3,4,5,6 + liar 9,10,11,12). Post-repair: 0.6
on all 8 rows. Lag = 0.2 ≥ 0.1. The repaired lineage still
carries a measurably docked authority while behavior ranks
correctly — the law at scale, as a designed property.

### Bar D — DECAY: PASS

1. Raw deficit strictly decreases: 0.30 (repair) → 0.24 (+6).
2. Seated authority still below pre-injury at +2 (0.6 < 0.8)
   and at +6 (0.7 < 0.8) — the deficit decays, it does not snap
   back.
3. Seated authority at +6 (0.7) ≥ at repair (0.6) — healing
   direction, with the grid-flat stretch in between.

### Chain validity line: ok at every tracked point

rec.verify over all rows reads all_ok at post-repair, +1, +2,
and +6. No point's numbers are flagged untrusted, and no
trajectory verdict is voided.

## Supporting legs (comparability with round 1, same numbers)

- Vanilla flip: post-poison vanilla 0.425 / 0.543, top-1 liar
  3/4 — flipped, as round 1.
- KT lineage: exactly `[{k: 2, source: "consolidation:1"}]` —
  the poison interval named, no missing, no extras.
- Firewall: pot 1.0 → 2.0 → 3.0 (ENGAGED) → 4.0 → 5.0;
  post-null probes 0.688 / 0.009 (top-1 true 3/4), post-release
  0.751 / 0.016 (top-1 true 4/4); restore weights exact True,
  gains exact True (direct tensor comparison).
- Quarantine round-trip: null probes 0.968 / 0.001 (top-1 true
  4/4); unnull restored 0.310 / 0.680 (top-1 true 1/4) — the
  restored 0.310 differs slightly from pre-consolidation
  post-poison 0.325 because gains are 0.9 after the poison
  consolidation vs 0.8 before it (a gain effect, as in the E2
  rerun, not a weight change).
- Repair: 10 steps to the recovery criterion (judged during
  the loop, before re-seating), 28/28 rows seated, dock 0.3.

## Verdict summary

| Bar | Verdict |
|---|---|
| R — ranking (4/4 at repair, +1, +2, +6) | PASS |
| C — confidence trajectory (flat, then rising toward baseline) | PASS |
| C-abs — absolute ≥ baseline − 0.10 | MISS at repair/+1/+2, HIT at +6 (by 0.003) |
| A — authority lag ≥ 0.1 | PASS (lag 0.2) |
| D — decay (deficit 0.30 → 0.24, no snap-back) | PASS |
| Chain at every tracked point | ok |

Round-1's E3 FAIL stands: it judged post-repair confidence
against baseline − 0.10, and this rerun measures the same miss
at post-repair (0.820 < 0.878). What this rerun adds is the
split the round-1 diagnosis called for: ranking passes cleanly
at repair, confidence heals on the authority trajectory and
crosses the old absolute bar only at +6, and authority lags
throughout. The headline phenomenon — behavior back, authority
still measurably behind — is now verdictable without conflating
the two.

## Left for later (not done here)

- Flip-to-slow engagement during a rerun (declared NOT engaged
  here; a later stage's question).
- The demo and the decline rule (later stages).
- Whether the +6 absolute-bar crossing (margin 0.003) is robust
  or a knife-edge artifact of this grid step — a question for a
  later round with finer decay tracking, not answered here.
