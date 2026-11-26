# R3-E2 — results: tail manipulation controls healing (2026-10-26)

Frozen contract: `experiments/preregistration-round3-2026-10-26.md`, R3-E2 section. Every number below comes from this run; the .md is generated from the same dict as `results-r3-e2.json`, not re-measured.

## Run command

From `~/workspace/kt-neuron`:

```
python3 experiments/run_r3_e2.py
```

Gradient gate: PASS (forward max abs diff 0.0, max rel err 5.99e-04, max abs err 4.16e-09, 1056 params checked). No new training code path exists: all training reuses `run_all.train_steps`; `kt_bench/tail.py` is pure arithmetic.

## Setup reproduced (per tail, identical seeds)

- Bench: vocab 32, block 16, n_embd 64, 4 heads, 2 layers, 105,248 params, 1,184 KIDs; seeds init 0 / data 1. Clean 400 steps, poison 200, repair dock 0.3.
- Clean baseline (KT): mean P(true) 0.9780, top-1 true 4/4.
- Post-poison (KT): mean P(true) 0.3254, P(liar) 0.6543, top-1 liar 3/4.
- Pre-injury lm_head authority (clean pool): [0.8].
- Repair: 10 steps to recovery (1-step checks, cap 200), recovered True; post-repair mean P(true) 0.8195, top-1 true 4/4, lm_head authority all [0.6].
- Chain verify at repair: True; at +30 (all tails): True.
- Pre-decay trajectories identical across tails: {'SLOW': {'baseline_mean_p_true_equal': True, 'post_poison_mean_p_true_equal': True, 'affected_rows_equal': True, 'repair_steps_equal': True, 'post_repair_mean_p_true_equal': True, 'lineage_equal': True}, 'PERMANENT': {'baseline_mean_p_true_equal': True, 'post_poison_mean_p_true_equal': True, 'affected_rows_equal': True, 'repair_steps_equal': True, 'post_repair_mean_p_true_equal': True, 'lineage_equal': True}}.

## Frozen verdicts (a)-(f)

- FAST (decay 0.05/consolidation): deficit at +6 = 0.0000 (authority mean 0.900), at +30 = 0.0000 (authority mean 0.900), time-to-healed 5.
- SLOW (decay 0.01/consolidation): deficit at +6 = 0.2400 (authority mean 0.700), at +30 = 0.0000 (authority mean 0.900), time-to-healed 25.
- PERMANENT (decay 0.0/consolidation): deficit at +6 = 0.3000 (authority mean 0.600), at +30 = 0.3000 (authority mean 0.600), time-to-healed None.
- (a) FAST healed by +6: PASS (deficit at +6 = 0.0000 <= 0.05; healed already at +5).
- (b) SLOW not healed by +6: PASS (deficit at +6 = 0.2400, within 0.24 +/- 0.005, reproducing round-1/2).
- (c) PERMANENT unchanged at +6: PASS (deficit at +6 = 0.3000, within 0.30 +/- 0.005).
- (d) Ordering FAST < SLOW < PERMANENT-never: PASS (times-to-healed: {'FAST': 5, 'SLOW': 25, 'PERMANENT': None}).
- (e) Behavior ranking 4/4 at repair and every tracked consolidation (+1..+30, all tails): PASS.
- (f) Lineage exactly ```[{k: 2, source: "consolidation:1"}]``` in all tails: True; repair seats all affected rows: True (28/28 per tail). Combined: PASS.
- **OVERALL: PASS**

## Re-measurement leg (i): confidence trajectory — record only, NO verdict

The absolute confidence bar at repair time is retired; only the trajectory is reported, and no verdict is drawn from these values. Deficit is the dock parameter from `kt_bench.tail` (exact); standing authority is q10(0.9 - deficit) on the 0.1 grid, reported alongside, never substituted for deficit (e.g. SLOW at +6: deficit 0.24, authority mean 0.7).

- FAST mean top-1 confidence, repair then +1..+30: 0.8195, 0.8195, 0.8813, 0.9181, 0.9181, 0.9181, 0.9415, 0.9415, 0.9415, 0.9415, 0.9415, 0.9415, 0.9415, 0.9415, 0.9415, 0.9415, 0.9415, 0.9415, 0.9415, 0.9415, 0.9415, 0.9415, 0.9415, 0.9415, 0.9415, 0.9415, 0.9415, 0.9415, 0.9415, 0.9415, 0.9415
  Non-decreasing observed: True.
- SLOW mean top-1 confidence, repair then +1..+30: 0.8195, 0.8195, 0.8195, 0.8195, 0.8195, 0.8195, 0.8813, 0.8813, 0.8813, 0.8813, 0.8813, 0.8813, 0.8813, 0.8813, 0.8813, 0.9181, 0.9181, 0.9181, 0.9181, 0.9181, 0.9181, 0.9181, 0.9181, 0.9181, 0.9181, 0.9181, 0.9415, 0.9415, 0.9415, 0.9415, 0.9415
  Non-decreasing observed: True.
- PERMANENT mean top-1 confidence, repair then +1..+30: 0.8195, 0.8195, 0.8195, 0.8195, 0.8195, 0.8195, 0.8195, 0.8195, 0.8195, 0.8195, 0.8195, 0.8195, 0.8195, 0.8195, 0.8195, 0.8195, 0.8195, 0.8195, 0.8195, 0.8195, 0.8195, 0.8195, 0.8195, 0.8195, 0.8195, 0.8195, 0.8195, 0.8195, 0.8195, 0.8195, 0.8195
  Non-decreasing observed: True.

## Re-measurement leg (ii): repair cost as work touched (FAST tail run)

- Surgical repair: 10 steps, 28 rows by design, 28 rows moved, 5276 params/step (mask ones), 52760 total param updates; leak outside mask: [].
- Full retrain (vanilla): 6 steps, 105248 params/step, 631488 total param updates.
- Ratio repair/retrain = 0.0835; replication bar ratio < 0.10: MET. NOT FLOPs, NOT steps: optimizer writes only, as pre-registered (this harness computes full gradients then masks).

## Notes / left for later

- Restored side-branch in rounds 1-2, not part of any R3-E2 frozen threshold; round 2 verified its restore bit-exact, so skipping cannot change the post-poison state.
- Repair-cost leg run on the FAST tail only; other tails share the identical pre-decay trajectory by determinism (see pre_decay_trajectories_identical).
- Decay consolidations are seating-only and cheap; tracking ran to +30 for every tail, past the +6 frozen minimum, to observe SLOW healing at +25 and confirm PERMANENT never heals.

