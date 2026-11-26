# R3-E3 results — tick mutability (plan §2)

Frozen contract: `experiments/preregistration-round3-2026-10-26.md`, R3-E3 section. Thresholds (a)–(d) evaluated, never moved.
Baselines are PER VARIANT. No cross-variant absolute probe comparison: the variants train different amounts, so only SHAPE is compared.

Gradient gate: PASS (forward_max_abs_diff=0.0, max_rel_err=0.000599123365147806, params_checked=1056).
Decay: SLOW tail, 0.01 per decay consolidation, all variants.

## Verdicts

- a_authority_lag_ge_0.1_all_variants: {'FINE': True, 'BASELINE': True, 'COARSE': True}
- b_deficit_nonincreasing_all_variants: {'FINE': True, 'BASELINE': True, 'COARSE': True}
- c_top1_4of4_post_repair_all_variants: {'FINE': True, 'BASELINE': True, 'COARSE': True}
- d_tick_count_to_healed: {'FINE': 25, 'BASELINE': 25, 'COARSE': 25}
- d_spread_max_minus_min: 0
- d_expected_ticks: 25
- d_identical_within_plus_minus_1_tick: True
- d_wiring_check_note: Decay priced per decay consolidation, never per training step; see run docstring and results .md.
- overall: PASS

## Per-variant results

### FINE — 400 clean steps, tick 100 steps, 4 tick(s)
- clean baseline (pre-final-consolidation): mean_p_true=0.9896, top1_true=4/4
- pre-injury lm_head authority mean: 0.800 (per row: {3: 0.8, 4: 0.8, 5: 0.8, 6: 0.8, 9: 0.8, 10: 0.8, 11: 0.8, 12: 0.8})
- post-poison: mean_p_true=0.2770, poison bench_k=5, lineage liar entries=[{'k': 5, 'source': 'consolidation:poison:FINE'}]
- affected rows: 28, repair steps: 13 (recovered: True), seated 28/28
- post-repair: mean_p_true=0.8407, top1_true=4/4, auth mean=0.600, lag=0.200
- chain OK at repair: True, at +30: True
- deficit series (n=0..30): [0.3, 0.29, 0.28, 0.27, 0.26, 0.25, 0.24, 0.23, 0.22, 0.21, 0.2, 0.19, 0.18, 0.17, 0.16, 0.15, 0.14, 0.13, 0.12, 0.11, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.0]
- time to healed (applied): 25, (tail arithmetic): 25
- top1_true across decay trajectory: [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]

### BASELINE — 400 clean steps, tick 400 steps, 1 tick(s)
- clean baseline (pre-final-consolidation): mean_p_true=0.9780, top1_true=4/4
- pre-injury lm_head authority mean: 0.800 (per row: {3: 0.8, 4: 0.8, 5: 0.8, 6: 0.8, 9: 0.8, 10: 0.8, 11: 0.8, 12: 0.8})
- post-poison: mean_p_true=0.3254, poison bench_k=2, lineage liar entries=[{'k': 2, 'source': 'consolidation:poison:BASELINE'}]
- affected rows: 28, repair steps: 10 (recovered: True), seated 28/28
- post-repair: mean_p_true=0.8195, top1_true=4/4, auth mean=0.600, lag=0.200
- chain OK at repair: True, at +30: True
- deficit series (n=0..30): [0.3, 0.29, 0.28, 0.27, 0.26, 0.25, 0.24, 0.23, 0.22, 0.21, 0.2, 0.19, 0.18, 0.17, 0.16, 0.15, 0.14, 0.13, 0.12, 0.11, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.0]
- time to healed (applied): 25, (tail arithmetic): 25
- top1_true across decay trajectory: [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]

### COARSE — 800 clean steps, tick 800 steps, 1 tick(s)
- clean baseline (pre-final-consolidation): mean_p_true=0.9790, top1_true=4/4
- pre-injury lm_head authority mean: 0.800 (per row: {3: 0.8, 4: 0.8, 5: 0.8, 6: 0.8, 9: 0.8, 10: 0.8, 11: 0.8, 12: 0.8})
- post-poison: mean_p_true=0.2123, poison bench_k=2, lineage liar entries=[{'k': 2, 'source': 'consolidation:poison:COARSE'}]
- affected rows: 28, repair steps: 18 (recovered: True), seated 28/28
- post-repair: mean_p_true=0.7789, top1_true=4/4, auth mean=0.600, lag=0.200
- chain OK at repair: True, at +30: True
- deficit series (n=0..30): [0.3, 0.29, 0.28, 0.27, 0.26, 0.25, 0.24, 0.23, 0.22, 0.21, 0.2, 0.19, 0.18, 0.17, 0.16, 0.15, 0.14, 0.13, 0.12, 0.11, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.0]
- time to healed (applied): 25, (tail arithmetic): 25
- top1_true across decay trajectory: [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]

## Wiring-check note (threshold (d))

Decay is priced per decay consolidation, never per training step: the decay counter n counts decay consolidations only and never sees the clean-phase step count. COARSE trained 800 clean steps against FINE/BASELINE's 400, yet all three variants heal in the same number of decay ticks (see time_to_healed above). If decay were per-step, the healed tick-counts would differ across variants. This leg therefore checks the wiring (decay counts ticks), not a discovered healing law -- labeled honestly as such.

Overall: PASS
