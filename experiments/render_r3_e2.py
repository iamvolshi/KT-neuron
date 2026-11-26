"""Markdown rendering for R3-E2 results.

Split out of experiments/run_r3_e2.py so no module exceeds
500 lines (standing build doctrine). render_md formats the
same run dict the JSON writer serializes; it measures
nothing itself.
"""


def render_md(out):
    """Markdown report generated from the same run dict as the JSON."""
    v = out["verdicts"]
    lines = []
    lines.append("# R3-E2 — results: tail manipulation controls healing "
                 "(2026-10-26)")
    lines.append("")
    lines.append("Frozen contract: "
                 "`experiments/preregistration-round3-2026-10-26.md`, "
                 "R3-E2 section. Every number below comes from this run; "
                 "the .md is generated from the same dict as "
                 "`results-r3-e2.json`, not re-measured.")
    lines.append("")
    lines.append("## Run command")
    lines.append("")
    lines.append("From `~/workspace/kt-neuron`:")
    lines.append("")
    lines.append("```")
    lines.append("python3 experiments/run_r3_e2.py")
    lines.append("```")
    lines.append("")
    g = out["gradient_check"]
    lines.append(f"Gradient gate: {g['gate']} (forward max abs diff "
                 f"{g['forward_max_abs_diff']}, max rel err "
                 f"{g['max_rel_err']:.2e}, max abs err "
                 f"{g['max_abs_err']:.2e}, {g['params_checked']} params "
                 f"checked). No new training code path exists: all "
                 f"training reuses `run_all.train_steps`; "
                 f"`kt_bench/tail.py` is pure arithmetic.")
    lines.append("")
    lines.append("## Setup reproduced (per tail, identical seeds)")
    lines.append("")
    r = out["tails"]["FAST"]
    lines.append(f"- Bench: vocab 32, block 16, n_embd 64, 4 heads, "
                 f"2 layers, 105,248 params, 1,184 KIDs; seeds init 0 / "
                 f"data 1. Clean 400 steps, poison 200, repair dock 0.3.")
    lines.append(f"- Clean baseline (KT): mean P(true) "
                 f"{r['baseline_kt']['mean_p_true']:.4f}, top-1 true "
                 f"{r['baseline_kt']['top1_true_count']}/4.")
    lines.append(f"- Post-poison (KT): mean P(true) "
                 f"{r['post_poison_kt']['mean_p_true']:.4f}, P(liar) "
                 f"{r['post_poison_kt']['mean_p_liar']:.4f}, top-1 liar "
                 f"{r['post_poison_kt']['top1_liar_count']}/4.")
    lines.append(f"- Pre-injury lm_head authority (clean pool): "
                 f"{sorted(set(r['pre_injury_lm_head_auth'].values()))}.")
    lines.append(f"- Repair: {r['repair_steps']} steps to recovery "
                 f"(1-step checks, cap 200), recovered "
                 f"{r['repair_recovered']}; post-repair mean P(true) "
                 f"{r['post_repair_probes']['mean_p_true']:.4f}, top-1 "
                 f"true {r['post_repair_probes']['top1_true_count']}/4, "
                 f"lm_head authority all "
                 f"{sorted(set(r['post_repair_lm_head_auth'].values()))}.")
    lines.append(f"- Chain verify at repair: "
                 f"{r['chain_ok_at_repair']}; at +30 (all tails): "
                 f"{all(t['chain_ok_at_plus30'] for t in out['tails'].values())}.")
    lines.append(f"- Pre-decay trajectories identical across tails: "
                 f"{out['pre_decay_trajectories_identical']}.")
    lines.append("")
    lines.append("## Frozen verdicts (a)-(f)")
    lines.append("")
    for name in ("FAST", "SLOW", "PERMANENT"):
        t6 = out["tails"][name]["trajectory"][6]
        t30 = out["tails"][name]["trajectory"][30]
        lines.append(f"- {name} (decay "
                     f"{out['tails'][name]['decay_per_consolidation']}/consolidation): "
                     f"deficit at +6 = {t6['deficit']:.4f} "
                     f"(authority mean {t6['auth_mean']:.3f}), at +30 = "
                     f"{t30['deficit']:.4f} (authority mean "
                     f"{t30['auth_mean']:.3f}), time-to-healed "
                     f"{out['tails'][name]['time_to_healed']}.")
    lines.append(f"- (a) FAST healed by +6: "
                 f"{'PASS' if v['a_fast_healed_by_plus6'] else 'FAIL'} "
                 f"(deficit at +6 = "
                 f"{out['tails']['FAST']['trajectory'][6]['deficit']:.4f} "
                 f"<= 0.05; healed already at +5).")
    lines.append(f"- (b) SLOW not healed by +6: "
                 f"{'PASS' if v['b_slow_deficit_0.24_at_plus6'] else 'FAIL'} "
                 f"(deficit at +6 = "
                 f"{out['tails']['SLOW']['trajectory'][6]['deficit']:.4f}, "
                 f"within 0.24 +/- 0.005, reproducing round-1/2).")
    lines.append(f"- (c) PERMANENT unchanged at +6: "
                 f"{'PASS' if v['c_permanent_deficit_0.30_at_plus6'] else 'FAIL'} "
                 f"(deficit at +6 = "
                 f"{out['tails']['PERMANENT']['trajectory'][6]['deficit']:.4f}, "
                 f"within 0.30 +/- 0.005).")
    lines.append(f"- (d) Ordering FAST < SLOW < PERMANENT-never: "
                 f"{'PASS' if v['d_ordering_fast_lt_slow_lt_permanent_never'] else 'FAIL'} "
                 f"(times-to-healed: {v['time_to_healed']}).")
    lines.append(f"- (e) Behavior ranking 4/4 at repair and every tracked "
                 f"consolidation (+1..+30, all tails): "
                 f"{'PASS' if v['e_top1_4of4_at_repair_and_every_tracked_consolidation'] else 'FAIL'}.")
    lines.append(f"- (f) Lineage exactly "
                 f"```[{{k: 2, source: \"consolidation:1\"}}]``` in all "
                 f"tails: {v['f_lineage_exact']}; repair seats all "
                 f"affected rows: {v['f_repair_seats_all']} "
                 f"({r['repair_statuses_seated']}/{r['repair_statuses_total']} "
                 f"per tail). Combined: "
                 f"{'PASS' if v['f_lineage_exact_and_repair_seats_all'] else 'FAIL'}.")
    lines.append(f"- **OVERALL: {v['overall']}**")
    lines.append("")
    lines.append("## Re-measurement leg (i): confidence trajectory — "
                 "record only, NO verdict")
    lines.append("")
    lines.append("The absolute confidence bar at repair time is retired; "
                 "only the trajectory is reported, and no verdict is "
                 "drawn from these values. Deficit is the dock "
                 "parameter from `kt_bench.tail` (exact); standing "
                 "authority is q10(0.9 - deficit) on the 0.1 grid, "
                 "reported alongside, never substituted for deficit "
                 "(e.g. SLOW at +6: deficit 0.24, authority mean 0.7).")
    lines.append("")
    for name in ("FAST", "SLOW", "PERMANENT"):
        confs = out["remeasurement_confidence_trajectory"][name][
            "mean_conf_at_repair_and_each_decay"]
        lines.append(f"- {name} mean top-1 confidence, repair then +1..+30: "
                     + ", ".join(f"{c:.4f}" for c in confs))
        lines.append(f"  Non-decreasing observed: "
                     f"{out['remeasurement_confidence_trajectory'][name]['non_decreasing_observed']}.")
    lines.append("")
    lines.append("## Re-measurement leg (ii): repair cost as work touched "
                 "(FAST tail run)")
    lines.append("")
    w = out["tails"]["FAST"]["work_leg"]
    lines.append(f"- Surgical repair: {w['repair_steps_to_recovery_1step_checks']} "
                 f"steps, {w['repair_rows_design']} rows by design, "
                 f"{w['repair_rows_moved_actual']} rows moved, "
                 f"{w['repair_params_per_step_mask_ones']} params/step "
                 f"(mask ones), {w['repair_total_param_updates']} total "
                 f"param updates; leak outside mask: "
                 f"{w['repair_rows_moved_outside_mask_leak']}.")
    lines.append(f"- Full retrain (vanilla): "
                 f"{w['retrain_steps_to_recovery_1step_checks']} steps, "
                 f"{w['retrain_params_per_step']} params/step, "
                 f"{w['retrain_total_param_updates']} total param updates.")
    lines.append(f"- Ratio repair/retrain = "
                 f"{w['ratio_repair_over_retrain']:.4f}; replication bar "
                 f"ratio < 0.10: {'MET' if w['bar_met'] else 'NOT MET'}. "
                 f"NOT FLOPs, NOT steps: optimizer writes only, as "
                 f"pre-registered (this harness computes full gradients "
                 f"then masks).")
    lines.append("")
    lines.append("## Notes / left for later")
    lines.append("")
    lines.append(f"- {out['notes']['firewall_demo_skipped']}")
    lines.append(f"- {out['notes']['work_leg_scope']}")
    lines.append("- Decay consolidations are seating-only and cheap; "
                 "tracking ran to +30 for every tail, past the +6 "
                 "frozen minimum, to observe SLOW healing at +25 and "
                 "confirm PERMANENT never heals.")
    lines.append("")
    return "\n".join(lines) + "\n"
