"""R3-E2 official run: tail manipulation controls healing (plan §6 step 7).

Frozen contract: experiments/preregistration-round3-2026-10-26.md,
R3-E2 section. Frozen thresholds (a)-(f) are evaluated, never moved.

Method, exactly:
- Gradient gate FIRST (experiments/trainer.gradient_check); abort
  on FAIL. No new training code path exists here: all training
  reuses experiments/run_all.train_steps (manual backprop + Adam,
  numpy-only), as in rounds 1-2.
- Per tail (FAST 0.05, SLOW 0.01, PERMANENT 0.00 per consolidation):
  the SAME injury as round-1 E2, run independently with identical
  seeds so the pre-decay trajectory is comparable: clean 400 steps
  (rng 1), poison 200 (rng 2), consolidation:0 (clean pool) then
  consolidation:1 (poison pool), affected rows = 8 lm_head fact rows
  + top-20 upstream rows moved during poison, surgical repair on
  clean batches (rng 3, 1-step recovery checks, cap 200) against the
  model's own clean baseline minus 0.05, then re-seat of all
  affected rows with the repair pool and dock 0.3 -> deficit 0.30
  at repair, authority 0.6.
- Decay consolidations are seating-only (direct rec.replace, as in
  round-1 E3 -- NOT Bench.consolidate, so no ledger/snapshot side
  effects): at decay n the dock is tail.deficit_at(n) from
  kt_bench.tail, the claim stays the repair pool (0.9), gains reset
  to standing authority. Tracked to +30 for every tail: +6 is the
  frozen minimum, +30 is needed to observe SLOW healing (25 ticks)
  and to confirm PERMANENT never heals.
- Smallest-choice notes (stated, not hidden): (1) the round-1/2
  firewall demo is a restored side-branch, not part of any R3-E2
  frozen threshold, and is skipped -- round 2 verified its restore
  bit-exact, so skipping cannot change the post-poison state;
  (2) the repair-cost re-measurement leg is run on the FAST tail
  only, with a lockstep vanilla twin retrained on clean batches
  (rng 4, 1-step checks, cap 400) -- the other tails share the
  identical pre-decay trajectory by determinism, verified below,
  so their work numbers would be the same run repeated.
- Deficit vs authority: the deficit is the dock parameter from
  kt_bench.tail, exact. Standing authority is q10(0.9 - deficit)
  on the 0.1 grid and is reported alongside, never substituted:
  e.g. SLOW at +6 has deficit exactly 0.24 but authority 0.7
  (q10(0.66)), so 0.9 - authority = 0.2 is NOT the deficit.

Run from ~/workspace/kt-neuron:
    python3 experiments/run_r3_e2.py
Writes experiments/results-r3-e2.json and results-r3-e2.md (the .md
is generated from the same JSON dict, copying numbers, not
re-measuring).
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from kt_bench.linear import row_hash
from kt_bench.seating import seat_authority
from kt_bench.tail import FAST_TAIL, PERMANENT_TAIL, SLOW_TAIL
from kt_bench.transformer import GPTConfig
from kt_bench.twin import build_pair
from experiments.corpus import (LIAR_TOKENS, TRUE_TOKENS, gen_clean_batch,
                                gen_poison_batch, probe_all_facts,
                                summarize_probes)
from experiments.playbook import (Bench, affected_upstream,
                                  build_repair_mask, capture_weights,
                                  layer_param_prefix)
from experiments.run_all import (CLEAN_POOL, POISON_POOL, REPAIR_POOL,
                                 train_steps)
from experiments.trainer import (Adam, collect_params, gradient_check)
from experiments.render_r3_e2 import render_md

CFG = GPTConfig(vocab_size=32, block_size=16, n_embd=64, n_head=4, n_layer=2)
REPAIR_DOCK = 0.3
CLEAN_STEPS, POISON_STEPS = 400, 200
REPAIR_CAP, RETRAIN_CAP = 200, 400
DECAY_TO = 30
TAILS = [FAST_TAIL, SLOW_TAIL, PERMANENT_TAIL]


def probe_summary(model):
    """Probes + mean top-1 confidence (confidence is NOT in summarize)."""
    probes = probe_all_facts(model)
    s = summarize_probes(probes)
    s["mean_conf"] = float(np.mean([p["conf"] for p in probes]))
    s["per_fact_p_true"] = [p["p_true"] for p in probes]
    s["per_fact_conf"] = [p["conf"] for p in probes]
    return s


def moved_counts(names, before, after, model):
    """(params_moved, rows_moved) mirroring the round-2 rerun's count."""
    params_moved = int(sum(int(np.count_nonzero(b != a))
                           for b, a in zip(before, after)))
    prefix_to_layer = {}
    for layer_name, _layer in model.all_linears():
        prefix_to_layer[layer_param_prefix(layer_name)] = layer_name
    before_map, after_map = dict(zip(names, before)), dict(zip(names, after))
    rows_moved = []
    for prefix, layer_name in prefix_to_layer.items():
        w0, w1 = before_map[f"{prefix}.W"], after_map[f"{prefix}.W"]
        b0, b1 = before_map[f"{prefix}.b"], after_map[f"{prefix}.b"]
        w_diff = np.any(w0 != w1, axis=1)
        b_diff = b0 != b1
        for i in range(w0.shape[0]):
            if w_diff[i] or b_diff[i]:
                rows_moved.append((layer_name, int(i)))
    return params_moved, rows_moved


def run_tail(tail, clean_batches, poison_batches, repair_batches,
             retrain_batches, do_work_leg):
    """Full injury + repair + decay pipeline for one tail deployment."""
    kt, vn = build_pair(CFG, 0)
    train_steps(kt, clean_batches)
    if do_work_leg:
        train_steps(vn, clean_batches)
    base_kt = summarize_probes(probe_all_facts(kt))
    base_vn = summarize_probes(probe_all_facts(vn)) if do_work_leg else None
    bench = Bench(kt)
    bench.consolidate(CLEAN_POOL, "consolidation:0")
    pre_injury_auth = bench.standing_authorities(
        "lm_head", TRUE_TOKENS + LIAR_TOKENS)
    pre_poison_w = capture_weights(kt)
    train_steps(kt, poison_batches)
    if do_work_leg:
        train_steps(vn, poison_batches)
    post_poison_w = capture_weights(kt)
    pois_kt = summarize_probes(probe_all_facts(kt))
    bench.consolidate(POISON_POOL, "consolidation:1")
    lineage = bench.lineage_query("liar")
    upstream = affected_upstream(pre_poison_w, post_poison_w, top_n=20)
    fact_rows = [("lm_head", t) for t in TRUE_TOKENS + LIAR_TOKENS]
    affected_rows = fact_rows + upstream

    # ---- Surgical repair (1-step recovery checks) ----------------------
    _names, mask = build_repair_mask(kt, set(affected_rows))
    mask_ones = int(sum(int(np.count_nonzero(m)) for m in mask))
    rep_names, rep_before = collect_params(kt)
    rep_before = [p.copy() for p in rep_before]
    _nr, _pr = collect_params(kt)
    adam_rep = Adam(_pr, lr=0.01)
    repair_steps = None
    for step in range(1, REPAIR_CAP + 1):
        train_steps(kt, repair_batches[step - 1:step], mask=mask,
                    adam=adam_rep)
        cur = summarize_probes(probe_all_facts(kt))
        if cur["mean_p_true"] >= base_kt["mean_p_true"] - 0.05:
            repair_steps = step
            break
    rep_recovered = repair_steps is not None
    if not rep_recovered:
        repair_steps = REPAIR_CAP
    _rn, rep_after = collect_params(kt)
    rep_params_moved, rep_rows_moved = moved_counts(
        rep_names, rep_before, rep_after, kt)
    rep_leak = sorted(set(rep_rows_moved) - set(affected_rows))
    repair_statuses = {}
    for name, row in affected_rows:
        layer = dict(bench.layers)[name]
        rec = layer.records[row]
        status, _r = rec.replace(
            "row", row_hash(layer.W[row], layer.b[row]),
            seat_authority(REPAIR_POOL), "repair:1",
            authority_dock=REPAIR_DOCK)
        layer.gain[row] = rec.standing("row")["authority"]
        repair_statuses[f"{name}:{row}"] = status
    post_repair_probes = probe_summary(kt)
    post_repair_auth = bench.standing_authorities(
        "lm_head", TRUE_TOKENS + LIAR_TOKENS)
    chain_ok, chain_bad = bench.verify_all()

    # ---- Vanilla full retrain (work leg, FAST tail only) ---------------
    work = None
    if do_work_leg:
        ret_names, ret_before = collect_params(vn)
        ret_before = [p.copy() for p in ret_before]
        _nret, _pret = collect_params(vn)
        adam_ret = Adam(_pret, lr=0.01)
        retrain_steps = None
        for step in range(1, RETRAIN_CAP + 1):
            train_steps(vn, retrain_batches[step - 1:step], adam=adam_ret)
            cur = summarize_probes(probe_all_facts(vn))
            if cur["mean_p_true"] >= base_vn["mean_p_true"] - 0.05:
                retrain_steps = step
                break
        ret_recovered = retrain_steps is not None
        if not ret_recovered:
            retrain_steps = RETRAIN_CAP
        _rn2, ret_after = collect_params(vn)
        ret_params_moved, ret_rows_moved = moved_counts(
            ret_names, ret_before, ret_after, vn)
        param_count = kt.param_count()
        rep_total = mask_ones * repair_steps
        ret_total = param_count * retrain_steps
        work = {
            "repair_steps_to_recovery_1step_checks": repair_steps,
            "repair_recovered": bool(rep_recovered),
            "repair_rows_design": len(affected_rows),
            "repair_rows_moved_actual": len(rep_rows_moved),
            "repair_rows_moved_outside_mask_leak": rep_leak,
            "repair_params_per_step_mask_ones": mask_ones,
            "repair_params_moved_actual": rep_params_moved,
            "repair_total_param_updates": rep_total,
            "retrain_steps_to_recovery_1step_checks": retrain_steps,
            "retrain_recovered": bool(ret_recovered),
            "retrain_rows_moved_actual": len(ret_rows_moved),
            "retrain_params_per_step": param_count,
            "retrain_params_moved_actual": ret_params_moved,
            "retrain_total_param_updates": ret_total,
            "ratio_repair_over_retrain": rep_total / ret_total,
            "replication_bar": "ratio < 0.10 (round-2 bar)",
            "bar_met": bool(rep_total < 0.10 * ret_total),
        }
        print(f"WORK [{tail.name}]: repair {rep_total} vs retrain "
              f"{ret_total}, ratio {rep_total / ret_total:.4f}",
              flush=True)

    # ---- Decay consolidations (seating-only) to +30 --------------------
    trajectory = [{"n": 0, "deficit": tail.deficit_at(0),
                   "probes": post_repair_probes,
                   "auth_mean": float(np.mean(list(post_repair_auth.values()))),
                   "lm_head_auth": post_repair_auth,
                   "healed": tail.is_healed(0)}]
    for n in range(1, DECAY_TO + 1):
        deficit = tail.deficit_at(n)
        seated_n = 0
        for name, row in affected_rows:
            layer = dict(bench.layers)[name]
            rec = layer.records[row]
            status, _r = rec.replace(
                "row", row_hash(layer.W[row], layer.b[row]),
                seat_authority(REPAIR_POOL), f"decay:{tail.name}:{n}",
                authority_dock=deficit)
            layer.gain[row] = rec.standing("row")["authority"]
            if status == "seated":
                seated_n += 1
        auth = bench.standing_authorities("lm_head",
                                          TRUE_TOKENS + LIAR_TOKENS)
        trajectory.append({
            "n": n, "deficit": deficit, "probes": probe_summary(kt),
            "auth_mean": float(np.mean(list(auth.values()))),
            "lm_head_auth": auth, "healed": tail.is_healed(n),
            "decay_reseat_seated": seated_n,
            "decay_reseat_total": len(affected_rows)})
        if n in (1, 2, 6, 25, 30):
            t = trajectory[-1]
            print(f"[{tail.name}] +{n}: deficit={deficit:.4f} "
                  f"auth_mean={t['auth_mean']:.3f} "
                  f"p_true={t['probes']['mean_p_true']:.4f} "
                  f"top1={t['probes']['top1_true_count']}/4 "
                  f"conf={t['probes']['mean_conf']:.4f} "
                  f"healed={t['healed']}", flush=True)
    chain_ok_end, chain_bad_end = bench.verify_all()
    return {
        "tail": tail.name, "decay_per_consolidation": tail.decay_per_consolidation,
        "baseline_kt": base_kt,
        "pre_injury_lm_head_auth": pre_injury_auth,
        "post_poison_kt": pois_kt,
        "lineage_liar_entries": [{"k": e["k"], "source": e["source"]}
                                 for e in lineage],
        "affected_rows": [list(r) for r in affected_rows],
        "affected_rows_count": len(affected_rows),
        "repair_steps": repair_steps, "repair_recovered": bool(rep_recovered),
        "repair_statuses_seated": sum(1 for s in repair_statuses.values()
                                      if s == "seated"),
        "repair_statuses_total": len(repair_statuses),
        "post_repair_probes": post_repair_probes,
        "post_repair_lm_head_auth": post_repair_auth,
        "chain_ok_at_repair": chain_ok, "chain_bad_at_repair": chain_bad,
        "chain_ok_at_plus30": chain_ok_end, "chain_bad_at_plus30": chain_bad_end,
        "trajectory": trajectory,
        "time_to_healed": tail.time_to_healed(n_cap=DECAY_TO),
        "work_leg": work,
    }


def main():
    out = {}
    fwd_diff, max_rel, max_abs, checked, gate_pass = gradient_check()
    out["gradient_check"] = {"forward_max_abs_diff": fwd_diff,
                             "max_rel_err": max_rel,
                             "max_abs_err": max_abs,
                             "params_checked": checked,
                             "gate": "PASS" if gate_pass else "FAIL"}
    print("gradient check:", out["gradient_check"], flush=True)
    if not gate_pass:
        print("GATE FAILED -- aborting.", flush=True)
        return out

    rng_c = np.random.default_rng(1)
    clean_batches = [gen_clean_batch(rng_c) for _ in range(CLEAN_STEPS)]
    rng_p = np.random.default_rng(2)
    poison_batches = [gen_poison_batch(rng_p) for _ in range(POISON_STEPS)]
    rng_r = np.random.default_rng(3)
    repair_batches = [gen_clean_batch(rng_r) for _ in range(REPAIR_CAP)]
    rng_ret = np.random.default_rng(4)
    retrain_batches = [gen_clean_batch(rng_ret) for _ in range(RETRAIN_CAP)]

    results = {}
    for i, tail in enumerate(TAILS):
        print(f"--- tail {tail.name} (decay {tail.decay_per_consolidation}) ---",
              flush=True)
        results[tail.name] = run_tail(
            tail, clean_batches, poison_batches, repair_batches,
            retrain_batches, do_work_leg=(i == 0))
    out["tails"] = results

    # Pre-decay trajectories must match across tails (same seeds).
    ref = results["FAST"]
    same = {}
    for name in ("SLOW", "PERMANENT"):
        r = results[name]
        same[name] = {
            "baseline_mean_p_true_equal": r["baseline_kt"]["mean_p_true"]
            == ref["baseline_kt"]["mean_p_true"],
            "post_poison_mean_p_true_equal": r["post_poison_kt"]["mean_p_true"]
            == ref["post_poison_kt"]["mean_p_true"],
            "affected_rows_equal": r["affected_rows"] == ref["affected_rows"],
            "repair_steps_equal": r["repair_steps"] == ref["repair_steps"],
            "post_repair_mean_p_true_equal":
                r["post_repair_probes"]["mean_p_true"]
                == ref["post_repair_probes"]["mean_p_true"],
            "lineage_equal": r["lineage_liar_entries"]
            == ref["lineage_liar_entries"],
        }
    out["pre_decay_trajectories_identical"] = same
    print("pre-decay identical:", same, flush=True)

    # ---- Frozen verdicts (a)-(f) ---------------------------------------
    def traj(tail_name, n):
        return results[tail_name]["trajectory"][n]  # n == index (0..30)

    a = traj("FAST", 6)["deficit"] <= 0.05
    b = abs(traj("SLOW", 6)["deficit"] - 0.24) <= 0.005
    c = abs(traj("PERMANENT", 6)["deficit"] - 0.30) <= 0.005
    t_fast = results["FAST"]["time_to_healed"]
    t_slow = results["SLOW"]["time_to_healed"]
    t_perm = results["PERMANENT"]["time_to_healed"]
    d = (t_fast is not None and t_slow is not None
         and t_fast < t_slow and t_perm is None)
    e = all(t["probes"]["top1_true_count"] == 4
            for r in results.values() for t in r["trajectory"])
    f_lineage = all(r["lineage_liar_entries"]
                    == [{"k": 2, "source": "consolidation:1"}]
                    for r in results.values())
    f_seating = all(r["repair_statuses_seated"] == r["repair_statuses_total"]
                    for r in results.values())
    f = f_lineage and f_seating
    out["verdicts"] = {
        "a_fast_healed_by_plus6": bool(a),
        "b_slow_deficit_0.24_at_plus6": bool(b),
        "c_permanent_deficit_0.30_at_plus6": bool(c),
        "d_ordering_fast_lt_slow_lt_permanent_never": bool(d),
        "e_top1_4of4_at_repair_and_every_tracked_consolidation": bool(e),
        "f_lineage_exact_and_repair_seats_all": bool(f),
        "f_lineage_exact": bool(f_lineage),
        "f_repair_seats_all": bool(f_seating),
        "time_to_healed": {"FAST": t_fast, "SLOW": t_slow,
                           "PERMANENT": t_perm},
        "overall": "PASS" if (a and b and c and d and e and f) else "FAIL",
    }
    print("VERDICTS:", out["verdicts"], flush=True)

    # ---- Re-measurement leg (i): confidence trajectory, NO verdict ----
    conf_leg = {}
    for name, r in results.items():
        confs = [t["probes"]["mean_conf"] for t in r["trajectory"]]
        conf_leg[name] = {
            "mean_conf_at_repair_and_each_decay": confs,
            "non_decreasing_observed": all(
                b >= a for a, b in zip(confs, confs[1:])),
            "note": "Reported for the record only. No verdict is drawn "
                    "from absolute confidence values (retired bar).",
        }
    out["remeasurement_confidence_trajectory"] = conf_leg
    print("confidence non-decreasing observed:",
          {k: v["non_decreasing_observed"] for k, v in conf_leg.items()},
          flush=True)

    out["notes"] = {
        "deficit_is_dock_parameter": "Deficit values are the exact dock "
            "from kt_bench.tail. Standing authority is q10(0.9 - deficit) "
            "and is reported alongside, never substituted for deficit.",
        "firewall_demo_skipped": "Restored side-branch in rounds 1-2, "
            "not part of any R3-E2 frozen threshold; round 2 verified "
            "its restore bit-exact, so skipping cannot change the "
            "post-poison state.",
        "work_leg_scope": "Repair-cost leg run on the FAST tail only; "
            "other tails share the identical pre-decay trajectory by "
            "determinism (see pre_decay_trajectories_identical).",
        "no_new_training_code_path": "All training reuses "
            "experiments/run_all.train_steps; kt_bench/tail.py is pure "
            "arithmetic. Gradient gate above covers the training path.",
    }

    exp_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(exp_dir, "results-r3-e2.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("results-r3-e2.json written", flush=True)
    with open(os.path.join(exp_dir, "results-r3-e2.md"), "w") as f:
        f.write(render_md(out))
    print("results-r3-e2.md written", flush=True)
    print("OVERALL:", out["verdicts"]["overall"], flush=True)
    return out


if __name__ == "__main__":
    main()
