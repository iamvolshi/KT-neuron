"""E2 RERUN (2026-10-26): poisoning, with the cost leg measured as
WORK TOUCHED, not steps alone.

Pre-registered in preregistration-e2-rerun-2026-10-26.md BEFORE this
script ran. Flip-to-slow is NOT engaged (round-1 behavior).
Reuses trainer.py / corpus.py / playbook.py unchanged; only this new
script is added. Writes results-e2-rerun.json (the .md is written
from that JSON after the run, copying numbers, not re-measuring).

Work definitions (frozen): rows/params updated by the optimizer.
Repair per-step params = ones in the gradient mask (counted);
retrain per-step params = total param count. Total param updates =
per-step params x steps to recovery, steps checked EVERY step.
FLOPs are NOT counted: this harness computes full gradients then
masks, so a repair step pays full backward cost here; any FLOP
saving needs a sparse implementation that does not exist here.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kt_bench.seating import seat_authority
from kt_bench.transformer import GPTConfig
from kt_bench.twin import build_pair
from kt_bench.linear import row_hash
from experiments.corpus import (LIAR_TOKENS, TRUE_TOKENS, gen_clean_batch,
                                gen_poison_batch, probe_all_facts,
                                summarize_probes)
from experiments.playbook import (Bench, FirewallPot, affected_upstream,
                                  build_repair_mask, capture_weights,
                                  null_rows, unnull_rows)
from experiments.trainer import (Adam, collect_params, gradient_check,
                                 loss_and_grads)

CFG = GPTConfig(vocab_size=32, block_size=16, n_embd=64, n_head=4, n_layer=2)
CLEAN_POOL = {"sources": {"textbook": 0.9, "web": 0.5}}   # -> 0.8
POISON_POOL = {"sources": {"liar": 0.9}}                  # -> 0.9
REPAIR_POOL = {"sources": {"textbook": 0.9}}              # -> 0.9, dock 0.3 -> 0.6
REPAIR_DOCK = 0.3
CLEAN_STEPS, POISON_STEPS = 400, 200
REPAIR_CAP, RETRAIN_CAP = 200, 400


def train_steps(model, batches, mask=None, adam=None):
    names, params = collect_params(model)
    if adam is None:
        adam = Adam(params, lr=0.01)
    losses = []
    for b in batches:
        loss, grads = loss_and_grads(model, b)
        glist = [grads[n] for n in names]
        if mask is not None:
            glist = [g * m for g, m in zip(glist, mask)]
        adam.step(glist)
        losses.append(loss)
    return losses, adam


def snapshot_params(model):
    names, params = collect_params(model)
    return names, [p.copy() for p in params]


def moved_counts(names, before, after, model):
    """(params_moved, rows_moved, rows_moved_detail).

    Rows counted over linear layers only (KID rows): a row moves if
    its W row or its bias differs bit-wise. Params moved counts every
    differing entry across all parameter arrays (incl. embeddings/LNs).
    """
    params_moved = 0
    for b, a in zip(before, after):
        params_moved += int(np.count_nonzero(b != a))
    # Map param name -> (layer_name, is_W, is_b) for linear layers.
    from experiments.playbook import layer_param_prefix
    prefix_to_layer = {}
    for layer_name, _layer in model.all_linears():
        prefix_to_layer[layer_param_prefix(layer_name)] = layer_name
    # Collect per-layer W/b before/after.
    before_map = dict(zip(names, before))
    after_map = dict(zip(names, after))
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


def gains_snapshot(model):
    return {name: layer.gain.copy() for name, layer in model.kt_linears()}


def main():
    out = {"flip_engaged": False,
           "flip_note": "NOT engaged per pre-registration: round-1 "
                        "behavior (firewall nulls gains, fast trace "
                        "continues) to preserve setup comparability."}
    fwd_diff, max_rel, max_abs, checked, gate_pass = gradient_check()
    out["gradient_check"] = {"forward_max_abs_diff": fwd_diff,
                             "max_rel_err": max_rel,
                             "max_abs_err": max_abs,
                             "params_checked": checked,
                             "gate": "PASS" if gate_pass else "FAIL"}
    print("gradient check:", out["gradient_check"], flush=True)
    if not gate_pass:
        print("GATE FAILED - training results void. Aborting.", flush=True)
        with open("experiments/results-e2-rerun.json", "w") as f:
            json.dump(out, f, indent=2, default=str)
        return

    rng_c = np.random.default_rng(1)
    clean_batches = [gen_clean_batch(rng_c) for _ in range(CLEAN_STEPS)]
    rng_p = np.random.default_rng(2)
    poison_batches = [gen_poison_batch(rng_p) for _ in range(POISON_STEPS)]
    rng_r = np.random.default_rng(3)
    repair_batches = [gen_clean_batch(rng_r) for _ in range(REPAIR_CAP)]
    rng_ret = np.random.default_rng(4)
    retrain_batches = [gen_clean_batch(rng_ret) for _ in range(RETRAIN_CAP)]

    kt, vn = build_pair(CFG, 0)
    train_steps(kt, clean_batches)
    train_steps(vn, clean_batches)
    base_kt = summarize_probes(probe_all_facts(kt))
    base_vn = summarize_probes(probe_all_facts(vn))
    print("baseline kt:", base_kt, "vn:", base_vn, flush=True)
    bench = Bench(kt)
    bench.consolidate(CLEAN_POOL, "consolidation:0")
    pre_injury_auth = bench.standing_authorities(
        "lm_head", TRUE_TOKENS + LIAR_TOKENS)
    pre_poison_w = capture_weights(kt)
    train_steps(kt, poison_batches)
    train_steps(vn, poison_batches)
    post_poison_w = capture_weights(kt)
    pois_kt = summarize_probes(probe_all_facts(kt))
    pois_vn = summarize_probes(probe_all_facts(vn))
    print("post-poison kt:", pois_kt, "vn:", pois_vn, flush=True)
    bench.consolidate(POISON_POOL, "consolidation:1")
    lineage = bench.lineage_query("liar")
    upstream = affected_upstream(pre_poison_w, post_poison_w, top_n=20)
    fact_rows = [("lm_head", t) for t in TRUE_TOKENS + LIAR_TOKENS]
    affected_rows = fact_rows + upstream
    total_rows = sum(layer.out_features for _n, layer in kt.kt_linears())
    param_count = kt.param_count()
    print("affected rows:", len(affected_rows), "total rows:", total_rows,
          "params:", param_count, flush=True)

    # ---- Firewall demo (branch; restored afterwards, as round 1) -----
    pre_fw_weights = capture_weights(kt)
    pre_fw_gains = gains_snapshot(kt)
    bench.take_snapshot("post-poison")
    pot = FirewallPot()
    pot_traj = []
    nulled = False
    _nfw, _pfw = collect_params(kt)
    adam_fw = Adam(_pfw, lr=0.01)
    for b in poison_batches[:5]:
        train_steps(kt, [b], adam=adam_fw)
        engaged = pot.tainted_batch()
        if engaged and not nulled:
            null_rows(kt, [("lm_head", t) for t in LIAR_TOKENS])
            nulled = True
        pot_traj.append({"pot": pot.pot, "engaged": pot.engaged})
    fw_null_probes = summarize_probes(probe_all_facts(kt))
    for b in clean_batches[:4]:
        train_steps(kt, [b], adam=adam_fw)
        pot.clean_batch()
        if pot.released and nulled:
            unnull_rows(bench, [("lm_head", t) for t in LIAR_TOKENS])
            nulled = False
    fw_end_probes = summarize_probes(probe_all_facts(kt))
    fw_pot_end = pot.pot
    bench.restore_snapshot("post-poison")
    bench.reset_gains_to_standing()
    post_fw_weights = capture_weights(kt)
    post_fw_gains = gains_snapshot(kt)
    fw_restore_weights_exact = all(
        np.array_equal(pre_fw_weights[n][0], post_fw_weights[n][0])
        and np.array_equal(pre_fw_weights[n][1], post_fw_weights[n][1])
        for n in pre_fw_weights)
    # Embeddings/LNs also moved in the demo: compare full params too.
    fw_restore_gains_exact = all(
        np.array_equal(pre_fw_gains[n], post_fw_gains[n]) for n in pre_fw_gains)
    print("firewall traj:", pot_traj, "restore weights exact:",
          fw_restore_weights_exact, "gains exact:", fw_restore_gains_exact,
          flush=True)

    # ---- Quarantine round-trip (null liar rows, probe, unnull) -------
    pre_q_weights = capture_weights(kt)
    null_rows(kt, [("lm_head", t) for t in LIAR_TOKENS])
    q_null_probes = summarize_probes(probe_all_facts(kt))
    unnull_rows(bench, [("lm_head", t) for t in LIAR_TOKENS])
    q_restored_probes = summarize_probes(probe_all_facts(kt))
    post_q_weights = capture_weights(kt)
    q_weights_exact = all(
        np.array_equal(pre_q_weights[n][0], post_q_weights[n][0])
        and np.array_equal(pre_q_weights[n][1], post_q_weights[n][1])
        for n in pre_q_weights)
    print("quarantine null probes:", q_null_probes, "restored:",
          q_restored_probes, "weights exact:", q_weights_exact, flush=True)

    # ---- Surgical repair, 1-step recovery checks, work counted -------
    _names, mask = build_repair_mask(kt, set(affected_rows))
    mask_ones = int(sum(int(np.count_nonzero(m)) for m in mask))
    rep_names, rep_before = snapshot_params(kt)
    _nr, _pr = collect_params(kt)
    adam_rep = Adam(_pr, lr=0.01)
    repair_steps = None
    for step in range(1, REPAIR_CAP + 1):
        train_steps(kt, repair_batches[step - 1:step], mask=mask, adam=adam_rep)
        cur = summarize_probes(probe_all_facts(kt))
        if cur["mean_p_true"] >= base_kt["mean_p_true"] - 0.05:
            repair_steps = step
            break
    rep_recovered = repair_steps is not None
    if not rep_recovered:
        repair_steps = REPAIR_CAP
    _rn, rep_after = snapshot_params(kt)
    rep_params_moved, rep_rows_moved = moved_counts(
        rep_names, rep_before, rep_after, kt)
    rep_rows_moved_set = set(rep_rows_moved)
    rep_leak = sorted(rep_rows_moved_set - set(affected_rows))
    rep_total_updates = mask_ones * repair_steps
    print("repair steps:", repair_steps, "recovered:", rep_recovered,
          "mask ones:", mask_ones, "rows moved:", len(rep_rows_moved),
          "params moved:", rep_params_moved, "leak:", rep_leak, flush=True)
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
    post_repair_probes = summarize_probes(probe_all_facts(kt))
    post_repair_auth = bench.standing_authorities(
        "lm_head", TRUE_TOKENS + LIAR_TOKENS)
    chain_ok, chain_bad = bench.verify_all()
    print("post-repair:", post_repair_probes, "auth:", post_repair_auth,
          "chain ok:", chain_ok, flush=True)

    # ---- Vanilla full retrain, 1-step checks, work counted -----------
    ret_names, ret_before = snapshot_params(vn)
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
    _rn2, ret_after = snapshot_params(vn)
    ret_params_moved, ret_rows_moved = moved_counts(
        ret_names, ret_before, ret_after, vn)
    ret_total_updates = param_count * retrain_steps
    print("retrain steps:", retrain_steps, "recovered:", ret_recovered,
          "params moved:", ret_params_moved, "rows moved:",
          len(ret_rows_moved), flush=True)

    work_ratio = rep_total_updates / ret_total_updates
    work_pass = bool(rep_recovered and ret_recovered
                      and rep_total_updates < 0.10 * ret_total_updates)
    out["E2_rerun"] = {
        "baseline_kt": base_kt, "baseline_vn": base_vn,
        "post_poison_kt": pois_kt, "post_poison_vn": pois_vn,
        "lineage_liar_entries": [{"k": e["k"], "source": e["source"]}
                                 for e in lineage],
        "affected_rows": [list(r) for r in affected_rows],
        "affected_rows_count": len(affected_rows),
        "total_linear_rows": total_rows,
        "param_count": param_count,
        "firewall_pot_trajectory": pot_traj,
        "firewall_probes_post_null": fw_null_probes,
        "firewall_probes_post_release": fw_end_probes,
        "firewall_pot_end": fw_pot_end,
        "firewall_restore_weights_exact": bool(fw_restore_weights_exact),
        "firewall_restore_gains_exact": bool(fw_restore_gains_exact),
        "quarantine_null_probes": q_null_probes,
        "quarantine_restored_probes": q_restored_probes,
        "quarantine_weights_exact": bool(q_weights_exact),
        "repair": {
            "steps_to_recovery_1step_checks": repair_steps,
            "recovered": rep_recovered,
            "rows_updated_design": len(affected_rows),
            "rows_moved_actual": len(rep_rows_moved),
            "rows_moved_outside_mask_leak": rep_leak,
            "params_updated_per_step_mask_ones": mask_ones,
            "params_moved_actual": rep_params_moved,
            "total_param_updates": rep_total_updates,
            "post_repair_probes": post_repair_probes,
            "post_repair_lm_head_auth": post_repair_auth,
            "repair_statuses_seated": sum(
                1 for s in repair_statuses.values() if s == "seated"),
            "repair_statuses_total": len(repair_statuses),
            "chain_ok": chain_ok, "chain_bad": chain_bad,
        },
        "full_retrain": {
            "steps_to_recovery_1step_checks": retrain_steps,
            "recovered": ret_recovered,
            "rows_updated_design": total_rows,
            "rows_moved_actual": len(ret_rows_moved),
            "params_updated_per_step": param_count,
            "params_moved_actual": ret_params_moved,
            "total_param_updates": ret_total_updates,
        },
        "work_comparison": {
            "repair_total_param_updates": rep_total_updates,
            "retrain_total_param_updates": ret_total_updates,
            "ratio_repair_over_retrain": work_ratio,
            "threshold": "repair < 0.10 x retrain",
            "verdict": "PASS" if work_pass else "FAIL",
        },
        "behavior_legs": {
            "vanilla_flip": bool(pois_vn["mean_p_liar"] > pois_vn["mean_p_true"]
                                 and pois_vn["top1_liar_count"] >= 3),
            "lineage_exact_round1_form": lineage and [
                {"k": e["k"], "source": e["source"]} for e in lineage]
                == [{"k": 2, "source": "consolidation:1"}],
            "firewall_engaged": any(t["engaged"] for t in pot_traj),
            "repair_seated_all": sum(
                1 for s in repair_statuses.values() if s == "seated")
                == len(repair_statuses),
            "chain_ok": chain_ok,
        },
        "flops_note": "NOT counted: harness computes full gradients then "
                      "masks, so a repair step pays full backward cost; "
                      "optimizer-write work is the counted quantity.",
    }
    print("WORK:", out["E2_rerun"]["work_comparison"], flush=True)
    with open("experiments/results-e2-rerun.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("results-e2-rerun.json written", flush=True)


if __name__ == "__main__":
    main()
