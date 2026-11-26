"""Run all Phase 3 experiments. Writes results.json + results.md.

Order: gradient-check gate FIRST (training is void if it fails),
then E1, then the mainline (E2/E3/E4), then E5. Deterministic:
every batch list is pre-generated from a fixed seed; both models in
a pair train on the same batch lists (lockstep).
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kt_bench.linear import KTLinear, Linear, row_hash
from kt_bench.seating import seat_authority
from kt_bench.transformer import GPT, GPTConfig
from kt_bench.twin import build_pair
from experiments.corpus import (FACTS, LIAR_ONLY, LIAR_TOKENS,
                                TRUE_TOKENS, cross_entropy_loss,
                                gen_clean_batch, gen_poison_batch,
                                probe_all_facts, probe_model,
                                probe_prefix_tokens,
                                summarize_probes)
from experiments.playbook import (Bench, FirewallPot, abstain_predict,
                                  affected_upstream, build_repair_mask,
                                  capture_weights, copy_nonrow_state,
                                  layer_param_prefix,
                                  null_rows, time_forwards, transfer_rows,
                                  unnull_rows)
from experiments.trainer import (Adam, collect_params, gradient_check,
                                 loss_and_grads)

CFG = GPTConfig(vocab_size=32, block_size=16, n_embd=64, n_head=4, n_layer=2)
CLEAN_POOL = {"sources": {"textbook": 0.9, "web": 0.5}}   # -> 0.8
POISON_POOL = {"sources": {"liar": 0.9}}                  # -> 0.9
REPAIR_POOL = {"sources": {"textbook": 0.9}}              # -> 0.9, dock 0.3 -> 0.6
REPAIR_DOCK = 0.3
TRANSFER_DOCK = 0.1
CLEAN_STEPS, POISON_STEPS = 400, 200
REPAIR_CAP, RETRAIN_CAP = 200, 400


def train_steps(model, batches, mask=None, adam=None):
    """Train on a batch list with ONE Adam state across all steps.

    Returns (losses, adam). A fresh Adam per step would silently turn
    Adam into signSGD, so callers doing stepwise loops must pass the
    same adam back in.
    """
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


def eval_loss(model, batches):
    return float(np.mean([cross_entropy_loss(model, b) for b in batches]))


def main():
    out = {}
    # ---- Gate: gradient check -------------------------------------------
    fwd_diff, max_rel, max_abs, checked, gate_pass = gradient_check()
    out["gradient_check"] = {"forward_max_abs_diff": fwd_diff,
                             "max_rel_err": max_rel,
                             "max_abs_err": max_abs,
                             "params_checked": checked,
                             "gate": "PASS" if gate_pass else "FAIL"}
    print("gradient check:", out["gradient_check"], flush=True)
    if out["gradient_check"]["gate"] != "PASS":
        print("GATE FAILED — training results would be void. Aborting.", flush=True)
        with open("experiments/results.json", "w") as f:
            json.dump(out, f, indent=2, default=str)
        return

    # ---- Fixed batch lists (lockstep) ------------------------------------
    # One rng per list, consumed in order: both models in a pair train
    # on the same batch lists, so lockstep holds by construction.
    rng_c = np.random.default_rng(1)
    clean_batches = [gen_clean_batch(rng_c) for _ in range(CLEAN_STEPS)]
    rng_p = np.random.default_rng(2)
    poison_batches = [gen_poison_batch(rng_p) for _ in range(POISON_STEPS)]
    rng_r = np.random.default_rng(3)
    repair_batches = [gen_clean_batch(rng_r) for _ in range(REPAIR_CAP)]
    rng_ret = np.random.default_rng(4)
    retrain_batches = [gen_clean_batch(rng_ret) for _ in range(RETRAIN_CAP)]
    rng_e = np.random.default_rng(99)
    eval_batches = [gen_clean_batch(rng_e) for _ in range(16)]
    eval_batch = eval_batches[0]

    # ---- E1: Cost ---------------------------------------------------------
    kt1, vn1 = build_pair(CFG, 0)
    parity_pre = bool(np.array_equal(kt1.forward(eval_batch), vn1.forward(eval_batch)))
    train_steps(kt1, clean_batches)
    train_steps(vn1, clean_batches)
    wpar = all(np.array_equal(a.W, b.W) and np.array_equal(a.b, b.b)
               for (_n1, a), (_n2, b) in zip(kt1.all_linears(), vn1.all_linears()))
    kt1_pre = summarize_probes(probe_all_facts(kt1))
    vn1_pre = summarize_probes(probe_all_facts(vn1))
    loss_kt_pre, loss_vn_pre = eval_loss(kt1, eval_batches), eval_loss(vn1, eval_batches)
    bench1 = Bench(kt1)
    bench1.consolidate(CLEAN_POOL, "consolidation:0")
    kt1_post = summarize_probes(probe_all_facts(kt1))
    loss_kt_post = eval_loss(kt1, eval_batches)
    t_kt = time_forwards(kt1, eval_batch)
    t_vn = time_forwards(vn1, eval_batch)
    n_kids = sum(layer.out_features for _n, layer in kt1.kt_linears())
    n_rows = sum(len(rec.rows) for _n, layer in kt1.kt_linears() for rec in layer.records)
    out["E1_cost"] = {
        "logit_parity_pre_consolidation": parity_pre,
        "weights_identical_post_clean_training": bool(wpar),
        "kt_probes_pre_consolidation": kt1_pre, "vn_probes": vn1_pre,
        "kt_probes_post_consolidation": kt1_post,
        "loss_kt_pre": loss_kt_pre, "loss_vn": loss_vn_pre,
        "loss_kt_post_consolidation": loss_kt_post,
        "forward_ms_kt": t_kt, "forward_ms_vn": t_vn,
        "forward_ratio": t_kt / t_vn,
        "param_count": kt1.param_count(), "n_kids": n_kids,
        "record_rows_after_1_consolidation": n_rows,
        "snapshot_bytes_each": kt1.param_count() * 8,
        "verdict": "PASS" if (parity_pre and t_kt / t_vn < 2.0) else "FAIL",
    }
    print("E1 done:", out["E1_cost"]["verdict"], flush=True)

    # ---- Mainline: clean -> poison -> null -> firewall -> repair -> decay
    kt, vn = build_pair(CFG, 0)
    train_steps(kt, clean_batches)
    train_steps(vn, clean_batches)
    base_kt = summarize_probes(probe_all_facts(kt))
    base_vn = summarize_probes(probe_all_facts(vn))
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
    bench.consolidate(POISON_POOL, "consolidation:1")
    lineage = bench.lineage_query("liar")
    upstream = affected_upstream(pre_poison_w, post_poison_w, top_n=20)
    fact_rows = [("lm_head", t) for t in TRUE_TOKENS + LIAR_TOKENS]
    affected_rows = fact_rows + upstream
    vanilla_lineage = {"queryable": False,
                       "reason": "vanilla model has no records: kt_linears() == [] "
                                 "and no layer carries a KRecord; there is nothing to query"}

    # ---- E4: abstention (measured here, post-poison, pre-repair) ---------
    _probs_lx, _top1_lx, _conf_lx = probe_prefix_tokens(vn, [1, 2, 3, 4, 28, 29])
    lx_vn = {"p_liar_only": float(_probs_lx[13]), "top1": _top1_lx, "conf": _conf_lx}
    lx_quarantine = [("lm_head", t) for t in LIAR_TOKENS + [13]]
    null_rows(kt, lx_quarantine)
    lx_kt_null = abstain_predict(kt, np.array([[1, 2, 3, 4, 28, 29]]),
                                 set(LIAR_TOKENS + [13]))
    null_probes = summarize_probes(probe_all_facts(kt))
    unnull_rows(bench, lx_quarantine)
    out["E4_abstention"] = {
        "vanilla_lx": lx_vn,
        "kt_lx_post_null": {"answer": lx_kt_null[0], "conf": lx_kt_null[1],
                            "top1": lx_kt_null[2]},
        "kt_fact_probes_post_null": null_probes,
        "verdict": "PASS" if (lx_vn["top1"] == 13 and lx_vn["conf"] >= 0.6
                               and (lx_kt_null[0] == "DECLINE" or lx_kt_null[1] <= 0.5))
                   else "FAIL",
    }
    print("E4 done:", out["E4_abstention"]["verdict"], flush=True)

    # ---- Firewall demo (branch; restored afterwards) ----------------------
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
    restored_probes = summarize_probes(probe_all_facts(kt))
    restored_match = (abs(restored_probes["mean_p_true"] - pois_kt["mean_p_true"]) < 1e-9
                      and restored_probes["top1_liar_count"] == pois_kt["top1_liar_count"])

    # ---- E2: surgical repair vs full retrain ------------------------------
    _names, mask = build_repair_mask(kt, set(affected_rows))
    _nr, _pr = collect_params(kt)
    adam_rep = Adam(_pr, lr=0.01)
    repair_steps = None
    for step in range(1, REPAIR_CAP + 1):
        train_steps(kt, repair_batches[step - 1:step], mask=mask, adam=adam_rep)
        if step % 5 == 0 or step == 1:
            cur = summarize_probes(probe_all_facts(kt))
            if cur["mean_p_true"] >= base_kt["mean_p_true"] - 0.05:
                repair_steps = step
                break
    if repair_steps is None:
        repair_steps = REPAIR_CAP
    # Re-seat repaired rows with dock (direct record ops, gate-honest).
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
    post_repair_auth = bench.standing_authorities("lm_head", TRUE_TOKENS + LIAR_TOKENS)
    repaired_auth_mean = float(np.mean(list(post_repair_auth.values())))
    chain_ok, chain_bad = bench.verify_all()
    # Vanilla full retrain to the same recovery criterion.
    _nret, _pret = collect_params(vn)
    adam_ret = Adam(_pret, lr=0.01)
    retrain_steps = None
    for step in range(1, RETRAIN_CAP + 1):
        train_steps(vn, retrain_batches[step - 1:step], adam=adam_ret)
        if step % 5 == 0 or step == 1:
            cur = summarize_probes(probe_all_facts(vn))
            if cur["mean_p_true"] >= base_vn["mean_p_true"] - 0.05:
                retrain_steps = step
                break
    if retrain_steps is None:
        retrain_steps = RETRAIN_CAP
    out["E2_poisoning"] = {
        "baseline_kt": base_kt, "baseline_vn": base_vn,
        "post_poison_kt": pois_kt, "post_poison_vn": pois_vn,
        "lineage_liar_entries": [{"k": e["k"], "source": e["source"]} for e in lineage],
        "vanilla_lineage": vanilla_lineage,
        "affected_upstream_rows": upstream,
        "firewall_pot_trajectory": pot_traj,
        "firewall_probes_post_null": fw_null_probes,
        "firewall_probes_post_release": fw_end_probes,
        "firewall_pot_end": fw_pot_end,
        "snapshot_restore_exact": restored_match,
        "repair_steps": repair_steps, "full_retrain_steps": retrain_steps,
        "repair_statuses_seated": sum(1 for s in repair_statuses.values() if s == "seated"),
        "repair_statuses_total": len(repair_statuses),
        "post_repair_probes": post_repair_probes,
        "post_repair_lm_head_auth": post_repair_auth,
        "chain_ok": chain_ok, "chain_bad": chain_bad,
        "verdict": "PASS" if (pois_vn["mean_p_liar"] > pois_vn["mean_p_true"]
                               and pois_vn["top1_liar_count"] >= 3
                               and len(lineage) >= 1
                               and repair_steps < retrain_steps) else "FAIL",
    }
    print("E2 done:", out["E2_poisoning"]["verdict"], flush=True)

    # ---- E3: law at scale + decay ------------------------------------------
    pre_injury_mean = float(np.mean(list(pre_injury_auth.values())))
    traj = {"post_repair": {"probes": post_repair_probes,
                            "auth_mean": repaired_auth_mean}}
    deficit = REPAIR_DOCK
    for n in range(1, 7):
        deficit = max(0.0, REPAIR_DOCK - 0.01 * n)
        for name, row in affected_rows:
            layer = dict(bench.layers)[name]
            rec = layer.records[row]
            rec.replace("row", row_hash(layer.W[row], layer.b[row]),
                        seat_authority(REPAIR_POOL), f"decay:{n}",
                        authority_dock=deficit)
            layer.gain[row] = rec.standing("row")["authority"]
        if n in (1, 2, 6):
            auth = bench.standing_authorities("lm_head", TRUE_TOKENS + LIAR_TOKENS)
            traj[f"+{n}"] = {"probes": summarize_probes(probe_all_facts(kt)),
                             "auth_mean": float(np.mean(list(auth.values()))),
                             "deficit": deficit}
    out["E3_law_at_scale"] = {
        "pre_injury_auth_mean": pre_injury_mean,
        "trajectory": traj,
        "verdict": "PASS" if (post_repair_probes["mean_p_true"] >= base_kt["mean_p_true"] - 0.10
                               and post_repair_probes["top1_true_count"] >= base_kt["top1_true_count"]
                               and repaired_auth_mean <= pre_injury_mean - 0.1
                               and traj["+2"]["auth_mean"] < pre_injury_mean) else "FAIL",
    }
    print("E3 done:", out["E3_law_at_scale"]["verdict"], flush=True)

    # ---- E5: C2C proving ground ---------------------------------------------
    # Vanilla sender/receiver.
    sender_v = GPT(CFG, np.random.default_rng(0), Linear)
    train_steps(sender_v, clean_batches)
    sender_v_clean = summarize_probes(probe_all_facts(sender_v))
    train_steps(sender_v, poison_batches)
    sender_v_pois = summarize_probes(probe_all_facts(sender_v))
    recv_v = GPT(CFG, np.random.default_rng(7), Linear)
    recv_v_pre = summarize_probes(probe_all_facts(recv_v))
    all_rows = [(n, r) for n, layer in sender_v.all_linears() for r in range(layer.out_features)]
    transfer_rows(sender_v, recv_v, all_rows, 0.0)
    copy_nonrow_state(sender_v, recv_v)
    recv_v_post = summarize_probes(probe_all_facts(recv_v))
    _nrv, _prv = collect_params(recv_v)
    adam_rv = Adam(_prv, lr=0.01)
    retrain_v_steps = None
    for step in range(1, RETRAIN_CAP + 1):
        train_steps(recv_v, retrain_batches[step - 1:step], adam=adam_rv)
        if step % 5 == 0 or step == 1:
            cur = summarize_probes(probe_all_facts(recv_v))
            if cur["mean_p_true"] >= sender_v_clean["mean_p_true"] - 0.05:
                retrain_v_steps = step
                break
    if retrain_v_steps is None:
        retrain_v_steps = RETRAIN_CAP
    # KT sender/receiver.
    sender_k = GPT(CFG, np.random.default_rng(0), KTLinear)
    train_steps(sender_k, clean_batches)
    bench_s = Bench(sender_k)
    bench_s.consolidate(CLEAN_POOL, "consolidation:0")
    sender_k_clean = summarize_probes(probe_all_facts(sender_k))
    pre_s = capture_weights(sender_k)
    train_steps(sender_k, poison_batches)
    post_s = capture_weights(sender_k)
    bench_s.consolidate(POISON_POOL, "consolidation:1")
    sender_k_pois = summarize_probes(probe_all_facts(sender_k))
    recv_k = GPT(CFG, np.random.default_rng(7), KTLinear)
    bench_r = Bench(recv_k)
    recv_k_pre = summarize_probes(probe_all_facts(recv_k))
    transfer_rows(sender_k, recv_k, all_rows, TRANSFER_DOCK, bench=bench_r,
                  sender_sources=["liar"])
    copy_nonrow_state(sender_k, recv_k)
    recv_k_post = summarize_probes(probe_all_facts(recv_k))
    transfer_auth = bench_r.standing_authorities("lm_head", TRUE_TOKENS + LIAR_TOKENS)
    transfer_auth_mean = float(np.mean(list(transfer_auth.values())))
    lineage_r = bench_r.lineage_query("liar")
    null_rows(recv_k, [("lm_head", t) for t in LIAR_TOKENS])
    recv_k_null = summarize_probes(probe_all_facts(recv_k))
    unnull_rows(bench_r, [("lm_head", t) for t in LIAR_TOKENS])
    upstream_s = affected_upstream(pre_s, post_s, top_n=20)
    affected_r = fact_rows + upstream_s
    _n2, mask_r = build_repair_mask(recv_k, set(affected_r))
    _nrr, _prr = collect_params(recv_k)
    adam_rr = Adam(_prr, lr=0.01)
    repair_r_steps = None
    for step in range(1, REPAIR_CAP + 1):
        train_steps(recv_k, repair_batches[step - 1:step], mask=mask_r, adam=adam_rr)
        if step % 5 == 0 or step == 1:
            cur = summarize_probes(probe_all_facts(recv_k))
            if cur["mean_p_true"] >= sender_k_clean["mean_p_true"] - 0.05:
                repair_r_steps = step
                break
    if repair_r_steps is None:
        repair_r_steps = REPAIR_CAP
    for name, row in affected_r:
        layer = dict(bench_r.layers)[name]
        rec = layer.records[row]
        rec.replace("row", row_hash(layer.W[row], layer.b[row]),
                    seat_authority(REPAIR_POOL), "repair:1",
                    authority_dock=REPAIR_DOCK)
        layer.gain[row] = rec.standing("row")["authority"]
    recv_k_repaired = summarize_probes(probe_all_facts(recv_k))
    repaired_r_auth = bench_r.standing_authorities("lm_head", TRUE_TOKENS + LIAR_TOKENS)
    repaired_r_mean = float(np.mean(list(repaired_r_auth.values())))
    chain_r_ok, _ = bench_r.verify_all()
    out["E5_c2c"] = {
        "vanilla_sender_clean": sender_v_clean, "vanilla_sender_poison": sender_v_pois,
        "vanilla_receiver_pre": recv_v_pre, "vanilla_receiver_post_transfer": recv_v_post,
        "vanilla_receiver_full_retrain_steps": retrain_v_steps,
        "vanilla_lineage": vanilla_lineage,
        "kt_sender_clean": sender_k_clean, "kt_sender_poison": sender_k_pois,
        "kt_receiver_pre": recv_k_pre, "kt_receiver_post_transfer": recv_k_post,
        "kt_transfer_auth_mean": transfer_auth_mean,
        "kt_lineage_liar_entries": len(lineage_r),
        "kt_receiver_post_null": recv_k_null,
        "kt_receiver_repair_steps": repair_r_steps,
        "kt_receiver_post_repair": recv_k_repaired,
        "kt_repaired_auth_mean": repaired_r_mean,
        "kt_chain_ok": chain_r_ok,
        "verdict": "PASS" if (recv_v_post["mean_p_liar"] > recv_v_post["mean_p_true"]
                               and recv_k_null["mean_p_true"] >= sender_k_clean["mean_p_true"] - 0.10
                               and len(lineage_r) >= 1
                               and repaired_r_mean <= transfer_auth_mean - 0.1) else "FAIL",
    }
    print("E5 done:", out["E5_c2c"]["verdict"], flush=True)

    with open("experiments/results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("results.json written", flush=True)


if __name__ == "__main__":
    main()
