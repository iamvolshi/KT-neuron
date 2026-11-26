"""E3 RERUN (2026-10-26): the law at scale, with RANKING and
CONFIDENCE split into separate, independently verdictable bars.

Pre-registered in preregistration-e3-rerun-2026-10-26.md BEFORE this
script ran. Flip-to-slow is NOT engaged (round-1 behavior).
Reuses trainer.py / corpus.py / playbook.py unchanged; only this new
script is added. Writes results-e3-rerun.json (the .md is written
from that JSON after the run, copying numbers, not re-measuring).

Mainline replicates round-1 E2 (and the stage-2 E2 rerun): clean
400, poison 200, firewall demo (branch, full-state snapshot
restored afterwards), surgical repair of affected rows only with
dock 0.3, then the E3 decay: re-seat affected rows at 6 further
clean consolidations with deficit 0.30 - 0.01*n (no training steps
between decay points, so probe changes across decay points are
pure gain effects). Tracked points: baseline, post-poison,
post-repair, +1, +2, +6.
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
REPAIR_CAP = 200


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


def gains_snapshot(model):
    return {name: layer.gain.copy() for name, layer in model.kt_linears()}


def point(bench, kt, label):
    """One tracked point: probes + per-probe detail + auth + chain."""
    probes = probe_all_facts(kt)
    summ = summarize_probes(probes)
    auth = bench.standing_authorities("lm_head", TRUE_TOKENS + LIAR_TOKENS)
    chain_ok, chain_bad = bench.verify_all()
    rec = {"summary": summ,
           "per_probe": [{"p_true": p["p_true"], "p_liar": p["p_liar"],
                          "top1": p["top1"], "conf": p["conf"]}
                         for p in probes],
           "lm_head_auth": {str(k): v for k, v in auth.items()},
           "lm_head_auth_mean": float(np.mean(list(auth.values()))),
           "chain_ok": bool(chain_ok),
           "chain_bad": [[n, r, s] for n, r, s in chain_bad]}
    print(label, ":", summ, "auth_mean:", rec["lm_head_auth_mean"],
          "chain_ok:", chain_ok, flush=True)
    return rec


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
        with open("experiments/results-e3-rerun.json", "w") as f:
            json.dump(out, f, indent=2, default=str)
        return

    rng_c = np.random.default_rng(1)
    clean_batches = [gen_clean_batch(rng_c) for _ in range(CLEAN_STEPS)]
    rng_p = np.random.default_rng(2)
    poison_batches = [gen_poison_batch(rng_p) for _ in range(POISON_STEPS)]
    rng_r = np.random.default_rng(3)
    repair_batches = [gen_clean_batch(rng_r) for _ in range(REPAIR_CAP)]

    kt, vn = build_pair(CFG, 0)
    train_steps(kt, clean_batches)
    train_steps(vn, clean_batches)
    bench = Bench(kt)
    base_kt = summarize_probes(probe_all_facts(kt))
    base_vn = summarize_probes(probe_all_facts(vn))
    print("baseline kt:", base_kt, "vn:", base_vn, flush=True)
    bench.consolidate(CLEAN_POOL, "consolidation:0")
    pre_injury_auth = bench.standing_authorities(
        "lm_head", TRUE_TOKENS + LIAR_TOKENS)
    pre_injury_mean = float(np.mean(list(pre_injury_auth.values())))
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
          "params:", param_count, "lineage:", lineage, flush=True)

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
    bench.restore_snapshot("post-poison")
    bench.reset_gains_to_standing()
    post_fw_weights = capture_weights(kt)
    post_fw_gains = gains_snapshot(kt)
    fw_restore_weights_exact = all(
        np.array_equal(pre_fw_weights[n][0], post_fw_weights[n][0])
        and np.array_equal(pre_fw_weights[n][1], post_fw_weights[n][1])
        for n in pre_fw_weights)
    fw_restore_gains_exact = all(
        np.array_equal(pre_fw_gains[n], post_fw_gains[n]) for n in pre_fw_gains)
    print("firewall traj:", pot_traj, "restore weights exact:",
          fw_restore_weights_exact, "gains exact:", fw_restore_gains_exact,
          flush=True)

    # ---- Quarantine round-trip (null liar rows, probe, unnull) -------
    null_rows(kt, [("lm_head", t) for t in LIAR_TOKENS])
    q_null_probes = summarize_probes(probe_all_facts(kt))
    unnull_rows(bench, [("lm_head", t) for t in LIAR_TOKENS])
    q_restored_probes = summarize_probes(probe_all_facts(kt))
    print("quarantine null probes:", q_null_probes, "restored:",
          q_restored_probes, flush=True)

    # ---- Surgical repair, 1-step recovery checks ----------------------
    _names, mask = build_repair_mask(kt, set(affected_rows))
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
    print("repair steps:", repair_steps, "recovered:", rep_recovered, flush=True)
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
    seated_n = sum(1 for s in repair_statuses.values() if s == "seated")
    print("repair seated:", seated_n, "/", len(repair_statuses), flush=True)

    # ---- E3 tracked points --------------------------------------------
    traj = {}
    traj["baseline"] = {"summary": base_kt}
    traj["post_poison"] = {"summary": pois_kt}
    traj["post_repair"] = point(bench, kt, "post-repair")
    traj["post_repair"]["deficit"] = REPAIR_DOCK
    deficit = REPAIR_DOCK
    for n in range(1, 7):
        deficit = max(0.0, REPAIR_DOCK - 0.01 * n)
        statuses = {}
        for name, row in affected_rows:
            layer = dict(bench.layers)[name]
            rec = layer.records[row]
            status, _r = rec.replace(
                "row", row_hash(layer.W[row], layer.b[row]),
                seat_authority(REPAIR_POOL), f"decay:{n}",
                authority_dock=deficit)
            layer.gain[row] = rec.standing("row")["authority"]
            statuses[f"{name}:{row}"] = status
        if n in (1, 2, 6):
            pt = point(bench, kt, f"+{n}")
            pt["deficit"] = deficit
            pt["decay_statuses_seated"] = sum(
                1 for s in statuses.values() if s == "seated")
            pt["decay_statuses_total"] = len(statuses)
            traj[f"+{n}"] = pt
    chain_all_ok = all(traj[k]["chain_ok"] for k in
                       ("post_repair", "+1", "+2", "+6"))

    # ---- Frozen-bar verdicts ------------------------------------------
    base_count = base_kt["top1_true_count"]
    bar_r = all(traj[k]["summary"]["top1_true_count"] == base_count
                for k in ("post_repair", "+1", "+2", "+6"))
    p_rep = traj["post_repair"]["summary"]["mean_p_true"]
    p_1 = traj["+1"]["summary"]["mean_p_true"]
    p_2 = traj["+2"]["summary"]["mean_p_true"]
    p_6 = traj["+6"]["summary"]["mean_p_true"]
    base_p = base_kt["mean_p_true"]
    bar_c = bool(p_rep <= p_1 <= p_2 <= p_6 and p_6 > p_rep
                 and abs(p_6 - base_p) < abs(p_rep - base_p))
    bar_c_abs = {k: bool(traj[k]["summary"]["mean_p_true"] >= base_p - 0.10)
                 for k in ("post_repair", "+1", "+2", "+6")}
    post_auth = traj["post_repair"]["lm_head_auth_mean"]
    bar_a = bool(post_auth <= pre_injury_mean - 0.1)
    bar_d = bool(traj["+6"]["deficit"] < REPAIR_DOCK
                 and traj["+2"]["lm_head_auth_mean"] < pre_injury_mean
                 and traj["+6"]["lm_head_auth_mean"] < pre_injury_mean
                 and traj["+6"]["lm_head_auth_mean"] >= post_auth)
    verdicts = {"bar_R_ranking": "PASS" if bar_r else "FAIL",
                "bar_C_confidence_trajectory": "PASS" if bar_c else "FAIL",
                "bar_C_abs_confidence_absolute": bar_c_abs,
                "bar_A_authority_lag": "PASS" if bar_a else "FAIL",
                "bar_D_decay": "PASS" if bar_d else "FAIL",
                "chain_all_tracked_points_ok": bool(chain_all_ok),
                "trajectory_verdicts_void_on_chain_failure":
                    not chain_all_ok}
    print("VERDICTS:", verdicts, flush=True)

    out["E3_rerun"] = {
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
        "firewall_restore_weights_exact": bool(fw_restore_weights_exact),
        "firewall_restore_gains_exact": bool(fw_restore_gains_exact),
        "quarantine_null_probes": q_null_probes,
        "quarantine_restored_probes": q_restored_probes,
        "repair": {"steps_to_recovery_1step_checks": repair_steps,
                   "recovered": rep_recovered,
                   "statuses_seated": seated_n,
                   "statuses_total": len(repair_statuses),
                   "dock": REPAIR_DOCK},
        "pre_injury_lm_head_auth": {str(k): v
                                    for k, v in pre_injury_auth.items()},
        "pre_injury_lm_head_auth_mean": pre_injury_mean,
        "trajectory": traj,
        "verdicts": verdicts,
        "absolute_bar_threshold": base_p - 0.10,
    }
    with open("experiments/results-e3-rerun.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("results-e3-rerun.json written", flush=True)


if __name__ == "__main__":
    main()
