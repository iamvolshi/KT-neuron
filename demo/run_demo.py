"""Side-by-side C2C demo (plan §11, evidence base: round-1 E5 run 2).

Re-runs the E5 scenario LIVE — nothing here displays stored numbers.
A sender model is built, poisoned by the liar source, then its trained
state is transferred to a fresh receiver BOTH ways, with the SAME
poison:

  VANILLA side: cache-style merge (weight copy + embeddings/norms),
    no record of any kind.
  KT side: record transfer (same weight copy + K-records, transfer
    dock 0.1), then lineage query, null-run, surgical repair.

Same seeds, same batch lists, same fact-set and protocol as round-1
E5 run 2 (experiments/run_all.py E5 section, reused, not rewritten):
GPTConfig(vocab 32, block 16, n_embd 64, n_head 4, n_layer 2);
clean 400 steps, poison 200, repair/retrain to the recovery
criterion (mean P(true) >= sender-clean mean P(true) - 0.05).
Run from ~/workspace/kt-neuron:

    python3 -m demo.run_demo
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kt_bench.linear import KTLinear, Linear, row_hash
from kt_bench.seating import seat_authority
from kt_bench.transformer import GPT, GPTConfig
from experiments.corpus import (LIAR_TOKENS, TRUE_TOKENS,
                                gen_clean_batch, gen_poison_batch,
                                probe_all_facts, summarize_probes)
from experiments.playbook import (Bench, affected_upstream,
                                  build_repair_mask, capture_weights,
                                  copy_nonrow_state, null_rows,
                                  transfer_rows, unnull_rows)
from experiments.run_all import train_steps

CFG = GPTConfig(vocab_size=32, block_size=16, n_embd=64, n_head=4, n_layer=2)
CLEAN_POOL = {"sources": {"textbook": 0.9, "web": 0.5}}   # -> seated 0.8
POISON_POOL = {"sources": {"liar": 0.9}}                  # -> seated 0.9
REPAIR_POOL = {"sources": {"textbook": 0.9}}              # -> 0.9, dock 0.3 -> 0.6
REPAIR_DOCK = 0.3
TRANSFER_DOCK = 0.1
CLEAN_STEPS, POISON_STEPS = 400, 200
REPAIR_CAP, RETRAIN_CAP = 200, 400


def fmt(probes):
    """'P(true)/P(liar), top-1 true x/4' — the probe summary in one line."""
    return (f"{probes['mean_p_true']:.3f} / {probes['mean_p_liar']:.3f}  "
            f"top-1 true {probes['top1_true_count']}/4, "
            f"top-1 liar {probes['top1_liar_count']}/4")


def _train_to_recovery(model, batches, criterion, mask=None):
    """Train stepwise; return steps used at the recovery criterion."""
    from experiments.trainer import Adam, collect_params
    _names, params = collect_params(model)
    adam = Adam(params, lr=0.01)
    for step in range(1, len(batches) + 1):
        train_steps(model, batches[step - 1:step], mask=mask, adam=adam)
        if step % 5 == 0 or step == 1:
            if summarize_probes(probe_all_facts(model))["mean_p_true"] >= criterion:
                return step
    return len(batches)


def run_demo():
    """Run both sides live; print the side-by-side; return all numbers."""
    rng_c = np.random.default_rng(1)
    clean_batches = [gen_clean_batch(rng_c) for _ in range(CLEAN_STEPS)]
    rng_p = np.random.default_rng(2)
    poison_batches = [gen_poison_batch(rng_p) for _ in range(POISON_STEPS)]
    rng_r = np.random.default_rng(3)
    repair_batches = [gen_clean_batch(rng_r) for _ in range(REPAIR_CAP)]
    rng_ret = np.random.default_rng(4)
    retrain_batches = [gen_clean_batch(rng_ret) for _ in range(RETRAIN_CAP)]

    # ---- VANILLA side ---------------------------------------------------
    print("VANILLA side: sender -> receiver, cache-style merge, no record",
          flush=True)
    sender_v = GPT(CFG, np.random.default_rng(0), Linear)
    train_steps(sender_v, clean_batches)
    sender_v_clean = summarize_probes(probe_all_facts(sender_v))
    train_steps(sender_v, poison_batches)
    sender_v_pois = summarize_probes(probe_all_facts(sender_v))
    recv_v = GPT(CFG, np.random.default_rng(7), Linear)
    recv_v_pre = summarize_probes(probe_all_facts(recv_v))
    all_rows = [(n, r) for n, layer in sender_v.all_linears()
                for r in range(layer.out_features)]
    transfer_rows(sender_v, recv_v, all_rows, 0.0)
    copy_nonrow_state(sender_v, recv_v)  # full trained state (run-2 fix)
    recv_v_post = summarize_probes(probe_all_facts(recv_v))
    vanilla_has_records = len(recv_v.kt_linears()) > 0
    retrain_steps = _train_to_recovery(
        recv_v, retrain_batches, sender_v_clean["mean_p_true"] - 0.05)
    print(f"  sender clean:    {fmt(sender_v_clean)}", flush=True)
    print(f"  sender poisoned: {fmt(sender_v_pois)}", flush=True)
    print(f"  receiver pre:    {fmt(recv_v_pre)}", flush=True)
    print(f"  receiver post-transfer: {fmt(recv_v_post)}", flush=True)
    print(f"  lineage query:   NOTHING — kt_linears() == []: "
          f"{not vanilla_has_records} (no records exist to query)", flush=True)
    print(f"  only repair path: full retraining, {retrain_steps} steps",
          flush=True)

    # ---- KT side ---------------------------------------------------------
    print("KT side: sender -> receiver, record transfer", flush=True)
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
    copy_nonrow_state(sender_k, recv_k)  # full trained state (run-2 fix)
    recv_k_post = summarize_probes(probe_all_facts(recv_k))
    transfer_auth = bench_r.standing_authorities(
        "lm_head", TRUE_TOKENS + LIAR_TOKENS)
    transfer_auth_mean = float(np.mean(list(transfer_auth.values())))
    lineage = bench_r.lineage_query("liar")
    print(f"  sender clean:    {fmt(sender_k_clean)}", flush=True)
    print(f"  sender poisoned: {fmt(sender_k_pois)}", flush=True)
    print(f"  receiver pre:    {fmt(recv_k_pre)}", flush=True)
    print(f"  receiver post-transfer: {fmt(recv_k_post)}", flush=True)
    print(f"  lineage query 'liar': {len(lineage)} transfer-level entries "
          f"(one per transferred row, tagged with the transfer event)",
          flush=True)
    print(f"  transfer authority (lm_head fact rows): {transfer_auth_mean:.1f}",
          flush=True)
    null_rows(recv_k, [("lm_head", t) for t in LIAR_TOKENS])
    recv_k_null = summarize_probes(probe_all_facts(recv_k))
    print(f"  post-null (quarantine): {fmt(recv_k_null)}", flush=True)
    unnull_rows(bench_r, [("lm_head", t) for t in LIAR_TOKENS])
    upstream_s = affected_upstream(pre_s, post_s, top_n=20)
    affected_r = [("lm_head", t) for t in TRUE_TOKENS + LIAR_TOKENS] + upstream_s
    _n2, mask_r = build_repair_mask(recv_k, set(affected_r))
    repair_steps = _train_to_recovery(
        recv_k, repair_batches, sender_k_clean["mean_p_true"] - 0.05,
        mask=mask_r)
    for name, row in affected_r:  # re-seat repaired rows, dock 0.3
        layer = dict(bench_r.layers)[name]
        rec = layer.records[row]
        rec.replace("row", row_hash(layer.W[row], layer.b[row]),
                    seat_authority(REPAIR_POOL), "repair:1",
                    authority_dock=REPAIR_DOCK)
        layer.gain[row] = rec.standing("row")["authority"]
    recv_k_repaired = summarize_probes(probe_all_facts(recv_k))
    repaired_auth = bench_r.standing_authorities(
        "lm_head", TRUE_TOKENS + LIAR_TOKENS)
    repaired_auth_mean = float(np.mean(list(repaired_auth.values())))
    chain_ok, _bad = bench_r.verify_all()
    print(f"  surgical repair: {repair_steps} steps, affected rows only "
          f"({len(affected_r)} rows; all other rows frozen)", flush=True)
    print(f"  post-repair:     {fmt(recv_k_repaired)}", flush=True)
    print(f"  repaired authority: {repaired_auth_mean:.1f} vs transfer "
          f"{transfer_auth_mean:.1f} (lag "
          f"{transfer_auth_mean - repaired_auth_mean:.1f})", flush=True)
    print(f"  chain verify: {'ok' if chain_ok else 'BAD'}", flush=True)

    # ---- Side-by-side summary -------------------------------------------
    print()
    print("=" * 78)
    print("SIDE BY SIDE — same poison, same transfer, two different worlds")
    print("=" * 78)
    rows = [
        ("Sender clean (P true / P liar)", fmt(sender_v_clean), fmt(sender_k_clean)),
        ("Sender poisoned", fmt(sender_v_pois), fmt(sender_k_pois)),
        ("Receiver pre-transfer", fmt(recv_v_pre), fmt(recv_k_pre)),
        ("Receiver post-transfer", fmt(recv_v_post), fmt(recv_k_post)),
        ("Inherits the poison?", "YES — exactly the sender's numbers",
         "YES — softer: transfer dock 0.1 lowers gains to 0.8"),
        ("Lineage query for 'liar'", "NOTHING — no record exists",
         f"{len(lineage)} transfer-level entries"),
        ("Quarantine (null-run)", "impossible — nothing to null",
         fmt(recv_k_null)),
        ("Surgical repair", f"none — full retrain, {retrain_steps} steps",
         f"{repair_steps} steps, affected rows only"),
        ("After repair", "— (retrain, no authority record)",
         f"{fmt(recv_k_repaired)}; authority "
         f"{repaired_auth_mean:.1f} vs {transfer_auth_mean:.1f} (lag "
         f"{transfer_auth_mean - repaired_auth_mean:.1f})"),
        ("Chain verification", "n/a — no chain", "ok" if chain_ok else "BAD"),
    ]
    print(f"{'':<28} {'VANILLA':<32} KT")
    for label, v, k in rows:
        print(f"{label:<28} {v:<32} {k}")
    print("=" * 78)
    print("THE DIFFERENTIAL: vanilla merges the poison invisibly and can")
    print("only retrain it away. KT transfers the receipts with the weights,")
    print("finds the poison by query, quarantines it, repairs only the")
    print("affected rows — and the repaired beliefs visibly lag in")
    print("authority after behavior returns. That lag is the phenomenon.")
    return {
        "vanilla_sender_clean": sender_v_clean,
        "vanilla_sender_poison": sender_v_pois,
        "vanilla_receiver_pre": recv_v_pre,
        "vanilla_receiver_post_transfer": recv_v_post,
        "vanilla_has_records": vanilla_has_records,
        "vanilla_retrain_steps": retrain_steps,
        "kt_sender_clean": sender_k_clean,
        "kt_sender_poison": sender_k_pois,
        "kt_receiver_pre": recv_k_pre,
        "kt_receiver_post_transfer": recv_k_post,
        "kt_transfer_auth_mean": transfer_auth_mean,
        "kt_lineage_liar_entries": len(lineage),
        "kt_receiver_post_null": recv_k_null,
        "kt_receiver_repair_steps": repair_steps,
        "kt_receiver_post_repair": recv_k_repaired,
        "kt_repaired_auth_mean": repaired_auth_mean,
        "kt_authority_lag": transfer_auth_mean - repaired_auth_mean,
        "kt_chain_ok": chain_ok,
    }


def main():
    run_demo()


if __name__ == "__main__":
    main()
