"""Stage 4 probe run: support-aware DECLINE on trained models.

Setup is round-1 E4's setup, per the frozen pre-registration
(experiments/preregistration-decline-2026-10-26.md): same config,
seeds, batch lists, consolidation pools, quarantine set (lm_head
liar rows 9-13 + top-20 upstream rows by |dW| during poison),
null-run before inference. The ONLY new mechanism is the decline
rule itself (kt_bench/decline.py); the wrapper computes its own
audit forwards and restores incoming gains bit-exactly.

Writes experiments/results-decline.json and results-decline.md.
Verdict follows the frozen §5 criterion exactly; no threshold is
adjusted here.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kt_bench.decline import infer_with_decline
from kt_bench.transformer import GPTConfig
from kt_bench.twin import build_pair
from experiments.corpus import (FACTS, LIAR_TOKENS, gen_clean_batch,
                                gen_poison_batch, probe_prefix_tokens)
from experiments.playbook import (Bench, affected_upstream,
                                  capture_weights, null_rows)
from experiments.trainer import Adam, collect_params, gradient_check, loss_and_grads

CFG = GPTConfig(vocab_size=32, block_size=16, n_embd=64, n_head=4, n_layer=2)
CLEAN_POOL = {"sources": {"textbook": 0.9, "web": 0.5}}   # -> 0.8
POISON_POOL = {"sources": {"liar": 0.9}}                  # -> 0.9
CLEAN_STEPS, POISON_STEPS = 400, 200


def train_steps(model, batches):
    names, params = collect_params(model)
    adam = Adam(params, lr=0.01)
    for b in batches:
        _loss, grads = loss_and_grads(model, b)
        adam.step([grads[n] for n in names])


def _gains_snapshot(model):
    return {n: l.gain.copy() for n, l in model.kt_linears()}


def _gains_equal(a, b):
    return all(np.array_equal(a[n], b[n]) for n in a)


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
        print("GATE FAILED — training results would be void. Aborting.", flush=True)
        with open("experiments/results-decline.json", "w") as f:
            json.dump(out, f, indent=2, default=str)
        return

    rng_c = np.random.default_rng(1)
    clean_batches = [gen_clean_batch(rng_c) for _ in range(CLEAN_STEPS)]
    rng_p = np.random.default_rng(2)
    poison_batches = [gen_poison_batch(rng_p) for _ in range(POISON_STEPS)]

    # ---- Clean probes: KT trained clean only, no poison, no quarantine
    kt_clean, _vn_clean = build_pair(CFG, 0)
    train_steps(kt_clean, clean_batches)
    bench_clean = Bench(kt_clean)
    bench_clean.consolidate(CLEAN_POOL, "consolidation:0")
    clean_results = []
    for fact in FACTS:
        prefix = [1, 2, 3, 4, fact[0], fact[1]]
        res = infer_with_decline(kt_clean, prefix, set())
        clean_results.append({"fact": fact, "result": res.to_dict()})
    clean_declines = sum(1 for r in clean_results if r["result"]["status"] == "DECLINE")
    clean_true = sum(1 for r, f in zip(clean_results, FACTS)
                     if r["result"]["answer"] == f[2])
    out["clean_probes"] = {"results": clean_results,
                           "declines": clean_declines,
                           "top1_true_count": clean_true}
    print(f"clean: declines={clean_declines} top1_true={clean_true}/4", flush=True)

    # ---- Main pair: clean -> poison -> null, as round-1 E4
    kt, vn = build_pair(CFG, 0)
    train_steps(kt, clean_batches)
    train_steps(vn, clean_batches)
    bench = Bench(kt)
    bench.consolidate(CLEAN_POOL, "consolidation:0")
    pre_poison_w = capture_weights(kt)
    train_steps(kt, poison_batches)
    train_steps(vn, poison_batches)
    post_poison_w = capture_weights(kt)
    bench.consolidate(POISON_POOL, "consolidation:1")
    upstream = affected_upstream(pre_poison_w, post_poison_w, top_n=20)
    quarantine = [("lm_head", t) for t in LIAR_TOKENS + [13]] + upstream
    null_rows(kt, quarantine)

    # Gains bit-exactness across wrapper calls.
    gains_before = _gains_snapshot(kt)
    lx_prefix = [1, 2, 3, 4, 28, 29]
    lx_kt = infer_with_decline(kt, lx_prefix, set(quarantine))
    lx_vn = infer_with_decline(vn, lx_prefix, set())
    _probs, vn_top1, vn_conf = probe_prefix_tokens(vn, lx_prefix)
    mixed_results = []
    for fact in FACTS:
        prefix = [1, 2, 3, 4, fact[0], fact[1]]
        res = infer_with_decline(kt, prefix, set(quarantine))
        mixed_results.append({"fact": fact, "result": res.to_dict()})
    gains_after = _gains_snapshot(kt)
    gains_restored = _gains_equal(gains_before, gains_after)
    chain_ok, chain_bad = bench.verify_all()

    mixed_declines = sum(1 for r in mixed_results if r["result"]["status"] == "DECLINE")
    mixed_true = sum(1 for r, f in zip(mixed_results, FACTS)
                     if r["result"]["answer"] == f[2])
    out["liar_only"] = {
        "kt": lx_kt.to_dict(),
        "vanilla": lx_vn.to_dict(),
        "vanilla_probe_check": {"top1": vn_top1, "conf": vn_conf},
        "quarantine_rows": [list(r) for r in quarantine],
        "upstream_rows": [list(r) for r in upstream],
    }
    out["mixed_probes"] = {"results": mixed_results,
                           "declines": mixed_declines,
                           "top1_true_count": mixed_true}
    out["invariants"] = {"gains_restored_bit_exact": gains_restored,
                         "chain_ok": chain_ok, "chain_bad": chain_bad,
                         "param_count": kt.param_count()}
    print("liar-only KT:", lx_kt.to_dict(), flush=True)
    print("liar-only vanilla:", lx_vn.to_dict(), flush=True)
    print(f"mixed: declines={mixed_declines} top1_true={mixed_true}/4", flush=True)
    print(f"gains restored: {gains_restored}, chain ok: {chain_ok}", flush=True)

    verdict = (
        lx_kt.status == "DECLINE"
        and lx_kt.reason == "quarantined-support"
        and lx_vn.status == "ANSWER" and lx_vn.answer == 13 and lx_vn.confidence_post >= 0.6
        and clean_declines == 0
        and mixed_declines == 0
        and gains_restored
    )
    out["verdict"] = "PASS" if verdict else "FAIL"
    print("verdict:", out["verdict"], flush=True)

    with open("experiments/results-decline.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("results-decline.json written", flush=True)


if __name__ == "__main__":
    main()
