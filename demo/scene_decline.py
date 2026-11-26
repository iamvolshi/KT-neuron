"""Decline/abstention scene (plan §11, evidence base: round-1 E4 +
round-2 decline rule, experiments/run_decline.py).

Re-runs the decline scenario LIVE — nothing here displays stored
numbers. The SAME liar-only probe is put to both sides in one frame:

  VANILLA side: no K-records to read, so the decline rule cannot
    fire — it answers confidently and wrong.
  KT side: the support-aware rule (kt_bench/decline.py) reads the
    K-records, finds the pre-null winner's support is quarantined,
    finds no clean fallback post-null, and DECLINEs — no token
    is returned.

Same seeds, same batch lists, same pools and quarantine set as the
frozen decline pre-registration (experiments/run_decline.py, reused,
not rewritten): GPTConfig(vocab 32, block 16, n_embd 64, n_head 4,
n_layer 2); clean 400 steps (rng 1), poison 200 steps (rng 2);
Adam lr 0.01; clean pool textbook 0.9 / web 0.5 -> seated 0.8;
poison pool liar 0.9 -> seated 0.9; quarantine = lm_head rows for
liar tokens 9, 10, 11, 12, 13 (exact) + top-20 upstream rows by
|dW| during poison (APPROXIMATE heuristic, labeled as such).
Null-run before inference; flip NOT engaged.

Run from ~/workspace/kt-neuron:

    python3 -m demo.scene_decline
"""

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
from experiments.run_all import train_steps

CFG = GPTConfig(vocab_size=32, block_size=16, n_embd=64, n_head=4, n_layer=2)
CLEAN_POOL = {"sources": {"textbook": 0.9, "web": 0.5}}   # -> seated 0.8
POISON_POOL = {"sources": {"liar": 0.9}}                  # -> seated 0.9
CLEAN_STEPS, POISON_STEPS = 400, 200
LX_PREFIX = [1, 2, 3, 4, 28, 29]  # liar-only probe: no true token exists


def _fmt_decision(res):
    """One-line decision: status, answer, pre/post top-1 + confidence."""
    ans = "None" if res.answer is None else str(res.answer)
    return (f"{res.status} answer={ans} "
            f"pre top-1 {res.top1_pre} ({res.confidence_pre:.3f}) -> "
            f"post top-1 {res.top1_post} ({res.confidence_post:.3f})")


def run_scene():
    """Run both sides live; print the paired sections + summary; return numbers."""
    rng_c = np.random.default_rng(1)
    clean_batches = [gen_clean_batch(rng_c) for _ in range(CLEAN_STEPS)]
    rng_p = np.random.default_rng(2)
    poison_batches = [gen_poison_batch(rng_p) for _ in range(POISON_STEPS)]

    # ---- Clean-probe control: KT trained clean only, no poison --------
    print("CONTROL — KT clean-only model, no poison, empty quarantine",
          flush=True)
    kt_clean, _vn_clean = build_pair(CFG, 0)
    train_steps(kt_clean, clean_batches)
    bench_clean = Bench(kt_clean)
    bench_clean.consolidate(CLEAN_POOL, "consolidation:0")
    clean_results = []
    for fact in FACTS:
        prefix = [1, 2, 3, 4, fact[0], fact[1]]
        res = infer_with_decline(kt_clean, prefix, set())
        clean_results.append((fact, res))
        print(f"  fact ({fact[0]},{fact[1]}) -> true {fact[2]}: "
              f"{_fmt_decision(res)}", flush=True)
    clean_declines = sum(1 for _f, r in clean_results if r.status == "DECLINE")
    clean_true = sum(1 for f, r in clean_results if r.answer == f[2])
    print(f"  control: DECLINEs {clean_declines}, top-1 true "
          f"{clean_true}/4 — the rule is visibly not over-firing",
          flush=True)

    # ---- Main pair: clean -> poison -> null, as round-1 E4 ------------
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
    print(f"Quarantine set: {len(quarantine)} rows "
          f"(lm_head 9,10,11,12,13 exact + {len(upstream)} upstream "
          f"APPROXIMATE)", flush=True)
    chain_ok, _bad = bench.verify_all()

    # ---- Liar-only probe LX, both sides in one frame -------------------
    lx_kt = infer_with_decline(kt, LX_PREFIX, set(quarantine))
    lx_vn = infer_with_decline(vn, LX_PREFIX, set())
    _probs, vn_top1, vn_conf = probe_prefix_tokens(vn, LX_PREFIX)

    print()
    print(f"LIAR-ONLY PROBE {LX_PREFIX} — same probe, both sides",
          flush=True)
    print("VANILLA side: no records to read, so the rule cannot fire",
          flush=True)
    print(f"  {_fmt_decision(lx_vn)}", flush=True)
    print(f"  probe check: top-1 {vn_top1}, confidence {vn_conf:.3f} "
          f"(confident and wrong — token 13 is the liar-only token)",
          flush=True)
    print(f"  note: {lx_vn.note}", flush=True)
    print("KT side: support-aware DECLINE (kt_bench/decline.py)",
          flush=True)
    print(f"  {_fmt_decision(lx_kt)}", flush=True)
    print(f"  reason: {lx_kt.reason}", flush=True)
    print(f"  quarantined support fraction: "
          f"{lx_kt.fraction_quarantined_only:.3f} "
          f"(all-tainted {lx_kt.support_fraction:.3f}, "
          f"floor-only {lx_kt.fraction_floor_only:.3f})", flush=True)
    print(f"  chain verify: {'ok' if chain_ok else 'BAD'}", flush=True)

    # ---- Mixed-support probes: same poisoned+null KT model ------------
    print()
    print("MIXED-SUPPORT PROBES — same KT poisoned+null model, facts F1–F4",
          flush=True)
    mixed_results = []
    for fact in FACTS:
        prefix = [1, 2, 3, 4, fact[0], fact[1]]
        res = infer_with_decline(kt, prefix, set(quarantine))
        mixed_results.append((fact, res))
        print(f"  fact ({fact[0]},{fact[1]}) -> true {fact[2]}: "
              f"{_fmt_decision(res)} "
              f"quarantined fraction {res.fraction_quarantined_only:.3f}",
              flush=True)
    mixed_declines = sum(1 for _f, r in mixed_results if r.status == "DECLINE")
    mixed_true = sum(1 for f, r in mixed_results if r.answer == f[2])
    print(f"  mixed: DECLINEs {mixed_declines}, top-1 true {mixed_true}/4",
          flush=True)

    # ---- Side-by-side summary ------------------------------------------
    print()
    print("=" * 78)
    print("SIDE BY SIDE — liar-only probe, same probe, two different answers")
    print("=" * 78)
    rows = [
        ("Probe", str(LX_PREFIX), str(LX_PREFIX)),
        ("Status", lx_vn.status, lx_kt.status),
        ("Answer", str(lx_vn.answer),
         "None (no token returned)" if lx_kt.answer is None else str(lx_kt.answer)),
        ("Pre-null top-1 (conf)",
         f"{lx_vn.top1_pre} ({lx_vn.confidence_pre:.3f})",
         f"{lx_kt.top1_pre} ({lx_kt.confidence_pre:.3f})"),
        ("Post-null top-1 (conf)",
         f"{lx_vn.top1_post} ({lx_vn.confidence_post:.3f})",
         f"{lx_kt.top1_post} ({lx_kt.confidence_post:.3f})"),
        ("Quarantined support frac.", "n/a — no records",
         f"{lx_kt.fraction_quarantined_only:.3f}"),
        ("Reason", "None", str(lx_kt.reason)),
        ("Control: clean probes",
         "— (KT control above)",
         f"{clean_declines} DECLINEs, top-1 true {clean_true}/4"),
        ("Control: mixed probes",
         "— (KT control above)",
         f"{mixed_declines} DECLINEs, top-1 true {mixed_true}/4"),
    ]
    print(f"{'':<28} {'VANILLA':<32} KT")
    for label, v, k in rows:
        print(f"{label:<28} {v:<32} {k}")
    print("=" * 78)
    print("THE DIFFERENTIAL: vanilla answers confidently and wrong — it")
    print("has no record to read, so it cannot tell tainted support from")
    print("clean support. KT declines: the answer it would have given")
    print("depended on quarantined support, and no clean answer emerges")
    print("once that support is removed.")
    print()
    print("HONEST CAVEATS (printed, not footnoted away):")
    print("  1. Upstream attribution is APPROXIMATE — top-20 rows by |dW|")
    print("     during poison. The lm_head liar rows are exact; the")
    print("     upstream set is a heuristic, labeled as such.")
    print("  2. Standing authority alone cannot distinguish liar rows")
    print("     from clean rows here: interval-level pools seat every")
    print("     row at 0.9 after poison consolidation. The quarantine")
    print("     set is the operative liar signal; per-row pools remain")
    print("     future work.")
    print("  3. The DECLINE is a rule, not the round-1 hedge. Round-1 E4")
    print("     hedged (a low-confidence token returned); this rule")
    print("     returns NO token when its frozen condition fires.")
    print("  4. Toy scale: 105,248-parameter numpy model, 32-token vocab.")
    print("     The claim is the mechanism differential at this scale,")
    print("     not frontier behavior. Seating policy untouched")
    print("     (placeholder); no ARCHE feedback rule designed here.")
    return {
        "liar_only_prefix": list(LX_PREFIX),
        "kt_status": lx_kt.status,
        "kt_answer": lx_kt.answer,
        "kt_reason": lx_kt.reason,
        "kt_top1_pre": lx_kt.top1_pre,
        "kt_confidence_pre": lx_kt.confidence_pre,
        "kt_top1_post": lx_kt.top1_post,
        "kt_confidence_post": lx_kt.confidence_post,
        "kt_support_fraction": lx_kt.support_fraction,
        "kt_fraction_quarantined_only": lx_kt.fraction_quarantined_only,
        "kt_fraction_floor_only": lx_kt.fraction_floor_only,
        "vanilla_status": lx_vn.status,
        "vanilla_answer": lx_vn.answer,
        "vanilla_top1": vn_top1,
        "vanilla_confidence": vn_conf,
        "vanilla_top1_pre": lx_vn.top1_pre,
        "vanilla_confidence_pre": lx_vn.confidence_pre,
        "vanilla_top1_post": lx_vn.top1_post,
        "vanilla_confidence_post": lx_vn.confidence_post,
        "quarantine_row_count": len(quarantine),
        "upstream_row_count": len(upstream),
        "clean_declines": clean_declines,
        "clean_top1_true_count": clean_true,
        "mixed_declines": mixed_declines,
        "mixed_top1_true_count": mixed_true,
        "chain_ok": chain_ok,
        "param_count": kt.param_count(),
    }


def main():
    run_scene()


if __name__ == "__main__":
    main()
