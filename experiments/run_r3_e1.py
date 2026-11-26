"""R3-E1 official run: one-signal regulation (plan §7).

Frozen contract: experiments/preregistration-round3-2026-10-26.md,
R3-E1 section. Frozen thresholds (a)-(e) are evaluated, never moved.

Design (build + demonstrate, NOT discovery -- stated plainly):
the bench before this round had NO slow-accumulation dynamics, so
this run builds the slow trace with kt_bench.signal.SignalTracker
and demonstrates the regulation mechanism. No crossing in
unmodified dynamics is claimed: there was no slow trace to cross.

Run from ~/workspace/kt-neuron:
    python3 experiments/run_r3_e1.py
Writes experiments/results-r3-e1.json and results-r3-e1.md.

Method, exactly:
- Gradient gate FIRST (experiments/trainer.gradient_check); abort
  on FAIL. No new training code path exists here: clean and poison
  training reuse experiments/run_all.train_steps (manual backprop +
  Adam, numpy-only) with a single Adam per phase, as in round 1.
- Clean phase: 400 steps, seeds init 0 / data 1 (round-1 batches),
  one Adam across all 400 steps. At each 100-step boundary, fast(k)
  = live-tensor probe mean P(true) on the 4 clean facts, sampled
  BEFORE that boundary's consolidation applies new gains. Then
  Bench.consolidate(CLEAN_POOL) seats, and the tracker records the
  boundary with seated(k) = fast(k).
- Poison phase (flip corollary, leg (e)): poison introduced AFTER
  k*, with a fresh Adam (round-1 style), probed every 5 steps, and
  NO consolidation before/during the measured steps -- so the
  signal reading is the frozen slow value throughout. Stops at the
  first probe where the fast-trace probe has moved >= 0.10 from
  the poison-start probe (cap 200 poison steps).
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from kt_bench.linear import KTLinear
from kt_bench.signal import SignalTracker
from kt_bench.transformer import GPT, GPTConfig
from experiments.corpus import (gen_clean_batch, gen_poison_batch,
                                probe_all_facts, summarize_probes)
from experiments.playbook import Bench
from experiments.run_all import CLEAN_POOL, train_steps
from experiments.trainer import gradient_check

CFG = GPTConfig(vocab_size=32, block_size=16, n_embd=64, n_head=4, n_layer=2)
CLEAN_STEPS, CONSOL_EVERY = 400, 100
POISON_CAP, POISON_PROBE_EVERY = 200, 5


def mean_p_true(model):
    probes = probe_all_facts(model)
    return summarize_probes(probes)["mean_p_true"], [p["p_true"] for p in probes]


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
    if not gate_pass:
        print("GATE FAILED -- aborting.", flush=True)
        return out

    # ---- Fixed batch lists (round-1 seeds) -------------------------------
    rng_c = np.random.default_rng(1)
    clean_batches = [gen_clean_batch(rng_c) for _ in range(CLEAN_STEPS)]
    rng_p = np.random.default_rng(2)
    poison_batches = [gen_poison_batch(rng_p) for _ in range(POISON_CAP)]

    # ---- Clean phase: 400 steps, consolidations every 100 ----------------
    kt = GPT(CFG, np.random.default_rng(0), KTLinear)
    bench = Bench(kt)
    tracker = SignalTracker()  # alpha 0.8, noise band 0.05, slow(0) 0.0
    adam = None
    boundaries = []
    for b in range(1, CLEAN_STEPS // CONSOL_EVERY + 1):
        chunk = clean_batches[(b - 1) * CONSOL_EVERY:b * CONSOL_EVERY]
        _losses, adam = train_steps(kt, chunk, adam=adam)
        fast, per_fact = mean_p_true(kt)  # BEFORE consolidation gains
        bench.consolidate(CLEAN_POOL, f"r3e1-clean:{b}")
        post, _ = mean_p_true(kt)  # after consolidation, for the record
        rec = tracker.consolidate(seated_value=fast, fast_value=fast)
        boundaries.append({"boundary": b, "step": b * CONSOL_EVERY,
                           "fast_pre_consolidation": fast,
                           "fast_per_fact": per_fact,
                           "fast_post_consolidation": post,
                           "slow": rec["slow"], "owner": rec["owner"],
                           "signal": rec["signal"]})
        print(f"boundary {b} (step {b * CONSOL_EVERY}): fast={fast:.6f} "
              f"slow={rec['slow']:.6f} owner={rec['owner']}", flush=True)
    out["clean_boundaries"] = boundaries
    out["tracker_verdicts_a_to_d"] = tracker.verdicts()
    print("verdicts (a)-(d):", out["tracker_verdicts_a_to_d"], flush=True)

    # ---- Poison phase: flip corollary (leg (e)) --------------------------
    # Signal at poison start = tracker reading after the clean phase
    # (slow-owned if handed off; slow value frozen, no consolidation
    # happens during this phase, so it cannot move).
    poison_start_fast, _ = mean_p_true(kt)
    poison_start_signal = tracker.current_signal(poison_start_fast)
    adam_p = None
    qualifying = None
    poison_probes = []
    for step in range(1, POISON_CAP + 1):
        _losses, adam_p = train_steps(kt, poison_batches[step - 1:step],
                                      adam=adam_p)
        if step % POISON_PROBE_EVERY == 0:
            fast, _ = mean_p_true(kt)
            signal = tracker.current_signal(fast)
            poison_probes.append({"poison_step": step, "fast": fast,
                                  "signal": signal,
                                  "fast_move": abs(fast - poison_start_fast),
                                  "signal_change": abs(signal - poison_start_signal)})
            print(f"poison step {step}: fast={fast:.6f} "
                  f"signal={signal:.6f} fast_move={abs(fast - poison_start_fast):.6f}",
                  flush=True)
            if qualifying is None and abs(fast - poison_start_fast) >= 0.10:
                qualifying = poison_probes[-1]
    out["poison_start"] = {"fast": poison_start_fast,
                           "signal": poison_start_signal,
                           "tracker_k_star": tracker.k_star,
                           "tracker_slow": tracker.slow}
    out["poison_probes"] = poison_probes
    if tracker.k_star is not None and qualifying is not None:
        out["flip_corollary_e"] = SignalTracker.flip_corollary(
            poison_start_signal, qualifying["signal"],
            poison_start_fast, qualifying["fast"])
        out["flip_corollary_e"]["qualifying_poison_step"] = qualifying["poison_step"]
    elif tracker.k_star is None:
        out["flip_corollary_e"] = {"pass": False,
                                   "reason": "no k*: poison-after-handoff "
                                             "corollary not evaluable"}
    else:
        out["flip_corollary_e"] = {"pass": False,
                                   "reason": "fast probe never moved >= 0.10 "
                                             "within 200 poison steps"}
    print("flip corollary (e):", out["flip_corollary_e"], flush=True)

    v = out["tracker_verdicts_a_to_d"]
    e_pass = bool(out["flip_corollary_e"].get("pass"))
    overall = bool(v["a_crossing_in_clean_phase"] and v["b_single_switch"]
                   and v["c_pass"] and v["d_pass"] and e_pass)
    out["verdict"] = "PASS" if overall else "FAIL"
    out["accumulation_formula"] = (
        "slow(0) = 0.0; slow(k) = slow(k-1) + 0.8 * max(0, seated(k) "
        "- slow(k-1)), updated ONLY at consolidation; seated(k) = "
        "fast(k) = live-tensor probe mean P(true) on the 4 clean "
        "facts sampled at boundary k before consolidation gains.")
    print("OVERALL:", out["verdict"], flush=True)

    exp_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(exp_dir, "results-r3-e1.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("results-r3-e1.json written", flush=True)
    return out


if __name__ == "__main__":
    main()
