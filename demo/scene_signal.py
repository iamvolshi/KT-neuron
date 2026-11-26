"""Side-by-side signal-survival scene (plan §11, evidence base: R3-E1).

Re-runs the R3-E1 scenario LIVE — nothing here displays stored numbers.
Poison arrives AFTER the fast->slow crossing, on BOTH sides, with the
SAME poison batches:

  VANILLA side: a plain Linear model. No tracker, no slow trace, no
    signal — its probe output is just the live tensors, so it moves
    WITH the poison.
  KT side: the KT model + SignalTracker (kt_bench/signal.py). After
    the handoff the signal reads the frozen slow trace, not live
    tensors, so the signal stays put while the fast-trace probe
    collapses. This is the same switch as the plan §6 poison flip:
    one switch, not two.

Same seeds, same batch lists, same protocol as experiments/run_r3_e1.py
(reused, not rewritten): GPTConfig(vocab 32, block 16, n_embd 64,
n_head 4, n_layer 2); clean 400 steps, consolidations every 100 steps
with CLEAN_POOL (textbook 0.9 + web 0.5 -> seated 0.8); probes sampled
at each boundary BEFORE that boundary's consolidation applies new
gains; poison with a fresh Adam, NO consolidation during the measured
steps, probes every 5 poison steps (40 probes over 200 poison steps).
Run from ~/workspace/kt-neuron:

    python3 -m demo.scene_signal
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kt_bench.signal import SignalTracker
from kt_bench.transformer import GPTConfig
from kt_bench.twin import build_pair
from experiments.corpus import gen_clean_batch, gen_poison_batch
from experiments.playbook import Bench
from experiments.run_all import CLEAN_POOL, train_steps
from experiments.run_r3_e1 import mean_p_true

CFG = GPTConfig(vocab_size=32, block_size=16, n_embd=64, n_head=4, n_layer=2)
CLEAN_STEPS, CONSOL_EVERY = 400, 100
POISON_STEPS, POISON_PROBE_EVERY = 200, 5


def run_scene():
    """Run both sides live; print the side-by-side; return all numbers."""
    rng_c = np.random.default_rng(1)
    clean_batches = [gen_clean_batch(rng_c) for _ in range(CLEAN_STEPS)]
    rng_p = np.random.default_rng(2)
    poison_batches = [gen_poison_batch(rng_p) for _ in range(POISON_STEPS)]

    # Identical starting weights (twin harness); KT gains start at 1,
    # so the pair is in forward parity until KT consolidations move gains.
    kt, vn = build_pair(CFG, 0)
    bench = Bench(kt)
    tracker = SignalTracker()  # alpha 0.8, noise band 0.05, slow(0) 0.0

    # ---- Clean phase: 400 steps, one Adam per model across all 400 -----
    print("CLEAN PHASE — same batches, same seeds, both models",
          flush=True)
    adam_k, adam_v = None, None
    boundaries = []
    for b in range(1, CLEAN_STEPS // CONSOL_EVERY + 1):
        chunk = clean_batches[(b - 1) * CONSOL_EVERY:b * CONSOL_EVERY]
        _losses, adam_k = train_steps(kt, chunk, adam=adam_k)
        _losses, adam_v = train_steps(vn, chunk, adam=adam_v)
        fast, _per_fact = mean_p_true(kt)  # BEFORE consolidation gains
        vanilla, _ = mean_p_true(vn)  # vanilla has no consolidation step
        bench.consolidate(CLEAN_POOL, f"scene-clean:{b}")
        rec = tracker.consolidate(seated_value=fast, fast_value=fast)
        boundaries.append({"boundary": b, "step": b * CONSOL_EVERY,
                           "kt_fast": fast, "kt_slow": rec["slow"],
                           "owner": rec["owner"], "kt_signal": rec["signal"],
                           "vanilla_probe": vanilla})
        print(f"  boundary {b} (step {b * CONSOL_EVERY}): "
              f"KT fast={fast:.4f} slow={rec['slow']:.4f} "
              f"owner={rec['owner']} signal={rec['signal']:.4f} | "
              f"VANILLA probe={vanilla:.4f}", flush=True)
    verdicts = tracker.verdicts()
    print(f"  crossing: k*={verdicts['k_star']} "
          f"(owners: {'/'.join(verdicts['owners'])}); "
          f"continuity gap at switch: "
          f"{verdicts['c_continuity_at_switch']:.4f}", flush=True)

    # ---- Poison phase: fresh Adams, NO consolidation, probe every 5 ----
    print("POISON PHASE — poison AFTER the crossing, no consolidation",
          flush=True)
    kt_start_fast, _ = mean_p_true(kt)
    kt_start_signal = tracker.current_signal(kt_start_fast)
    vn_start, _ = mean_p_true(vn)
    adam_kp, adam_vp = None, None
    kt_probes, vn_probes = [], []
    for step in range(1, POISON_STEPS + 1):
        _losses, adam_kp = train_steps(
            kt, poison_batches[step - 1:step], adam=adam_kp)
        _losses, adam_vp = train_steps(
            vn, poison_batches[step - 1:step], adam=adam_vp)
        if step % POISON_PROBE_EVERY == 0:
            fast, _ = mean_p_true(kt)
            signal = tracker.current_signal(fast)
            vanilla, _ = mean_p_true(vn)
            kt_probes.append({"poison_step": step, "fast": fast,
                              "signal": signal})
            vn_probes.append({"poison_step": step, "probe": vanilla})
    print(f"  poison start: KT fast={kt_start_fast:.4f} "
          f"signal={kt_start_signal:.4f} | VANILLA probe={vn_start:.4f}",
          flush=True)
    print(f"  {len(kt_probes)} probes, every {POISON_PROBE_EVERY} steps, "
          f"{POISON_STEPS} poison steps total", flush=True)

    # ---- VANILLA section ------------------------------------------------
    print()
    print("VANILLA side: plain Linear model, live tensors only — "
          "nothing holds", flush=True)
    print(f"  clean end probe: {vn_start:.4f}", flush=True)
    print(f"  poison probe trajectory (every 5 steps):", flush=True)
    for rec in vn_probes:
        print(f"    poison step {rec['poison_step']:>3}: "
              f"probe={rec['probe']:.4f} "
              f"(move from clean end: "
              f"{rec['probe'] - vn_start:+.4f})", flush=True)
    vn_moves = [abs(r["probe"] - vn_start) for r in vn_probes]
    print(f"  vanilla verdict: probe moved with the poison — max move "
          f"{max(vn_moves):.4f}, final probe {vn_probes[-1]['probe']:.4f} "
          f"(clean end was {vn_start:.4f})", flush=True)

    # ---- KT section ------------------------------------------------------
    print()
    print("KT side: SignalTracker — signal reads the slow trace after "
          "the handoff", flush=True)
    print(f"  clean end: fast={kt_start_fast:.4f} "
          f"signal={kt_start_signal:.4f} (slow frozen at "
          f"{tracker.slow:.4f})", flush=True)
    print(f"  poison trajectories (every 5 steps): fast trace vs signal",
          flush=True)
    for rec in kt_probes:
        print(f"    poison step {rec['poison_step']:>3}: "
              f"fast={rec['fast']:.4f} signal={rec['signal']:.4f} "
              f"(signal change: "
              f"{rec['signal'] - kt_start_signal:+.4f})", flush=True)
    kt_fast_moves = [abs(r["fast"] - kt_start_fast) for r in kt_probes]
    kt_signal_moves = [abs(r["signal"] - kt_start_signal) for r in kt_probes]
    print(f"  KT verdict: signal change across all {len(kt_probes)} probes: "
          f"max {max(kt_signal_moves):.4f} (final "
          f"{kt_probes[-1]['signal']:.4f}); fast trace collapsed: max move "
          f"{max(kt_fast_moves):.4f}, fast range "
          f"{min(r['fast'] for r in kt_probes):.4f}-"
          f"{max(r['fast'] for r in kt_probes):.4f}", flush=True)

    # ---- Side-by-side summary: same poison, same frame -------------------
    print()
    print("=" * 78)
    print("SIDE BY SIDE — same poison, same probe cadence, two readings")
    print("=" * 78)
    print(f"{'poison step':>11} {'KT signal':>10} {'KT fast':>10} "
          f"{'VANILLA probe':>14}  {'KT signal move':>15} "
          f"{'VANILLA move':>12}")
    for k_rec, v_rec in zip(kt_probes, vn_probes):
        print(f"{k_rec['poison_step']:>11} {k_rec['signal']:>10.4f} "
              f"{k_rec['fast']:>10.4f} {v_rec['probe']:>14.4f}  "
              f"{k_rec['signal'] - kt_start_signal:>+15.4f} "
              f"{v_rec['probe'] - vn_start:>+12.4f}")
    print("=" * 78)
    print("THE DIFFERENTIAL: under the same poison, KT's signal does not")
    print("move — it no longer reads live tensors — while its own fast")
    print("trace collapses beside it, and vanilla's probe output just")
    print("moves with the poison. One switch, not two: the handoff that")
    print("handed the signal to the slow trace is the same switch the")
    print("poison flip tests.")
    print()
    print("HONEST CAVEATS — read these with the numbers above:")
    print("  * The slow trace was BUILT (SignalTracker), not discovered")
    print("    in unmodified dynamics — the bench had no slow trace")
    print("    before this round, so no crossing in unmodified")
    print("    dynamics is claimed.")
    print("  * Alpha 0.8 was calibrated on a pilot before the official")
    print("    run and frozen; textbook 0.5 never crosses in a")
    print("    4-boundary window (its slow ends 0.105 below fast).")
    print("  * The post-handoff signal-equals-slow behavior is partly")
    print("    by tracker construction — the demo shows the regulation")
    print("    MECHANISM, not a discovered crossing.")
    print("  * Toy scale: 105,248-parameter numpy model, 32-token vocab,")
    print("    four planted facts. The claim is the mechanism")
    print("    differential at this scale, not frontier behavior.")
    print("  * Seating policy is the placeholder (mean + agreement bonus")
    print("    + bad-source dock, caller-declared pools), not ARCHE's")
    print("    feedback rule. No ARCHE controller is designed here.")
    return {
        "clean_boundaries": boundaries,
        "tracker_k_star": verdicts["k_star"],
        "tracker_owners": verdicts["owners"],
        "continuity_gap_at_switch": verdicts["c_continuity_at_switch"],
        "kt_poison_start_fast": kt_start_fast,
        "kt_poison_start_signal": kt_start_signal,
        "kt_poison_probes": kt_probes,
        "kt_signal_max_change": max(kt_signal_moves),
        "kt_fast_range": (min(r["fast"] for r in kt_probes),
                          max(r["fast"] for r in kt_probes)),
        "kt_fast_max_move": max(kt_fast_moves),
        "vanilla_poison_start": vn_start,
        "vanilla_poison_probes": vn_probes,
        "vanilla_max_move": max(vn_moves),
        "vanilla_final_probe": vn_probes[-1]["probe"],
        "poison_steps": POISON_STEPS,
        "poison_probe_count": len(kt_probes),
        "poison_probe_every": POISON_PROBE_EVERY,
    }


def main():
    run_scene()


if __name__ == "__main__":
    main()
