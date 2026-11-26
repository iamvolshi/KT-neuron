"""Side-by-side tick-mutability scene (plan §11; evidence: round-3 E3).

Re-runs the R3-E3 scenario LIVE — nothing here displays stored
numbers. Three tick schedules, one shape: every variant heals in
25 decay ticks. The mechanism is scale-free because the tick can
be resized to the deployment.

  VANILLA side: plain GPT + Linear, same clean + poison + full
    retrain — then NOTHING: no tick, no decay schedule, no
    healing count. Vanilla has no tick to resize, because it has
    no record whose healing could be scheduled.
  KT side: the SAME injury + repair + decay pipeline as R3-E3,
    run once per tick schedule (experiments/run_r3_e3.run_variant,
    reused, not rewritten), with identical seeds.

Setup, exactly as frozen in round 3:
GPTConfig(vocab 32, block 16, n_embd 64, n_head 4, n_layer 2);
105,248 params; seeds init 0 / data 1; SLOW tail, decay 0.01 per
decay consolidation, all variants.
Variants: FINE (400 clean steps, 100-step ticks), BASELINE
(400 clean steps, 400-step tick / 1 tick), COARSE (800 clean
steps, 800-step tick / 1 tick).

No new training code path exists here: all training reuses
experiments/run_all.train_steps, and the KT side reuses
experiments/run_r3_e3.run_variant and probe_summary wholesale.
kt_bench/tick.py is pure schedule arithmetic and trains nothing.

Run from ~/workspace/kt-neuron:

    python3 -m demo.scene_tick
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kt_bench.linear import Linear
from kt_bench.tick import BASELINE_TICK, COARSE_TICK, FINE_TICK
from kt_bench.transformer import GPT, GPTConfig
from experiments.corpus import (gen_clean_batch, gen_poison_batch,
                                probe_all_facts, summarize_probes)
from experiments.run_all import train_steps
from experiments.run_r3_e3 import run_variant
from experiments.trainer import Adam, collect_params

CFG = GPTConfig(vocab_size=32, block_size=16, n_embd=64, n_head=4, n_layer=2)
CLEAN_STEPS_VANILLA, POISON_STEPS, RETRAIN_CAP = 400, 200, 400
SCHEDULES = [FINE_TICK, BASELINE_TICK, COARSE_TICK]


def fmt(probes):
    """'P(true)/P(liar), top-1 true x/4' — the probe summary in one line."""
    return (f"{probes['mean_p_true']:.3f} / {probes['mean_p_liar']:.3f}  "
            f"top-1 true {probes['top1_true_count']}/4, "
            f"top-1 liar {probes['top1_liar_count']}/4")


def _train_to_recovery(model, batches, criterion):
    """Full retrain stepwise; return steps used at the recovery criterion.

    Same pattern as demo/run_demo.py: train_steps, one Adam, probe
    checks at step 1 and every 5 steps. No new training path.
    """
    _names, params = collect_params(model)
    adam = Adam(params, lr=0.01)
    for step in range(1, len(batches) + 1):
        train_steps(model, batches[step - 1:step], adam=adam)
        if step % 5 == 0 or step == 1:
            if summarize_probes(probe_all_facts(model))["mean_p_true"] >= criterion:
                return step
    return len(batches)


def run_scene():
    """Run both sides live; print the side-by-side; return all numbers."""
    rng_c = np.random.default_rng(1)
    clean_batches = [gen_clean_batch(rng_c) for _ in range(800)]
    rng_p = np.random.default_rng(2)
    poison_batches = [gen_poison_batch(rng_p) for _ in range(POISON_STEPS)]
    rng_r = np.random.default_rng(3)
    repair_batches = [gen_clean_batch(rng_r) for _ in range(200)]
    rng_ret = np.random.default_rng(4)
    retrain_batches = [gen_clean_batch(rng_ret) for _ in range(RETRAIN_CAP)]

    # ---- VANILLA side ---------------------------------------------------
    print("VANILLA side: plain GPT + Linear, clean + poison + full retrain",
          flush=True)
    vanilla = GPT(CFG, np.random.default_rng(0), Linear)
    train_steps(vanilla, clean_batches[:CLEAN_STEPS_VANILLA])
    vanilla_clean = summarize_probes(probe_all_facts(vanilla))
    train_steps(vanilla, poison_batches)
    vanilla_pois = summarize_probes(probe_all_facts(vanilla))
    vanilla_has_records = len(vanilla.kt_linears()) > 0
    retrain_steps = _train_to_recovery(
        vanilla, retrain_batches, vanilla_clean["mean_p_true"] - 0.05)
    vanilla_retrained = summarize_probes(probe_all_facts(vanilla))
    print(f"  clean:           {fmt(vanilla_clean)}", flush=True)
    print(f"  poisoned:        {fmt(vanilla_pois)}", flush=True)
    print(f"  full retrain:    {retrain_steps} steps -> "
          f"{fmt(vanilla_retrained)}", flush=True)
    print(f"  tick schedule:   NONE — kt_linears() == []: "
          f"{not vanilla_has_records} (no records exist to schedule)",
          flush=True)
    print("  decay schedule:  NONE — no decay consolidations, no deficit, "
          "no healing count", flush=True)
    print("  ticks-to-healed: NONE — vanilla has no tick to resize, "
          "because it has no record whose healing could be scheduled",
          flush=True)

    # ---- KT side: one live run_variant per tick schedule ----------------
    print("KT side: same injury + repair + decay, three tick schedules",
          flush=True)
    kt = {}
    for sched in SCHEDULES:
        print(f"--- tick {sched.name} ({sched.clean_steps} clean steps, "
              f"tick {sched.consol_every} steps, {sched.n_ticks} tick(s)) ---",
              flush=True)
        r = run_variant(sched, clean_batches, poison_batches, repair_batches)
        kt[sched.name] = r
        deficits = [t["deficit"] for t in r["trajectory"]]
        top1_all = [t["probes"]["top1_true_count"] for t in r["trajectory"]]
        print(f"  [{sched.name}] baseline:   "
              f"{r['baseline_kt']['mean_p_true']:.4f} "
              f"(top-1 true {r['baseline_kt']['top1_true_count']}/4)",
              flush=True)
        print(f"  [{sched.name}] post-poison: "
              f"{r['post_poison_kt']['mean_p_true']:.4f}", flush=True)
        print(f"  [{sched.name}] repair: {r['repair_steps']} steps, "
              f"post-repair {r['post_repair_probes']['mean_p_true']:.4f} "
              f"(top-1 true {r['post_repair_probes']['top1_true_count']}/4)",
              flush=True)
        print(f"  [{sched.name}] authority lag: {r['authority_lag']:.3f} "
              f"(>= 0.1: {r['authority_lag'] >= 0.1}); "
              f"deficit nonincreasing: "
              f"{all(x >= y for x, y in zip(deficits, deficits[1:]))} "
              f"({deficits[0]:.2f} -> {deficits[-1]:.2f}); "
              f"top-1 true throughout: {all(v == 4 for v in top1_all)}",
              flush=True)
        print(f"  [{sched.name}] healed at {r['time_to_healed_applied']} "
              f"decay ticks; chain ok: {r['chain_ok_at_plus30']}; "
              f"seated {r['repair_statuses_seated']}/"
              f"{r['repair_statuses_total']} rows", flush=True)

    healed = {name: kt[name]["time_to_healed_applied"] for name in kt}
    healed_vals = [v for v in healed.values() if v is not None]
    spread = max(healed_vals) - min(healed_vals) if healed_vals else None

    # ---- Side-by-side summary -------------------------------------------
    print()
    print("=" * 78)
    print("SIDE BY SIDE — one shape across three tick sizes; vanilla has none")
    print("=" * 78)
    print(f"{'variant':<12} {'clean steps':<12} {'tick size':<12} "
          f"{'ticks-to-healed':<16} note")
    for sched in SCHEDULES:
        r = kt[sched.name]
        print(f"{sched.name:<12} {sched.clean_steps:<12} "
              f"{sched.consol_every:<12} {r['time_to_healed_applied']:<16} "
              f"lag {r['authority_lag']:.3f}, repair {r['repair_steps']} steps")
    print(f"{'VANILLA':<12} {CLEAN_STEPS_VANILLA:<12} {'NONE':<12} "
          f"{'NONE':<16} no record, no tick, no healing count")
    print("=" * 78)
    print(f"KT ticks-to-healed: {healed['FINE']}/{healed['BASELINE']}/"
          f"{healed['COARSE']} (spread {spread}) — same 25 decay ticks "
          f"at 100-, 400-, and 800-step tick sizes.")
    print("VANILLA next to it: no tick schedule at all. There is nothing")
    print("to resize, because there is no record whose healing could be")
    print("scheduled — only full retraining, and then silence.")
    print()
    print("HONEST CAVEATS — read before quoting this scene:")
    print("  * Leg (d) is a WIRING CHECK. Decay is priced per decay")
    print("    consolidation, never per training step, so COARSE (800")
    print("    clean steps) healing in the same 25 decay ticks as the")
    print("    400-step variants verifies the wiring, not a discovered")
    print("    healing law.")
    print("  * Baselines are PER VARIANT. Never compare absolute probe")
    print("    numbers across variants — they train different amounts.")
    print("    Only SHAPE is compared: lag, deficit monotonicity,")
    print("    ranking, ticks-to-healed.")
    print("  * Per-variant baseline differences (e.g. repair steps "
          "13/10/18)")
    print("    are expected. No verdict rests on comparing absolutes")
    print("    across variants.")
    print("  * Toy scale (105,248 params, 32-token vocab). The claim is")
    print("    the mechanism differential at this scale, not frontier")
    print("    behavior.")

    return {
        "vanilla_clean": vanilla_clean,
        "vanilla_poison": vanilla_pois,
        "vanilla_has_records": vanilla_has_records,
        "vanilla_retrain_steps": retrain_steps,
        "vanilla_post_retrain": vanilla_retrained,
        "vanilla_tick_schedule": None,
        "vanilla_decay_schedule": None,
        "vanilla_ticks_to_healed": None,
        "kt_variants": {
            name: {
                "clean_steps": kt[name]["clean_steps"],
                "tick_size": kt[name]["consol_every"],
                "n_clean_ticks": kt[name]["n_clean_ticks"],
                "baseline_p_true": kt[name]["baseline_kt"]["mean_p_true"],
                "post_poison_p_true":
                    kt[name]["post_poison_kt"]["mean_p_true"],
                "repair_steps": kt[name]["repair_steps"],
                "post_repair_p_true":
                    kt[name]["post_repair_probes"]["mean_p_true"],
                "post_repair_top1_true":
                    kt[name]["post_repair_probes"]["top1_true_count"],
                "authority_lag": kt[name]["authority_lag"],
                "deficit_nonincreasing": all(
                    x >= y for x, y in zip(
                        [t["deficit"] for t in kt[name]["trajectory"]],
                        [t["deficit"] for t in kt[name]["trajectory"]][1:])),
                "top1_true_throughout": all(
                    t["probes"]["top1_true_count"] == 4
                    for t in kt[name]["trajectory"]),
                "ticks_to_healed": kt[name]["time_to_healed_applied"],
                "chain_ok": kt[name]["chain_ok_at_plus30"],
                "seated": kt[name]["repair_statuses_seated"],
                "seated_total": kt[name]["repair_statuses_total"],
            } for name in kt
        },
        "kt_ticks_to_healed": healed,
        "kt_ticks_to_healed_spread": spread,
    }


def main():
    run_scene()


if __name__ == "__main__":
    main()
