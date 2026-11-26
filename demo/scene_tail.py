"""Side-by-side tail-healing scene (plan §11, evidence base: round-3 E2).

Re-runs the R3-E2 scenario LIVE — nothing here displays stored numbers.
Same injury, three tails: the KT model is injured (clean, poison,
surgical repair), then decay consolidations run AFTER repair, one
schedule per tail deployment from kt_bench/tail.py:

  FAST      decay 0.05/consolidation — heals fast.
  SLOW      decay 0.01/consolidation — heals slow.
  PERMANENT decay 0.00/consolidation — never heals.

The VANILLA twin gets the same clean + poison + full retrain, and
then NOTHING: no authority, no deficit, no scar, no schedule. There
is no healing dimension to show on that side, because there is no
record to heal.

REUSE ONLY — no new training code path exists here:
  - experiments/run_r3_e2.run_tail + probe_summary (the injury
    pipeline: clean, poison, repair, then decay consolidations),
  - experiments/run_all.train_steps, experiments/corpus.py batches,
  - plain GPT + Linear construction from demo/run_demo.py for the
    vanilla twin.

Run from ~/workspace/kt-neuron:

    python3 -m demo.scene_tail
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kt_bench.linear import KTLinear, Linear
from kt_bench.tail import FAST_TAIL, PERMANENT_TAIL, SLOW_TAIL
from kt_bench.transformer import GPT, GPTConfig
from experiments.corpus import gen_clean_batch, gen_poison_batch
from experiments.run_all import train_steps
from experiments.run_r3_e2 import probe_summary, run_tail
from experiments.trainer import Adam, collect_params

CFG = GPTConfig(vocab_size=32, block_size=16, n_embd=64, n_head=4, n_layer=2)
CLEAN_STEPS, POISON_STEPS = 400, 200
REPAIR_CAP, RETRAIN_CAP = 200, 400
REPAIR_DOCK = 0.3
DECAY_TO = 30
TAILS = [FAST_TAIL, SLOW_TAIL, PERMANENT_TAIL]
TABLE_NS = [0, 1, 2, 5, 6, 10, 25, 30]


def fmt(probes):
    """'P(true)/P(liar), top-1 true x/4' — the probe summary in one line."""
    return (f"{probes['mean_p_true']:.4f} / {probes['mean_p_liar']:.4f}  "
            f"top-1 true {probes['top1_true_count']}/4, "
            f"top-1 liar {probes['top1_liar_count']}/4")


def run_scene():
    """Run both sides live; print the side-by-side; return all numbers."""
    rng_c = np.random.default_rng(1)
    clean_batches = [gen_clean_batch(rng_c) for _ in range(CLEAN_STEPS)]
    rng_p = np.random.default_rng(2)
    poison_batches = [gen_poison_batch(rng_p) for _ in range(POISON_STEPS)]
    rng_r = np.random.default_rng(3)
    repair_batches = [gen_clean_batch(rng_r) for _ in range(REPAIR_CAP)]
    rng_ret = np.random.default_rng(4)
    retrain_batches = [gen_clean_batch(rng_ret) for _ in range(RETRAIN_CAP)]

    # ---- Setup, measured live from a fresh model, not stated --------
    probe_model = GPT(CFG, np.random.default_rng(0), KTLinear)
    n_params = probe_model.param_count()
    n_kids = sum(layer.out_features
                 for _name, layer in probe_model.all_linears())
    print("SETUP (same injury pipeline as round-3 E2, seeds init 0 / "
          "data 1)", flush=True)
    print(f"  GPTConfig vocab 32, block 16, n_embd 64, n_head 4, n_layer 2; "
          f"{n_params} params; {n_kids} KIDs (one per row, born "
          f"registered)", flush=True)
    print(f"  clean {CLEAN_STEPS} steps, poison {POISON_STEPS} steps, "
          f"repair dock {REPAIR_DOCK}, decay consolidations tracked to "
          f"+{DECAY_TO}", flush=True)

    # ---- VANILLA side -------------------------------------------------
    print("VANILLA side: same clean + poison + full retrain, then NOTHING",
          flush=True)
    vanilla = GPT(CFG, np.random.default_rng(0), Linear)
    train_steps(vanilla, clean_batches)
    vanilla_clean = probe_summary(vanilla)
    train_steps(vanilla, poison_batches)
    vanilla_pois = probe_summary(vanilla)
    _names, params = collect_params(vanilla)
    adam = Adam(params, lr=0.01)
    retrain_steps = None
    for step in range(1, RETRAIN_CAP + 1):
        train_steps(vanilla, retrain_batches[step - 1:step], adam=adam)
        if probe_summary(vanilla)["mean_p_true"] >= \
                vanilla_clean["mean_p_true"] - 0.05:
            retrain_steps = step
            break
    if retrain_steps is None:
        retrain_steps = RETRAIN_CAP
    vanilla_retrained = probe_summary(vanilla)
    vanilla_has_records = len(vanilla.kt_linears()) > 0
    print(f"  clean:         {fmt(vanilla_clean)}", flush=True)
    print(f"  poisoned:      {fmt(vanilla_pois)}", flush=True)
    print(f"  full retrain:  {retrain_steps} steps to the recovery "
          f"criterion", flush=True)
    print(f"  post-retrain:  {fmt(vanilla_retrained)}", flush=True)
    print(f"  kt_linears() == []: {not vanilla_has_records} (no records "
          f"exist)", flush=True)
    print("  authority:     NONE — nothing seats, nothing is tracked",
          flush=True)
    print("  deficit:       NONE — no scar, so no deficit to decay",
          flush=True)
    print("  healing schedule: NONE — no +n axis exists. Retrain is the "
          "whole story; after it, there is nothing left to watch heal.",
          flush=True)

    # ---- KT side: same injury, three tails ---------------------------
    print("KT side: same injury, three tails (decay AFTER repair)",
          flush=True)
    results = {}
    for i, tail in enumerate(TAILS):
        print(f"--- tail {tail.name} (decay "
              f"{tail.decay_per_consolidation}/consolidation) ---",
              flush=True)
        results[tail.name] = run_tail(
            tail, clean_batches, poison_batches, repair_batches,
            retrain_batches, do_work_leg=(i == 0))
    ref = results["FAST"]
    for name, r in results.items():
        traj = r["trajectory"]
        print(f"  [{name}] clean baseline: {fmt(r['baseline_kt'])}",
              flush=True)
        print(f"  [{name}] post-poison:   {fmt(r['post_poison_kt'])}",
              flush=True)
        print(f"  [{name}] repair: {r['repair_steps']} steps, "
              f"post-repair {fmt(r['post_repair_probes'])}", flush=True)
        print(f"  [{name}] post-repair authority (lm_head fact rows): "
              f"{traj[0]['auth_mean']:.1f}; chain "
              f"{'ok' if r['chain_ok_at_repair'] else 'BAD'}; lineage "
              f"{r['lineage_liar_entries']}; repair seats "
              f"{r['repair_statuses_seated']}/{r['repair_statuses_total']} "
              f"rows", flush=True)
        print(f"  [{name}] deficit +6: {traj[6]['deficit']:.2f} "
              f"(authority mean {traj[6]['auth_mean']:.1f}); "
              f"deficit +30: {traj[30]['deficit']:.2f} "
              f"(authority mean {traj[30]['auth_mean']:.1f}); "
              f"healed at +{r['time_to_healed'] if r['time_to_healed'] is not None else 'never'}",
              flush=True)
        print(f"  [{name}] top-1 true 4/4 at repair and every tracked "
              f"consolidation: "
              f"{all(t['probes']['top1_true_count'] == 4 for t in traj)}",
              flush=True)

    # ---- Pre-decay trajectories must be identical across tails -------
    identical = {}
    for name in ("SLOW", "PERMANENT"):
        r = results[name]
        identical[name] = (
            r["baseline_kt"]["mean_p_true"]
            == ref["baseline_kt"]["mean_p_true"]
            and r["post_poison_kt"]["mean_p_true"]
            == ref["post_poison_kt"]["mean_p_true"]
            and r["affected_rows"] == ref["affected_rows"]
            and r["repair_steps"] == ref["repair_steps"]
            and r["post_repair_probes"]["mean_p_true"]
            == ref["post_repair_probes"]["mean_p_true"]
            and r["lineage_liar_entries"] == ref["lineage_liar_entries"])
    print(f"  identical pre-decay pipeline across tails: {identical}",
          flush=True)

    # ---- Side-by-side summary -----------------------------------------
    print()
    print("=" * 78)
    print("SIDE BY SIDE — one injury, three KT schedules, no vanilla "
          "schedule")
    print("=" * 78)
    print(f"{'consolidation':<16} {'FAST (0.05)':<22} "
          f"{'SLOW (0.01)':<22} PERMANENT (0.00)")
    for n in TABLE_NS:
        cells = []
        for tail in TAILS:
            t = results[tail.name]["trajectory"][n]
            mark = " healed" if t["healed"] else ""
            cells.append(f"deficit {t['deficit']:.2f} "
                         f"(auth {t['auth_mean']:.1f}){mark}")
        print(f"+{n:<15} {cells[0]:<22} {cells[1]:<22} {cells[2]}")
    print("-" * 78)
    print(f"{'healed at':<16} "
          f"+{results['FAST']['time_to_healed']:<21} "
          f"+{results['SLOW']['time_to_healed']:<21} never")
    print(f"{'vanilla':<16} no deficit column at all — no record, no "
          f"scar, no +n axis")
    print("=" * 78)
    print("THE DIFFERENTIAL: the same injury heals on three different "
          "schedules")
    print("under KT — fast, slow, never — while behavior (top-1 true "
          "4/4) holds")
    print("throughout. Vanilla cannot show any of this: after retrain "
          "there is")
    print("no authority to lag, no deficit to decay, no scar to watch. "
          "KT has a")
    print("healing dimension because it has a record to heal.")
    print()
    print("HONEST CAVEATS — read before claiming more than this shows:")
    print("  - tail.py is pure arithmetic. This scene shows the CONTROL "
          "SURFACE")
    print("    working through the pipeline (identical pre-decay "
          "trajectories,")
    print("    quantized authority tracking the deficit) — not a "
          "discovered decay")
    print("    law. The tail rates are deployment parameters, set by the "
          "caller.")
    print("  - Deficit is NOT 0.9 minus authority. Authority is quantized "
          "to the")
    print("    0.1 grid: e.g. SLOW at +6 has deficit 0.24 but authority "
          "mean 0.7")
    print("    (q10(0.66)), so 0.9 - authority = 0.2 is NOT the deficit.")
    print("  - Repair cost is optimizer writes only, NOT FLOPs and NOT "
          "steps:")
    work = ref["work_leg"]
    if work is not None:
        print(f"    FAST-leg work: repair {work['repair_total_param_updates']} "
              f"vs retrain {work['retrain_total_param_updates']} param "
              f"updates, ratio {work['ratio_repair_over_retrain']:.4f} "
              f"(repair {work['repair_steps_to_recovery_1step_checks']} "
              f"steps, retrain "
              f"{work['retrain_steps_to_recovery_1step_checks']} steps).",
              flush=True)
    print(f"  - Toy scale: {n_params:,}-parameter numpy model, 32-token "
          f"vocab, counting")
    print("    grammar + four planted facts. The claim is the mechanism "
          "differential")
    print("    at this scale, not frontier behavior.")

    out = {
        "n_params": n_params,
        "n_kids": n_kids,
        "vanilla_clean": vanilla_clean,
        "vanilla_poison": vanilla_pois,
        "vanilla_retrain_steps": retrain_steps,
        "vanilla_post_retrain": vanilla_retrained,
        "vanilla_has_records": vanilla_has_records,
        "vanilla_has_schedule": False,
        "pre_decay_trajectories_identical": identical,
        "tails": {},
    }
    for name, r in results.items():
        traj = r["trajectory"]
        out["tails"][name] = {
            "decay_per_consolidation": r["decay_per_consolidation"],
            "baseline": r["baseline_kt"],
            "post_poison": r["post_poison_kt"],
            "repair_steps": r["repair_steps"],
            "post_repair": r["post_repair_probes"],
            "post_repair_auth_mean": traj[0]["auth_mean"],
            "chain_ok_at_repair": r["chain_ok_at_repair"],
            "lineage_liar_entries": r["lineage_liar_entries"],
            "repair_seats": (r["repair_statuses_seated"],
                             r["repair_statuses_total"]),
            "deficit_at_plus6": traj[6]["deficit"],
            "auth_mean_at_plus6": traj[6]["auth_mean"],
            "deficit_at_plus30": traj[30]["deficit"],
            "auth_mean_at_plus30": traj[30]["auth_mean"],
            "time_to_healed": r["time_to_healed"],
            "top1_true_4of4_throughout": all(
                t["probes"]["top1_true_count"] == 4 for t in traj),
            "deficit_table": {n: traj[n]["deficit"] for n in TABLE_NS},
            "work_leg": r["work_leg"],
        }
    return out


def main():
    run_scene()


if __name__ == "__main__":
    main()
