"""Saturation check on the V2 3-arm KTN experiment (NEW file, 2026-10-28).

Question: does the 200-step poison phase saturate the flip and
compress the measurable P2 gap (recovery-1 -> resistance fall,
arm 1 minus arm 3)?

Frozen files are NEVER touched: preregistration-3arm-v2-2026-10-28.md,
run_3arm_v2.py, results-3arm-v2.json/.md (and all v1 files). This
runner IMPORTS the exact V2 code paths (train_phase,
run_phase_with_driver, run_phase_arm3_with_handoff, affected_rows,
rank helpers) and only parameterizes poison-phase length. Clean 400,
restores 200, seeds, batch-list construction, fresh Adam per phase
(lr 0.01), and the arm-3 handoff step ordering are all identical to
V2. Poison batches for length N are the FIRST N of the seed-2 list
(V2 uses the same list for poison 1 and poison 2), so shorter runs
are strict prefixes of the baseline, not new draws.

Two driver variants, because the driver rate is itself a confound:
  A "frozen-driver": PressureDriver rates unchanged (poison
     0.75/200 per step, restore 0.375/200). Shorter phases then reach
     a LOWER peak pressure/scar and may never cross the handoff
     gate -- that is reported, not hidden. This is the literal
     "keep everything else identical" sweep.
  B "peak-matched": poison rate scaled to 0.75/N per step, so every
     length reaches the same peak pressure 0.75 (scar 1.2) and the
     same restore-1 end state (pressure 0.375, scar 0.2) as the
     200-step baseline; only the number of training steps varies.
     This isolates steps-beyond-flip from driver-peak differences.
     Implemented by patching kt_bench.scar.POISON_RATE for the run
     (PressureDriver reads the module global at step time) and
     restoring it afterwards. Restore rate is never changed.

Trajectory (part b): during each poison phase, arm probes
(mean P(true)/P(liar), top-1 counts) are recorded at phase start,
every 10 steps (poison 2) / every 25 steps (poison 1), and at the
final step. Probes are forward-only and cannot change weights, so
the phase-boundary numbers of the (A, 200) config must reproduce
V2 exactly -- that replication is the validation check.

Scope notes (same honesty as V2):
- Mock-RK selection is OMITTED: selection is read-only (it never
  writes weights, gains, or records) and cannot affect raw falls.
  Only raw probes are reported here.
- Fact-record seating is kept exactly as V2 (birth + teacher
  seating at consolidations); fact records do not affect training.
- Bench scale only. This check measures how the gap varies with
  phase length; it does NOT re-grade P2 and moves no bar.
- Gradient gate FIRST; init parity (bit-identical weights across
  arms within each run) asserted per config before training.

Run from ~/workspace/kt-neuron:
    python3 experiments/saturation-check-3arm-v2.py [lengths...]
Default lengths: 50 100 150 200 (both variants, arms 1+3; arm 2
included only in the (A, 200) validation config). Writes
experiments/results-3arm-v2-saturation.json.
"""

import itertools
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import kt_bench.scar as scar_mod
from experiments.corpus import (FACTS, LIAR_TOKENS, TRUE_TOKENS,
                                gen_clean_batch, gen_poison_batch,
                                probe_all_facts, summarize_probes)
from experiments.playbook import Bench, capture_weights, layer_param_prefix
from experiments.run_3arm_v2 import (CFG, FACT_ROWS, affected_rows,
                                     run_phase_arm3_with_handoff,
                                     run_phase_with_driver, train_phase)
from experiments.run_all import (CLEAN_POOL, REPAIR_DOCK, REPAIR_POOL,
                                 train_steps)
from experiments.trainer import (Adam, collect_params, gradient_check,
                                 loss_and_grads)
from kt_bench.handoff import HandoffTracker
from kt_bench.linear import KTLinear, Linear
from kt_bench.scar import PressureDriver, ScarLinear
from kt_bench.seating import seat_authority
from kt_bench.transformer import GPT
from kt_neuron.kid import make_kid
from kt_neuron.record import KRecord

CLEAN_STEPS, RESTORE_STEPS, FULL_POISON = 400, 200, 200
ORIG_POISON_RATE = scar_mod.POISON_RATE
ORIG_RESTORE_RATE = scar_mod.RESTORE_RATE

_fact_seq = itertools.count(200000)


def run_poison_arm(model, batches, kind, driver, tracker, prefix_of,
                   interval):
    """One poison phase for ONE arm, stepwise, with probe checkpoints.

    Mirrors the V2 loops exactly (arm 1: train_steps per batch with
    one shared Adam; arm 2: driver loop; arm 3: handoff loop with
    frozen step ordering). Returns trajectory checkpoints:
    [{step, mean_p_true, mean_p_liar, top1_true, top1_liar}].
    Probes are forward-only and do not touch weights, gains, or
    driver state.
    """
    traj = []

    def probe(step):
        s = summarize_probes(probe_all_facts(model))
        traj.append({"step": step, "mean_p_true": s["mean_p_true"],
                     "mean_p_liar": s["mean_p_liar"],
                     "top1_true_count": s["top1_true_count"],
                     "top1_liar_count": s["top1_liar_count"]})

    names, params = collect_params(model)
    adam = Adam(params, lr=0.01)
    probe(0)
    for i, b in enumerate(batches):
        step = i + 1
        if driver is None:  # arm 1: plain training
            train_steps(model, [b], adam=adam)
        elif tracker is None:  # arm 2: scar driver loop (as V2)
            train_steps(model, [b], adam=adam)
            driver.poison_step()
            driver.apply_to(model)
        else:  # arm 3: handoff loop (frozen V2 ordering)
            lr_scale = tracker.lr_scale_list(names, prefix_of)
            _loss, grads = loss_and_grads(model, b)
            adam.step([grads[n] for n in names], lr_scale=lr_scale)
            prev_F = driver.scar
            driver.poison_step()
            new_F = driver.apply_to(model)
            tracker.check(prev_F, new_F)
        if step % interval == 0 or step == len(batches):
            probe(step)
    return traj


def run_config(poison_len, variant, include_arm2):
    """One full V2-shaped run at the given poison length/variant."""
    if variant == "peak-matched":
        scar_mod.POISON_RATE = 0.75 / poison_len
    else:
        scar_mod.POISON_RATE = ORIG_POISON_RATE
    scar_mod.RESTORE_RATE = ORIG_RESTORE_RATE
    try:
        return _run_config_inner(poison_len, variant, include_arm2)
    finally:
        scar_mod.POISON_RATE = ORIG_POISON_RATE
        scar_mod.RESTORE_RATE = ORIG_RESTORE_RATE


def _run_config_inner(poison_len, variant, include_arm2):
    rng_c = np.random.default_rng(1)
    clean_batches = [gen_clean_batch(rng_c) for _ in range(CLEAN_STEPS)]
    rng_p = np.random.default_rng(2)
    poison_full = [gen_poison_batch(rng_p) for _ in range(FULL_POISON)]
    poison_batches = poison_full[:poison_len]
    rng_r = np.random.default_rng(3)
    restore1_batches = [gen_clean_batch(rng_r) for _ in range(RESTORE_STEPS)]
    rng_r2 = np.random.default_rng(4)
    restore2_batches = [gen_clean_batch(rng_r2) for _ in range(RESTORE_STEPS)]

    arm1 = GPT(CFG, np.random.default_rng(0), Linear)
    arm3 = GPT(CFG, np.random.default_rng(0), KTLinear)
    arms = {"arm1_conventional": arm1, "arm3_full_ktn": arm3}
    arm2 = None
    if include_arm2:
        arm2 = GPT(CFG, np.random.default_rng(0), ScarLinear)
        arms["arm2_scar_only"] = arm2
    parity = all(
        np.array_equal(a.W, b.W) and np.array_equal(a.b, b.b)
        for (_n1, a), (_n2, b) in zip(arm1.all_linears(), arm3.all_linears()))
    if include_arm2:
        parity = parity and all(
            np.array_equal(a.W, b.W) and np.array_equal(a.b, b.b)
            for (_n1, a), (_n2, b) in zip(arm1.all_linears(),
                                          arm2.all_linears()))
    assert parity, "init parity FAILED -- aborting config"
    bench3 = Bench(arm3)
    driver3 = PressureDriver()
    driver2 = PressureDriver() if include_arm2 else None
    tracker = HandoffTracker(bench3.layers)
    prefix_of = {name: layer_param_prefix(name)
                 for name, _layer in bench3.layers}
    fact_records = []
    for fact in FACTS:
        kid = make_kid(birth_tick=0, seq=next(_fact_seq))
        rec = KRecord(kid)
        rec.open_belief("fact", fact[2], 0.5, "birth")
        fact_records.append(rec)

    phases = {}

    def boundary(name):
        rec = {}
        for aname, model in arms.items():
            rec[aname] = summarize_probes(probe_all_facts(model))
        rec["scar_arm3"] = driver3.scar
        rec["pressure_arm3"] = driver3.pressure
        rec["latched_total"] = tracker.latched_total()
        rec["mean_scale_latched"] = tracker.mean_scale_latched()
        phases[name] = rec
        return rec

    # Phase 1: clean (V2 batch path for every arm).
    for model in arms.values():
        train_phase(model, clean_batches)
    bench3.consolidate(CLEAN_POOL, "sat-clean")
    tracker.set_baseline(seat_authority(CLEAN_POOL))
    for rec, fact in zip(fact_records, FACTS):
        rec.replace("fact", fact[2], seat_authority(CLEAN_POOL), "teacher")
    boundary("baseline")

    # Phase 2: poison 1 (stepwise + trajectory every 25 steps).
    pre_w3 = capture_weights(arm3)
    tracker.set_phase("poison1")
    traj1 = {}
    for aname, model in arms.items():
        if aname == "arm2_scar_only":
            traj1[aname] = run_poison_arm(model, poison_batches, "poison",
                                          driver2, None, None, 25)
        elif aname == "arm3_full_ktn":
            traj1[aname] = run_poison_arm(model, poison_batches, "poison",
                                          driver3, tracker, prefix_of, 25)
        else:
            traj1[aname] = run_poison_arm(model, poison_batches, "poison",
                                          None, None, None, 25)
    boundary("injury")
    post_w3 = capture_weights(arm3)
    affected1 = affected_rows(pre_w3, post_w3)
    scar_after_p1 = driver3.scar
    pressure_after_p1 = driver3.pressure

    # Phase 3: restore 1 (V2 batch paths, unchanged length 200).
    tracker.set_phase("restore1")
    for aname, model in arms.items():
        if aname == "arm2_scar_only":
            run_phase_with_driver(model, restore1_batches, driver2, "restore")
        elif aname == "arm3_full_ktn":
            run_phase_arm3_with_handoff(model, restore1_batches, driver3,
                                        "restore", tracker, prefix_of)
        else:
            train_phase(model, restore1_batches)
    dock1 = {r: REPAIR_DOCK for r in affected1}
    bench3.consolidate(REPAIR_POOL, "sat-restore1", dock_map=dock1)
    tracker.set_baseline(seat_authority(REPAIR_POOL))
    for rec, fact in zip(fact_records, FACTS):
        rec.replace("fact", fact[2], seat_authority(REPAIR_POOL), "teacher",
                    authority_dock=REPAIR_DOCK)
    boundary("recovery1")

    # Phase 4: poison 2, SAME truncated list (stepwise, every 10).
    pre_w3_2 = capture_weights(arm3)
    tracker.set_phase("poison2")
    traj2 = {}
    for aname, model in arms.items():
        if aname == "arm2_scar_only":
            traj2[aname] = run_poison_arm(model, poison_batches, "poison",
                                          driver2, None, None, 10)
        elif aname == "arm3_full_ktn":
            traj2[aname] = run_poison_arm(model, poison_batches, "poison",
                                          driver3, tracker, prefix_of, 10)
        else:
            traj2[aname] = run_poison_arm(model, poison_batches, "poison",
                                          None, None, None, 10)
    boundary("resistance")
    post_w3_2 = capture_weights(arm3)
    affected2 = affected_rows(pre_w3_2, post_w3_2)
    fact_move = []
    for name, row in FACT_ROWS:
        w0, b0 = pre_w3_2[name]
        w1, b1 = post_w3_2[name]
        fact_move.append(float(np.linalg.norm(w1[row] - w0[row])
                               + abs(b1[row] - b0[row])))
    scar_after_p2 = driver3.scar

    # Phase 5: restore 2 (V2 batch paths, unchanged length 200).
    tracker.set_phase("restore2")
    for aname, model in arms.items():
        if aname == "arm2_scar_only":
            run_phase_with_driver(model, restore2_batches, driver2, "restore")
        elif aname == "arm3_full_ktn":
            run_phase_arm3_with_handoff(model, restore2_batches, driver3,
                                        "restore", tracker, prefix_of)
        else:
            train_phase(model, restore2_batches)
    dock2 = {r: REPAIR_DOCK for r in affected2}
    bench3.consolidate(REPAIR_POOL, "sat-restore2", dock_map=dock2)
    tracker.set_baseline(seat_authority(REPAIR_POOL))
    for rec, fact in zip(fact_records, FACTS):
        rec.replace("fact", fact[2], seat_authority(REPAIR_POOL), "teacher",
                    authority_dock=REPAIR_DOCK)
    boundary("recovery2")

    def mp(arm, phase):
        return phases[phase][arm]["mean_p_true"]

    fall1 = mp("arm1_conventional", "recovery1") - mp("arm1_conventional", "resistance")
    fall3 = mp("arm3_full_ktn", "recovery1") - mp("arm3_full_ktn", "resistance")
    out = {
        "poison_length": poison_len,
        "variant": variant,
        "driver_poison_rate_per_step": (0.75 / poison_len
                                        if variant == "peak-matched"
                                        else ORIG_POISON_RATE),
        "init_parity_identical_weights": bool(parity),
        "phases": phases,
        "trajectory_poison1": traj1,
        "trajectory_poison2": traj2,
        "affected_rows_poison1": len(affected1),
        "affected_rows_poison2": len(affected2),
        "scar_after_poison1": scar_after_p1,
        "scar_after_poison2": scar_after_p2,
        "pressure_after_poison1": pressure_after_p1,
        "handoff_events_poison1_plus_2": len(tracker.events),
        "handoff_latched_total": tracker.latched_total(),
        "arm3_fact_row_movement_poison2_mean": float(np.mean(fact_move)),
        "arm1_fall_recovery1_to_resistance": fall1,
        "arm3_fall_recovery1_to_resistance": fall3,
        "p2_gap_arm1_minus_arm3": fall1 - fall3,
    }
    if include_arm2:
        fall2 = (mp("arm2_scar_only", "recovery1")
                 - mp("arm2_scar_only", "resistance"))
        out["arm2_fall_recovery1_to_resistance"] = fall2
    print(f"[{variant} N={poison_len}] arm1 fall={fall1:.4f} "
          f"arm3 fall={fall3:.4f} gap={fall1 - fall3:.4f} "
          f"scar_p1={scar_after_p1} scar_p2={scar_after_p2} "
          f"latched={tracker.latched_total()} "
          f"affected1={len(affected1)} affected2={len(affected2)}",
          flush=True)
    return out


def main():
    lengths = [int(a) for a in sys.argv[1:]] or [50, 100, 150, 200]
    fwd_diff, max_rel, max_abs, checked, gate_pass = gradient_check()
    out = {"gradient_check": {"forward_max_abs_diff": fwd_diff,
                              "max_rel_err": max_rel,
                              "max_abs_err": max_abs,
                              "params_checked": checked,
                              "gate": "PASS" if gate_pass else "FAIL"}}
    print("gradient check:", out["gradient_check"], flush=True)
    if not gate_pass:
        print("GATE FAILED -- aborting.", flush=True)
        return out
    configs = []
    for variant in ("frozen-driver", "peak-matched"):
        for n in lengths:
            include_arm2 = (variant == "frozen-driver" and n == 200)
            configs.append(run_config(n, variant, include_arm2))
    # Merge with any configs already in the results file (e.g. the
    # 200-step validation run), keyed by (variant, poison_length),
    # so a partial sweep never clobbers earlier configs.
    exp_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(exp_dir, "results-3arm-v2-saturation.json")
    merged = {}
    if os.path.exists(out_path):
        try:
            prev = json.load(open(out_path))
            for c in prev.get("configs", []):
                merged[(c["variant"], c["poison_length"])] = c
        except (json.JSONDecodeError, KeyError):
            pass
    for c in configs:
        merged[(c["variant"], c["poison_length"])] = c
    out["configs"] = [merged[k] for k in sorted(merged)]
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("results-3arm-v2-saturation.json written", flush=True)
    return out


if __name__ == "__main__":
    main()
