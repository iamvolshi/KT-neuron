"""3-arm KTN experiment, v2 — REPLICATION across seeds (variance study).

NOT the official run. Frozen contract for this study:
experiments/replication-plan-3arm-v2.md. This file mirrors
experiments/run_3arm_v2.py's orchestration exactly, parameterized
only by a seed offset (1000 * rep_seed added to every V2 RNG seed:
init 0, clean 1, poison 2, restore-1 3, restore-2 4, counting 77,
eval 99). Rep seed 0 has offset 0 and must reproduce V2 bit-for-bit.

Run from ~/workspace/kt-neuron:
    python3 experiments/run_3arm_v2_replication.py 0 1 2 3 4
Writes experiments/results-3arm-v2-replication.json (per-seed falls
and gaps; descriptive only — this study does NOT re-grade P2 and
the 0.05 bar stays untouched). V2's files are never touched.

V2 change vs v1 (only this): arm 3 gains the fast->slow handoff
switch + three-number tracking (kt_bench/handoff.py). V1 wired the
deficit to forward gain only and followed the fast trace throughout,
so P2 (gradient damping) failed. V2 switches a row to slow governance
at the crossing (authority gate S > 0.5 + upward crossing
prev_F < S <= new_F): from then on the row's Adam step size is
multiplied by its current standing authority S (the slow number).
Gradients are NOT scaled — Adam normalizes per-element gradients, so
damping must act on the step (see the v2 preregistration for the
frozen reasoning).

Arms, phases, seeds, batch lists: identical to v1. Arms 1-2 train
through the exact v1 code paths. Arm 3's clean phase uses the v1
path; its driver phases use the handoff loop below (fresh Adam per
phase, lr 0.01, as v1).

Disclosed harness fix vs v1: fact records open a birth belief at
construction ("fact", true token, 0.5, "birth"), mirroring KTLinear
birth rows, so fact records carry birth + 3 seating rows = 4 each
(16 total, matching the P6 leg definition). V1's 12 is preserved in
v1's files. No probe value changes: standing("fact") is the latest
interval either way.
"""

import itertools
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from kt_bench.handoff import HandoffTracker
from kt_bench.linear import KTLinear, Linear
from kt_bench.mock_rk import MockRK
from kt_bench.scar import PressureDriver, ScarLinear
from kt_bench.seating import seat_authority
from kt_bench.transformer import GPT, GPTConfig
from kt_neuron.kid import make_kid
from kt_neuron.record import KRecord
from experiments.corpus import (FACTS, LIAR_TOKENS, TRUE_TOKENS,
                                gen_clean_batch, gen_poison_batch,
                                probe_all_facts, summarize_probes)
from experiments.playbook import (Bench, capture_weights,
                                  layer_param_prefix, time_forwards)
from experiments.run_all import (CLEAN_POOL, REPAIR_DOCK, REPAIR_POOL,
                                 train_steps)
from experiments.trainer import (Adam, collect_params, gradient_check,
                                 loss_and_grads)

CFG = GPTConfig(vocab_size=32, block_size=16, n_embd=64, n_head=4, n_layer=2)
CLEAN_STEPS, POISON_STEPS, RESTORE_STEPS = 400, 200, 200
FACT_ROWS = [("lm_head", t) for t in TRUE_TOKENS + LIAR_TOKENS]

_fact_seq = itertools.count(100000)


def rank_moved_rows(pre, post, top_n=20):
    """Same as v1: rows ranked by movement, INCLUDING lm_head."""
    moves = []
    for name in pre:
        w0, b0 = pre[name]
        w1, b1 = post[name]
        dw = np.linalg.norm(w1 - w0, axis=1) + np.abs(b1 - b0)
        for row, m in enumerate(dw):
            moves.append((float(m), name, row))
    moves.sort(reverse=True)
    return [(name, row) for _m, name, row in moves[:top_n]]


def affected_rows(pre, post):
    """Fact rows + top-20 moved rows, deduplicated, fact rows first."""
    out = list(FACT_ROWS)
    seen = set(out)
    for r in rank_moved_rows(pre, post, 20):
        if r not in seen:
            out.append(r)
            seen.add(r)
    return out


def counting_sequences(seed=77, n=32, seq_len=16):
    """Fresh counting runs: [s, s+1, ...] mod 16."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        s = int(rng.integers(0, 16))
        out.append([(s + i) % 16 for i in range(seq_len)])
    return np.array(out, dtype=np.int64)


def counting_accuracy(model, sequences):
    """Teacher-forced next-token accuracy over all positions."""
    logits = model.forward(sequences)
    pred = np.argmax(logits[:, :-1, :], axis=-1)
    target = sequences[:, 1:]
    return float((pred == target).mean())


def probe_arm(model):
    """Raw probes: summary + per-fact detail."""
    probes = probe_all_facts(model)
    return {"summary": summarize_probes(probes), "per_fact": probes}


def selection_probe(model, fact_records, mockrk):
    """Mock-RK selection per fact. Returns selected counts + entries."""
    probes = probe_all_facts(model)
    entries = []
    for i, (fact, probe) in enumerate(zip(FACTS, probes)):
        standing = fact_records[i].standing("fact")["authority"]
        e = mockrk.select(f"fact-{i}", fact[2], standing, probe["top1"],
                          liar_token=fact[3])
        entries.append(e)
    t, li = mockrk.selected_counts(entries)
    return {"selected_true_count": t, "selected_liar_count": li,
            "entries": entries}


def train_phase(model, batches):
    """One phase with a fresh Adam (no driver). Identical to v1."""
    names, params = collect_params(model)
    adam = Adam(params, lr=0.01)
    train_steps(model, batches, adam=adam)


def run_phase_with_driver(model, batches, driver, kind):
    """V1 driver loop (arms 1-2, and arm-3 clean-phase equivalent).

    kind: 'poison' or 'restore'. Trains stepwise, advancing the
    driver per step and applying the scar after each step.
    """
    names, params = collect_params(model)
    adam = Adam(params, lr=0.01)
    for b in batches:
        train_steps(model, [b], adam=adam)
        if kind == "poison":
            driver.poison_step()
        else:
            driver.restore_step()
        driver.apply_to(model)


def run_phase_arm3_with_handoff(model, batches, driver, kind, tracker,
                                prefix_of):
    """V2 arm-3 driver loop: handoff-governed Adam steps.

    Frozen step ordering (v2 preregistration):
    1. Build the step-size scale list from the CURRENT latch set and
       CURRENT standing authorities (gains).
    2. One batch: loss_and_grads + Adam.step with that scale list.
    3. Driver step, then apply the scar.
    4. Crossing check with new F -> newly-crossed rows latch for the
       NEXT step (the crossing step itself trains unscaled).
    """
    names, params = collect_params(model)
    adam = Adam(params, lr=0.01)
    for b in batches:
        lr_scale = tracker.lr_scale_list(names, prefix_of)
        loss, grads = loss_and_grads(model, b)
        adam.step([grads[n] for n in names], lr_scale=lr_scale)
        prev_F = driver.scar
        if kind == "poison":
            driver.poison_step()
        else:
            driver.restore_step()
        new_F = driver.apply_to(model)
        tracker.check(prev_F, new_F)


def run_seed(rep_seed):
    """One V2 replication at replication seed rep_seed (offset 1000*s)."""
    global _fact_seq
    _fact_seq = itertools.count(100000)
    offset = 1000 * rep_seed
    print(f"=== replication seed {rep_seed} (offset {offset}) ===", flush=True)
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
        raise RuntimeError("gradient gate FAILED -- aborting seed")

    # ---- Fixed batch lists (v1 seeds, lockstep) ---------------------------
    rng_c = np.random.default_rng(1 + offset)
    clean_batches = [gen_clean_batch(rng_c) for _ in range(CLEAN_STEPS)]
    rng_p = np.random.default_rng(2 + offset)
    poison_batches = [gen_poison_batch(rng_p) for _ in range(POISON_STEPS)]
    rng_r = np.random.default_rng(3 + offset)
    restore1_batches = [gen_clean_batch(rng_r) for _ in range(RESTORE_STEPS)]
    rng_r2 = np.random.default_rng(4 + offset)
    restore2_batches = [gen_clean_batch(rng_r2) for _ in range(RESTORE_STEPS)]
    rng_e = np.random.default_rng(99 + offset)
    eval_batches = [gen_clean_batch(rng_e) for _ in range(16)]
    eval_batch = eval_batches[0]
    count_seqs = counting_sequences(seed=77 + offset)

    # ---- Build arms (same seed -> identical init weights) -----------------
    arm1 = GPT(CFG, np.random.default_rng(offset), Linear)
    arm2 = GPT(CFG, np.random.default_rng(offset), ScarLinear)
    arm3 = GPT(CFG, np.random.default_rng(offset), KTLinear)
    init_parity = all(
        np.array_equal(a.W, b.W) and np.array_equal(a.b, b.b)
        for (_n1, a), (_n2, b) in zip(arm1.all_linears(), arm2.all_linears())
    ) and all(
        np.array_equal(a.W, b.W) and np.array_equal(a.b, b.b)
        for (_n1, a), (_n2, b) in zip(arm1.all_linears(), arm3.all_linears())
    )
    out["init_parity_identical_weights"] = bool(init_parity)
    print("init parity:", init_parity, flush=True)
    bench3 = Bench(arm3)
    driver2 = PressureDriver()
    driver3 = PressureDriver()
    mockrk = MockRK()
    tracker = HandoffTracker(bench3.layers)
    prefix_of = {name: layer_param_prefix(name)
                 for name, _layer in bench3.layers}
    fact_records = []
    for fact in FACTS:
        kid = make_kid(birth_tick=0, seq=next(_fact_seq))
        rec = KRecord(kid)
        # V2 construction fix (disclosed in prereg): birth belief at
        # construction, mirroring KTLinear birth rows.
        rec.open_belief("fact", fact[2], 0.5, "birth")
        fact_records.append(rec)
    arms = {"arm1_conventional": arm1, "arm2_scar_only": arm2,
            "arm3_full_ktn": arm3}
    phases = {}
    wall = {}

    def probe_all(phase_name):
        """Probe every arm at a phase boundary; arm 3 also runs
        mock-RK selection + a three-number snapshot. Returns the
        phase record."""
        rec = {}
        for name, model in arms.items():
            rec[name] = probe_arm(model)
        sel = selection_probe(arm3, fact_records, mockrk)
        rec["arm3_full_ktn"]["selection"] = {
            "selected_true_count": sel["selected_true_count"],
            "selected_liar_count": sel["selected_liar_count"],
            "entries": sel["entries"],
        }
        rec["arm3_full_ktn"]["scar"] = driver3.scar
        rec["arm2_scar_only"]["scar"] = driver2.scar
        rec["arm3_full_ktn"]["three_numbers"] = {
            "F": driver3.scar,
            "fact_rows": tracker.snapshot(driver3.scar, FACT_ROWS),
            "latched_total": tracker.latched_total(),
            "mean_scale_latched": tracker.mean_scale_latched(),
        }
        phases[phase_name] = rec
        for name in arms:
            s = rec[name]["summary"]
            line = (f"{phase_name} {name}: mean_p_true={s['mean_p_true']:.4f} "
                    f"mean_p_liar={s['mean_p_liar']:.4f} "
                    f"top1_true={s['top1_true_count']} top1_liar={s['top1_liar_count']}")
            if name == "arm3_full_ktn":
                line += (f" selected_true={sel['selected_true_count']} "
                         f"selected_liar={sel['selected_liar_count']} "
                         f"scar={driver3.scar} "
                         f"latched={tracker.latched_total()}")
            if name == "arm2_scar_only":
                line += f" scar={driver2.scar}"
            print(line, flush=True)
        return rec

    def arm3_mean_gain(rows=None):
        """Mean gain over rows (all rows, or a given row list)."""
        vals = []
        layers = dict(bench3.layers)
        if rows is None:
            for _n, layer in bench3.layers:
                vals.extend(float(g) for g in layer.gain)
        else:
            for name, row in rows:
                vals.append(float(layers[name].gain[row]))
        return float(np.mean(vals)) if vals else None

    # ---- Phase 1: clean ----------------------------------------------------
    t_arm = {}
    for name, model in arms.items():
        t0 = time.perf_counter()
        train_phase(model, clean_batches)
        t_arm[name] = time.perf_counter() - t0
    # Arm 3: clean consolidation + fact-record seating (teacher, 0.8).
    bench3.consolidate(CLEAN_POOL, "3arm-v2-clean")
    tracker.set_baseline(seat_authority(CLEAN_POOL))
    for rec, fact in zip(fact_records, FACTS):
        rec.replace("fact", fact[2], seat_authority(CLEAN_POOL), "teacher")
    probe_all("baseline")
    phases["baseline"]["counting_accuracy"] = {
        name: counting_accuracy(model, count_seqs)
        for name, model in arms.items()
    }
    print("baseline counting:", phases["baseline"]["counting_accuracy"], flush=True)

    # ---- Phase 2: poison 1 ---------------------------------------------------
    pre_w3 = capture_weights(arm3)
    tracker.set_phase("poison1")
    for name, model in arms.items():
        t0 = time.perf_counter()
        if name == "arm2_scar_only":
            run_phase_with_driver(model, poison_batches, driver2, "poison")
        elif name == "arm3_full_ktn":
            run_phase_arm3_with_handoff(model, poison_batches, driver3,
                                        "poison", tracker, prefix_of)
        else:
            train_phase(model, poison_batches)
        t_arm[name] = time.perf_counter() - t0
    mockrk.activate_liar()  # liar's challenge stands from now on
    probe_all("injury")
    post_w3 = capture_weights(arm3)
    affected1 = affected_rows(pre_w3, post_w3)
    print(f"affected rows (poison 1): {len(affected1)}", flush=True)
    print(f"handoff events (poison 1): {len(tracker.events)}", flush=True)

    # ---- Phase 3: restore 1 ----------------------------------------------------
    tracker.set_phase("restore1")
    for name, model in arms.items():
        t0 = time.perf_counter()
        if name == "arm2_scar_only":
            run_phase_with_driver(model, restore1_batches, driver2, "restore")
        elif name == "arm3_full_ktn":
            run_phase_arm3_with_handoff(model, restore1_batches, driver3,
                                        "restore", tracker, prefix_of)
        else:
            train_phase(model, restore1_batches)
        t_arm[name] += time.perf_counter() - t0
    dock1 = {r: REPAIR_DOCK for r in affected1}
    bench3.consolidate(REPAIR_POOL, "3arm-v2-restore1", dock_map=dock1)
    tracker.set_baseline(seat_authority(REPAIR_POOL))
    for rec, fact in zip(fact_records, FACTS):
        rec.replace("fact", fact[2], seat_authority(REPAIR_POOL), "teacher",
                    authority_dock=REPAIR_DOCK)
    mockrk.flag_liar_known_bad()  # quarantine decision, not discovery
    probe_all("recovery1")
    phases["recovery1"]["arm3_full_ktn"]["affected_rows"] = affected1
    phases["recovery1"]["arm3_full_ktn"]["affected_mean_gain"] = arm3_mean_gain(affected1)
    phases["recovery1"]["arm3_full_ktn"]["overall_mean_gain"] = arm3_mean_gain()
    phases["recovery1"]["arm3_full_ktn"]["fact_authorities"] = [
        rec.standing("fact")["authority"] for rec in fact_records]
    phases["recovery1"]["arm3_full_ktn"]["affected_three_numbers"] = \
        tracker.snapshot(driver3.scar, affected1)

    # ---- Phase 4: poison 2 (SAME seed-2 list) ------------------------------------
    pre_w3_2 = capture_weights(arm3)
    tracker.set_phase("poison2")
    for name, model in arms.items():
        t0 = time.perf_counter()
        if name == "arm2_scar_only":
            run_phase_with_driver(model, poison_batches, driver2, "poison")
        elif name == "arm3_full_ktn":
            run_phase_arm3_with_handoff(model, poison_batches, driver3,
                                        "poison", tracker, prefix_of)
        else:
            train_phase(model, poison_batches)
        t_arm[name] += time.perf_counter() - t0
    probe_all("resistance")
    post_w3_2 = capture_weights(arm3)
    affected2 = affected_rows(pre_w3_2, post_w3_2)
    print(f"affected rows (poison 2): {len(affected2)}", flush=True)
    # Latched-row movement during poison 2 (mechanism check): mean
    # per-row movement on fact rows, arm 3.
    fact_move = []
    for name, row in FACT_ROWS:
        w0, b0 = pre_w3_2[name]
        w1, b1 = post_w3_2[name]
        fact_move.append(float(np.linalg.norm(w1[row] - w0[row])
                               + abs(b1[row] - b0[row])))
    phases["resistance"]["arm3_full_ktn"]["fact_row_movement_poison2"] = {
        "mean": float(np.mean(fact_move)), "per_row": fact_move}

    # ---- Phase 5: restore 2 -------------------------------------------------------
    tracker.set_phase("restore2")
    for name, model in arms.items():
        t0 = time.perf_counter()
        if name == "arm2_scar_only":
            run_phase_with_driver(model, restore2_batches, driver2, "restore")
        elif name == "arm3_full_ktn":
            run_phase_arm3_with_handoff(model, restore2_batches, driver3,
                                        "restore", tracker, prefix_of)
        else:
            train_phase(model, restore2_batches)
        t_arm[name] += time.perf_counter() - t0
    dock2 = {r: REPAIR_DOCK for r in affected2}
    bench3.consolidate(REPAIR_POOL, "3arm-v2-restore2", dock_map=dock2)
    tracker.set_baseline(seat_authority(REPAIR_POOL))
    for rec, fact in zip(fact_records, FACTS):
        rec.replace("fact", fact[2], seat_authority(REPAIR_POOL), "teacher",
                    authority_dock=REPAIR_DOCK)
    probe_all("recovery2")
    phases["recovery2"]["arm3_full_ktn"]["affected_rows"] = affected2
    phases["recovery2"]["arm3_full_ktn"]["affected_mean_gain"] = arm3_mean_gain(affected2)
    phases["recovery2"]["arm3_full_ktn"]["overall_mean_gain"] = arm3_mean_gain()
    phases["recovery2"]["arm3_full_ktn"]["fact_authorities"] = [
        rec.standing("fact")["authority"] for rec in fact_records]
    phases["recovery2"]["arm3_full_ktn"]["affected_three_numbers"] = \
        tracker.snapshot(driver3.scar, affected2)
    phases["recovery2"]["counting_accuracy"] = {
        name: counting_accuracy(model, count_seqs)
        for name, model in arms.items()
    }
    print("final counting:", phases["recovery2"]["counting_accuracy"], flush=True)

    # ---- Cost + integrity ---------------------------------------------------------
    out["phases"] = phases
    out["wall_seconds_per_arm"] = t_arm
    out["forward_ms"] = {name: time_forwards(model, eval_batch)
                         for name, model in arms.items()}
    row_rows = sum(len(rec.rows) for _n, layer in bench3.layers
                   for rec in layer.records)
    fact_rows_n = sum(len(rec.rows) for rec in fact_records)
    out["record_rows"] = {"row_records": row_rows, "fact_records": fact_rows_n,
                          "total": row_rows + fact_rows_n}
    chain_ok, chain_bad = bench3.verify_all()
    fact_chain = [rec.verify() for rec in fact_records]
    out["chain"] = {"row_records_ok": chain_ok, "row_records_bad": chain_bad,
                    "fact_records": fact_chain}
    out["mockrk_log_total_entries"] = len(mockrk.log)
    out["mockrk_overrides_total"] = sum(1 for e in mockrk.log if e["overridden"])
    out["handoff_events"] = tracker.events
    out["handoff_latched_total"] = tracker.latched_total()
    print("forward ms:", out["forward_ms"], flush=True)
    print("record rows:", out["record_rows"], flush=True)
    print("chain:", chain_ok, fact_chain, flush=True)
    print("handoff latched total:", tracker.latched_total(),
          "events:", len(tracker.events), flush=True)

    # ---- Replication summary (descriptive ONLY; this study does NOT
    # re-grade P2 and the 0.05 bar stays untouched) ----------------------
    def mp(arm, phase):
        return phases[phase][arm]["summary"]["mean_p_true"]

    # Falls are MAGNITUDES (positive = fall), as in V2.
    fall1 = mp("arm1_conventional", "recovery1") - mp("arm1_conventional", "resistance")
    fall2 = mp("arm2_scar_only", "recovery1") - mp("arm2_scar_only", "resistance")
    fall3 = mp("arm3_full_ktn", "recovery1") - mp("arm3_full_ktn", "resistance")
    summary = {
        "rep_seed": rep_seed,
        "seed_offset": offset,
        "rng_seeds": {"init": offset, "clean": 1 + offset,
                      "poison": 2 + offset, "restore1": 3 + offset,
                      "restore2": 4 + offset, "counting": 77 + offset,
                      "eval": 99 + offset},
        "gradient_gate": out["gradient_check"],
        "init_parity_identical_weights": out["init_parity_identical_weights"],
        "mean_p_true": {
            arm: {phase: phases[phase][arm]["summary"]["mean_p_true"]
                  for phase in ["baseline", "injury", "recovery1",
                                "resistance", "recovery2"]}
            for arm in arms},
        "arm1_fall": fall1,
        "arm2_fall": fall2,
        "arm3_fall": fall3,
        "p2_gap": fall1 - fall3,
        "handoff_latched_total": tracker.latched_total(),
        "handoff_event_count": len(tracker.events),
        "first_handoff_event": tracker.events[0] if tracker.events else None,
        "scar_trajectory_arm3": [phases[p]["arm3_full_ktn"]["scar"] for p in
                                 ["baseline", "injury", "recovery1",
                                  "resistance", "recovery2"]],
        "record_rows": out["record_rows"],
        "chain": out["chain"],
        "wall_seconds_per_arm": t_arm,
        "forward_ms": out["forward_ms"],
    }
    print(f"[rep_seed={rep_seed}] arm1_fall={fall1:.6f} "
          f"arm3_fall={fall3:.6f} p2_gap={fall1 - fall3:.6f}", flush=True)
    return summary

def main(argv):
    seeds = [int(a) for a in argv[1:]] or [0, 1, 2, 3, 4]
    exp_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(exp_dir, "results-3arm-v2-replication.json")
    results = {"plan": "experiments/replication-plan-3arm-v2.md",
               "note": ("Variance study only: this file does NOT re-grade "
                        "P2 and the 0.05 bar is untouched."),
               "per_seed": []}
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f)
    done = {s["rep_seed"] for s in results["per_seed"]}
    for s in seeds:
        if s in done:
            print(f"[rep_seed={s}] already in {out_path}; skipping",
                  flush=True)
            continue
        results["per_seed"].append(run_seed(s))
        results["per_seed"].sort(key=lambda d: d["rep_seed"])
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"results-3arm-v2-replication.json updated ({out_path})",
              flush=True)
    return results


if __name__ == "__main__":
    main(sys.argv)
