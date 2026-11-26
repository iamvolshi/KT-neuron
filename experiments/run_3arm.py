"""3-arm controlled KTN experiment (official run).

Frozen contract: experiments/preregistration-3arm-2026-10-28.md.
Run from ~/workspace/kt-neuron:
    python3 experiments/run_3arm.py
Writes experiments/results-3arm.json and results-3arm.md.

Arms (same GPT config, same seed 0, identical init weights, same
batch lists lockstep):
  arm1 CONVENTIONAL: GPT + Linear. No scar, no record, no selection.
  arm2 SCAR-ONLY: GPT + ScarLinear. Scar driver only. No record,
    no authority, no selection. The ablation arm.
  arm3 FULL KTN: GPT + KTLinear + per-row K-records + consolidations
    + deficit dock + scar + mock-RK selection at probe time.
    Selection reads fact-belief records and writes only its own
    MOCK/DERIVED log; it never writes K-record beliefs.

Phases (fresh Adam per phase per arm, lr 0.01):
  clean 400 (seed 1) -> arm3 clean consolidation -> BASELINE probe
  poison 1, 200 (seed 2, scar rises, no consolidation) -> INJURY
  restore 1, 200 clean (seed 3, scar falls, arm3 restore
    consolidation + liar flagged known-bad) -> RECOVERY-1
  poison 2, 200 (SAME seed-2 list, scar rises, no consolidation)
    -> RESISTANCE
  restore 2, 200 clean (seed 4 first 200, scar falls, arm3 restore
    consolidation 2) -> RECOVERY-2
  retention + cost probes.

No threshold in the preregistration moves. Verdicts are computed
against the frozen predictions P1-P8 in this script's verdict
section only.
"""

import itertools
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

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
from experiments.playbook import Bench, capture_weights, time_forwards
from experiments.run_all import (CLEAN_POOL, REPAIR_DOCK, REPAIR_POOL,
                                 train_steps)
from experiments.trainer import Adam, collect_params, gradient_check

CFG = GPTConfig(vocab_size=32, block_size=16, n_embd=64, n_head=4, n_layer=2)
CLEAN_STEPS, POISON_STEPS, RESTORE_STEPS = 400, 200, 200
FACT_ROWS = [("lm_head", t) for t in TRUE_TOKENS + LIAR_TOKENS]

_fact_seq = itertools.count(100000)


def rank_moved_rows(pre, post, top_n=20):
    """Rows ranked by movement (||dW_row||_2 + |db|) during an
    interval, INCLUDING lm_head. Deviation from
    playbook.affected_upstream (which excludes lm_head) is
    deliberate and frozen in the preregistration: fact decisions
    live in lm_head rows, so excluding them would dock around the
    decision instead of at it."""
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
    """One phase with a fresh Adam (no driver)."""
    names, params = collect_params(model)
    adam = Adam(params, lr=0.01)
    train_steps(model, batches, adam=adam)


def run_phase_with_driver(model, batches, driver, kind):
    """kind: 'poison' or 'restore'. Trains stepwise, advancing the
    driver per step and applying the scar after each step."""
    names, params = collect_params(model)
    adam = Adam(params, lr=0.01)
    for b in batches:
        train_steps(model, [b], adam=adam)
        if kind == "poison":
            driver.poison_step()
        else:
            driver.restore_step()
        driver.apply_to(model)


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

    # ---- Fixed batch lists (round-1 seeds, lockstep) ----------------------
    rng_c = np.random.default_rng(1)
    clean_batches = [gen_clean_batch(rng_c) for _ in range(CLEAN_STEPS)]
    rng_p = np.random.default_rng(2)
    poison_batches = [gen_poison_batch(rng_p) for _ in range(POISON_STEPS)]
    rng_r = np.random.default_rng(3)
    restore1_batches = [gen_clean_batch(rng_r) for _ in range(RESTORE_STEPS)]
    rng_r2 = np.random.default_rng(4)
    restore2_batches = [gen_clean_batch(rng_r2) for _ in range(RESTORE_STEPS)]
    rng_e = np.random.default_rng(99)
    eval_batches = [gen_clean_batch(rng_e) for _ in range(16)]
    eval_batch = eval_batches[0]
    count_seqs = counting_sequences()

    # ---- Build arms (same seed -> identical init weights) -----------------
    arm1 = GPT(CFG, np.random.default_rng(0), Linear)
    arm2 = GPT(CFG, np.random.default_rng(0), ScarLinear)
    arm3 = GPT(CFG, np.random.default_rng(0), KTLinear)
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
    fact_records = []
    for fact in FACTS:
        kid = make_kid(birth_tick=0, seq=next(_fact_seq))
        fact_records.append(KRecord(kid))
    arms = {"arm1_conventional": arm1, "arm2_scar_only": arm2,
            "arm3_full_ktn": arm3}
    phases = {}
    wall = {}

    def probe_all(phase_name):
        """Probe every arm at a phase boundary; arm 3 also runs
        mock-RK selection. Returns the phase record."""
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
        phases[phase_name] = rec
        for name in arms:
            s = rec[name]["summary"]
            line = (f"{phase_name} {name}: mean_p_true={s['mean_p_true']:.4f} "
                    f"mean_p_liar={s['mean_p_liar']:.4f} "
                    f"top1_true={s['top1_true_count']} top1_liar={s['top1_liar_count']}")
            if name == "arm3_full_ktn":
                line += (f" selected_true={sel['selected_true_count']} "
                         f"selected_liar={sel['selected_liar_count']} "
                         f"scar={driver3.scar}")
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
    bench3.consolidate(CLEAN_POOL, "3arm-clean")
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
    for name, model in arms.items():
        t0 = time.perf_counter()
        if name == "arm2_scar_only":
            run_phase_with_driver(model, poison_batches, driver2, "poison")
        elif name == "arm3_full_ktn":
            run_phase_with_driver(model, poison_batches, driver3, "poison")
        else:
            train_phase(model, poison_batches)
        t_arm[name] = time.perf_counter() - t0
    mockrk.activate_liar()  # liar's challenge stands from now on
    probe_all("injury")
    post_w3 = capture_weights(arm3)
    affected1 = affected_rows(pre_w3, post_w3)
    print(f"affected rows (poison 1): {len(affected1)}", flush=True)

    # ---- Phase 3: restore 1 ----------------------------------------------------
    for name, model in arms.items():
        t0 = time.perf_counter()
        if name == "arm2_scar_only":
            run_phase_with_driver(model, restore1_batches, driver2, "restore")
        elif name == "arm3_full_ktn":
            run_phase_with_driver(model, restore1_batches, driver3, "restore")
        else:
            train_phase(model, restore1_batches)
        t_arm[name] += time.perf_counter() - t0
    dock1 = {r: REPAIR_DOCK for r in affected1}
    bench3.consolidate(REPAIR_POOL, "3arm-restore1", dock_map=dock1)
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

    # ---- Phase 4: poison 2 (SAME seed-2 list) ------------------------------------
    pre_w3_2 = capture_weights(arm3)
    for name, model in arms.items():
        t0 = time.perf_counter()
        if name == "arm2_scar_only":
            run_phase_with_driver(model, poison_batches, driver2, "poison")
        elif name == "arm3_full_ktn":
            run_phase_with_driver(model, poison_batches, driver3, "poison")
        else:
            train_phase(model, poison_batches)
        t_arm[name] += time.perf_counter() - t0
    probe_all("resistance")
    post_w3_2 = capture_weights(arm3)
    affected2 = affected_rows(pre_w3_2, post_w3_2)
    print(f"affected rows (poison 2): {len(affected2)}", flush=True)

    # ---- Phase 5: restore 2 -------------------------------------------------------
    for name, model in arms.items():
        t0 = time.perf_counter()
        if name == "arm2_scar_only":
            run_phase_with_driver(model, restore2_batches, driver2, "restore")
        elif name == "arm3_full_ktn":
            run_phase_with_driver(model, restore2_batches, driver3, "restore")
        else:
            train_phase(model, restore2_batches)
        t_arm[name] += time.perf_counter() - t0
    dock2 = {r: REPAIR_DOCK for r in affected2}
    bench3.consolidate(REPAIR_POOL, "3arm-restore2", dock_map=dock2)
    for rec, fact in zip(fact_records, FACTS):
        rec.replace("fact", fact[2], seat_authority(REPAIR_POOL), "teacher",
                    authority_dock=REPAIR_DOCK)
    probe_all("recovery2")
    phases["recovery2"]["arm3_full_ktn"]["affected_rows"] = affected2
    phases["recovery2"]["arm3_full_ktn"]["affected_mean_gain"] = arm3_mean_gain(affected2)
    phases["recovery2"]["arm3_full_ktn"]["overall_mean_gain"] = arm3_mean_gain()
    phases["recovery2"]["arm3_full_ktn"]["fact_authorities"] = [
        rec.standing("fact")["authority"] for rec in fact_records]
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
    print("forward ms:", out["forward_ms"], flush=True)
    print("record rows:", out["record_rows"], flush=True)
    print("chain:", chain_ok, fact_chain, flush=True)

    # ---- Verdicts vs frozen predictions P1-P8 ----------------------------------------
    def mp(arm, phase):
        return phases[phase][arm]["summary"]["mean_p_true"]

    def top1_true(arm, phase):
        return phases[phase][arm]["summary"]["top1_true_count"]

    def top1_liar(arm, phase):
        return phases[phase][arm]["summary"]["top1_liar_count"]

    def sel_true(phase):
        return phases[phase]["arm3_full_ktn"]["selection"]["selected_true_count"]

    def sel_liar(phase):
        return phases[phase]["arm3_full_ktn"]["selection"]["selected_liar_count"]

    v = {}
    v["P1_recovery_raw"] = {
        arm: bool(mp(arm, "recovery1") >= mp(arm, "baseline") - 0.10)
        for arm in arms}
    v["P1_recovery_raw"]["pass"] = all(v["P1_recovery_raw"][a] for a in arms)
    drop1 = mp("arm1_conventional", "resistance") - mp("arm1_conventional", "recovery1")
    drop2 = mp("arm2_scar_only", "resistance") - mp("arm2_scar_only", "recovery1")
    drop3 = mp("arm3_full_ktn", "resistance") - mp("arm3_full_ktn", "recovery1")
    # Prereg defines "drop" as the MAGNITUDE of the fall (positive).
    # P2a (signed-change form is algebraically identical to the
    # magnitude form): arm3's fall smaller than arm1's by >= 0.05.
    # P2b: arm2 must NOT beat arm1 (fall less) by more than 0.05,
    # i.e. mag2 >= mag1 - 0.05, i.e. signed drop2 <= drop1 + 0.05.
    # CORRECTION (pre-verdict, disclosed in results-3arm.md): the
    # harness originally coded P2b as drop2 >= drop1 - 0.05, which
    # applies the magnitude inequality to signed changes and inverts
    # the leg. The prereg's rationale ("if arm 2 beats arm 1 by
    # more, that leg fails") plus the magnitude reading agree on
    # the form below. Both readings are reported in the JSON.
    v["P2_resistance_raw"] = {
        "arm1_drop_signed": drop1, "arm2_drop_signed": drop2,
        "arm3_drop_signed": drop3,
        "arm1_drop_magnitude": -drop1, "arm2_drop_magnitude": -drop2,
        "arm3_drop_magnitude": -drop3,
        "arm3_beats_arm1_by_ge_0.05": bool((drop3 - drop1) >= 0.05),
        "arm2_not_beating_arm1_by_gt_0.05": bool(drop2 <= drop1 + 0.05),
        "arm2_not_beating_arm1_by_gt_0.05_literal_signed_inequality": bool(drop2 >= drop1 - 0.05),
        "p2b_correction_note": ("Harness originally coded P2b as "
            "drop2 >= drop1 - 0.05 (signed); prereg defines drop as "
            "magnitude and its rationale is 'if arm 2 beats arm 1 by "
            "more, that leg fails'. Corrected pre-verdict to "
            "drop2 <= drop1 + 0.05. Raw numbers reported under both "
            "readings; P2 overall fails via P2a regardless."),
    }
    v["P2_resistance_raw"]["pass"] = bool(
        v["P2_resistance_raw"]["arm3_beats_arm1_by_ge_0.05"]
        and v["P2_resistance_raw"]["arm2_not_beating_arm1_by_gt_0.05"])
    sel_traj = [sel_true(p) for p in
                ["baseline", "injury", "recovery1", "resistance", "recovery2"]]
    v["P3_selection_adds"] = {
        "selected_trajectory": sel_traj,
        "trajectory_is_4_0_4_4_4": sel_traj == [4, 0, 4, 4, 4],
        "selected_gt_scar_raw_at_resistance": bool(
            sel_true("resistance") > top1_true("arm2_scar_only", "resistance")),
        "selected_ge_own_raw_at_resistance": bool(
            sel_true("resistance") >= top1_true("arm3_full_ktn", "resistance")),
    }
    v["P3_selection_adds"]["pass"] = bool(
        v["P3_selection_adds"]["trajectory_is_4_0_4_4_4"]
        and v["P3_selection_adds"]["selected_gt_scar_raw_at_resistance"]
        and v["P3_selection_adds"]["selected_ge_own_raw_at_resistance"])
    aff_mean_r2 = phases["recovery2"]["arm3_full_ktn"]["affected_mean_gain"]
    fact_auth_r2 = phases["recovery2"]["arm3_full_ktn"]["fact_authorities"]
    v["P4_deficit_persists"] = {
        "affected_mean_gain_recovery2": aff_mean_r2,
        "affected_mean_is_0.6": bool(abs(aff_mean_r2 - 0.6) < 1e-9),
        "fact_authorities_recovery2": fact_auth_r2,
        "fact_authorities_all_0.6": bool(
            all(abs(a - 0.6) < 1e-9 for a in fact_auth_r2)),
        "behavior_recovered_arm3": bool(v["P1_recovery_raw"]["arm3_full_ktn"]),
    }
    v["P4_deficit_persists"]["pass"] = bool(
        v["P4_deficit_persists"]["affected_mean_is_0.6"]
        and v["P4_deficit_persists"]["fact_authorities_all_0.6"]
        and v["P4_deficit_persists"]["behavior_recovered_arm3"])
    v["P5_retention"] = {
        arm: bool(abs(phases["recovery2"]["counting_accuracy"][arm]
                      - phases["baseline"]["counting_accuracy"][arm]) <= 0.05)
        for arm in arms}
    v["P5_retention"]["pass"] = all(v["P5_retention"][a] for a in arms)
    v["P6_cost"] = {
        "forward_ms": out["forward_ms"],
        "forward_ratio_arm3_over_arm1": out["forward_ms"]["arm3_full_ktn"] / out["forward_ms"]["arm1_conventional"],
        "forward_ratio_lt_2": bool(out["forward_ms"]["arm3_full_ktn"] < 2.0 * out["forward_ms"]["arm1_conventional"]),
        "row_rows": row_rows, "row_rows_is_4736": row_rows == 4736,
        "fact_rows": fact_rows_n, "fact_rows_is_16": fact_rows_n == 16,
    }
    v["P6_cost"]["pass"] = bool(
        v["P6_cost"]["forward_ratio_lt_2"]
        and v["P6_cost"]["row_rows_is_4736"]
        and v["P6_cost"]["fact_rows_is_16"])
    scar_traj2 = [phases[p]["arm2_scar_only"]["scar"] for p in
                  ["baseline", "injury", "recovery1", "resistance", "recovery2"]]
    scar_traj3 = [phases[p]["arm3_full_ktn"]["scar"] for p in
                  ["baseline", "injury", "recovery1", "resistance", "recovery2"]]
    v["P7_scar_trajectory"] = {
        "arm2": scar_traj2, "arm3": scar_traj3,
        "expected": [0.0, 1.2, 0.2, 1.2, 0.2],
        "arm2_match": all(abs(a - b) <= 1e-6 for a, b in zip(scar_traj2, [0.0, 1.2, 0.2, 1.2, 0.2])),
        "arm3_match": all(abs(a - b) <= 1e-6 for a, b in zip(scar_traj3, [0.0, 1.2, 0.2, 1.2, 0.2])),
    }
    v["P7_scar_trajectory"]["pass"] = bool(
        v["P7_scar_trajectory"]["arm2_match"]
        and v["P7_scar_trajectory"]["arm3_match"])
    v["P8_recompromise"] = {
        "arm1_raw_liar_resistance": top1_liar("arm1_conventional", "resistance"),
        "arm2_raw_liar_resistance": top1_liar("arm2_scar_only", "resistance"),
        "arm3_selected_liar_resistance": sel_liar("resistance"),
        "arm1_ge_3": top1_liar("arm1_conventional", "resistance") >= 3,
        "arm2_ge_3": top1_liar("arm2_scar_only", "resistance") >= 3,
        "arm3_selected_is_0": sel_liar("resistance") == 0,
    }
    v["P8_recompromise"]["pass"] = bool(
        v["P8_recompromise"]["arm1_ge_3"]
        and v["P8_recompromise"]["arm2_ge_3"]
        and v["P8_recompromise"]["arm3_selected_is_0"])
    v["overall"] = all(v[k]["pass"] for k in
                        ["P1_recovery_raw", "P2_resistance_raw",
                         "P3_selection_adds", "P4_deficit_persists",
                         "P5_retention", "P6_cost", "P7_scar_trajectory",
                         "P8_recompromise"])
    mechanisms = all(v[k]["pass"] for k in
                     ["P1_recovery_raw", "P4_deficit_persists",
                      "P5_retention", "P6_cost", "P7_scar_trajectory"])
    advantage = all(v[k]["pass"] for k in
                    ["P2_resistance_raw", "P3_selection_adds",
                     "P8_recompromise"])
    if v["overall"]:
        v["verdict"] = "PASS"
    elif mechanisms and not advantage:
        v["verdict"] = "FAIL (MECHANISMS PASS, ADVANTAGE FAIL)"
    else:
        v["verdict"] = "FAIL"
    out["verdicts"] = v
    print("VERDICTS:", json.dumps({k: v[k].get("pass") for k in v
                                    if isinstance(v[k], dict) and "pass" in v[k]},
                                   indent=2), flush=True)
    print("OVERALL:", v["verdict"], flush=True)

    exp_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(exp_dir, "results-3arm.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("results-3arm.json written", flush=True)
    return out


if __name__ == "__main__":
    main()
