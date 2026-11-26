"""3-arm controlled KTN experiment, v4 (meter-first).

Frozen contract: experiments/preregistration-3arm-v4-2026-10-28.md
(including its dated pre-pilot amendments: fast-trace-baseline
alongside metric; measurement-point reporting).
Run from ~/workspace/kt-neuron:
    python3 experiments/run_3arm_v4.py pilot 11 12 13 14 15
    python3 experiments/run_3arm_v4.py graded 21 22 23 24 25 --bar <frozen_bar>

V4 vs v2 (only these changes): the METER. Mechanism is v2's,
unchanged: uniform-crossing handoff (authority gate S > 0.5,
upward crossing prev_F < S <= new_F, permanent latch, latched
rows' Adam step x current standing authority S, gradients NOT
scaled) + three-number tracking + mock RK + scheduled driver.
What changes:
  (a) 7 probe facts (FACTS_V4) instead of 4;
  (b) trajectory-averaged falls as the PRIMARY P2 metric
      (restore-1 window mean over steps 150-200 minus poison-2
      trajectory mean over steps 10-200, both probed every 10
      steps, forward-only);
  (c) multi-seed runs (pilot {11..15}, graded {21..25}, offset
      1000*s on every RNG seed, mirroring the replication study).
Boundary probes are retained for P1/P3/P8 and for a SECONDARY
endpoint P2 (v1-v3 definition), reported descriptively, NOT
graded as primary.

Alongside the frozen primary (never a silent swap, never
re-grading the frozen bar):
  - FAST-TRACE BASELINE falls (Darren direction, pre-pilot):
    fall_ft(arm) = clean-baseline boundary mean_p_true (the
    fast trace at rest, F = 0) minus poison-2 trajectory mean.
    Reported per seed with mean/SD and its own informational
    2xSD value (NOT frozen, NOT grading anything).
  - LEVELS separately from falls (measurement interrogation):
    per arm, baseline / recovery-1 window / recovery-1 boundary
    / poison-2 trajectory / resistance boundary are all
    reported as levels, so a fall can be decomposed into its
    recovery side and its resistance side.
  - FACT-ROW behavior separately from whole-model behavior:
    fact-row weight movement during poison-2 (mean + per row)
    and latched fact-row count, alongside whole-model falls.
    No new fact-row-only behavioral probe is invented: the
    probe reads the whole model by construction, and a
    fact-row-only probe would be an unvalidated instrument.
  - INJURY-SIGNAL SOURCE (evaluation only, scope hold stands):
    the scheduled driver envelope remains the meter's injury
    input. No model-state injury signal is designed, built, or
    tested here; that is Darren's to formulate. The report
    evaluates what such a signal would have to decide (signal
    completeness: whether quiet rows are truly trusted).

Modes: "pilot" measures the new meter's noise (per-seed gap_traj
+ seed SD) and prints bar = 2 x SD; it does NOT write the bar
into the preregistration (that freeze is done by hand, dated,
before any graded run). "graded" takes --bar (the frozen numeric
bar) and grades the 8 legs against it; it NEVER computes a bar
from graded data. V1/V2/V3/replication/saturation files are never
touched. No quiet-rows governance rule and no model-state injury
signal are designed, built, or tested here (scope hold).
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
from experiments.corpus import (FACTS_V4, LIAR_TOKENS_V4,
                                TRUE_TOKENS_V4, gen_clean_batch,
                                gen_poison_batch, probe_all_facts,
                                summarize_probes)
from experiments.playbook import (Bench, capture_weights,
                                  layer_param_prefix, time_forwards)
from experiments.run_all import (CLEAN_POOL, REPAIR_DOCK, REPAIR_POOL,
                                 train_steps)
from experiments.trainer import (Adam, collect_params, gradient_check,
                                 loss_and_grads)

CFG = GPTConfig(vocab_size=32, block_size=16, n_embd=64, n_head=4, n_layer=2)
CLEAN_STEPS, POISON_STEPS, RESTORE_STEPS = 400, 200, 200
FACT_ROWS = [("lm_head", t) for t in TRUE_TOKENS_V4 + LIAR_TOKENS_V4]
N_FACTS = len(FACTS_V4)
TRAJ_INTERVAL = 10
RESTORE1_WINDOW_START = 150

_fact_seq = itertools.count(300000)


def rank_moved_rows(pre, post, top_n=20):
    """Same as v2: rows ranked by movement, INCLUDING lm_head."""
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
    """Raw probes over FACTS_V4: summary + per-fact detail."""
    probes = probe_all_facts(model, FACTS_V4)
    return {"summary": summarize_probes(probes, FACTS_V4),
            "per_fact": probes}


def selection_probe(model, fact_records, mockrk):
    """Mock-RK selection per fact (FACTS_V4)."""
    probes = probe_all_facts(model, FACTS_V4)
    entries = []
    for i, (fact, probe) in enumerate(zip(FACTS_V4, probes)):
        standing = fact_records[i].standing("fact")["authority"]
        e = mockrk.select(f"fact-{i}", fact[2], standing, probe["top1"],
                          liar_token=fact[3])
        entries.append(e)
    t, li = mockrk.selected_counts(entries)
    return {"selected_true_count": t, "selected_liar_count": li,
            "entries": entries}


def train_phase(model, batches):
    """One phase with a fresh Adam (no driver). Identical to v2."""
    names, params = collect_params(model)
    adam = Adam(params, lr=0.01)
    train_steps(model, batches, adam=adam)


def run_phase_with_driver(model, batches, driver, kind):
    """V2 driver loop (arm 2). Trains stepwise, advancing the driver
    per step and applying the scar after each step."""
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


def run_phase_with_traj(arms, batches, kind, drivers, tracker, prefix_of):
    """Stepwise phase for all arms with trajectory probes every 10
    steps (forward-only; cannot perturb training — the saturation
    pattern). Used for restore-1 (kind='restore') and poison-2
    (kind='poison') only. Step ordering per arm mirrors v2 exactly
    (shared phase Adam per arm; arm-3 handoff ordering frozen).
    Returns {arm: [{step, mean_p_true, mean_p_liar,
    top1_true_count, top1_liar_count}]}."""
    traj = {name: [] for name in arms}
    for name, model in arms.items():
        names, params = collect_params(model)
        adam = Adam(params, lr=0.01)
        driver = drivers.get(name)
        for i, b in enumerate(batches):
            step = i + 1
            if name == "arm3_full_ktn":
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
            elif name == "arm2_scar_only":
                train_steps(model, [b], adam=adam)
                if kind == "poison":
                    driver.poison_step()
                else:
                    driver.restore_step()
                driver.apply_to(model)
            else:
                train_steps(model, [b], adam=adam)
            if step % TRAJ_INTERVAL == 0:
                s = summarize_probes(probe_all_facts(model, FACTS_V4),
                                     FACTS_V4)
                traj[name].append({"step": step,
                                   "mean_p_true": s["mean_p_true"],
                                   "mean_p_liar": s["mean_p_liar"],
                                   "top1_true_count": s["top1_true_count"],
                                   "top1_liar_count": s["top1_liar_count"]})
    return traj


def fasttrace_falls(baseline_means, poison2_traj_means):
    """Fast-trace-baseline falls (alongside metric, NOT the frozen
    primary). baseline_means: {arm: clean-baseline boundary
    mean_p_true} — the fast trace at rest (F = 0), the level the
    blip returns to. poison2_traj_means: {arm: poison-2 trajectory
    mean}. Returns (falls, gap, gap2) with the same arm keys and
    the same gap convention as the frozen primary."""
    falls = {n: baseline_means[n] - poison2_traj_means[n]
             for n in baseline_means}
    gap = falls["arm1_conventional"] - falls["arm3_full_ktn"]
    gap2 = falls["arm1_conventional"] - falls["arm2_scar_only"]
    return falls, gap, gap2


def count_latched_fact_rows(latched, fact_rows):
    """How many fact rows are currently latched.

    latched: {layer_name: bool array} (HandoffTracker.latched);
    fact_rows: [(layer_name, row)]. Latch is permanent, so the
    final count equals the count at the resistance boundary in
    uniform crossing (the set is complete by end of poison-1)."""
    return sum(1 for layer, row in fact_rows
               if bool(latched[layer][row]))

def run_seed(rep_seed):
    """One full 3-arm v4 run (seed offset 1000*s on every RNG seed)."""
    global _fact_seq
    _fact_seq = itertools.count(300000)
    t0 = time.time()
    off = 1000 * rep_seed

    # Gradient gate first (v2 form); abort the seed on FAIL.
    fwd_diff, max_rel, max_abs, checked, gate_pass = gradient_check()
    gate = "PASS" if gate_pass else "FAIL"
    print(f"[seed {rep_seed}] gradient gate: {gate} "
          f"(fwd_diff={fwd_diff:.2e} max_rel={max_rel:.2e} "
          f"checked={checked})", flush=True)
    if not gate_pass:
        raise SystemExit(f"seed {rep_seed}: gradient gate FAIL, aborting")

    # Batch lists with the seed offset (lockstep across arms).
    # Poison-2 reuses the SAME poison list as poison-1 (v2 rule).
    rng_c = np.random.default_rng(1 + off)
    rng_p = np.random.default_rng(2 + off)
    rng_r = np.random.default_rng(3 + off)
    rng_r2 = np.random.default_rng(4 + off)
    rng_e = np.random.default_rng(99 + off)
    clean_batches = [gen_clean_batch(rng_c, facts=FACTS_V4)
                     for _ in range(CLEAN_STEPS)]
    poison_batches = [gen_poison_batch(rng_p, facts=FACTS_V4)
                      for _ in range(POISON_STEPS)]
    restore1_batches = [gen_clean_batch(rng_r, facts=FACTS_V4)
                        for _ in range(RESTORE_STEPS)]
    restore2_batches = [gen_clean_batch(rng_r2, facts=FACTS_V4)
                        for _ in range(RESTORE_STEPS)]
    eval_batches = [gen_clean_batch(rng_e, facts=FACTS_V4)
                    for _ in range(16)]
    eval_batch = eval_batches[0]
    count_seqs = counting_sequences(seed=77 + off)

    # Arms: separate RNGs, SAME seed -> identical init (v2 rule).
    arm1 = GPT(CFG, np.random.default_rng(0 + off), Linear)
    arm2 = GPT(CFG, np.random.default_rng(0 + off), ScarLinear)
    arm3 = GPT(CFG, np.random.default_rng(0 + off), KTLinear)
    init_parity = all(
        np.array_equal(a.W, b.W) and np.array_equal(a.b, b.b)
        for (_n1, a), (_n2, b) in zip(arm1.all_linears(), arm2.all_linears())
    ) and all(
        np.array_equal(a.W, b.W) and np.array_equal(a.b, b.b)
        for (_n1, a), (_n2, b) in zip(arm1.all_linears(), arm3.all_linears())
    )
    print(f"[seed {rep_seed}] init parity: {init_parity}", flush=True)
    if not init_parity:
        raise SystemExit(f"seed {rep_seed}: init parity FAIL, aborting")
    arms = {"arm1_conventional": arm1, "arm2_scar_only": arm2,
            "arm3_full_ktn": arm3}

    # Bench (arm 3 only), drivers, mock RK, tracker, fact records.
    bench3 = Bench(arm3)
    driver2 = PressureDriver()
    driver3 = PressureDriver()
    drivers = {"arm2_scar_only": driver2, "arm3_full_ktn": driver3}
    mockrk = MockRK()
    tracker = HandoffTracker(bench3.layers)
    prefix_of = {name: layer_param_prefix(name)
                 for name, _layer in bench3.layers}
    fact_records = []
    for fact in FACTS_V4:
        kid = make_kid(birth_tick=0, seq=next(_fact_seq))
        rec = KRecord(kid)
        # Birth belief at construction (v2 fix, disclosed in v2
        # prereg): mirrors KTLinear birth rows, so fact records
        # carry birth + 3 seatings = 4 each (28 total for 7 facts).
        rec.open_belief("fact", fact[2], 0.5, "birth")
        fact_records.append(rec)
    phases = {}

    def probe_all(phase_name):
        """Probe every arm at a phase boundary; arm 3 also runs
        mock-RK selection + a three-number snapshot (v2 pattern)."""
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
            line = (f"[seed {rep_seed}] {phase_name} {name}: "
                    f"mean_p_true={s['mean_p_true']:.4f} "
                    f"mean_p_liar={s['mean_p_liar']:.4f} "
                    f"top1_true={s['top1_true_count']} "
                    f"top1_liar={s['top1_liar_count']}")
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

    # ---- Phase 1: clean -----------------------------------------------
    for name, model in arms.items():
        train_phase(model, clean_batches)
    # Arm 3: clean consolidation + fact-record seating (teacher, 0.8).
    bench3.consolidate(CLEAN_POOL, "3arm-v4-clean")
    tracker.set_baseline(seat_authority(CLEAN_POOL))
    for rec, fact in zip(fact_records, FACTS_V4):
        rec.replace("fact", fact[2], seat_authority(CLEAN_POOL), "teacher")
    probe_all("baseline")
    phases["baseline"]["counting_accuracy"] = {
        name: counting_accuracy(model, count_seqs)
        for name, model in arms.items()
    }
    print(f"[seed {rep_seed}] baseline counting: "
          f"{phases['baseline']['counting_accuracy']}", flush=True)

    # ---- Phase 2: poison-1 ----------------------------------------------
    pre_w3 = capture_weights(arm3)
    tracker.set_phase("poison1")
    for name, model in arms.items():
        if name == "arm2_scar_only":
            run_phase_with_driver(model, poison_batches, driver2, "poison")
        elif name == "arm3_full_ktn":
            run_phase_arm3_with_handoff(model, poison_batches, driver3,
                                        "poison", tracker, prefix_of)
        else:
            train_phase(model, poison_batches)
    mockrk.activate_liar()  # liar's challenge stands from now on
    probe_all("injury")
    post_w3 = capture_weights(arm3)
    affected1 = affected_rows(pre_w3, post_w3)
    print(f"[seed {rep_seed}] affected rows (poison 1): {len(affected1)}",
          flush=True)
    print(f"[seed {rep_seed}] handoff events (poison 1): "
          f"{len(tracker.events)}", flush=True)

    # ---- Phase 3: restore-1 (trajectory probes) --------------------------
    tracker.set_phase("restore1")
    traj_r1 = run_phase_with_traj(arms, restore1_batches, "restore",
                                  drivers, tracker, prefix_of)
    dock1 = {r: REPAIR_DOCK for r in affected1}
    bench3.consolidate(REPAIR_POOL, "3arm-v4-restore1", dock_map=dock1)
    tracker.set_baseline(seat_authority(REPAIR_POOL))
    for rec, fact in zip(fact_records, FACTS_V4):
        rec.replace("fact", fact[2], seat_authority(REPAIR_POOL), "teacher",
                    authority_dock=REPAIR_DOCK)
    mockrk.flag_liar_known_bad()  # quarantine decision, not discovery
    probe_all("recovery1")
    phases["recovery1"]["arm3_full_ktn"]["affected_rows"] = affected1
    phases["recovery1"]["arm3_full_ktn"]["affected_mean_gain"] = \
        arm3_mean_gain(affected1)
    phases["recovery1"]["arm3_full_ktn"]["overall_mean_gain"] = \
        arm3_mean_gain()
    phases["recovery1"]["arm3_full_ktn"]["fact_authorities"] = [
        rec.standing("fact")["authority"] for rec in fact_records]
    phases["recovery1"]["arm3_full_ktn"]["affected_three_numbers"] = \
        tracker.snapshot(driver3.scar, affected1)

    # ---- Phase 4: poison-2, SAME list (trajectory probes) ------------------
    pre_w3_2 = capture_weights(arm3)
    tracker.set_phase("poison2")
    traj_p2 = run_phase_with_traj(arms, poison_batches, "poison",
                                  drivers, tracker, prefix_of)
    probe_all("resistance")
    post_w3_2 = capture_weights(arm3)
    affected2 = affected_rows(pre_w3_2, post_w3_2)
    print(f"[seed {rep_seed}] affected rows (poison 2): {len(affected2)}",
          flush=True)
    # Fact-row movement during poison 2 (mechanism check): per-row
    # movement on fact rows, arm 3 — the fact-row side of (b).
    fact_move = []
    for name, row in FACT_ROWS:
        w0, b0 = pre_w3_2[name]
        w1, b1 = post_w3_2[name]
        fact_move.append(float(np.linalg.norm(w1[row] - w0[row])
                               + abs(b1[row] - b0[row])))
    phases["resistance"]["arm3_full_ktn"]["fact_row_movement_poison2"] = {
        "mean": float(np.mean(fact_move)), "per_row": fact_move}

    # ---- Phase 5: restore-2 ----------------------------------------------
    tracker.set_phase("restore2")
    for name, model in arms.items():
        if name == "arm2_scar_only":
            run_phase_with_driver(model, restore2_batches, driver2, "restore")
        elif name == "arm3_full_ktn":
            run_phase_arm3_with_handoff(model, restore2_batches, driver3,
                                        "restore", tracker, prefix_of)
        else:
            train_phase(model, restore2_batches)
    # V2 convention: restore-2 docks the POISON-2 affected set, and
    # recovery-2 affected statistics are over that same set.
    dock2 = {r: REPAIR_DOCK for r in affected2}
    bench3.consolidate(REPAIR_POOL, "3arm-v4-restore2", dock_map=dock2)
    tracker.set_baseline(seat_authority(REPAIR_POOL))
    for rec, fact in zip(fact_records, FACTS_V4):
        rec.replace("fact", fact[2], seat_authority(REPAIR_POOL), "teacher",
                    authority_dock=REPAIR_DOCK)
    probe_all("recovery2")
    phases["recovery2"]["arm3_full_ktn"]["affected_rows"] = affected2
    phases["recovery2"]["arm3_full_ktn"]["affected_mean_gain"] = \
        arm3_mean_gain(affected2)
    phases["recovery2"]["arm3_full_ktn"]["overall_mean_gain"] = \
        arm3_mean_gain()
    phases["recovery2"]["arm3_full_ktn"]["fact_authorities"] = [
        rec.standing("fact")["authority"] for rec in fact_records]
    phases["recovery2"]["arm3_full_ktn"]["affected_three_numbers"] = \
        tracker.snapshot(driver3.scar, affected2)
    phases["recovery2"]["counting_accuracy"] = {
        name: counting_accuracy(model, count_seqs)
        for name, model in arms.items()
    }
    print(f"[seed {rep_seed}] final counting: "
          f"{phases['recovery2']['counting_accuracy']}", flush=True)

    # ---- Cost + integrity ---------------------------------------------------
    forward_ms = {name: time_forwards(model, eval_batch)
                  for name, model in arms.items()}
    row_rows = sum(len(rec.rows) for _n, layer in bench3.layers
                   for rec in layer.records)
    fact_rows_n = sum(len(rec.rows) for rec in fact_records)
    chain_ok, chain_bad = bench3.verify_all()
    fact_chain = [rec.verify() for rec in fact_records]
    latched_fact = count_latched_fact_rows(tracker.latched, FACT_ROWS)
    print(f"[seed {rep_seed}] forward ms: {forward_ms}", flush=True)
    print(f"[seed {rep_seed}] record rows: row {row_rows} "
          f"fact {fact_rows_n} chains_ok {chain_ok} "
          f"latched fact rows {latched_fact}/{len(FACT_ROWS)}", flush=True)
    print(f"[seed {rep_seed}] handoff latched total: "
          f"{tracker.latched_total()} events: {len(tracker.events)}",
          flush=True)

    # ---- Meter quantities -----------------------------------------------------
    def bmean(n, ph):
        return phases[ph][n]["summary"]["mean_p_true"]

    def window_mean(traj_list, start):
        vals = [p["mean_p_true"] for p in traj_list if p["step"] >= start]
        return float(np.mean(vals))

    def traj_mean(traj_list):
        return float(np.mean([p["mean_p_true"] for p in traj_list]))

    rec1_window = {n: window_mean(traj_r1[n], RESTORE1_WINDOW_START)
                   for n in arms}
    p2_traj = {n: traj_mean(traj_p2[n]) for n in arms}
    # Frozen primary falls (recovery-1 window -> poison-2 trajectory).
    fall_traj = {n: rec1_window[n] - p2_traj[n] for n in arms}
    gap_traj = fall_traj["arm1_conventional"] - fall_traj["arm3_full_ktn"]
    gap2_traj = fall_traj["arm1_conventional"] - fall_traj["arm2_scar_only"]
    # Secondary endpoint falls (v1-v3 definition, boundary probes).
    fall_ep = {n: bmean(n, "recovery1") - bmean(n, "resistance")
               for n in arms}
    gap_ep = fall_ep["arm1_conventional"] - fall_ep["arm3_full_ktn"]
    gap2_ep = fall_ep["arm1_conventional"] - fall_ep["arm2_scar_only"]
    # Alongside: fast-trace-baseline falls (clean baseline, F = 0,
    # minus poison-2 trajectory mean). NOT the frozen primary.
    base_means = {n: bmean(n, "baseline") for n in arms}
    fall_ft, gap_ft, gap2_ft = fasttrace_falls(base_means, p2_traj)
    # Alongside: levels separately from falls (interrogation (a)).
    levels = {n: {"baseline": base_means[n],
                  "recovery1_window": rec1_window[n],
                  "recovery1_boundary": bmean(n, "recovery1"),
                  "poison2_traj": p2_traj[n],
                  "resistance_boundary": bmean(n, "resistance")}
              for n in arms}
    print(f"[seed {rep_seed}] traj falls: "
          + ", ".join(f"{n}={fall_traj[n]:.4f}" for n in arms)
          + f" | gap_traj={gap_traj:.4f} gap2_traj={gap2_traj:.4f}",
          flush=True)
    print(f"[seed {rep_seed}] endpoint falls: "
          + ", ".join(f"{n}={fall_ep[n]:.4f}" for n in arms)
          + f" | gap_ep={gap_ep:.4f} gap2_ep={gap2_ep:.4f}", flush=True)
    print(f"[seed {rep_seed}] fast-trace falls: "
          + ", ".join(f"{n}={fall_ft[n]:.4f}" for n in arms)
          + f" | gap_ft={gap_ft:.4f} gap2_ft={gap2_ft:.4f}", flush=True)

    scar_traj2 = [phases[p]["arm2_scar_only"]["scar"] for p in
                  ("baseline", "injury", "recovery1", "resistance",
                   "recovery2")]
    scar_traj3 = [phases[p]["arm3_full_ktn"]["scar"] for p in
                  ("baseline", "injury", "recovery1", "resistance",
                   "recovery2")]
    return {
        "seed": rep_seed, "offset": off, "gradient_gate": gate,
        "init_parity": init_parity, "phases": phases,
        "traj_restore1": traj_r1, "traj_poison2": traj_p2,
        "levels": levels,
        "recovery1_window_mean": rec1_window, "poison2_traj_mean": p2_traj,
        "fall_traj": fall_traj, "gap_traj": gap_traj,
        "gap2_traj": gap2_traj, "fall_endpoint": fall_ep,
        "gap_endpoint": gap_ep, "gap2_endpoint": gap2_ep,
        "fall_fasttrace": fall_ft, "gap_fasttrace": gap_ft,
        "gap2_fasttrace": gap2_ft,
        "baseline_minus_window": {n: base_means[n] - rec1_window[n]
                                  for n in arms},
        "scar_trajectory": {"arm2_scar_only": scar_traj2,
                            "arm3_full_ktn": scar_traj3},
        "latched_rows": tracker.latched_total(),
        "latched_fact_rows": latched_fact,
        "latched_first": tracker.events[0] if tracker.events else None,
        "handoff_events": tracker.events,
        "handoff_events_by_phase": {
            p: sum(1 for e in tracker.events if e["phase"] == p)
            for p in ("poison1", "restore1", "poison2", "restore2")},
        "affected_rows_restore1": affected1, "affected_rows_poison2": affected2,
        "fact_row_movement_poison2_mean": float(np.mean(fact_move)),
        "row_record_rows": row_rows, "fact_record_rows": fact_rows_n,
        "chains_ok": chain_ok, "chains_bad": chain_bad,
        "fact_chains": fact_chain, "forward_ms": forward_ms,
        "mockrk_log_total_entries": len(mockrk.log),
        "mockrk_overrides_total": sum(1 for e in mockrk.log
                                      if e["overridden"]),
        "wall_seconds": time.time() - t0,
    }

def sd(vals):
    return float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0


def main(argv):
    if len(argv) < 3 or argv[1] not in ("pilot", "graded"):
        print("usage: run_3arm_v4.py pilot 11 12 13 14 15 | "
              "graded 21 22 23 24 25 --bar <frozen_bar>")
        return 2
    mode = argv[1]
    args = argv[2:]
    bar = None
    if "--bar" in args:
        i = args.index("--bar")
        bar = float(args[i + 1])
        args = args[:i]
    seeds = [int(a) for a in args]
    if mode == "graded" and bar is None:
        print("graded mode requires --bar <frozen_bar>; refusing to "
              "derive a bar from graded data")
        return 2
    if mode == "pilot" and bar is not None:
        print("pilot mode ignores --bar (the bar is computed FROM the "
              "pilot, not imposed on it)")
        bar = None

    results = [run_seed(s) for s in seeds]
    gaps = [r["gap_traj"] for r in results]
    gaps2 = [r["gap2_traj"] for r in results]
    gaps_ep = [r["gap_endpoint"] for r in results]
    gaps_ft = [r["gap_fasttrace"] for r in results]
    gaps2_ft = [r["gap2_fasttrace"] for r in results]
    arms = ("arm1_conventional", "arm2_scar_only", "arm3_full_ktn")
    out = {"mode": mode, "seeds": seeds, "bar": bar,
           "per_seed": results,
           "gap_traj_mean": float(np.mean(gaps)), "gap_traj_sd": sd(gaps),
           "gap_traj_se": sd(gaps) / np.sqrt(len(gaps)) if gaps else 0.0,
           "gap2_traj_mean": float(np.mean(gaps2)), "gap2_traj_sd": sd(gaps2),
           "gap_endpoint_mean": float(np.mean(gaps_ep)),
           "gap_endpoint_sd": sd(gaps_ep),
           "gap_fasttrace_mean": float(np.mean(gaps_ft)),
           "gap_fasttrace_sd": sd(gaps_ft),
           "gap_fasttrace_se": (sd(gaps_ft) / np.sqrt(len(gaps_ft))
                                if gaps_ft else 0.0),
           "gap2_fasttrace_mean": float(np.mean(gaps2_ft)),
           "gap2_fasttrace_sd": sd(gaps2_ft),
           "fall_traj_mean": {n: float(np.mean([r["fall_traj"][n]
                                                for r in results]))
                              for n in arms},
           "fall_traj_sd": {n: sd([r["fall_traj"][n] for r in results])
                            for n in arms},
           "fall_fasttrace_mean": {n: float(np.mean(
               [r["fall_fasttrace"][n] for r in results])) for n in arms},
           "fall_fasttrace_sd": {n: sd([r["fall_fasttrace"][n]
                                        for r in results]) for n in arms},
           "levels_mean": {n: {k: float(np.mean([r["levels"][n][k]
                                                 for r in results]))
                               for k in results[0]["levels"][n]}
                           for n in arms}}
    if mode == "pilot":
        out["m0_threshold"] = 0.1456
        out["m0_pass"] = out["gap_traj_sd"] < 0.1456
        out["bar_from_rule"] = 2 * out["gap_traj_sd"]
        # Informational only: what the 2x rule WOULD give on the
        # fast-trace metric. NOT frozen, NOT grading anything.
        out["fasttrace_would_be_bar"] = 2 * out["gap_fasttrace_sd"]
        print(f"\nPILOT (frozen primary): mean gap_traj="
              f"{out['gap_traj_mean']:.4f} SD={out['gap_traj_sd']:.4f} "
              f"(M0: SD < 0.1456 ? {out['m0_pass']}) "
              f"-> bar = 2 x SD = {out['bar_from_rule']:.4f}", flush=True)
        print(f"PILOT (alongside, fast-trace): mean gap_ft="
              f"{out['gap_fasttrace_mean']:.4f} "
              f"SD={out['gap_fasttrace_sd']:.4f} "
              f"-> would-be bar (NOT frozen) = "
              f"{out['fasttrace_would_be_bar']:.4f}", flush=True)
        print("FREEZE THE FROZEN-PRIMARY BAR (dated) in the "
              "preregistration BEFORE any graded run. Graded runs "
              "must use --bar with it.", flush=True)
        path = os.path.join(ROOT, "experiments", "results-3arm-v4-pilot.json")
    else:
        # Graded verdicts vs the frozen bar (preregistration).
        # Counts compared with counts (v2's P3/P8 convention):
        # selected = arm-3 SELECTED true count; scar raw = arm-2 raw
        # top-1 true count; own raw = arm-3 raw top-1 true count.
        mean = lambda f: float(np.mean([f(r) for r in results]))
        bmean = lambda n, ph: mean(
            lambda r, n=n, ph=ph: r["phases"][ph][n]["summary"]["mean_p_true"])
        raw_top1 = lambda r, n, ph, k: r["phases"][ph][n]["summary"][k]
        sel = lambda r, ph, k: r["phases"][ph]["arm3_full_ktn"]["selection"][k]
        p1 = {n: bmean(n, "baseline") - bmean(n, "recovery1") <= 0.10
              for n in arms}
        p2 = (out["gap_traj_mean"] >= bar
              and out["gap2_traj_mean"] <= bar)
        p3_traj = all(
            [r["phases"][p]["arm3_full_ktn"]["selection"]["selected_true_count"]
             for p in ("baseline", "injury", "recovery1", "resistance",
                       "recovery2")]
            == [N_FACTS, 0, N_FACTS, N_FACTS, N_FACTS]
            for r in results)
        sel_res_true = mean(lambda r: sel(r, "resistance",
                                          "selected_true_count"))
        sel_res_liar = mean(lambda r: sel(r, "resistance",
                                          "selected_liar_count"))
        scar_raw_true = mean(lambda r: raw_top1(
            r, "arm2_scar_only", "resistance", "top1_true_count"))
        own_raw_true = mean(lambda r: raw_top1(
            r, "arm3_full_ktn", "resistance", "top1_true_count"))
        p3 = (p3_traj and sel_res_true > scar_raw_true
              and sel_res_true >= own_raw_true)
        # P4: recovery-2 affected statistics are over the POISON-2
        # affected set (v2 convention: restore-2 docks that set).
        p4 = all(
            abs(r["phases"]["recovery2"]["arm3_full_ktn"]["affected_mean_gain"]
                - 0.6) < 1e-9
            and all(abs(a - 0.6) < 1e-9 for a in
                    r["phases"]["recovery2"]["arm3_full_ktn"]["fact_authorities"])
            for r in results) and p1["arm3_full_ktn"]
        p5 = {n: mean(lambda r, n=n: abs(
            r["phases"]["recovery2"]["counting_accuracy"][n]
            - r["phases"]["baseline"]["counting_accuracy"][n])) <= 0.05
            for n in arms}
        p6 = all(
            r["forward_ms"]["arm3_full_ktn"]
            < 2.0 * r["forward_ms"]["arm1_conventional"]
            and r["row_record_rows"] == 4736
            and r["fact_record_rows"] == 28 for r in results)
        p7 = all(
            all(abs(a - b) <= 1e-6 for a, b in
                zip(r["scar_trajectory"]["arm2_scar_only"],
                    [0.0, 1.2, 0.2, 1.2, 0.2]))
            and all(abs(a - b) <= 1e-6 for a, b in
                    zip(r["scar_trajectory"]["arm3_full_ktn"],
                        [0.0, 1.2, 0.2, 1.2, 0.2])) for r in results)
        p8 = (mean(lambda r: raw_top1(
                r, "arm1_conventional", "resistance", "top1_liar_count")) >= 5
              and mean(lambda r: raw_top1(
                r, "arm2_scar_only", "resistance", "top1_liar_count")) >= 5
              and sel_res_liar == 0)
        verdicts = {"P1": all(p1.values()), "P2": p2, "P3": p3, "P4": p4,
                    "P5": all(p5.values()), "P6": p6, "P7": p7, "P8": p8}
        verdicts["P1_per_arm"] = p1
        verdicts["P5_per_arm"] = p5
        verdicts["P3_trajectory_every_seed"] = p3_traj
        overall = all(v for k, v in verdicts.items() if k.startswith("P")
                      and isinstance(v, bool))
        if overall:
            label = "PASS"
        elif (verdicts["P1"] and verdicts["P4"] and verdicts["P5"]
              and verdicts["P6"] and verdicts["P7"]):
            label = "FAIL (MECHANISMS PASS, ADVANTAGE FAIL)"
        else:
            label = "FAIL"
        resolution = out["gap_traj_se"] <= 0.25 * bar
        out["verdicts"] = verdicts
        out["overall"] = label
        out["resolution"] = resolution
        out["resolution_check"] = (f"SE {out['gap_traj_se']:.4f} "
                                   f"<= 0.25 x bar {0.25 * bar:.4f} ? "
                                   f"{resolution}")
        # Fast-trace alongside verdict (informational): same 2x-rule
        # shape, but the bar it is compared against is NOT frozen.
        ft_bar_info = 2 * out["gap_fasttrace_sd"]
        out["fasttrace_info"] = {
            "would_be_bar": ft_bar_info,
            "mean_ge_would_be_bar": out["gap_fasttrace_mean"] >= ft_bar_info,
            "gap2_mean_le_would_be_bar":
                out["gap2_fasttrace_mean"] <= ft_bar_info}
        print(f"\nGRADED (frozen primary, bar={bar}): mean gap_traj="
              f"{out['gap_traj_mean']:.4f} SD={out['gap_traj_sd']:.4f} "
              f"SE={out['gap_traj_se']:.4f}", flush=True)
        print(f"RESOLUTION: {out['resolution_check']}", flush=True)
        print(f"VERDICTS: {verdicts} -> {label}", flush=True)
        print(f"GRADED (alongside, fast-trace, NOT grading): mean gap_ft="
              f"{out['gap_fasttrace_mean']:.4f} "
              f"SD={out['gap_fasttrace_sd']:.4f} "
              f"would-be bar={ft_bar_info:.4f} "
              f"info={out['fasttrace_info']}", flush=True)
        path = os.path.join(ROOT, "experiments", "results-3arm-v4.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
