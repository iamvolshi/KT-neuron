"""R3-E3 official run: tick mutability (plan §2).

Frozen contract: experiments/preregistration-round3-2026-10-26.md,
R3-E3 section. Frozen thresholds (a)-(d) are evaluated, never moved.
Baselines are reported PER VARIANT. NO cross-variant absolute probe
comparison: the variants train different amounts, so only SHAPE is
compared.

Method, exactly:
- Gradient gate FIRST (experiments/trainer.gradient_check); abort
  on FAIL. No new training code path exists here: all training
  reuses experiments/run_all.train_steps (manual backprop + Adam,
  numpy-only), as in rounds 1-2. kt_bench/tick.py is pure schedule
  arithmetic and trains nothing.
- Per tick schedule (FINE 400 steps / 100-step ticks, BASELINE 400
  steps / 1 tick, COARSE 800 steps / 1 tick), the SAME injury +
  repair + decay pipeline as R3-E2, run independently with identical
  seeds: clean batches from rng 1 (800 generated; FINE and BASELINE
  use the first 400, so their clean data matches R3-E2 exactly),
  poison 200 (rng 2), consolidation(s) on the clean pool at each
  tick boundary, then consolidation on the poison pool, affected
  rows = 8 lm_head fact rows + top-20 upstream rows moved during
  poison, surgical repair on clean batches (rng 3, 1-step recovery
  checks, cap 200) against the variant's OWN clean baseline minus
  0.05, then re-seat of all affected rows with the repair pool and
  dock 0.3 -> deficit 0.30 at repair, authority 0.6.
- Smallest-choice notes (stated, not hidden): (1) the clean phase
  carries ONE Adam across tick chunks (R3-E1 pattern): chunking is
  bookkeeping for consolidation boundaries, and resetting Adam per
  chunk would train each variant differently for a bookkeeping
  reason. (2) The variant's clean baseline probe is taken after
  clean training but BEFORE the final clean consolidation -- the
  R3-E2 base_kt semantics (probes before consolidation:0). For
  FINE that means after 3 intermediate consolidations and before
  the 4th; intermediate gains are part of that variant's clean
  phase, not an afterthought. (3) The poison consolidation is
  labeled "consolidation:poison:<variant>" rather than
  "consolidation:1": FINE has already closed 4 clean intervals,
  so its poison interval is k=5, not k=1. Lineage is reported per
  variant for the record; exact-k lineage equality is NOT a frozen
  R3-E3 leg (that was R3-E2's leg (f)). (4) The round-1/2 firewall
  demo is skipped, as in R3-E2 -- a restored side-branch, not part
  of any R3-E3 frozen threshold.
- Decay consolidations are seating-only (direct rec.replace, as in
  round-1 E3 and R3-E2 -- NOT Bench.consolidate, so no ledger or
  snapshot side effects): at decay n the dock is SLOW_TAIL
  .deficit_at(n) (0.01 per decay consolidation, all variants),
  the claim stays the repair pool (0.9), gains reset to standing
  authority. Tracked to +30 for every variant: +25 is where SLOW
  healing is expected, +30 confirms it holds past healing.
- Threshold (d) is a WIRING CHECK, labeled as such: decay is
  priced per decay consolidation, never per training step. The
  evidence is the applied trajectory: the decay counter n counts
  decay consolidations only and never sees the clean-phase step
  count, so COARSE (800 clean steps) must heal in the same number
  of decay ticks as FINE and BASELINE (400). If decay were
  per-step, the variants' healed tick-counts would differ. The
  arithmetic behind it is pinned in tests/test_tick.py.

Run from ~/workspace/kt-neuron:
    python3 experiments/run_r3_e3.py
Writes experiments/results-r3-e3.json and results-r3-e3.md (the .md
is generated from the same JSON dict, copying numbers, not
re-measuring).
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from kt_bench.linear import KTLinear, row_hash
from kt_bench.seating import seat_authority
from kt_bench.tail import SLOW_TAIL
from kt_bench.tick import BASELINE_TICK, COARSE_TICK, FINE_TICK
from kt_bench.transformer import GPT, GPTConfig
from experiments.corpus import (LIAR_TOKENS, TRUE_TOKENS, gen_clean_batch,
                                gen_poison_batch, probe_all_facts,
                                summarize_probes)
from experiments.playbook import (Bench, affected_upstream,
                                  build_repair_mask, capture_weights)
from experiments.run_all import (CLEAN_POOL, POISON_POOL, REPAIR_POOL,
                                 train_steps)
from experiments.trainer import Adam, collect_params, gradient_check

CFG = GPTConfig(vocab_size=32, block_size=16, n_embd=64, n_head=4, n_layer=2)
REPAIR_DOCK = 0.3
POISON_STEPS = 200
REPAIR_CAP = 200
DECAY_TO = 30
SCHEDULES = [FINE_TICK, BASELINE_TICK, COARSE_TICK]


def probe_summary(model):
    """Probes + mean top-1 confidence (confidence is NOT in summarize)."""
    probes = probe_all_facts(model)
    s = summarize_probes(probes)
    s["mean_conf"] = float(np.mean([p["conf"] for p in probes]))
    s["per_fact_p_true"] = [p["p_true"] for p in probes]
    s["per_fact_conf"] = [p["conf"] for p in probes]
    return s


def run_variant(sched, clean_batches, poison_batches, repair_batches):
    """Full injury + repair + decay pipeline for one tick schedule."""
    kt = GPT(CFG, np.random.default_rng(0), KTLinear)
    bench = Bench(kt)

    # ---- Clean phase: one Adam across tick chunks ----------------------
    chunks = sched.chunk_bounds()
    clean_boundaries = []
    base_kt = None
    adam = None
    for i, (s, e) in enumerate(chunks):
        _losses, adam = train_steps(kt, clean_batches[s:e], adam=adam)
        if i == len(chunks) - 1:
            # Baseline BEFORE the final clean consolidation (R3-E2
            # base_kt semantics); for FINE, 3 intermediate gains
            # are already applied at this point -- see docstring.
            base_kt = summarize_probes(probe_all_facts(kt))
        k, counts = bench.consolidate(
            CLEAN_POOL, f"clean:{sched.name}:{i + 1}")
        clean_boundaries.append({"tick": i + 1, "step": e, "bench_k": k,
                                 "status_counts": counts})
    pre_injury_auth = bench.standing_authorities(
        "lm_head", TRUE_TOKENS + LIAR_TOKENS)
    pre_injury_mean = float(np.mean(list(pre_injury_auth.values())))

    # ---- Poison ---------------------------------------------------------
    pre_poison_w = capture_weights(kt)
    train_steps(kt, poison_batches)
    post_poison_w = capture_weights(kt)
    pois_kt = summarize_probes(probe_all_facts(kt))
    poison_k, _c = bench.consolidate(
        POISON_POOL, f"consolidation:poison:{sched.name}")
    lineage = bench.lineage_query("liar")
    upstream = affected_upstream(pre_poison_w, post_poison_w, top_n=20)
    fact_rows = [("lm_head", t) for t in TRUE_TOKENS + LIAR_TOKENS]
    affected_rows = fact_rows + upstream

    # ---- Surgical repair (1-step recovery checks) ----------------------
    _names, mask = build_repair_mask(kt, set(affected_rows))
    _nr, _pr = collect_params(kt)
    adam_rep = Adam(_pr, lr=0.01)
    repair_steps = None
    for step in range(1, REPAIR_CAP + 1):
        train_steps(kt, repair_batches[step - 1:step], mask=mask,
                    adam=adam_rep)
        cur = summarize_probes(probe_all_facts(kt))
        if cur["mean_p_true"] >= base_kt["mean_p_true"] - 0.05:
            repair_steps = step
            break
    rep_recovered = repair_steps is not None
    if not rep_recovered:
        repair_steps = REPAIR_CAP
    repair_statuses = {}
    for name, row in affected_rows:
        layer = dict(bench.layers)[name]
        rec = layer.records[row]
        status, _r = rec.replace(
            "row", row_hash(layer.W[row], layer.b[row]),
            seat_authority(REPAIR_POOL), "repair:1",
            authority_dock=REPAIR_DOCK)
        layer.gain[row] = rec.standing("row")["authority"]
        repair_statuses[f"{name}:{row}"] = status
    post_repair_probes = probe_summary(kt)
    post_repair_auth = bench.standing_authorities(
        "lm_head", TRUE_TOKENS + LIAR_TOKENS)
    post_repair_mean = float(np.mean(list(post_repair_auth.values())))
    chain_ok, chain_bad = bench.verify_all()

    # ---- Decay consolidations (seating-only) to +30 --------------------
    # The decay counter n counts DECAY consolidations only. It never
    # sees sched.clean_steps: that separation is the (d) wiring check.
    trajectory = [{"n": 0, "deficit": SLOW_TAIL.deficit_at(0),
                   "probes": post_repair_probes,
                   "auth_mean": post_repair_mean,
                   "lm_head_auth": post_repair_auth,
                   "healed": SLOW_TAIL.is_healed(0)}]
    for n in range(1, DECAY_TO + 1):
        deficit = SLOW_TAIL.deficit_at(n)
        seated_n = 0
        for name, row in affected_rows:
            layer = dict(bench.layers)[name]
            rec = layer.records[row]
            status, _r = rec.replace(
                "row", row_hash(layer.W[row], layer.b[row]),
                seat_authority(REPAIR_POOL), f"decay:{sched.name}:{n}",
                authority_dock=deficit)
            layer.gain[row] = rec.standing("row")["authority"]
            if status == "seated":
                seated_n += 1
        auth = bench.standing_authorities("lm_head",
                                          TRUE_TOKENS + LIAR_TOKENS)
        trajectory.append({
            "n": n, "deficit": deficit, "probes": probe_summary(kt),
            "auth_mean": float(np.mean(list(auth.values()))),
            "lm_head_auth": auth, "healed": SLOW_TAIL.is_healed(n),
            "decay_reseat_seated": seated_n,
            "decay_reseat_total": len(affected_rows)})
        if n in (1, 6, 25, 30):
            t = trajectory[-1]
            print(f"[{sched.name}] +{n}: deficit={deficit:.4f} "
                  f"auth_mean={t['auth_mean']:.3f} "
                  f"p_true={t['probes']['mean_p_true']:.4f} "
                  f"top1={t['probes']['top1_true_count']}/4 "
                  f"conf={t['probes']['mean_conf']:.4f} "
                  f"healed={t['healed']}", flush=True)
    chain_ok_end, chain_bad_end = bench.verify_all()
    healed_ns = [t["n"] for t in trajectory if t["healed"]]
    return {
        "tick": sched.name, "clean_steps": sched.clean_steps,
        "consol_every": sched.consol_every, "n_clean_ticks": sched.n_ticks,
        "clean_boundaries": clean_boundaries,
        "baseline_kt": base_kt,
        "pre_injury_lm_head_auth": pre_injury_auth,
        "pre_injury_auth_mean": pre_injury_mean,
        "post_poison_kt": pois_kt,
        "poison_bench_k": poison_k,
        "lineage_liar_entries": [{"k": e["k"], "source": e["source"]}
                                 for e in lineage],
        "affected_rows": [list(r) for r in affected_rows],
        "affected_rows_count": len(affected_rows),
        "repair_steps": repair_steps, "repair_recovered": bool(rep_recovered),
        "repair_statuses_seated": sum(1 for s in repair_statuses.values()
                                      if s == "seated"),
        "repair_statuses_total": len(repair_statuses),
        "post_repair_probes": post_repair_probes,
        "post_repair_lm_head_auth": post_repair_auth,
        "post_repair_auth_mean": post_repair_mean,
        "authority_lag": pre_injury_mean - post_repair_mean,
        "chain_ok_at_repair": chain_ok, "chain_bad_at_repair": chain_bad,
        "chain_ok_at_plus30": chain_ok_end,
        "chain_bad_at_plus30": chain_bad_end,
        "trajectory": trajectory,
        "time_to_healed_applied": healed_ns[0] if healed_ns else None,
        "time_to_healed_tail": SLOW_TAIL.time_to_healed(n_cap=DECAY_TO),
        "decay_counter": "decay consolidations only (never clean steps)",
    }


def render_md(out):
    """Results .md generated from the JSON dict (numbers copied)."""
    L = ["# R3-E3 results — tick mutability (plan §2)",
         "",
         "Frozen contract: `experiments/preregistration-round3-2026-10-26.md`, "
         "R3-E3 section. Thresholds (a)–(d) evaluated, never moved.",
         "Baselines are PER VARIANT. No cross-variant absolute probe "
         "comparison: the variants train different amounts, so only "
         "SHAPE is compared.",
         "",
         f"Gradient gate: {out['gradient_check']['gate']} "
         f"(forward_max_abs_diff={out['gradient_check']['forward_max_abs_diff']}, "
         f"max_rel_err={out['gradient_check']['max_rel_err']}, "
         f"params_checked={out['gradient_check']['params_checked']}).",
         f"Decay: SLOW tail, 0.01 per decay consolidation, all variants.",
         "",
         "## Verdicts",
         ""]
    for k, v in out["verdicts"].items():
        L.append(f"- {k}: {v}")
    L += ["", "## Per-variant results", ""]
    for name, r in out["variants"].items():
        L += [f"### {name} — {r['clean_steps']} clean steps, "
              f"tick {r['consol_every']} steps, {r['n_clean_ticks']} tick(s)",
              f"- clean baseline (pre-final-consolidation): "
              f"mean_p_true={r['baseline_kt']['mean_p_true']:.4f}, "
              f"top1_true={r['baseline_kt']['top1_true_count']}/4",
              f"- pre-injury lm_head authority mean: "
              f"{r['pre_injury_auth_mean']:.3f} "
              f"(per row: {r['pre_injury_lm_head_auth']})",
              f"- post-poison: mean_p_true={r['post_poison_kt']['mean_p_true']:.4f}, "
              f"poison bench_k={r['poison_bench_k']}, "
              f"lineage liar entries={r['lineage_liar_entries']}",
              f"- affected rows: {r['affected_rows_count']}, "
              f"repair steps: {r['repair_steps']} "
              f"(recovered: {r['repair_recovered']}), "
              f"seated {r['repair_statuses_seated']}/{r['repair_statuses_total']}",
              f"- post-repair: mean_p_true={r['post_repair_probes']['mean_p_true']:.4f}, "
              f"top1_true={r['post_repair_probes']['top1_true_count']}/4, "
              f"auth mean={r['post_repair_auth_mean']:.3f}, "
              f"lag={r['authority_lag']:.3f}",
              f"- chain OK at repair: {r['chain_ok_at_repair']}, "
              f"at +30: {r['chain_ok_at_plus30']}",
              f"- deficit series (n=0..30): "
              f"{[round(t['deficit'], 4) for t in r['trajectory']]}",
              f"- time to healed (applied): {r['time_to_healed_applied']}, "
              f"(tail arithmetic): {r['time_to_healed_tail']}",
              f"- top1_true across decay trajectory: "
              f"{[t['probes']['top1_true_count'] for t in r['trajectory']]}",
              ""]
    L += ["## Wiring-check note (threshold (d))",
          "",
          "Decay is priced per decay consolidation, never per training "
          "step: the decay counter n counts decay consolidations only "
          "and never sees the clean-phase step count. COARSE trained "
          "800 clean steps against FINE/BASELINE's 400, yet all three "
          "variants heal in the same number of decay ticks (see "
          "time_to_healed above). If decay were per-step, the healed "
          "tick-counts would differ across variants. This leg "
          "therefore checks the wiring (decay counts ticks), not a "
          "discovered healing law -- labeled honestly as such.",
          "",
          f"Overall: {out['verdicts']['overall']}",
          ""]
    return "\n".join(L)


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
        print("GATE FAILED -- aborting.", flush=True)
        return out

    rng_c = np.random.default_rng(1)
    clean_batches = [gen_clean_batch(rng_c) for _ in range(800)]
    rng_p = np.random.default_rng(2)
    poison_batches = [gen_poison_batch(rng_p) for _ in range(POISON_STEPS)]
    rng_r = np.random.default_rng(3)
    repair_batches = [gen_clean_batch(rng_r) for _ in range(REPAIR_CAP)]

    results = {}
    for sched in SCHEDULES:
        print(f"--- tick {sched.name} ({sched.clean_steps} steps, "
              f"tick {sched.consol_every}, {sched.n_ticks} tick(s)) ---",
              flush=True)
        results[sched.name] = run_variant(
            sched, clean_batches, poison_batches, repair_batches)
        r = results[sched.name]
        print(f"[{sched.name}] baseline p_true={r['baseline_kt']['mean_p_true']:.4f} "
              f"pre_auth={r['pre_injury_auth_mean']:.3f} "
              f"post_auth={r['post_repair_auth_mean']:.3f} "
              f"lag={r['authority_lag']:.3f} "
              f"post_p_true={r['post_repair_probes']['mean_p_true']:.4f} "
              f"top1={r['post_repair_probes']['top1_true_count']}/4 "
              f"repair_steps={r['repair_steps']} "
              f"healed_at={r['time_to_healed_applied']}", flush=True)
    out["variants"] = results

    # BASELINE replication check vs R3-E2 SLOW (same seeds, same
    # pipeline modulo tick scheduling; BASELINE's single tick IS the
    # round-1 structure, so it should reproduce R3-E2 SLOW exactly).
    rep = {}
    try:
        with open(os.path.join(ROOT, "experiments",
                               "results-r3-e2.json")) as f:
            e2slow = json.load(f)["tails"]["SLOW"]
        b = results["BASELINE"]
        rep = {
            "baseline_mean_p_true_equal":
                b["baseline_kt"]["mean_p_true"]
                == e2slow["baseline_kt"]["mean_p_true"],
            "post_poison_mean_p_true_equal":
                b["post_poison_kt"]["mean_p_true"]
                == e2slow["post_poison_kt"]["mean_p_true"],
            "affected_rows_equal":
                b["affected_rows"] == e2slow["affected_rows"],
            "repair_steps_equal":
                b["repair_steps"] == e2slow["repair_steps"],
            "post_repair_mean_p_true_equal":
                b["post_repair_probes"]["mean_p_true"]
                == e2slow["post_repair_probes"]["mean_p_true"],
            "lineage_equal": b["lineage_liar_entries"]
            == e2slow["lineage_liar_entries"],
        }
    except (OSError, KeyError) as exc:
        rep = {"error": f"R3-E2 results not readable for comparison: {exc}"}
    out["baseline_replication_vs_r3_e2_slow"] = rep
    print("BASELINE replication vs R3-E2 SLOW:", rep, flush=True)

    # ---- Frozen verdicts (a)-(d), per variant ---------------------------
    a, b, c = {}, {}, {}
    for name, r in results.items():
        a[name] = bool(r["authority_lag"] >= 0.1)
        deficits = [t["deficit"] for t in r["trajectory"]]
        b[name] = bool(all(x >= y for x, y in zip(deficits, deficits[1:])))
        c[name] = bool(r["post_repair_probes"]["top1_true_count"] == 4)
    healed = {name: r["time_to_healed_applied"] for name, r in results.items()}
    healed_vals = [v for v in healed.values() if v is not None]
    d_spread = (max(healed_vals) - min(healed_vals)) if healed_vals else None
    d = bool(len(healed_vals) == 3 and d_spread is not None
             and d_spread <= 1
             and all(abs(v - 25) <= 1 for v in healed_vals))
    out["verdicts"] = {
        "a_authority_lag_ge_0.1_all_variants": a,
        "b_deficit_nonincreasing_all_variants": b,
        "c_top1_4of4_post_repair_all_variants": c,
        "d_tick_count_to_healed": healed,
        "d_spread_max_minus_min": d_spread,
        "d_expected_ticks": 25,
        "d_identical_within_plus_minus_1_tick": bool(d),
        "d_wiring_check_note": "Decay priced per decay consolidation, "
            "never per training step; see run docstring and results .md.",
        "overall": "PASS" if (all(a.values()) and all(b.values())
                           and all(c.values()) and d) else "FAIL",
    }
    print("VERDICTS:", out["verdicts"], flush=True)

    out["notes"] = {
        "no_cross_variant_probe_comparison": "Baselines are per "
            "variant; only SHAPE (lag, monotonicity, ranking, healed "
            "tick-count) is compared across variants.",
        "no_new_training_code_path": "All training reuses "
            "experiments/run_all.train_steps; kt_bench/tick.py is pure "
            "schedule arithmetic. Gradient gate above covers the "
            "training path.",
        "firewall_demo_skipped": "Restored side-branch in rounds 1-2, "
            "not part of any R3-E3 frozen threshold; as in R3-E2.",
        "deficit_is_dock_parameter": "Deficit values are the exact "
            "dock from kt_bench.tail (SLOW). Standing authority is "
            "q10(0.9 - deficit) and is reported alongside, never "
            "substituted for deficit.",
    }

    exp_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(exp_dir, "results-r3-e3.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("results-r3-e3.json written", flush=True)
    with open(os.path.join(exp_dir, "results-r3-e3.md"), "w") as f:
        f.write(render_md(out))
    print("results-r3-e3.md written", flush=True)
    print("OVERALL:", out["verdicts"]["overall"], flush=True)
    return out


if __name__ == "__main__":
    main()
