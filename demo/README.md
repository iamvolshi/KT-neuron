# Side-by-side demo (plan §11) — extended

For Darren's eyes only. Stealth: nothing here leaves this machine.

## Run it

From `~/workspace/kt-neuron`:

```
python3 -m demo.run_demo        # scene 0 only: the original E5 C2C scene
python3 -m demo.run_extended    # all five scenes, one console run (~20 min)
```

Each extension scene also runs standalone:

```
python3 -m demo.scene_decline   # ~3.5 min
python3 -m demo.scene_signal    # ~3 min
python3 -m demo.scene_tail      # ~6 min
python3 -m demo.scene_tick      # ~6 min
```

Console output is the demo: both sides' probe numbers, lineage, repair,
authority, signal, schedules, and chain status printed in paired
sections plus a side-by-side summary per scene. No HTML, no server, no
external calls — every number comes from the live run, not from stored
results. Each scene reuses the existing harnesses (vanilla twin,
kt_bench modules, experiment run code); no new training code path was
invented for the demo.

## The scenes

Same inputs, same injuries, both behaviors visible — five ways.

**Scene 0 — E5 C2C transfer (the original scene, untouched).**

VANILLA — cache-style merge, no record. The receiver inherits the
poison exactly (post-transfer probes identical to the sender's
poisoned probes). The lineage query returns nothing:
`kt_linears() == []`, no records exist, so there is nothing to query,
nothing to quarantine, and no surgical repair — the only path back is
full retraining.

KT — record transfer. The receiver inherits the poison (slightly
softer: the transfer dock 0.1 seats transferred beliefs at authority
0.8 instead of the sender's 0.9, so gains are lower). Then, in order:
the lineage query names the transfer/poison entries (one per
transferred row); the null-run (quarantine) restores clean behavior;
surgical repair touches only the affected rows (lm_head fact rows,
exact by construction, plus the top-20 upstream rows by |ΔW| during
poison, labeled approximate); repaired beliefs re-seat at docked
authority, so authority visibly lags the transfer authority after
behavior returns; chain verification ok.

**Scene 1 — Decline / abstention pair (E4 + decline rule).**

Same liar-only probe LX [1,2,3,4,28,29], both sides. VANILLA answers
token 13 at 0.983 — confident and wrong. KT DECLINEs: answer None,
reason quarantined-support (pre-null winner 13 at 0.996, post-null
winner 6 at 0.165, quarantined support fraction 0.978). Controls in the
same run: clean probes 0 DECLINEs, top-1 true 4/4; mixed-support probes
0 DECLINEs, top-1 true 4/4 — the rule visibly does not over-fire.

**Scene 2 — Signal survival (R3-E1, one-signal regulation).**

Poison introduced AFTER the fast->slow crossing. KT's signal follows
fast through boundaries 1–2, crosses to slow at boundary 3
(continuity gap 0.031), then stays at 0.982000 through all 40 poison
probes — change exactly 0.0 — while the fast trace collapses beside
it (range 0.2457–0.7542). VANILLA's probe moves with the poison at
every probe (clean end 0.9780, final 0.4246, max move 0.7778): no
signal, no slow trace, nothing holds. Same switch as the plan §6
poison flip.

**Scene 3 — Tail healing (R3-E2, tail controls healing).**

Same injury, three tails, one schedule table. FAST (decay 0.05):
deficit 0.00 at +6, healed at +5. SLOW (0.01): deficit 0.24 at +6,
healed at +25. PERMANENT (0.00): deficit 0.30 at +6 and +30, never
heals. Ranking top-1 true 4/4 throughout, all tails; lineage exact;
28/28 rows seated; chain ok. VANILLA retrains in 6 steps and then has
nothing left to watch heal: authority NONE, deficit NONE, schedule
NONE — no +n axis exists, because it has no record to heal.

**Scene 4 — Tick mutability (R3-E3, shape holds across tick sizes).**

Three tick schedules, one shape. FINE (400 clean steps, 100-step
ticks), BASELINE (400, 400), COARSE (800, 800): each shows authority
lag 0.200, deficit nonincreasing 0.30 -> 0.00, top-1 true 4/4, and
ticks-to-healed 25/25/25 (spread 0). VANILLA's row sits under KT's
three schedules: tick NONE, ticks-to-healed NONE — no tick to resize,
because it has no record whose healing could be scheduled.

## Honest limits (carried from the results files, not new claims)

- **Toy scale.** 105,248-parameter numpy model, 32-token vocab,
  counting grammar + four planted facts. The claim is the mechanism
  differential at this scale, not frontier behavior.
- **Transfer unit is fact rows + full trained state, labeled as
  such.** It is not a KV-cache. The comparison is cache-style
  transfer (weight copy, no record) vs record transfer (weight copy +
  K-records) — that is exactly the C2C audit question. The run-1 rig
  bug (rows-only transfer; receivers' fresh embeddings/norms
  mismatched) is fixed here by copying embeddings and LayerNorms, as
  round-1 run 2 did.
- **Interval/transfer-level lineage, not per-row.** The KT lineage
  count is one entry per transferred row, each tagged with the
  transfer event and the sender source "liar" — coarser than per-row
  provenance, reported as such.
- **Seating policy is the placeholder** (mean + agreement bonus +
  bad-source dock, caller-declared pools), not ARCHE's feedback rule.
  No ARCHE controller is designed here.
- **Repair/retrain step tie is disclosed, not hidden:** round-1 E5
  measured KT repair 10 steps vs vanilla full retrain 10 steps. The
  E5 claim is the lineage/quarantine/repair/authority differential,
  not a step-cost win. Repair cost as work touched (optimizer writes
  only, NOT FLOPs, NOT steps): 52,760 vs 631,488 param updates,
  ratio 0.0835.
- **Scene 2 (signal) was built, not discovered.** The bench had no
  slow-accumulation dynamics before round 3; SignalTracker implements
  the slow trace and the scene demonstrates the regulation mechanism.
  Alpha 0.8 was calibrated on a pilot before the official run and
  frozen (textbook 0.5 never crosses in a 4-boundary window).
  Post-handoff signal-equals-slow behavior is partly by tracker
  construction; making it empirical needs the slow trace to be a real
  kernel value — a Rust-engine question.
- **Scene 3 (tail) shows a control surface, not a discovered decay
  law.** tail.py is pure arithmetic; the empirical content is the
  pipeline integration (identical pre-decay trajectories, quantized
  authority tracking the deficit). Deficit is NOT 0.9 minus authority
  (0.1 grid quantization — SLOW at +6: deficit 0.24, authority mean
  0.7).
- **Scene 4 (tick) leg (d) is a wiring check.** Decay is priced per
  decay consolidation, never per training step, so COARSE healing in
  the same 25 ticks verifies the wiring, not a discovered healing law.
  Baselines are per-variant; no verdict rests on comparing absolutes
  across variants (repair-step differences 13/10/18 are expected).
- **Scene 1 (decline) upstream attribution is APPROXIMATE** (top-20
  by |ΔW|); lm_head liar rows are exact. Standing authority alone
  cannot distinguish liar rows (interval-level pools seat every row
  at 0.9 after poison consolidation); the quarantine set is the
  operative liar signal. The DECLINE is a rule, not the round-1 hedge
  (round 1 returned a low-confidence token; this returns no token).
- **Deterministic.** Fixed seeds throughout; the same demo every run
  (probe numbers reproduce exactly; only wall-clock time varies).
