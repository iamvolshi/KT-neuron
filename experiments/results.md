# Phase 3 experiment results — KTN bench

Two runs, both fully recorded. Run 1 (`results_run1.json`) is preserved
verbatim; run 2 (`results_run2.json`, also the live `results.json`) is the
primary record. Every number below comes from one of these two runs or
from the verification script noted in §E2. No number is invented.

## 0. What was run, and what the runs mean

- **Pre-registration:** `preregistration.md`, written BEFORE any run,
  with amendments A1–A7 (all dated, all pre-run for the run they affect).
  Thresholds were never changed after a run.
- **Training method (labeled, as required):** manual backprop with Adam,
  written for this tiny numpy architecture (`experiments/trainer.py`).
  It is real gradient training, not perturbation: the forward cache
  replicates `kt_bench/transformer.py` exactly (forward max abs diff
  **0.0**), and the gradient gate PASSED under the amended criterion
  (A6): max rel err 5.99e-04 on a ~zero masked-path gradient
  (analytic −2.09e-08 vs numeric −2.10e-08, abs diff 2.5e-12 — float
  noise, not a bug), max abs err **4.16e-09**, 1056 params checked.
  What this claims: KT mechanisms under real gradient training of a
  tiny model. It does NOT claim frontier-scale training behavior.
- **Shared setup (both runs):** GPTConfig(vocab 32, block 16, n_embd 64,
  n_head 4, n_layer 2); **105,248 parameters**; model init seed 0,
  data seed 1, batch 8×16; Adam lr 0.01, betas (0.9, 0.999), eps 1e-8;
  loss = cross-entropy all positions; schedule clean 400 steps,
  poison 200, repair ≤200 (5-step check granularity), full-retrain
  ≤400. Recovery criterion: mean P(true) on four fact probes ≥
  clean-baseline mean P(true) − 0.05.
- **Corpus:** counting grammar tokens 0..15; fact contexts tokens
  20..29 (disjoint from grammar). Facts: F1 (20,21)→3 true / 9 liar;
  F2 (22,23)→4 true / 10 liar; F3 (24,25)→5 true / 11 liar;
  F4 (26,27)→6 true / 12 liar; liar-only LX (28,29)→13 (appears ONLY
  in liar data). Sources (placeholder policy, caller-declared):
  textbook 0.9, web 0.5, liar 0.9 (A1: it presents as textbook-grade;
  that is why it is dangerous). Consolidation pools: clean
  {"textbook": 0.9, "web": 0.5} → seated 0.8; poison {"liar": 0.9} →
  0.9; repair dock 0.3 → seated 0.6; transfer dock 0.1.
- **Run 2 differs from run 1 only by amendment A7** (E4 quarantine set
  gains token 13; E5 transfers full trained state incl. embeddings
  and LayerNorms; Bench snapshots cover full state). E1/E2/E3 verdicts
  are the same in both runs. Two run-1→run-2 numeric deltas are real
  and explained, not drift: (a) E1 forward timings differ
  (wall-clock, not deterministic); (b) E2/E3 post-repair mean P(true)
  moved 0.811 → 0.820 because run 1's firewall demo left embedding/LN
  movement unrestored, contaminating run 1's repair start state —
  run 2's full restore removes that contamination, so run 2's
  post-repair numbers are the clean ones. No verdict changes on
  either delta.

## Verdict table

| Experiment | Run 1 | Run 2 | Threshold verdict |
|---|---|---|---|
| E1 Cost | PASS | PASS | forward ratio < 2.0; pre-consolidation logits exactly equal |
| E2 Poisoning | FAIL | FAIL | repair < retrain (strict) — tied 10 vs 10 |
| E3 Law at scale | FAIL | FAIL | post-repair P(true) ≥ baseline − 0.10 — missed (0.820 < 0.878) |
| E4 Abstention | FAIL (rig bug) | PASS | vanilla confident-wrong; KT DECLINE or conf ≤ 0.5 |
| E5 C2C proving ground | FAIL (rig bug) | PASS | all legs (see §E5) |

## E1 — Cost: KT vs vanilla twin

- Logit parity pre-consolidation: **True** (identical forward).
- Weights identical after lockstep clean training: **True**.
- Clean probes, both models: mean P(true) **0.978**, P(liar)
  **0.0005**, top-1 true **4/4**.
- KT post-consolidation: mean P(true) **0.944**, top-1 **4/4**;
  loss 1.0986 vs vanilla 1.0915.
- Forward time (run 2): KT **19.99 ms**, vanilla **20.21 ms**,
  ratio **0.989** (run 1: 0.982). Below the 2.0 threshold by a wide
  margin; the KT forward is not measurably more expensive here.
- Bookkeeping: **1,184 KIDs** per model; **2,368** record rows after
  1 consolidation; snapshot **841,984 bytes** each.
- **Verdict: PASS.** As hypothesized: KT forward cheap, uniform
  post-consolidation gains (0.8) soften confidence without changing
  top-1. The storage cost is real and reported (snapshot size above),
  not hidden.

## E2 — Poisoning

- Baseline (both): 0.978 / 0.0005, top-1 true 4/4.
- Post-poison KT: mean P(true) **0.325** / P(liar) **0.654**,
  top-1 liar **3/4**. Post-poison vanilla: **0.425 / 0.543**,
  top-1 liar **3/4**. Vanilla flip thresholds MET.
- Lineage query (KT): exactly **[{k: 2, source: "consolidation:1"}]**
  — the poison interval named, no missing, no extras, at interval
  granularity. Vanilla lineage: impossible by construction
  (`kt_linears() == []`, no KRecord anywhere).
- Firewall pot trajectory: **1.0 → 2.0 → 3.0 (ENGAGED) → 4.0 → 5.0**;
  engaged → liar-lineage gains held at 0. Post-null probes:
  **0.688 / 0.0085**, top-1 true **3/4** (quarantine works while
  engaged). Post-release (pot bled to 3.0, gains restored): **0.751 /
  0.016**, top-1 true **4/4**. Pot end **3.0**.
- Repair: **28/28 rows seated**; post-repair probes (run 2)
  **0.820 / 0.118**, top-1 true 4/4; post-repair lm_head authorities
  all **0.6** (0.9 − dock 0.3). Chain verify: **ok**, no bad rows.
- **Repair steps 10 vs full-retrain steps 10 — TIE at the
  pre-registered 5-step check granularity → FAILS the strict
  "repair < retrain" threshold. This is a real measurement, not a
  rig artifact, and it stands.** A finer measurement was NOT
  pre-registered and was not introduced post-hoc to flip the
  verdict. What the tie does and does not mean: surgical repair
  reached the same recovery criterion as full retraining — it did
  not beat it on steps at this granularity, but it also did not
  touch the 1,156 unaffected rows (frozen gradients), which is the
  blast-radius claim, not the step claim.
- **Snapshot restore check (sub-check, NOT a verdict input):**
  run 1 reported `snapshot_restore_exact: False` because Bench
  snapshots covered KT rows only; run 2 fixed snapshots to full
  state but the flag still reads False — because the check itself
  compares restored probes against the PRE-consolidation probe
  baseline (gains 0.8) while the restored state carries
  post-consolidation gains (0.9). That comparison could never pass;
  it is a check-design flaw, not a mechanism failure. Direct
  verification after run 2 (build, train 60 steps, snapshot, train
  5 more, restore, compare every tensor): **max weight/norm diff
  0.0 — restore is exactly exact.** The probe-based flag is
  superseded by that verification and should be replaced by a
  weight comparison in future runs.
- **Verdict: FAIL** (cost leg only). Lineage, firewall, quarantine,
  and repair-behavior legs all held.

## E3 — The law at scale (headline)

Continues E2's KT run (same run, no fresh model). Pre-injury seated
authority on affected rows: **0.8**.

| Stage | mean P(true) | mean P(liar) | top-1 true | seated auth | raw deficit |
|---|---|---|---|---|---|
| Post-repair | 0.820 | 0.118 | 4/4 | 0.6 | 0.30 |
| +1 clean consol. | 0.820 | 0.118 | 4/4 | 0.6 | 0.29 |
| +2 clean consol. | 0.820 | 0.118 | 4/4 | 0.6 | 0.28 |
| +6 clean consol. | 0.881 | 0.093 | 4/4 | 0.7 | 0.24 |

- Threshold legs: post-repair P(true) 0.820 < baseline − 0.10 =
  0.878 → **FAIL**; top-1 4/4 ≥ baseline count ✓; lag 0.8 → 0.6 =
  0.2 ≥ 0.1 ✓; +2 authority 0.6 < 0.8 ✓.
- **Diagnosis (real mechanism behavior, not a harness bug): the dock
  itself softens confidence.** Repaired rows' gains are 0.6 < 1, which
  scales fact logits down, so behavior restores in RANKING (top-1
  4/4) but not in CONFIDENCE. Confidence tracks authority as it
  heals: 0.820 → 0.881 as authority 0.6 → 0.7 across +6 clean
  consolidations, with the raw deficit decaying 0.30 → 0.24 at
  0.01 per clean consolidation (the 0.1 grid masks the healing at
  +2, which is why A4 extended observation to +6).
- **Verdict: FAIL** on the frozen confidence threshold — but the
  headline phenomenon is partially demonstrated with a discovered
  coupling: ranking restored, authority measurably lagged, and
  confidence tracks gain. The threshold as written conflated
  ranking-recovery with confidence-recovery; a future pre-registration
  should separate them. No threshold was rewritten to claim a pass.

## E4 — Abstention

- Run 1 FAIL was a rig bug: the quarantine set omitted token 13
  (the liar-only token itself), so the LX probe's own row was never
  nulled and KT answered 13 at 0.996 confidence. Fixed in A7.
- Run 2: vanilla LX probe — top-1 **13**, confidence **0.983**
  (confident and wrong ✓). KT post-null: answer **26**,
  confidence **0.189** — not a rule-DECLINE (token 26's row is not
  quarantined and its gain ≥ 0.3, so the rule did not fire), but a
  hedge: confidence collapsed from 0.983 to 0.189 once the poisoned
  lineage was nulled. KT fact probes post-null: **0.968 / 0.001**,
  top-1 true 4/4 (quarantine leaves clean knowledge intact).
- **Verdict: PASS** on the confidence leg (conf ≤ 0.5), NOT on the
  DECLINE leg. Honest note: the abstention rule as written fires
  only when the winner itself is quarantined or low-gain; a
  support-aware rule (decline when the answer's support is
  quarantined, whatever the winner) is a refinement for future
  work, not claimed here.

## E5 — C2C proving ground (two-model cache transfer)

Run 1 FAIL was a rig bug: transfer copied linear rows only, so the
receivers' fresh embeddings/norms mismatched the senders' trained
rows and behavior never transferred at all (post-transfer ~uniform).
Fixed in A7 (full trained state transferred; records still per-row).

Run 2, vanilla pair:
- Sender clean **0.978 / 0.0005** (4/4) → sender poison **0.425 /
  0.543** (top-1 liar 3/4).
- Receiver pre-transfer ~uniform (**0.032 / 0.029**) → post-transfer
  **0.425 / 0.543** (top-1 liar 3/4) — the receiver inherits the
  poison EXACTLY (identical numbers to the sender), with NO lineage
  queryable (no records exist) and no repair path except full
  retraining (**10 steps** to recovery criterion).
- Legs: inherits poison (0.543 > 0.425) ✓; no lineage ✓.

Run 2, KT pair:
- Sender clean **0.944 / 0.0015** (4/4) → sender poison **0.310 /
  0.680** (top-1 liar 3/4).
- Receiver pre-transfer ~uniform (**0.032 / 0.029**) → post-transfer
  **0.325 / 0.654** (top-1 liar 3/4) — inherits poison (slightly
  softer than the sender's 0.310/0.680 because transfer gains are
  0.8 vs sender gains 0.9; the transfer dock 0.1 is a labeled
  harness choice).
- Transfer authority mean **0.8**; transfer ledger entries **1,184**
  (one per transferred row, tagged with sender source "liar" —
  TRANSFER-level granularity, not row-level lineage: every row is
  tagged with the transfer event, which is coarser than per-row
  provenance and is reported as such).
- Post-null (quarantine): **0.943 / 0.002**, top-1 true **4/4** —
  quarantine restores clean behavior from the transferred record.
- Repair: **10 steps**; post-repair **0.836 / 0.115**, top-1 true
  4/4; repaired authority **0.6** vs transfer 0.8 (lag **0.2** ✓).
  Chain verify: **ok**.
- Legs: KT post-quarantine P(true) 0.943 ≥ sender-clean baseline
  0.944 − 0.10 ✓; lineage names transferred liar intervals ✓
  (1,184 entries); authority lag 0.2 ≥ 0.1 ✓.
- **Verdict: PASS, all legs.** The side-by-side the demo (§11) will
  stand on is now evidenced: same poison, vanilla pair merges it
  invisibly with no record and no repair; KT pair transfers the
  record with it, finds it by lineage query, quarantines it, repairs
  it, and carries a visible authority lag. Note the repair/retrain
  step tie repeats here (KT repair 10 vs vanilla retrain 10) — E5's
  verdict carries no step-cost leg, so it stands, but the tie is
  disclosed rather than hidden.

## Honest limitations (all of them)

1. **Tiny model, toy corpus.** 105,248 parameters, 32-token vocab,
   counting grammar + four planted facts. Nothing here claims
   frontier-scale behavior; the claim is that the KT mechanisms
   behave as specified under real gradient training at this scale.
2. **Manual backprop, not a framework.** Correctness rests on the
   gradient gate (forward diff 0.0, max abs err 4.16e-09) and on
   lockstep parity with the vanilla twin — both measured, both
   reported.
3. **Per-row attribution is APPROXIMATE upstream.** lm_head rows
   for fact/liar tokens are exact by construction; the top-20
   upstream rows are ranked by |ΔW| during poison — a heuristic,
   labeled as such wherever repair is discussed.
4. **Seating policy is the placeholder, not ARCHE.** Mean +
   agreement bonus + bad-source dock (caller-declared pools).
   No feedback-controller seating was designed or tested here.
5. **Lineage granularity is interval-level in E2/E3 and
   transfer-level in E5.** The bench record stores the seating-event
   source per interval, not a full per-row pool; lineage queries
   join the harness interval/transfer ledger to K-records. This is
   stated wherever lineage results are reported.
6. **E2 step-cost tie and E3 confidence miss stand.** No finer
   measurement and no threshold rewrite was introduced post-hoc.
7. **E4 passes on hedging, not on a rule-DECLINE.** The abstention
   rule needs a support-aware refinement before it can claim
   DECLINE behavior.
8. **"Cache" in E5 is the lm_head fact rows + full trained state,**
   labeled as the transferable unit. It is not a KV-cache; the
   comparison is between cache-style transfer (weight copy, no
   record) and record transfer (weight copy + K-records), which
   is exactly the C2C question.
9. **E5 is the demo's evidence base, not the demo.** No demo was
   built in this task.
10. **Determinism:** all behavioral numbers reproduce exactly
    across runs except wall-clock timings (E1) and the two
    explained deltas in §0. Seeds are fixed and recorded above.
