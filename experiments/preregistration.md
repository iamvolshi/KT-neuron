# Phase 3 Experiments — Pre-registration

Frozen before running, 2026-10-26. Darren's style: hypotheses frozen
before running, results reported honestly whatever they are. Any change
after this file is written must be added as a dated amendment below,
never by editing the frozen sections.

## Training method (labeled, because it sets what results can claim)

The bench is numpy-only and forward-only: no torch, no autograd, no
training loop existed. For these experiments we implement MANUAL
BACKPROP (hand-derived gradients for this exact architecture) with
Adam, verified by a NUMERICAL GRADIENT CHECK before any training run.

- Gradient-check gate: on a reduced config, max relative error between
  analytic and numerical gradients must be < 1e-4. If the gate fails,
  all training results are VOID and this file is amended to fall back
  to perturbation-based training, disclosed as such. Never present
  perturbation runs as gradient training.
- What this means for claims: results demonstrate the KT MECHANISMS
  (gains, scars, records, lineage, repair) under real gradient
  training of a tiny model. They do NOT demonstrate frontier training
  dynamics, and nothing here claims scale.
- KT/vanilla lockstep: same init seed, same batch order, same data.
  Models diverge only when KT mechanisms engage (gains/scars after
  consolidation). Divergence after consolidation is the measured
  effect, not a bug.

## Shared setup (all experiments unless stated otherwise)

- Config: GPTConfig(vocab_size=32, block_size=16, n_embd=64, n_head=4,
  n_layer=2). Exact parameter count is computed and reported by the
  harness, not hand-asserted here.
- Model init seed: 0 (build_pair). Data seed: 1. All other randomness
  (probe noise, batch shuffling) is seeded and reported.
- Batch: 8 sequences x 16 tokens. Optimizer: Adam, lr 0.01,
  betas (0.9, 0.999), eps 1e-8. Loss: cross-entropy on all positions.
- Schedule: clean phase 400 steps; poison phase 200 steps;
  repair up to 200 steps (stops early at recovery, see E2/E3);
  full-retrain up to 400 steps (same recovery criterion).
- Recovery criterion (repair and full-retrain): mean P(true) on the
  four fact probes >= clean-baseline mean P(true) - 0.05. Steps used
  are the measurement; hitting the cap without recovery is a FAIL
  reported as such, not a silent extension.

## Corpus and the planted liar (all experiments)

Toy language over tokens 0..31, two disjoint parts so grammar and
facts never fight each other:

- Counting grammar (tokens 0..15 only): runs [s, s+1, ...] mod 16.
- Fact contexts (tokens 20..29, never inside counting runs):
  - F1: (20,21) -> 3 true / 9 liar
  - F2: (22,23) -> 4 true / 10 liar
  - F3: (24,25) -> 5 true / 11 liar
  - F4: (26,27) -> 6 true / 12 liar
  - Liar-only fact LX: (28,29) -> 13 (appears ONLY in liar data;
    used by E4 abstention, never in clean data)
- Sources (declared authority, placeholder policy):
  - textbook (curated): 0.9 — clean grammar + true facts
  - web (ordinary): 0.5 — clean grammar + true facts + noise
  - liar: declared 0.5 at train time (confident, fluent, wrong) —
    fact contexts with liar tokens, repeated consistently
- Clean corpus mix: 50% counting sequences, 50% fact sequences
  (fact context placed at varied positions 3..10 so probes at
  position 6 are trained). Poison corpus mix: 50% liar-fact
  sequences, 50% clean sequences (grammar must not collapse).
- Probes: fixed prefix [1,2,3,4, a, b] (length 6), next-token
  distribution at the last position. Reported per probe: P(true),
  P(liar), top-1 token, top-1 confidence (max softmax prob).
  Identical probes across models and phases — comparisons are the
  measurement, never absolute scores alone.

## Validity criterion (applies to E2–E5)

Clean-phase fact top-1 must be >= 3/4 facts correct for the poison
comparisons to be labeled STRONG evidence. Below that, results are
still reported in full but labeled WEAK evidence with this caveat.
Absolute learning levels are always reported alongside differentials.

## Experiment 1 — Cost: KT vs vanilla twin

- Hypothesis: the KT machinery costs little forward-pass time (gain
  is one multiply per row) but costs record/snapshot storage, and
  uniform post-consolidation gains soften confidence (higher CE loss)
  without changing fact top-1.
- Setup: both models, clean phase only, then one consolidation for
  KT (pool textbook+web). Vanilla has no consolidation.
- Measurements: (a) mean forward-pass ms per batch over 200 forwards
  of each model on the same batch; (b) record overhead: KID count,
  total record rows, snapshot parameter copies; (c) fact-probe
  top-1 and mean P(true) for both models post-clean-phase;
  (d) pre-consolidation logit parity (must be exact).
- Thresholds: forward overhead ratio KT/vanilla < 2.0 (PASS).
  Pre-consolidation logits exactly equal (MUST hold; else the rig
  is broken and nothing else is trusted). Quality delta is
  REPORT-ONLY, no threshold — honest reporting either way.

## Experiment 2 — Poisoning

- Hypothesis: mid-training liar data flips the vanilla model's fact
  behavior invisibly (no structure exists to query), while the KT
  model names the poison intervals by lineage query, quarantines by
  null-run, and repairs surgically for fewer steps than a full
  retrain.
- Setup: clean phase (both) -> baseline probes -> poison phase
  (both, same liar-mixed batches) -> post-poison probes.
  KT lineage: harness interval ledger (bench k -> declared source
  mix) joined to K-records; vanilla lineage: asserted impossible
  by construction (no records exist to query).
- Firewall engagement (KT, post-exposure): present liar-tainted
  batches after exposure; per-source pot fills 1.0 per tainted
  batch, threshold 3.0, bleeds 0.5 per clean batch, release below
  1.0. While engaged, liar-lineage gains are held at 0 (null).
- Surgical repair (KT): retrain ONLY affected rows (lm_head rows
  for fact tokens — exact by construction — plus top-20 upstream
  rows by |dW| during the poison interval, labeled APPROXIMATE),
  all other rows frozen (zeroed gradients). Re-seat repaired rows
  with authority dock 0.3. Full retrain (vanilla): whole model on
  clean data to the recovery criterion.
- Measurements: post-poison mean P(true)/P(liar), fact top-1 for
  both; lineage query result (interval ids returned); firewall pot
  trajectory; repair steps vs full-retrain steps.
- Thresholds (PASS requires all): vanilla flips — post-poison mean
  P(liar) > mean P(true) AND top-1 = liar on >= 3/4 facts; KT
  lineage query returns exactly the poison intervals (no missing,
  no extras at interval granularity); repair steps < full-retrain
  steps (strictly).

## Experiment 3 — The law at scale (headline measurement)

- Hypothesis: after repair, KT probe behavior returns to baseline
  while seated authority on the repaired lineage stays measurably
  below its pre-injury value, decaying slowly on the consolidation
  timescale.
- Setup: continues the KT run from E2 (same run, same numbers —
  no fresh model). Pre-injury authority = seated authority on
  affected rows after the clean phase. Repair as in E2 (dock 0.3).
  Then 2 further clean consolidations; deficit heals 0.01 per
  clean consolidation (re-seat repaired rows with the reduced
  dock each time).
- Measurements: fact probes (mean P(true), top-1 count) at
  baseline, post-poison, post-repair, +1 consolidation, +2
  consolidations; seated authority on repaired rows at the same
  five points; chain verification (all_ok) at each point.
- Thresholds (PASS requires all): post-repair mean P(true) >=
  baseline - 0.10 AND fact top-1 count >= baseline count;
  repaired authority <= pre-injury - 0.1; authority after +2
  clean consolidations still < pre-injury (deficit decaying,
  not snapped back).

## Experiment 4 — Abstention

- Hypothesis: on a probe whose only support is the poisoned
  lineage, vanilla answers confidently and wrong, while KT hedges
  or declines by the math, not by a system prompt.
- Setup: liar-only fact LX (28,29) -> 13, present only in poison
  data. After poisoning, KT path: exposure + null-run (poison
  lineage gains -> 0). Abstention rule (harness mechanism,
  labeled): predict_with_abstention returns DECLINE if the top-1
  token's lm_head row is in the quarantined lineage OR its gain
  < 0.3; else it returns the token with its confidence.
- Measurements: LX probe top-1 token and confidence for vanilla
  (post-poison) and KT (post-null-run, via the abstention rule).
- Thresholds (PASS requires both): vanilla top-1 = 13 AND
  confidence >= 0.6; KT returns DECLINE OR confidence <= 0.5.
  Anything between is reported as a FAIL on the threshold, with
  the numbers.

## Experiment 5 — C2C proving ground (side-by-side cache transfer)

- Hypothesis: a vanilla cache transfer carries poison invisibly
  (no lineage, no repair short of retraining), while a KATIE
  record transfer carries its receipts, so the receiver can
  quarantine, repair surgically, and show the authority lag.
- Setup: sender A trained clean (facts learned). Receiver B fresh
  init. The "cache" = the lm_head rows encoding the fact-set
  (true tokens 3,4,5,6 and, in the poison variant, liar tokens
  9,10,11,12) — the transferable unit, labeled as such.
  - Vanilla path: direct weight copy of sender's fact rows into
    receiver (overwrite). No record of any kind.
  - KT path: copy rows AND records — receiver rows adopt the
    transferred intervals (source "transfer:sender", authority =
    sender's seated authority docked 0.1 for transfer, labeled
    harness choice), with the transfer recorded in the harness
    ledger (sender KID -> receiver KID, intervals, sources).
  - Poison variant: sender trained with liar facts (same poison
    protocol as E2); transfer its fact rows; then KT receiver
    path: lineage query, null-run, surgical repair (as E2, dock
    0.3), authority trajectory.
- Measurements: receiver fact probes (mean P(true)/P(liar)) at
  pre-transfer, post-transfer, post-quarantine (KT only),
  post-repair (KT only); lineage query results on both paths;
  KT repair steps vs vanilla full-retrain steps to recovery.
- Thresholds (PASS requires all): vanilla receiver inherits
  poison — post-transfer mean P(liar) > mean P(true) — with no
  lineage queryable; KT post-quarantine mean P(true) >=
  clean-baseline - 0.10; KT lineage query names the transferred
  liar intervals; KT post-repair authority lag >= 0.1 vs
  pre-transfer seated authority.

## What is NOT claimed by any experiment here

- No frontier-scale claim. No claim that per-row attribution is
  exact (upstream attribution is labeled APPROXIMATE everywhere).
- No ARCHE seating claim: the placeholder policy is used as-is.
- No demo claim: E5 is the demonstration's evidence base, not
  the demo itself (§11 demo is separate work).
- The bench record stores the seating-event source per interval,
  not a full per-row source pool; the lineage query joins the
  harness interval ledger to the K-records. That join is stated
  wherever lineage results are reported.

## Amendments (all dated, all written BEFORE any runs — 2026-10-26)

- A1 — Liar declared authority: 0.9, not 0.5. The liar presents as
  textbook-grade (confident, fluent, apparently authoritative) —
  that is WHY it is dangerous. Exposure reveals it as bad only
  post-hoc, from outside. Supersedes the "liar: declared 0.5" line
  in the corpus section.
- A2 — Interval seating pools (caller-declared, placeholder policy):
  clean {"textbook": 0.9, "web": 0.5} -> 0.8; poison {"liar": 0.9}
  -> 0.9; repair claim {"textbook": 0.9} -> 0.9 with dock 0.3 ->
  seated 0.6; transfer dock 0.1 (see E5). Pools are declared per
  interval by the harness, not measured per row.
- A3 — Consolidation driver: phase transitions drive rec.replace
  directly (harness function), NOT Consolidator.consolidate,
  because Consolidator raises on candidate outcomes and a candidate
  is legitimate gate data (a weaker claim not seating), not a
  failure. The harness logs seated/candidate/reaffirmed per row,
  sets each row's gain to its standing authority, snapshots via
  SnapshotStore, and advances its own bench-k counter. Chain
  verification is unchanged (rec.verify / RecordStore).
- A4 — E3 decay observation extended to +6 clean consolidations
  (the 0.1 grid masks 0.01-per-consolidation deficit healing at
  +2). Threshold unchanged: authority still < pre-injury at +2.
  Raw deficit trajectory is reported alongside quantized seated
  authority.
- A6 — Gradient-check gate criterion refined (not loosened on
  substance): an entry passes if rel err < 1e-4 OR abs diff < 1e-9.
  Reason, found in the check itself: relative error is meaningless
  on ~zero gradients. The worst entry was qkv.W (6,1) with analytic
  -2.09e-08 vs numeric -2.10e-08 — same sign, same order, absolute
  difference 2.5e-12 (float noise on a masked-path gradient that is
  essentially zero). Every other entry was rel <= 3.5e-05. A mask
  leak would show as a LARGE discrepancy, not 2.5e-12. The combined
  criterion is standard check_grad practice and the gate still
  fails on any real gradient bug.
- A7 — Run-2 harness fixes (setup-vs-intent mismatches found in run
  1; THRESHOLDS UNCHANGED; run 1 preserved at results_run1.json):
  (a) E4 quarantine set now includes the liar-only token row 13 —
  run 1 nulled only tokens 9-12, so the LX probe's own row was
  never quarantined; the pre-registered intent ("null liar rows",
  "only support is the poisoned lineage") plainly includes 13.
  (b) E5 transfer now copies full trained state — linear rows PLUS
  embeddings and LayerNorms — because run 1 copied rows only and
  the receivers' fresh embeddings/norms mismatched the sender's
  trained rows, so behavior never transferred at all.
  (c) Bench snapshots now cover full model state (rows +
  embeddings + norms) for the same reason: run 1's firewall
  restore check failed only because embeddings/norms moved during
  the demo and were never restored. Records are still never
  snapshotted: the ledger is never rolled back, only the fast
  trace is.
