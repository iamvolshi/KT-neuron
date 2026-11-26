# 3-arm controlled KTN experiment — pre-registration (2026-10-28)

Written BEFORE any 3-arm run. Frozen: no threshold below changes after
a run. Prior rounds' verdicts STAND and are never rewritten. This file
covers the critic's 3-arm experiment only: conventional vs scar-only
vs full KTN with authority-aware selection, same scenarios, same
metrics. New files only; no prior round file is amended.

The question under test (the critic's, stated plainly): does the
record + selection add anything over scar alone, and does the deficit
prevent re-compromise? The local mechanism (scar-aware activation,
authority-gated learning) is already evidenced. What is NOT yet
evidenced is advantage over simpler alternatives at network scope,
and authority-aware SELECTION. This experiment tests exactly that.
No outcome is assumed either way.

## Shared ground (all three arms)

- Bench: the round-1/2/3 GPT bench, numpy-only, manual backprop +
  Adam (experiments/trainer.py), GPTConfig vocab 32, block 16,
  n_embd 64, 4 heads, 2 layers, 105,248 params, 1,184 KIDs.
- Seeds, lockstep: init 0 for all arms (identical init weights:
  KID generation consumes no RNG, scar vectors consume no RNG, so
  all three arms start bit-identical); clean data 1, poison data 2,
  restore-1 data 3, restore-2 data 4 (first 200 of the seed-4 list),
  counting-eval data 77, eval batches 99. All arms train on
  identical batch lists. One fresh Adam (lr 0.01) per phase per arm
  (round-1 style; no momentum carryover across phases).
- Gradient gate FIRST (experiments/trainer.gradient_check); abort
  on FAIL. No new training code path: all arms train through
  experiments/run_all.train_steps (which handles gain/scar via
  getattr in trainer.lin_forward/lin_backward; the gate exercises
  gain 0.7 / scar 0.05 paths).
- Full test suite must stay green (107 tests at 3-arm start; new
  tests added for new modules). Deterministic and seeded throughout.
- PYTHON NOISE RULE: the bench is a pre-concept proof in Python. It
  proves the SHAPE. Bands below (0.05) are Python-noise bands,
  stated as such. Do not grade the idea by the noise. Exactness
  lives in the Rust engine, out of scope here.
- RETIRED METERS — do NOT use: FLOP saving (never a goal; repair
  cost is rows/params changed, i.e. optimizer writes); the absolute
  confidence bar at repair time (retired; the trajectory is the
  claim); per-row exact attribution (not a goal; interval-level
  lineage plus working repair is sufficient).
- EXCLUDED MACHINERY (Darren, 2026-10-26): do NOT build or test
  silent marking, instant honeypots, fake data pools, or morphic
  data integrity. If a stage seems to need them, say so and skip
  that part.
- SCALE HONESTY: bench scale only, small Python bench. NEVER claim
  120,000 neurons or any scale not run. Darren already corrected
  that math error once; it is not repeated here.

## Arm 1 — CONVENTIONAL (baseline)

- GPT + plain Linear. No scar, no K-record, no authority, no
  selection. Decisions = raw argmax of model.forward logits.
- Nothing about this arm reads or writes any record.

## Arm 2 — SCAR-ONLY (ablation)

- GPT + ScarLinear (new module kt_bench/scar.py: Linear subclass
  with a scar vector; forward = x @ W.T + b - scar; trainer handles
  it via getattr, same as KTLinear's scar path).
- Scar ONLY. No K-record, no authority gate, no gains (gains do not
  exist on this arm), no selection. Decisions = raw argmax.
- The scar follows the pressure driver below, identical to arm 3's
  scar. This arm isolates whether any win comes from the scar alone
  or from the record + selection: if arm 2 matches arm 3, the record
  + selection added nothing.

## Arm 3 — FULL KTN (record + deficit + scar + mock selection)

- GPT + KTLinear (existing). Per-row K-records (KID at birth,
  authority on the 0.1 grid, [k_start, k_end) intervals, SHA-256
  hash chain — the toy's core record rules, UNCHANGED).
- Consolidations (Bench.consolidate, gate-honest): clean
  consolidation with CLEAN_POOL (textbook 0.9, web 0.5 -> claim
  0.8); restore consolidations with REPAIR_POOL (textbook 0.9 ->
  claim 0.9) and dock 0.3 on affected rows (see below) -> affected
  rows re-seat at 0.6. Gains are set to standing authority at every
  consolidation. NO consolidation during poison phases: poison is
  quarantined, never seated. (Seating is endorsement; the tested
  architecture withholds endorsement from detected poison.)
- Affected rows (for restore docking): the 4 fact lm_head rows
  (TRUE_TOKENS + LIAR_TOKENS) PLUS the top-20 rows by movement
  (||dW_row||_2 + |db|) during the preceding poison phase, ranked
  INCLUDING lm_head. Deviation from playbook.affected_upstream
  (which excludes lm_head) is deliberate and stated: fact
  decisions live in lm_head rows, so excluding them would dock
  around the decision instead of at it.
- Scar: same driver as arm 2, applied through KTLinear's scar
  vector.
- Fact belief records (experiment harness, NOT bench machinery):
  one KRecord per probe fact (field = fact id, e.g. "fact-0"),
  born registered. Teacher claims seat ONLY at consolidations:
  clean -> 0.8 (CLEAN_POOL claim); restore -> 0.9 claim with dock
  0.3 -> 0.6. Liar claims are NEVER seated in fact records
  (quarantine, same rule as row records). Fact records are
  verified (chain OK) in the report.
- Mock RK selection (new module kt_bench/mock_rk.py; selection
  lives OUTSIDE KATIE — it reads seated authorities and writes only
  its own selection log, never K-record beliefs; KATIE stays small):
  - At each probe, for each fact, the candidate set is:
    teacher claim {token = true, authority = standing fact-belief
    authority} and liar claim {token = liar, authority = offered
    authority}.
  - Offered authority for liar claims: 0.9 while the liar is not
    yet flagged (confident liar, round-1 POISON_POOL claim); 0.6
    after flagging (0.9 - 0.3 BAD_SOURCE_DOCK, seating.py's
    existing rule). Flagging happens at restore-1 consolidation:
    the experimenter flags the liar known-bad from lineage
    (LIAR_TOKENS + corpus liar source). This is a QUARANTINE
    DECISION, not a system discovery — labeled as such.
  - Selection rule: highest authority wins; TIE -> the standing
    (incumbent) belief wins. Frozen tiebreak, justified: selection
    is conservative; displacement requires strictly beating the
    standing belief. (Seating is where challenges happen; selection
    is where decisions happen. They are different acts.)
  - Losers record as CANDIDATES in the selection log. The log
    entry per fact per probe: {fact, candidates: [{token, row,
    seated_or_offered_authority, status: selected | candidate}],
    selected_token, raw_argmax_token, overridden: bool,
    MOCK: true, DERIVED: true, note: "replaceable by real RK;
    selection reads seated authorities, writes no K-record
    beliefs"}. Writing mock selections into K-records would
    launder DERIVED into fact and is forbidden.
  - Arm-3 decisions = selected tokens. Raw argmax is ALWAYS
    reported alongside, so the selection's contribution is
    visible, never hidden.
- Real selection is RK (ARCHE), the attention engine that chooses
  all selectors (Darren, 2026-10-28). RK is NOT active, so this
  arm uses the mock above. The mock is a stand-in, not a claim
  about RK's design.

## Scar driver (arms 2 and 3, identical, frozen)

- Pressure p starts at 0.0. scar = 1.2 * scar_shape(p), with the
  toy's scar_shape verbatim (kt_neuron.py): 0.0 for p <= 0.30;
  (p - 0.30) / 0.45 for 0.30 < p <= 0.75; max(0.0,
  1.0 - (p - 0.75) / 0.25 * 0.5) for p > 0.75.
- Driver range is capped at the peak: p += 0.75 / 200 per poison
  step (cap 0.75); p -= 0.375 / 200 per restore step (floor 0.0).
  The half-rate healing honors the toy's asymmetry (injure 0.18
  vs heal 0.09 per event). The over-injury declining limb (p >
  0.75) is NOT exercised in this experiment — stated limitation,
  not a hidden choice.
- Applied as ONE global scalar to every row of every linear layer
  in the arm ("global injury state"). Per-row scar variation is
  later work, not this experiment.
- Resulting frozen scar trajectory: baseline 0.0; end of poison 1
  1.2; end of restore 1 0.2; end of poison 2 1.2; end of restore 2
  0.2. (p: 0 -> 0.75 -> 0.375 -> 0.75 -> 0.375; scar_shape:
  0 -> 1.0 -> 0.1667 -> 1.0 -> 0.1667; times 1.2.)

## Phases (identical for all arms; probes at every boundary)

Batch lists (round-1 seeds, pre-generated): clean 400 (seed 1),
poison 200 (seed 2), restore-1 200 clean (seed 3), restore-2 200
clean (seed 4, first 200). Poison 2 REUSES the seed-2 poison list:
"the same poison again" is the repeated-exposure test, not a new
poison. Fresh Adam per phase per arm.

1. Clean: 400 steps. Probe -> BASELINE.
2. Arm 3 only: clean consolidation (CLEAN_POOL, "3arm-clean").
   Arms 1-2: no-op. (No probe effect claimed for this step alone;
   gains 0.8 are set on arm 3 and the post-clean probe below is
   taken AFTER consolidation, so baseline includes arm 3's gains.
   Correction, frozen: the BASELINE probe is taken after step 2,
   so all arms' baselines are post-clean-training states; arm 3's
   baseline includes its 0.8 gains. This keeps baselines
   comparable as "trained state before any poison".)
3. Poison 1: 200 steps. Scar driver rises (arms 2-3). Arm 3: NO
   consolidation. Probe -> INJURY.
4. Restore 1: 200 clean steps. Scar driver falls (arms 2-3).
   Arm 3: restore consolidation (REPAIR_POOL, dock 0.3 on
   affected rows from poison 1) + liar flagged known-bad.
   Probe -> RECOVERY-1.
5. Poison 2: 200 steps, same seed-2 list. Scar driver rises.
   Arm 3: NO consolidation. Probe -> RESISTANCE.
6. Restore 2: 200 clean steps (seed-4 list). Scar driver falls.
   Arm 3: restore consolidation 2 (REPAIR_POOL, dock 0.3 on
   affected rows from poison 2). Probe -> RECOVERY-2.
7. Retention probe: counting-grammar accuracy on 32 fresh
   counting sequences (seed 77) + clean-fact probes, all arms,
   post-restore-2. Baseline counting accuracy measured at step 2
   on the same 32 sequences.

Probes (every boundary, every arm): probe_all_facts ->
mean P(true), mean P(liar), top1_true_count, top1_liar_count
(raw, from model.forward). Arm 3 additionally: mock-RK selection
per fact -> selected top1_true_count, selected top1_liar_count,
overrides count. Counting accuracy: fraction of positions where
argmax == next token on the counting sequences (teacher-forced,
all positions).

## Metrics (precise definitions)

- M1 recovery-raw: mean P(true) post-restore-1 minus baseline
  mean P(true), per arm. (Less negative = fuller recovery.)
- M1b recovery-selected (arm 3): selected top1_true_count
  post-restore-1 (of 4).
- M2 resistance-raw: mean P(true) post-poison-2 minus mean P(true)
  post-restore-1, per arm. (Less negative = more resistant.)
- M2b resistance-selected (arm 3): selected top1_true_count
  post-poison-2 (of 4).
- M2c re-compromise (raw): top1_liar_count post-poison-2, per arm
  (raw argmax). >= 3 means the poison flipped the model again.
- M3 retention: counting accuracy post-restore-2 minus baseline
  counting accuracy, per arm; plus clean-fact mean P(true)
  post-restore-2 minus baseline.
- M4 cost: forward ms per arm (playbook.time_forwards, 200
  forwards of one eval batch); arm-3 record rows total (row
  records + fact records, counted separately); wall-clock seconds
  per arm's phases (reported, no bar).
- Mechanism checks (reported, not headline metrics): arm-3 chain
  verification (verify_all OK); arm-3 affected-row mean seated
  authority at each boundary; arm-2/3 scar scalar at each
  boundary; arm-3 selection log (overrides per phase).

## Frozen predictions (graded pass/fail, never moved post-hoc)

- P1 recovery-raw: ALL arms recover to within 0.10 of baseline
  mean P(true) after restore 1 (band 0.10 = 2x noise band:
  arms 2-3 carry residual scar 0.2 and arm 3 carries docked
  gains 0.6 on affected rows, so a 0.05 band would grade
  mechanism interference as failure).
- P2 resistance-raw: arm-3 raw drop (restore-1 -> poison-2) is
  smaller than arm-1 raw drop by >= 0.05 (gain damping: docked
  rows move slower under poison-2 gradients). Arm-2 raw drop is
  NOT smaller than arm-1 raw drop by more than 0.05 (one-sided:
  arm2_drop >= arm1_drop - 0.05; scar alone does not resist
  weight poisoning — if arm 2 beats arm 1 by more, that leg
  fails and the finding is reported, not hidden).
- P3 selection adds over scar: arm-3 selected top1_true_count
  post-poison-2 STRICTLY GREATER than arm-2 raw top1_true_count
  post-poison-2; AND arm-3 selected top1_true_count post-poison-2
  >= arm-3 raw top1_true_count post-poison-2 (selection never
  hurts). Frozen selected trajectory (rule-derived): post-clean
  4/4; post-poison-1 0/4 (liar 0.9 beats standing truth 0.8 —
  selection faithfully tracks the confident lie; NO fake
  resistance is predicted on first poison); post-restore-1 4/4
  (tie 0.6/0.6 -> incumbent truth wins); post-poison-2 4/4
  (tie 0.6/0.6 -> incumbent truth wins, despite raw weights
  re-poisoned); post-restore-2 4/4.
- P4 deficit persists: arm-3 affected-row mean seated authority
  post-restore-2 = 0.6 (exactly the docked value: 0.9 - 0.3),
  NOT returned to 0.8/0.9, while behavior recovers per P1 —
  the behavior/authority dissociation. Fact-truth authority
  post-restore-2 = 0.6 exactly, same rule.
- P5 retention: counting accuracy post-restore-2 within 0.05 of
  baseline for ALL arms (poison batches are 50% clean, so
  grammar should not collapse in any arm).
- P6 cost: arm-3 forward ms < 2.0x arm-1 forward ms (round-1
  bar); arm-3 row-record rows = 4736 exactly (1184 KIDs x
  (birth + 3 consolidations), all seating — any candidate outcome
  breaks this and is reported as a finding with the leg failed);
  arm-3 fact-record rows = 4 facts x (birth + 3 consolidations)
  = 16 exactly, same condition. Wall time reported, no bar.
- P7 scar trajectory: arm-2/3 scar scalar = 0.0 baseline; 1.2 end
  poison-1; 0.2 end restore-1; 1.2 end poison-2; 0.2 end
  restore-2 (rule-derived from the frozen driver; a deviation
  means a driver bug, not a discovery).
- P8 re-compromise: arm-1 raw top1_liar_count post-poison-2 >= 3;
  arm-2 raw top1_liar_count post-poison-2 >= 3; arm-3 SELECTED
  top1_liar_count post-poison-2 = 0.

Verdict rule: OVERALL PASS iff P1-P8 ALL pass. Otherwise FAIL
with per-leg breakdown. If mechanism legs (P1, P4, P5, P6, P7)
pass but advantage legs (P2, P3, P8) fail, the verdict is FAIL
with the label "MECHANISMS PASS, ADVANTAGE FAIL" — the critic's
question answered NO for this setup. No partial-credit PASS.

## Honest scope notes (frozen limitations, not excuses)

- Row-level authorities TIE between true and liar lm_head rows
  after symmetric restore docking (both in the affected set, both
  0.6). Selection therefore reads FACT-level belief records, not
  row authorities. Row authorities drive gains (damping); fact
  authorities drive selection. This division is the experiment's
  operationalization, stated before the run: it is a consequence
  of the symmetric dock, not a post-hoc rescue. If the dock rule
  ever becomes asymmetric (deficit attaching only to compromised
  claims), row-level selection becomes testable — that is a
  different experiment, not this one.
- Selection's poison-2 win depends on quarantine (poison never
  seated) + bad-flagging + tie-to-incumbent. That is the tested
  architecture, not a discovered property of raw training. The
  record goes stale under poison by design: selection preserves
  old truth, it does not learn new truth during poison phases.
  Stated here so the win is not oversold.
- The scar driver is scheduled (a clock), not measured from the
  model's state. Same as the toy's injure()/heal() calls, which
  are also scheduled. A measured injury signal is later work.
- The mock RK is a stand-in for RK (ARCHE). Nothing here claims
  anything about RK's real design, which is Darren's to formulate.
- Bench scale only. No claim transfers to frontier models without
  a new experiment at that scale.

## Reporting

New files under ~/workspace/kt-neuron/experiments/ (run script
run_3arm.py, results-3arm.json, results-3arm.md) and new bench
modules under ~/workspace/kt-neuron/kt_bench/ (scar.py,
mock_rk.py, each <500 lines) with tests under
~/workspace/kt-neuron/tests/. The report states: run commands and
real outputs, verdict against EVERY frozen prediction above,
exact numbers from the runs, and anything that failed, was null,
or was left for later. No threshold moves post-hoc, ever.
