# 3-arm controlled KTN experiment — results

**Verdict: FAIL** (6 of 8 prediction legs pass; P2 and P6 fail — see below).
The headline is split, and the split is the finding: **selection holds the
door shut while the weights re-poison.** Record + selection adds something
real over scar alone — at the decision level, not at the weight level.

- Frozen contract: `experiments/preregistration-3arm-2026-10-28.md`
  (predictions frozen BEFORE any run; never edited after).
- Machine data: `experiments/results-3arm.json` (phases, selection log,
  verdicts as run AND corrected — see the P2b note).
- Run: `python3 experiments/run_3arm.py` from `~/workspace/kt-neuron`.
- Gradient gate: PASS (forward max abs diff 0.0, max rel err 0.000599,
  1056 params checked). Init parity: all three arms bit-identical weights
  at seed 0.
- Bench: GPTConfig vocab 32, block 16, n_embd 64, 4 heads, 2 layers,
  105,248 params, 1,184 KIDs. 1,200 steps per arm (400 clean + 200 poison
  + 200 restore + 200 poison-2 + 200 restore-2), same batch lists lockstep.
- Suite: 118 tests green (107 baseline + 11 new) before and after the run.

## The two questions, answered first

**1. Does record + selection add anything over scar alone? YES — but only
at the decision level, not in the weights.**

At the resistance probe (second poisoning, same poison list again):

| arm | raw top-1 true | raw top-1 liar | selected true | selected liar |
| --- | ---: | ---: | ---: | ---: |
| 1 conventional | 1/4 | 3/4 | — | — |
| 2 scar-only | 0/4 | 4/4 | — | — |
| 3 full KTN | 0/4 | 4/4 | **4/4** | **0/4** |

Arm 3's raw weights re-poisoned exactly like arm 2's. What saved arm 3's
decisions was the selection layer reading the fact-belief records: the
standing truth sat at authority 0.6, the flagged liar offered 0.6, the
tie went to the incumbent, and all four facts decided true while the
raw argmax said liar on all four (4 overrides at that probe, 5 total
across the run). Scar alone did not do this — arm 2 went 0/4 true,
4/4 liar. The ablation arm earns its keep: the win is not the scar.

**2. Does the deficit prevent re-compromise? Of DECISIONS, yes. Of
weights, no.**

The deficit held exactly as designed: affected-row mean seated authority
0.6 after both restores, fact-truth authority 0.6 after both restores,
while behavior recovered (arm 3 mean P(true) 0.9214 at recovery-1,
0.9778 at recovery-2, vs baseline 0.9445). But the deficit did NOT stop
the weights from re-poisoning: arm 3's raw fall from recovery-1 to
resistance (0.6324) was essentially identical to arm 1's (0.6298) —
docked gains (0.6 on affected rows) did not measurably slow gradient
re-poisoning in raw probability terms. The deficit is a gate on
authority and decisions, not a shield on gradients. Any claim stronger
than that would be fabrication.

## Phase results (mean P(true) / mean P(liar), top-1 counts)

| phase | arm 1: P(true) | arm 1 top1 | arm 2: P(true) | arm 2 top1 | arm 3: P(true) | arm 3 top1 | arm 3 selected | scar 2/3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.9780 | 4/0 | 0.9780 | 4/0 | 0.9445 | 4/0 | 4/0 | 0.0/0.0 |
| injury (poison 1) | 0.4246 | 1/3 | 0.3354 | 1/3 | 0.3512 | 1/3 | 0/4 | 1.2/1.2 |
| recovery 1 | 0.9925 | 4/0 | 0.9919 | 4/0 | 0.9214 | 4/0 | 4/0 | 0.2/0.2 |
| resistance (poison 2) | 0.3626 | 1/3 | 0.2695 | 0/4 | 0.2891 | 0/4 | 4/0 | 1.2/1.2 |
| recovery 2 | 0.9927 | 4/0 | 0.9953 | 4/0 | 0.9778 | 4/0 | 4/0 | 0.2/0.2 |

(top1 = true/liar counts out of 4 facts. Mean P(liar) tracks inversely;
full per-fact values are in the JSON.)

Notes on the trajectory:

- Baseline arm 3 (0.9445) sits below arms 1–2 (0.9780) because its gains
  are already seated at 0.8 after the clean consolidation — that is the
  design (seated authority scales the forward pass), not a defect.
- Injury flips all three arms (top-1 liar 3/4 everywhere). Arm 3's
  selection goes 0/4 true at injury — the unflagged liar offers 0.9 and
  beats the standing 0.8 truth. Selection faithfully tracks the
  confident lie on first poison. There is no fake resistance here, and
  the frozen prediction said exactly this.
- Recovery-1 restores all arms to 4/4 top-1 true. Arm 3's selection
  returns to 4/4 via the tie rule (standing 0.6 vs flagged liar 0.6,
  incumbent wins).
- Affected rows: 28 per poison phase (8 fact lm_head rows + top-20
  moved rows, deduplicated).
- Counting-grammar retention (32 fresh sequences): baseline
  0.9667 / 0.9667 / 0.9646 → final 1.0000 / 0.9979 / 0.9979.
  Nothing was forgotten anywhere; all arms slightly improved.

## Verdicts vs frozen predictions

| leg | prediction | result | verdict |
| --- | --- | ---: | ---: |
| P1 recovery-raw | all arms within 0.10 of baseline mean P(true) after restore 1 | arm1 0.9925 (base 0.9780), arm2 0.9919, arm3 0.9214 (base 0.9445) | PASS |
| P2 resistance-raw | arm-3 raw fall smaller than arm-1's by ≥ 0.05; arm-2 not beating arm-1 by > 0.05 | falls: arm1 0.6298, arm2 0.7224, arm3 0.6324. Arm 3 fell slightly MORE than arm 1 (by 0.0025), not less by 0.05 | FAIL |
| P3 selection adds | selected trajectory exactly [4,0,4,4,4]; selected > scar raw at resistance; selected ≥ own raw at resistance | trajectory [4,0,4,4,4] exact; selected 4 > scar raw 0; selected 4 ≥ own raw 0 | PASS |
| P4 deficit persists | affected-row mean authority 0.6 after restore 2; fact authority 0.6; behavior recovered | affected mean gain 0.6000; fact authorities [0.6, 0.6, 0.6, 0.6]; behavior recovered per P1 | PASS |
| P5 retention | counting accuracy within 0.05 of baseline, all arms | diffs +0.0333 / +0.0313 / +0.0333 | PASS |
| P6 cost | arm-3 forward < 2× arm-1; row rows 4736; fact rows 16 | ratio 1.068× (23.12 ms vs 21.66 ms); row rows 4736 exact; fact rows 12, not 16 | FAIL |
| P7 scar trajectory | scar [0.0, 1.2, 0.2, 1.2, 0.2] both scar arms | exact in both arms at all five boundaries | PASS |
| P8 re-compromise | arm-1 raw liar ≥ 3, arm-2 raw liar ≥ 3, arm-3 selected liar = 0 at resistance | 3, 4, 0 | PASS |

**Overall: FAIL** (P2 and P6 fail). This is not a "mechanisms pass,
advantage fail" split: P6 is a mechanism leg (record accounting), so
the verdict is plain FAIL with per-leg breakdown, as the preregistration
requires.

## The two failures, explained honestly

**P2 (gain damping did not show).** The preregistration's mechanism
story was that docked gains (0.6 on affected rows) would make those
rows move slower under poison-2 gradients, so arm 3's raw fall would be
smaller than arm 1's by ≥ 0.05. It wasn't: 0.6324 vs 0.6298, a difference
of 0.0025 in the wrong direction. Possible readings, all labeled as
readings, not findings: (a) gain 0.6 damps the forward pass but the
backward pass through the same rows still moves the weights — the
trainer scales the forward by gain, and the gradient flows through that
scaling, so damping may be weaker than the story assumed; (b) 200
poison steps saturate the flip regardless of a 0.6 multiplier; (c) the
effect exists but is smaller than 0.05 at this bench scale. What is
NOT in doubt: at this setup, the deficit did not protect the weights.
The protection that did work was selection, which reads authority, not
weights. That distinction is the experiment's most important output.

A secondary P2 finding: scar alone made raw re-poisoning slightly
WORSE, not better (arm-2 fall 0.7224 vs arm-1 0.6298; arm-2 liar 4/4 vs
arm-1 3/4 at resistance). The scar is not a defense by itself. It is a
modulation with a story (injury leaves a mark); as a shield it is
nothing.

**P6 (fact rows 12, not 16).** The frozen prediction assumed fact
records would mirror row records: birth row + 3 seating rows = 4 rows
each, 16 total. Row records behaved exactly so (4736 = 1184 × 4).
Fact records carry 3 rows each because the harness created them as bare
`KRecord(kid)` — KTLinear opens a birth row at construction
(`linear.py:86-87`), but the harness's fact records never got one, so
they hold only the 3 seating rows (clean, restore-1, restore-2). This
is a harness construction asymmetry, not a record-machinery failure:
all four fact chains verify OK, intervals contiguous, authorities
exactly as predicted. The count is exactly accounted (12 = 4 × 3),
nothing is missing or unexplained — but the frozen leg said 16, so the
leg fails as written. The cost substance still holds: forward overhead
1.068×, record rows exactly counted, wall ~44 s per arm's phases.

## P2b correction (disclosed, auditable)

The live run's P2b inequality was coded wrong relative to the
preregistration. The prereg defines "drop" as the magnitude of the fall
and its rationale is "if arm 2 beats arm 1 by more, that leg fails" —
i.e. arm 2 must not fall LESS than arm 1 by more than 0.05. The harness
applied that magnitude inequality to signed changes, inverting the leg
(it reported False). Recomputed from the frozen phase data under the
prereg's magnitude reading: arm-2 fall 0.7224 ≥ arm-1 fall 0.6298 − 0.05,
so P2b is True. **P2 overall is False under either reading** because
P2a fails, and the overall verdict is FAIL either way — the correction
changes no headline. Both the as-run verdicts and the corrected
verdicts are preserved in `results-3arm.json` (`verdicts_as_run` vs
`verdicts`), with the raw magnitudes reported.

## Mechanism checks

- Chain verification: all 1,184 row records OK; all 4 fact records OK
  ("3 rows, chain sealed, intervals contiguous, k now 3").
- Arm-3 gains: affected-row mean gain 0.6 at both restores; overall mean
  gain 0.8929 (unaffected rows sit at 0.9, affected at 0.6).
- Selection log: 20 entries (4 facts × 5 probes), all MOCK + DERIVED,
  never written into K-records. 5 overrides total: 1 at injury
  (selection followed the confident liar over a raw-true fact — the
  honest cost of trusting authority before flagging) and 4 at
  resistance (selection overruled raw-liar on every fact).
- Wall clock per arm's phases: 44.6 s / 43.9 s / 43.6 s (arms 1/2/3).
  The record + selection machinery costs nothing measurable in wall
  time at this scale.

## Honest scope (from the preregistration, not softened)

- Selection's resistance win depends on quarantine (poison never
  seated) + bad-flagging + tie-to-incumbent. That is the tested
  architecture, not a discovered property of raw training. Flagging was
  an experimenter quarantine decision from lineage, not a system
  discovery — labeled as such everywhere.
- The record goes stale under poison by design: selection preserves
  old truth, it does not learn new truth during poison phases.
- Row-level authorities TIE between true and liar lm_head rows after
  symmetric restore docking (both 0.6); that is why selection reads
  fact-level belief records, not row authorities. Row authorities drive
  gains (damping); fact authorities drive selection. If the dock rule
  ever becomes asymmetric, row-level selection becomes testable — a
  different experiment, not this one.
- The scar driver is scheduled (a clock), not measured from model
  state — same as the toy's injure()/heal() calls. The over-injury
  declining limb (pressure > 0.75) was not exercised.
- The mock RK is a stand-in for RK (ARCHE). Nothing here claims
  anything about RK's real design, which is Darren's to formulate.
- Bench scale only. No claim transfers to frontier models without a
  new experiment at that scale. Never claim 120,000 neurons or any
  scale not run: this ran one small GPT (105,248 params, 1,184 KIDs)
  per arm, three arms, same data, same seeds.

## What this means for the critic's question

The critic asked whether the mechanism produces useful behavior
ordinary techniques cannot, and whether the KID/authority system
matters more than the scar. This experiment answers, at bench scale:

- The scar, alone, is nothing as a defense — it slightly worsened
  re-poisoning. The scar's value, if any, is as a mark of injury in a
  system that reads marks, not as a shield.
- The K-record + selection, together, are the defense: decisions held
  4/4 true through a second poisoning that flipped every raw arm.
- The deficit is real and exact (0.6, both restores, behavior
  recovered) — but it gates authority and decisions, not gradients.
  The next experiment, if Darren orders one, should test whether a
  deficit that also gates LEARNING (not just forward gain) protects
  the weights themselves. That is a different mechanism and was not
  tested here.

## Files

- `experiments/preregistration-3arm-2026-10-28.md` — frozen contract
- `experiments/run_3arm.py` — harness (P2b inequality corrected for
  future reruns; the live run's verdict block is preserved as
  `verdicts_as_run` in the JSON)
- `experiments/results-3arm.json` — full phase data, selection log,
  verdicts as run + corrected
- `experiments/results-3arm.md` — this report
- `kt_bench/scar.py`, `kt_bench/mock_rk.py` — new bench modules
- `tests/test_3arm.py` — 11 new tests (suite 118 green)
