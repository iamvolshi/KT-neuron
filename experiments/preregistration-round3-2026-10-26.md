# Round 3 pre-registration — KTN bench, new knowledge (2026-10-26)

Written BEFORE any Round 3 run. Frozen: no threshold below changes after
a run. Round-1 verdicts (E1 PASS, E2 FAIL strict step-cost leg, E3 FAIL
frozen confidence threshold, E4 PASS hedge leg, E5 PASS all legs) and
Round-2 verdicts (E2-rerun work leg PASS narrow 0.0835, E3-rerun split
verdicts, decline PASS all frozen legs, demo green) STAND and are never
rewritten. This file covers Round 3 only. New files only; no round-1 or
round-2 file is amended.

## Shared ground (all experiments)

- Bench: numpy-only, manual backprop + Adam (`experiments/trainer.py`),
  vocab 32, block 16, n_embd 64, 4 heads, 2 layers, 105,248 params,
  1,184 KIDs. Seeds init 0 / data 1 (same as rounds 1–2 for
  comparability). Clean 400 steps / poison 200 steps unless an
  experiment below states otherwise. Repair dock 0.3; repair pool
  textbook 0.9 → post-repair authority 0.6.
- Gradient gate must PASS for any new training code path. Full test
  suite must stay green (69 tests at round-3 start; new tests added
  for new modules). Deterministic and seeded throughout.
- PYTHON NOISE RULE: the bench is a pre-concept proof in Python. It
  proves the SHAPE. Bands below (0.05) are Python-noise bands, stated
  as such. Do not grade the idea by the noise. Exactness lives in the
  Rust engine, out of scope here.
- RETIRED METERS — do NOT use: FLOP saving (never a goal; repair cost
  is rows/params changed, i.e. optimizer writes); the absolute
  confidence bar at repair time (retired; the trajectory is the
  claim); per-row exact attribution (not a goal; interval-level
  lineage plus working repair is sufficient).
- EXCLUDED MACHINERY (Darren, 2026-10-26): do NOT build or test silent
  marking, instant honeypots, fake data pools, or morphic data
  integrity. Those are plan design notes, not this round. If a stage
  seems to need them, say so and skip that part.

## R3-E1 — One-signal regulation (plan §7)

Claim (Darren, Oct 26): the authority/trust signal follows the fast
trace until it intersects the slow trace, then the slow takes over.
Fast phase carries impact, slow phase carries influence — one signal
throughout, never merged. The handoff is seamless in value (traces
equal at intersection) and total in provenance (live reading becomes
record). RESOLVED: this handoff IS the same switch as the poison flip
(plan §6 step 4) — one switch, not two.

Honest scope note (frozen): the current bench has NO slow-accumulation
dynamics — consolidation snapshots live tensors, it does not accumulate
a slow trace. So R3-E1 is build + demonstrate, not discovery: implement
a SignalTracker (new module, <500 lines, with tests) and demonstrate
the regulation mechanism as specified. If a crossing is found in
UNMODIFIED dynamics under some trace definition, report that as the
stronger result; do not redefine the claim post-hoc to manufacture one.

Frozen operationalization:
- fast(t) = live-tensor probe mean P(true) on the 4 clean facts,
  sampled at each consolidation boundary during the clean phase.
- slow(k) = kernel value: an accumulating slow trace updated ONLY at
  consolidation from the seated probe value. The agent picks the
  accumulation formula (report it verbatim); it must satisfy: slow
  updates only at consolidation; slow is non-decreasing during clean
  training; slow starts below fast.
- Clean phase: 400 steps with consolidations every 100 steps (4
  boundaries) so a crossing sequence is observable.
- Crossing k* = first boundary with slow(k*) >= fast(k*) - 0.05
  (0.05 = Python-noise band).
- Ownership flag: signal(k) = fast for k < k*, slow for k >= k*.

Frozen thresholds (ALL required for PASS):
- (a) A crossing occurs within the clean phase (k* exists among the
  4 boundaries).
- (b) Ownership is a single switch: flag sequence is some fasts then
  all slows, no flip-flop.
- (c) Continuity at the switch: |slow(k*) - fast(k*)| <= 0.05.
- (d) Post-handoff the signal equals slow within 0.05 at every later
  boundary.
- Flip corollary (same switch): poison introduced AFTER k* (mature,
  slow-owned system). Across poison steps BEFORE the next
  consolidation: signal change <= 0.05 WHILE fast-trace probe moves
  >= 0.10 (insulation demonstrated: fast moves, signal doesn't).
  This is a wiring/integration check (signal reads the slow trace,
  not live tensors, post-handoff) — label it as such.

## R3-E2 — Tail manipulation controls healing (plan §6 step 7)

Claim (Darren, Oct 26): whether a scar heals at all, and how fast,
is set by manipulating the tail's parameters. Tail rules are
deployment-set; the law gives the shape, the deployment sets the
parameters.

Frozen setup: same injury as round-1 E2 (clean 400, poison 200,
repair dock 0.3 → deficit 0.30 at repair). Then decay consolidations
with THREE tail deployments, deficit decay per consolidation:
- FAST tail: 0.05 per consolidation.
- SLOW tail: 0.01 per consolidation (replicates round-1/2's tail).
- PERMANENT tail: 0.00 per consolidation (never heals).
Healed is defined as deficit <= 0.05. Track to +6 consolidations
minimum; extend to +30 if needed to observe healing (decay
consolidations are seating-only, cheap).

Frozen thresholds (ALL required for PASS):
- (a) FAST healed by +6 (0.30 - 6*0.05 = 0.00 <= 0.05).
- (b) SLOW not healed by +6 (deficit ≈ 0.24 ± 0.005, reproducing
  round-1/2).
- (c) PERMANENT deficit = 0.30 ± 0.005 at +6 (unchanged).
- (d) Ordering: time-to-healed FAST < SLOW < PERMANENT-never.
- (e) Behavior ranking holds throughout healing: top-1 true 4/4 at
  repair and at every tracked consolidation (trajectory leg).
- (f) Lineage at interval level names the poison interval exactly
  (round-1 E2 pattern), and repair seats all affected rows.

Re-measurement legs (report separately, same run):
- Confidence trajectory (NOT the retired absolute bar): post-repair
  confidence non-decreasing across tracked consolidations. Report
  absolute values for the record only — no verdict on them.
- Repair cost as work touched (NOT FLOPs, NOT steps): report
  repair vs full-retrain rows/params changed. Replication bar:
  ratio < 0.10 (round-2's bar). Report the exact ratio either way.

## R3-E3 — Tick mutability (plan §2)

Claim (Darren, Oct 26): scale is in the mechanism — the tick unit is
mutable. k's granularity adapts to the deployment; the law doesn't
change.

Frozen setup: same injury + repair + decay pipeline as R3-E2, with
three clean-phase tick sizes (tick = steps per clean consolidation):
- FINE tick: consolidate every 100 clean steps (400 steps, 4 ticks).
- BASELINE tick: 1 consolidation after 400 clean steps (round-1
  structure).
- COARSE tick: 1 consolidation after 800 clean steps (800 steps,
  1 tick; clean length adjusted so each variant holds ≥1 full tick).
Per-tick deficit decay fixed at 0.01 for all variants. Baselines are
reported per variant; NO cross-variant absolute probe comparison
(the variants train different amounts — only SHAPE is compared).

Frozen thresholds (ALL required for PASS):
- (a) Post-repair authority lag >= 0.1 below pre-injury in ALL
  variants.
- (b) Deficit non-increasing across decay consolidations in ALL
  variants.
- (c) Behavior ranking 4/4 post-repair in ALL variants.
- (d) Tick-count to healed (deficit <= 0.05 from 0.30 at 0.01/tick)
  identical across variants within ±1 tick (expected 25 ticks).
  This is the scale-invariance check: decay is per-TICK, not
  per-step — if decay were per-step, coarse-tick runs would heal in
  fewer ticks. Label the wiring-check nature honestly.

## Reporting (each experiment)

New files under ~/workspace/kt-neuron/experiments/ (run scripts,
results-*.md, results-*.json). New bench modules under
~/workspace/kt-neuron/kt_bench/ (<500 lines each) with tests under
~/workspace/kt-neuron/tests/. Each report states: run commands and
real outputs, verdict against the frozen thresholds above, exact
numbers from the runs, and anything that failed, was null, or was
left for later. No threshold moves post-hoc, ever.
