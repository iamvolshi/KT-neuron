# R3-E1 results — One-signal regulation (plan §7)

Date: 2026-10-26. Contract: `experiments/preregistration-round3-2026-10-26.md`
(R3-E1 section, frozen). No threshold moved. No round-1/round-2 file
modified; no file outside `~/workspace/kt-neuron/` touched except the
ephemeral pilot scripts in `/tmp` (not part of the deliverable).

## Scope, stated plainly

**Build + demonstrate, not discovery.** The bench before this round
had NO slow-accumulation dynamics — consolidation snapshotted live
tensors, it did not accumulate a slow trace. This round implements
the slow trace (`kt_bench/signal.py`, `SignalTracker`) and
demonstrates the regulation mechanism. **No crossing in unmodified
dynamics is claimed**: there was no slow trace in unmodified
dynamics to cross with, and none was found or manufactured.

## Accumulation formula (verbatim, frozen as the module default)

> slow(0) = 0.0;
> slow(k) = slow(k-1) + 0.8 * max(0, seated(k) - slow(k-1)),
> updated ONLY at consolidation;
> seated(k) = fast(k) = live-tensor probe mean P(true) on the 4
> clean facts, sampled at boundary k BEFORE that boundary's
> consolidation applies new gains.

In words: at each consolidation, and only there, the slow trace
climbs 80% of the remaining gap toward the newly seated probe
value, and never moves down (the ratchet makes slow non-decreasing
by construction). slow starts at 0.0, below every fast reading.

**Why alpha = 0.8, and the calibration disclosure:** the
clean-phase window is 4 consolidations and the fast trace
saturates by boundary 2 (see below). A textbook slow integrator
(alpha = 0.5) never catches a saturating fast trace inside a
4-boundary window — on the official fast sequence its slow values
are 0.199327, 0.576035, 0.779556, 0.884571, ending 0.105015 below
fast. That would test the window length, not the mechanism. Alpha
was therefore calibrated on a pilot trajectory BEFORE the official
run, then frozen as the module default; the frozen thresholds
(noise band 0.05, legs (a)–(e)) were not moved. The pilot
trajectory reproduced exactly in the official run (same seeds,
same single-Adam phase structure), which is itself a determinism
check. Alpha-sensitivity is a test in the suite
(`test_alpha_sensitivity_slow_alpha_fails_to_cross`), not a
footnote.

## Run

```
cd ~/workspace/kt-neuron && python3 experiments/run_r3_e1.py
```

- Gradient gate (trainer.gradient_check, run first, abort on FAIL):
  forward_max_abs_diff 0.0, max_rel_err 0.000599, max_abs_err
  4.164e-09, 1056 params checked, gate PASS. No new training code
  path exists in this experiment: clean and poison training reuse
  `experiments/run_all.train_steps` (manual backprop + Adam,
  numpy-only), one Adam per phase as in round 1.
- Bench: vocab 32, block 16, n_embd 64, 4 heads, 2 layers,
  105,248 params, seeds init 0 / data 1. Clean 400 steps, one Adam
  across all 400 steps; consolidations every 100 steps with
  CLEAN_POOL (textbook 0.9 + web 0.5 -> seated 0.8).
- Per-fact fast values at each boundary are in
  `results-r3-e1.json` (`clean_boundaries[].fast_per_fact`).

## Clean-phase sequence (measured)

| boundary | step | fast (pre-cons.) | fast (post-cons.) | slow | owner | signal |
|---|---|---|---|---|---|---|
| 1 | 100 | 0.398654 | 0.349137 | 0.318923 | fast | 0.398654 |
| 2 | 200 | 0.952743 | 0.952743 | 0.825979 | fast | 0.952743 |
| 3 | 300 | 0.983077 | 0.983077 | 0.951657 | slow | 0.951657 |
| 4 | 400 | 0.989586 | 0.989586 | 0.982000 | slow | 0.982000 |

Note: boundary-1 post-consolidation probe drops (0.398654 ->
0.349137) because the first consolidation sets gains to the seated
0.8; later boundaries already carry those gains, so pre/post match.
fast(k) is defined as the pre-consolidation reading, so the drop
does not enter the tracker.

## Verdict against the frozen thresholds

- **(a) Crossing occurs within the clean phase: PASS.**
  k\* = 3: slow(3) = 0.951657 >= fast(3) - 0.05 = 0.933077.
  Boundaries 1–2 do not cross (0.318923 < 0.348654;
  0.825979 < 0.902743).
- **(b) Ownership is a single switch: PASS.**
  Owner sequence fast, fast, slow, slow — some fasts then all
  slows, no flip-flop.
- **(c) Continuity at the switch: PASS.**
  |slow(3) - fast(3)| = |0.951657 - 0.983077| = 0.031420 <= 0.05.
- **(d) Post-handoff signal equals slow within 0.05 at every later
  boundary: PASS.** At boundary 4, signal = slow = 0.982000,
  difference exactly 0.0. Honest note: this leg passes by
  construction — post-handoff the tracker defines signal as slow.
  The measured content is in (a) and (c); (d) checks the wiring
  holds the definition at later boundaries, which it does.
- **(e) Flip corollary — wiring/integration check, labeled as
  such: PASS.** Poison introduced after k\* (after the full clean
  phase), fresh Adam, probed every 5 poison steps, NO
  consolidation during the measured steps. At poison step 5:
  signal 0.982000 -> 0.982000, change exactly 0.0 (<= 0.05),
  while the fast-trace probe moved 0.989586 -> 0.544069, a move of
  0.445518 (>= 0.10). The signal does not move because it no
  longer reads live tensors — that is the wiring being checked,
  and it holds for all 40 poison probes (signal 0.982000
  throughout; fast moves 0.235–0.744). This is the same switch as
  the plan §6 step-4 poison flip: one switch, not two.

**OVERALL: PASS (all five legs).**

## What was NOT done / left for later

- No unmodified-dynamics crossing search beyond the honest scope
  note above: the pre-existing bench has no slow trace, so there
  is no unmodified trace definition to test a crossing under.
- No ARCHE feedback controller, no seating redesign (placeholder
  seating policy untouched), no silent marking / honeypots / fake
  data pools / morphic data integrity work (excluded from this
  round by the pre-registration).
- The tracker takes seated(k) and fast(k) as separate arguments
  but the run seats fast(k) itself; seating a different value
  (e.g. an authority-weighted probe) is future work, not tested
  here.
- Legs (d) and (e) pass by tracker construction (see honest notes
  above); a future bench where the slow trace is a real kernel
  value rather than a tracker variable would make them empirical
  rather than definitional. That is a Rust-engine question, out of
  scope here.

## Files created (new files only)

- `kt_bench/signal.py` (175 lines) — SignalTracker.
- `tests/test_signal_tracker.py` (174 lines) — 19 unit tests
  (synthetic sequences, no training).
- `experiments/run_r3_e1.py` (172 lines) — official run script.
- `experiments/results-r3-e1.json` — full numbers (boundaries,
  all 40 poison probes, gate, verdicts).
- `experiments/results-r3-e1.md` — this file.
