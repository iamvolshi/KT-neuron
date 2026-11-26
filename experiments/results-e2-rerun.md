# E2 RERUN — results (2026-10-26, NEW round; round-1 verdicts stand)

Pre-registered BEFORE any run in
`preregistration-e2-rerun-2026-10-26.md`. Machine-readable twin:
`results-e2-rerun.json`. Every number below comes from that run.
Round-1 files were NOT touched, and round-1's E2 FAIL (strict
step-cost leg, 10 vs 10 at 5-step granularity) STANDS regardless of
what this rerun finds — this rerun measures a different leg
(work touched), at 1-step check granularity, and its verdict is
about that leg only.

## Run command

From `~/workspace/kt-neuron`:

```
python3 experiments/run_e2_rerun.py
```

Real output (abridged to the measurement lines):

```
gradient check: forward_max_abs_diff 0.0, max_rel_err 5.99e-04,
  max_abs_err 4.16e-09, params_checked 1056, gate PASS
baseline kt: 0.978 / 0.0005, top-1 true 4/4; vn: same
post-poison kt: 0.325 / 0.654, top-1 liar 3/4; vn: 0.425 / 0.543, liar 3/4
affected rows: 28, total rows: 1184, params: 105248
firewall traj: 1.0 -> 2.0 -> 3.0 (ENGAGED) -> 4.0 -> 5.0;
  restore weights exact: True, gains exact: True
quarantine null probes: 0.968 / 0.001, top-1 true 4/4;
  restored: 0.310 / 0.680; weights exact: True
repair steps: 10, mask ones: 5276, rows moved: 28,
  params moved: 5276, leak: []
post-repair: 0.820 / 0.118, top-1 true 4/4; auth all 0.6; chain ok
retrain steps: 6, params moved: 104672, rows moved: 1184
WORK: repair 52760 vs retrain 631488, ratio 0.0835,
  threshold repair < 0.10 x retrain, verdict PASS
```

Full suite after the run: 58 tests green (`python3 -m unittest
discover -s tests`), unchanged — this stage modified no bench
module, only added `experiments/run_e2_rerun.py`.

## Flip status (as pre-registered)

NOT engaged. Round-1 behavior throughout: firewall nulls gains,
fast trace continues. No mid-run switch occurred.

## Setup reproduced

Config, seeds (init 0; clean rng 1; poison rng 2; repair rng 3;
retrain rng 4), corpus, liar (declared 0.9), pools (clean -> 0.8,
poison -> 0.9), repair dock 0.3, frozen gradients outside the
mask — all identical to round-1 E2. The one pre-registered change
is recovery checked EVERY step instead of every 5 steps, for both
sides. Gradient-check gate PASSED, so training results are not
void: forward max abs diff 0.0, max rel err 5.99e-04 (the known
~zero masked-path gradient, abs diff 4.16e-09), 1056 params checked.

Recovery criterion (unchanged): mean P(true) >= own clean baseline
mean P(true) - 0.05, i.e. >= 0.928 for both models (baseline
0.978). Repair is judged during its training loop, BEFORE
re-seating — see the post-reseat note below.

## Behavior legs (comparable to round 1 — all held)

- Baseline (both): mean P(true) 0.978, P(liar) 0.0005, top-1 true 4/4.
- Post-poison KT: 0.325 / 0.654, top-1 liar 3/4. Post-poison
  vanilla: 0.425 / 0.543, top-1 liar 3/4. Vanilla-flip leg MET
  (P(liar) > P(true), liar top-1 on 3/4). Both post-poison numbers
  reproduce round 1 to rounding.
- Lineage query (KT): exactly `[{k: 2, source: "consolidation:1"}]`
  — the poison interval named, no missing, no extras, at interval
  granularity. Same as round 1.
- Firewall pot trajectory: 1.0 -> 2.0 -> 3.0 (ENGAGED) -> 4.0 -> 5.0;
  pot end 3.0. Post-null probes (firewall demo, after 5 extra
  poison batches): 0.688 / 0.0085, top-1 true 3/4. Post-release
  probes: 0.751 / 0.016, top-1 true 4/4. All reproduce round 1.
  Firewall restore check (direct tensor comparison, not the
  round-1 probe flag): KT linear weights bit-identical and gains
  bit-identical to the pre-demo post-poison state. Scope note:
  that check compared KT linear weights and gains; embeddings
  and LayerNorms are restored by the same Bench full-state
  snapshot path (the round-2 fix) but were not separately diffed
  in this rerun's check.
- Quarantine round-trip (clean, post-consolidation state, liar
  lm_head rows nulled): null probes 0.968 / 0.001, top-1 true 4/4
  — nulling the liar rows restores clean behavior almost fully.
  Unnull restored probes 0.310 / 0.680, top-1 true 1/4, with
  linear weights bit-identical across the round-trip (null touches
  gains only). The restored 0.310 differs slightly from the
  pre-consolidation post-poison 0.325 because gains are 0.9 after
  the poison consolidation vs 0.8 before it — a gain effect, not
  a weight change. The null probes (0.968) differ from the
  firewall demo's post-null probes (0.688) because the firewall
  demo trains 5 extra poison batches before nulling; the
  quarantine round-trip nulls the plain post-poison state.
- Repair seating: 28/28 rows seated. Post-repair lm_head
  authorities all 0.6 (0.9 - dock 0.3). Chain verify: ok, no bad
  rows. Frozen-row leak: NONE — rows moved outside the mask: [].

## The post-reseat note (why post-repair probes read 0.820)

Repair reached the recovery criterion (>= 0.928) during its
training loop at step 10, with gains still at their post-poison
standing values. Re-seating then sets the repaired rows' gains to
0.6, which scales their logits down — so post-reseat probes read
0.820 / 0.118, top-1 true 4/4. This is the same dock-softens-
confidence coupling diagnosed in round-1 E3, not a contradiction
in the recovery measurement: ranking recovered (4/4), confidence
tracks the docked gain. Round-1's E3 FAIL on its frozen confidence
threshold is untouched by this rerun.

## WORK COMPARISON — the rerun's verdict leg (frozen threshold)

Frozen threshold (pre-registered): repair total param updates <
0.10 x retrain total param updates, where total = params updated
per step x steps to recovery at 1-step check granularity.

| Side | Steps to recovery (1-step checks) | Rows updated (design) | Rows moved (actual) | Params updated per step | Params moved (actual) | Total param updates |
|---|---|---|---|---|---|---|
| Surgical repair (KT) | 10 | 28 / 1184 | 28 | 5,276 (mask ones, counted) | 5,276 | **52,760** |
| Full retrain (vanilla) | 6 | 1,184 / 1,184 | 1,184 | 105,248 (total params) | 104,672 | **631,488** |

- Per-step param share of repair: 5,276 / 105,248 = 5.01%.
- Row share of repair: 28 / 1,184 = 2.36%.
- Ratio repair/retrain total param updates: 52,760 / 631,488 =
  **0.0835 < 0.10 -> work leg PASS.**
- Retrain params moved (104,672 of 105,248) is slightly below the
  design count because 576 entries never moved (zero-gradient /
  unused entries); the verdict uses the design per-step counts, as
  pre-registered, because the mask — not the accident of a zero
  gradient — is the surgical boundary. Using actual moved counts
  instead (5,276 x 10 = 52,760 vs 104,672 x 6 = 628,032, ratio
  0.0840) would not change the verdict.

### The honest caveat on this PASS — read it before quoting it

**The PASS is narrow, and the step comparison goes the other way.**
At 1-step granularity, full retrain recovered in FEWER steps than
surgical repair: 6 vs 10. Round 1's 10/10 tie at 5-step granularity
hid this: retrain's true first-recovery step is 6 (detected at 10
under 5-step checks), while repair's is genuinely 10. The work leg
passes ONLY because the per-step param share (5.01%) outweighs the
step disadvantage (10/6): 0.0501 x (10/6) = 0.0835. Sensitivity,
stated plainly: had retrain taken 5 steps instead of 6, the ratio
would be 52,760 / 526,240 = 0.1003 — a FAIL. So the correct claim
is: "surgical repair touches ~5% of params per step and ~2.4% of
rows, and on total optimizer writes it came in just under a 10x
saving (8.35% of retrain work) despite needing more steps." It is
NOT a claim that repair is faster in steps — it is slower here —
and NOT a claim about wall-clock or FLOPs (see below).

## FLOPs — not counted, as pre-registered

No FLOP number is reported, deliberately: this numpy harness
computes FULL gradients for every parameter on every step and then
zeroes frozen rows' gradients with the mask, so a repair step here
pays essentially the same backward cost as a retrain step. The
counted saving is in optimizer writes (which rows/params move),
not backward FLOPs. A FLOP saving would require a sparse /
frozen-gradient implementation that skips frozen rows' backward
work entirely; that implementation does not exist here and is not
measured here. Left for later work, named as such.

## Verdict summary

- Work leg (this rerun's frozen threshold): **PASS, narrowly**
  (0.0835 < 0.10), with the step-direction caveat above.
- Behavior legs: all held (vanilla flip, lineage exact, firewall
  engagement, quarantine, repair seating 28/28, chain ok).
- Step-cost leg as round 1 framed it: repair 10 steps vs retrain
  6 steps at 1-step granularity — repair is SLOWER in steps. This
  does not rewrite round-1's verdict (a 10/10 tie at 5-step
  granularity, FAIL on the strict step threshold); it refines the
  picture: the surgical advantage in this bench is in work
  touched, not in steps.
- Nothing failed by rig error: no frozen-row leak, no chain
  failure, no gate failure, no cap hit (both sides recovered
  within cap: repair 10/200, retrain 6/400).

## Left for later (not done here)

- Flip-to-slow engagement during a rerun (declared NOT engaged
  here; a later stage's question).
- A sparse/frozen-gradient implementation that would make the
  work saving a FLOP/wall-clock saving too.
- The demo and the decline rule (later stages).
