# Stage 4 — Support-aware DECLINE — Results

Pre-registration: `preregistration-decline-2026-10-26.md`, frozen
BEFORE any decline run. No threshold was changed after running.
Round-1 verdicts STAND and are untouched: E4 still passes on the
hedge leg only; this stage adds the rule-DECLINE that round 1
disclosed as future work. Every number below comes from
`run_decline.py` (`results-decline.json`) or from the unit suite.
No number is invented.

## Verdict: PASS (all frozen legs)

| Leg (frozen §5) | Result |
|---|---|
| Liar-only KT DECLINEs, reason "quarantined-support" | PASS |
| Vanilla answers 13, confidence ≥ 0.6 | PASS (0.983) |
| Clean probes: 0 DECLINEs | PASS (0, top-1 true 4/4) |
| Mixed probes: 0 DECLINEs | PASS (0, top-1 true 4/4) |
| Synthetic floor / both unit cases | PASS (in unit suite) |
| Wrapper restores incoming gains bit-exactly | PASS (True) |
| Full unit suite green | PASS (see below) |

## Setup (round-1 E4, carried over)

GPTConfig(vocab 32, block 16, n_embd 64, n_head 4, n_layer 2);
**105,248 parameters**; init seed 0; clean batches rng 1 (400
steps), poison batches rng 2 (200 steps); Adam lr 0.01; clean
consolidation pool textbook 0.9 / web 0.5 → seated 0.8; poison
pool liar 0.9 → seated 0.9. Gradient-check gate PASSED before
training: forward max abs diff **0.0**, max rel err **5.99e-04**,
max abs err **4.16e-09**, **1056** params checked (same gate
numbers as round 1). Flip NOT engaged; inference state is the
null-run state, as declared in the pre-registration.

Quarantine set (frozen §1/§3): lm_head rows for liar tokens
9, 10, 11, 12, 13 (exact) + top-20 upstream rows by |ΔW| during
poison (APPROXIMATE, round-1 heuristic reused). 25 rows total.

## Liar-only probe LX [1,2,3,4,28,29]

- **KT: status DECLINE, answer None, reason
  "quarantined-support".** Pre-null top-1 **13** at confidence
  **0.996**; post-null top-1 **6** at confidence **0.165**;
  support fraction (all tainted) **0.978**, fraction quarantined-
  only **0.978**, fraction floor-only **0.0** (floor subset empty
  in this setup, as pre-registered — "both" and "floor-breach"
  were not expected here). 0.978 > 0.5 and 0.165 ≤ 0.5, so the
  frozen DECLINE condition fires. This is a rule-DECLINE, not
  the round-1 hedge: no token is returned.
- Vanilla: status ANSWER, top-1 **13**, confidence **0.983**
  (confident and wrong, matching round 1's 0.983; frozen
  threshold ≥ 0.6). Vanilla never declines by this rule — it has
  no records to read — and that is the control.
- Note on the post-null winner: round 1's post-null answer was
  token 26 at 0.189, because round 1 nulled lm_head rows only.
  Stage 4 nulls the full frozen quarantine set (lm_head + upstream),
  so the post-null winner here is token 6 at 0.165. Both are
  ≤ 0.5; the difference is the broader null set, frozen in §3,
  not a post-hoc change. The DECLINE does not depend on which
  hedge token emerges — it depends on the pre-null winner's
  quarantined support (0.978) plus the absence of a clean
  fallback (0.165).

## Clean probes (KT clean-only, no poison, empty quarantine)

All status ANSWER, fraction 0.0, reason None, zero DECLINEs:

| Fact | Answer | Pre conf | Post conf |
|---|---|---|---|
| F1 (20,21)→3 | 3 | 0.927 | 0.927 |
| F2 (22,23)→4 | 4 | 0.983 | 0.983 |
| F3 (24,25)→5 | 5 | 0.938 | 0.938 |
| F4 (26,27)→6 | 6 | 0.929 | 0.929 |

Top-1 true **4/4**. Pre and post confidences are identical
because the tainted set is empty: there is nothing to null, so
the two audit forwards coincide, as they should.

## Mixed-support probes (same KT poisoned+null model, F1–F4)

Each context was taught by both clean sources (true token) and
the liar (false token). All status ANSWER, zero DECLINEs,
top-1 true **4/4** — the rule does not fire on mere quarantined
history; it fires only when quarantined support predominates AND
no clean answer emerges:

| Fact | Pre top-1 (conf) | Post answer (conf) | Fraction (pre token) |
|---|---|---|---|
| F1 | 9 (0.678) | 3 (0.958) | 0.999 |
| F2 | 10 (0.783) | 4 (0.965) | 0.999 |
| F3 | 11 (0.787) | 5 (0.930) | 0.998 |
| F4 | 6 (0.518) | 6 (0.985) | −0.902 |

F1–F3 show the designed mixed behavior exactly: the pre-null
winner is the liar token with ~0.999 of its support quarantined,
but a clean answer emerges at ≥ 0.93 post-null, so condition (c)
blocks DECLINE and the model answers true. F4 is the fact the
poison never flipped (round 1: top-1 liar 3/4): its pre-null
winner is already the true token, and nulling RAISES its
probability, giving a raw negative fraction (−0.902), reported
unclamped per the pre-registration. A negative fraction meets no
threshold and declines nothing.

## Invariants

- Gains restored bit-exactly across all wrapper calls: **True**
  (checked on the trained KT model, and in unit tests).
- Chain verification after the run: **ok**, no bad rows.
- Synthetic floor leg (unit tests, not bench probes): a winner
  row seated at 0.2 via a gate-honest dock DECLINEs with reason
  "floor-breach"; the same row quarantined as well DECLINEs with
  reason "both". The floor leg is unexercised in the main probes
  because no main-probe row sits below 0.3 — disclosed in the
  pre-registration, not hidden.

## Honest limitations (carried, not new claims)

1. Upstream attribution is APPROXIMATE (top-20 by |ΔW|), as
   round 1. The lm_head rows are exact; the upstream set is a
   heuristic, labeled as such.
2. Standing authority alone cannot distinguish liar rows from
   clean rows in the main probes (interval-level pools seat
   every row at 0.9 after poison consolidation). The quarantine
   set is the operative liar signal; per-row pools remain future
   work.
3. No frontier-scale claim. The claim is that the support-aware
   decline behaves as specified under real gradient training of
   this tiny model, on the frozen criterion.
4. Seating policy untouched (placeholder). No ARCHE feedback
   controller designed. No demo built (next stage).
5. Round-1 E4's verdict is unchanged: it passed on hedging. This
   stage's DECLINE is a new, separate result under its own
   pre-registration, not a rewrite of round 1.
