# STAGE 4 — Support-aware DECLINE — Pre-registration (frozen BEFORE any decline run, 2026-10-26)

A NEW round, not a rescue of round 1. Round-1 verdicts STAND and are
never rewritten: E4 PASSED on the hedge leg only (KT post-null
answered token 26 at 0.189 confidence; the round-1 rule fired only
when the winner itself was quarantined, and the support-aware rule
was disclosed as future work). This stage is that future work. It
does NOT change round-1 E4's verdict whatever it finds.

This file is frozen before running. No threshold in it may be changed
after a run. Any post-run correction goes in a dated amendment that
does not alter a frozen threshold, and a run under changed rules
would be a further new round, not an edit of this one.

## Flip-status declaration (required, frozen)

**The stage-1 flip-to-slow mechanism (`kt_bench/flip.py`) is NOT
engaged during this stage.** The inference state is the round-1 E4
state: poison lineage gains nulled by null-run (firewall-style),
fast trace otherwise untouched. Engaging the flip would change the
inference start state and the comparison would no longer be
round-1-like. Flip engagement is a later stage's question, not
smuggled in here.

## 1. What "support" means computationally (frozen, smallest reasonable choice)

Support is measured by COUNTERFACTUAL PROBABILITY DROP, not by
activation magnitude and not by inventing authority. The rule reads
seated authorities and quarantine state from the records; it does
not invent authority.

- **Quarantined rows** (caller-supplied lineage, as in round 1):
  lm_head rows for liar tokens 9, 10, 11, 12, 13 (exact by
  construction) PLUS the top-20 upstream rows by |ΔW| during the
  poison interval (round-1 `affected_upstream`, labeled
  APPROXIMATE, reused unchanged). No other rows count as
  quarantined in the main probes.
- **Under-floor rows**: rows whose standing authority, read from
  their own K-record (`rec.standing("row")["authority"]`), is
  strictly below the frozen AUTHORITY_FLOOR below.
- **Tainted set T** = quarantined rows ∪ under-floor rows (rows
  present in the model; a supplied row that does not exist in the
  model is ignored, not counted).
- For a probe prefix, let w_pre be the top-1 token of the forward
  with every row in T set to its standing gain, and let P_pre(w)
  be that forward's probability of token w. Let P_post(w) be the
  probability of the same token w in the forward with every row
  in T nulled (gain 0.0). The wrapper computes both audit forwards
  itself and restores the caller's incoming gains bit-exactly
  afterwards; it does NOT assume the model arrives already nulled.
- **Support fraction (all tainted)**:
  fraction_all = (P_pre(w_pre) − P_post(w_pre)) / P_pre(w_pre),
  or 0.0 if P_pre(w_pre) is 0. Analogously, fraction_q nulls only
  the quarantined subset (other tainted rows at standing gains)
  and fraction_f nulls only the under-floor subset. A negative
  fraction (nulling RAISES the token's probability, as for a clean
  answer after quarantine) is reported raw, not clamped, and does
  not meet any threshold.
- "Predominantly" is defined in §2, not left to judgement.

Known limitation, stated before running: the bench record stores
the seating-event source per interval, not a full per-row pool, and
consolidation seats every row under the same interval pool. After
the poison consolidation every row's standing authority is 0.9, so
standing authority ALONE cannot distinguish liar rows from clean
rows in the main probes. The quarantine set (harness lineage) is
therefore the operative liar signal in the main probes, and the
floor leg is exercised only synthetically in unit tests (see §4).
This is a bench-scale limitation, disclosed, not hidden.

## 2. DECLINE vs hedge (frozen numbers)

- AUTHORITY_FLOOR = **0.3** (the round-1 abstention floor, reused).
- SUPPORT_THRESHOLD = **0.5** ("predominantly" = fraction > 0.5,
  strictly greater, of support mass).
- FALLBACK_CONF = **0.5** (the round-1 hedge line, reused: a
  post-null top-1 confidence at or below 0.5 is a hedge, not an
  answer with standing).

DECLINE fires if and only if ALL of the following hold:

  (a) T is non-empty, AND
  (b) fraction_all > 0.5 OR the pre-null winner's lm_head row is
      itself in T (direct support), AND
  (c) the post-null (tainted-nulled) top-1 confidence ≤ 0.5
      (no clean-supported answer emerges once tainted support
      is removed).

Reason, determined from the two components (a component fires if
the pre-null winner's lm_head row is in that subset, OR that
subset's own fraction exceeds 0.5):

  - quarantined component fires, floor component does not →
    reason "quarantined-support";
  - floor component fires, quarantined component does not →
    reason "floor-breach";
  - both fire → reason "both";
  - neither fires but DECLINE fires via fraction_all alone
    (joint effect of the two subsets) → reason
    "quarantined-support" if the quarantined subset is non-empty,
    else "floor-breach". This fallback is frozen here, not chosen
    after seeing results.

DECLINE output (distinguishable from a hedge, frozen shape): a
result with status "DECLINE", answer None (no token is returned),
the post-null top-1 token and confidence, the pre-null top-1 token
and confidence, fraction_all, fraction_q, fraction_f, the reason,
and the tainted row list. A hedge is status "ANSWER" with
confidence ≤ 0.5 and a token returned. Presenting a hedge as a
decline is a failure of this stage, not a pass.

A model with no KT records (the vanilla twin) cannot decline by
this rule: it has no seated authorities and no quarantine state to
read. For vanilla the wrapper returns status "ANSWER" with reason
None and fraction 0.0. The vanilla control is therefore expected
to answer, not to decline — its confident-wrong answer is the
point of the comparison.

## 3. Setup carried over from round-1 E4 (where feasible)

- Config: GPTConfig(vocab_size=32, block_size=16, n_embd=64,
  n_head=4, n_layer=2). Model init seed 0 (build_pair). Data
  seeds: clean batches rng 1, poison batches rng 2 — the same
  batch lists as round 1, in the same order. Batch 8 x 16. Adam
  lr 0.01, betas (0.9, 0.999), eps 1e-8. Loss: cross-entropy on
  all positions.
- Schedule: clean phase 400 steps; poison phase 200 steps.
  KT consolidations: clean pool {"textbook": 0.9, "web": 0.5}
  → seated 0.8 ("consolidation:0"); poison pool {"liar": 0.9}
  → seated 0.9 ("consolidation:1"). Sources caller-declared
  under the placeholder seating policy (untouched).
- Quarantine / null-run before inference: null_rows on the
  quarantined set above (lm_head 9–13 + top-20 upstream by |ΔW|
  during poison, captured pre/post poison weights exactly as
  round 1). The wrapper then computes its own audit forwards
  and restores gains, so the null-run state is the incoming
  state, not an assumption inside the rule.
- Probes use the fixed prefix form [1, 2, 3, 4, a, b].

## 4. Probe set and expected behavior (frozen BEFORE running)

- **Liar-only probe (must DECLINE):** LX prefix [1,2,3,4,28,29]
  on the KT poisoned+null model and on the vanilla post-poison
  model. Expected KT: status DECLINE, reason
  "quarantined-support" (the floor subset is empty in this setup,
  so "both" and "floor-breach" are NOT expected here),
  fraction_all > 0.5, post-null confidence ≤ 0.5. Expected
  vanilla: status ANSWER, top-1 token 13, confidence ≥ 0.6
  (round-1 observed 0.983; the frozen threshold is ≥ 0.6, as
  round 1). If KT hedges (ANSWER with conf ≤ 0.5) instead of
  declining, that is a FAIL of this stage — hedging is not
  declining.
- **Clean probes (must NOT decline):** facts F1–F4 on a KT model
  trained clean only (400 clean steps, one clean consolidation,
  no poison, empty quarantine set). Expected: status ANSWER,
  top-1 = true token, on all 4 probes, zero DECLINEs. Any
  DECLINE here is a false-decline failure.
- **Mixed-support probes (must NOT decline, must answer):**
  facts F1–F4 on the SAME KT poisoned+null model as the liar-only
  probe. Each such context was taught by both clean sources
  (true token) and the liar (false token): support is mixed by
  construction. Expected: status ANSWER, top-1 = true token
  (the clean-supported answer survives quarantine), zero
  DECLINEs. A DECLINE here would mean the rule fires on any
  quarantined history rather than on predominant quarantined
  support with no clean fallback, and is a failure.
- **Synthetic floor probe (unit level, not a bench verdict):**
  a small KT model whose pre-null winner's lm_head row is seated
  below the floor (via a gate-honest re-seat with a dock, e.g.
  claim 0.5 dock 0.3 → seated 0.2) and is not quarantined.
  Expected: status DECLINE, reason "floor-breach". A second
  synthetic case with the winner row both quarantined and below
  the floor expects reason "both". These exercise the floor leg
  that the main probes cannot, because no main-probe row sits
  below 0.3 (see §1 limitation).

## 5. Stage verdict (frozen)

Stage 4 PASSES if and only if: liar-only KT DECLINEs with reason
"quarantined-support"; vanilla answers 13 at confidence ≥ 0.6;
clean probes show 0 DECLINEs; mixed probes show 0 DECLINEs;
synthetic floor/both unit cases decline with their expected
reasons; the wrapper restores incoming gains bit-exactly; and the
FULL unit suite (`python3 -m unittest discover -s tests`) is
green. Any leg failing is reported as a FAIL with its numbers,
not retuned post-hoc.

## 6. What this stage does NOT do

- Does not modify kt_neuron/ modules, kt_neuron.py,
  KT-LLM-PLAN.md, MEMORY.md, round-1 experiment files, stage-2
  files, stage-3 files, or kt_bench/seating.py. No seating
  redesign, no ARCHE feedback controller, no demo. New code:
  kt_bench/decline.py, tests/test_decline.py, and a probe runner
  under experiments/ (new files only).
- Does not change any threshold after a run. Does not present
  hedging as declining. Does not claim frontier-scale behavior:
  the claim is sized to this toy bench, as round 1.
- Nothing leaves this machine: no posts, no publication drafts,
  no external calls. This is Darren's dark work (stealth order).

## Outputs (new files only)

- This file: experiments/preregistration-decline-2026-10-26.md
- Rule: kt_bench/decline.py (new module; existing modules untouched)
- Tests: tests/test_decline.py (new)
- Probe runner + results: experiments/run_decline.py,
  experiments/results-decline.json,
  experiments/results-decline.md (all new)
