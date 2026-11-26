# KT-LLM — Plan: an LLM built from KT neurons

STEALTH. Originated by Darren Hall, 2026-10-23 (the KT neuron: "in the math..
include a KATIE"). Nothing here leaves this machine. No drafts for publication,
no posts. Planning document only — not a build order.

## 0. The one-sentence version

Take a small, ordinary transformer and replace its layers with KT layers, so
every neuron's weights are beliefs in a K-record — born with a KID, seated by
consolidation, graded by the authority of what taught them, sealed in a hash
chain — and the model can discount, refuse on, quarantine, and surgically
repair its own knowledge.

## 1. The two hard problems (named first, on purpose)

**Scale.** A frontier model has hundreds of billions of weights. One belief row
per weight per update is storage-absurd and always will be. The design must
never pretend otherwise. Granularity decision below (Section 2) is what makes
the bench version real and keeps a scale path open.

**The clock.** Gradient descent touches every weight on every step. An
append-only record cannot tick per step. Solution — and this is the core
architectural move: **k advances at consolidation, not at the step.**
Live training steps are the fast trace (scratch, unrecorded, exactly like a
human's unconsolidated experience). At a consolidation boundary, the current
values are *seated* as beliefs: one interval closes, the next opens. The
record holds what the model believed between consolidations, at what
authority, taught by which sources. Fast trace forgets; slow trace is the
record. This is the tritemporal split, used as an engineering budget.

## 2. Granularity: the KID sits on the neuron, not the weight

- One KID per **neuron** (one output row of a weight matrix, plus its bias).
  The belief's value = the row itself (stored as a hash + a pointer to the
  tensor snapshot; the full row lives in periodic snapshots, Section 5).
- A 12-layer, 512-wide bench model has tens of thousands of neurons →
  tens of thousands of KIDs, each accumulating one belief interval per
  consolidation epoch (hundreds) → low millions of record rows. That is a
  Postgres-sized problem, and Postgres is the home stack.
- Scale path (later, unproven, labeled as such): KID per matrix *block*,
  Merkle root over all KID chains per snapshot. The bench proves the
  discipline; it does not prove frontier scale.
- (Darren, Oct 26: scale is in there -- the tick unit is mutable.
  The mechanism is scale-free because the tick can be resized to
  the deployment; k's granularity adapts, the law doesn't change.)

## 3. Anatomy of a KT layer

A drop-in wrapper around a standard linear layer:

- **Weights** — ordinary tensors during steps (fast trace).
- **Gain vector** — one scalar per neuron, folded in at inference:
  activation × authority-gain(neuron). Precomputed at each consolidation;
  hot path cost ≈ one multiply the layer was doing anyway. The deterministic
  floor adds no latency of its own (standing doctrine).
- **Scar offset** — one scalar per neuron, added to the bias. Nonzero only
  for neurons whose beliefs sit in an injured lineage; follows the hump
  (v1.8 landmarks) as consolidation heals it. Temperament, as a vector.
- **Record** (cold, beside the tensors) — per KID: belief intervals
  [k_start, k_end) with value-hash, seated authority, source pool (which
  sources taught this neuron during the interval, with weights), prev-hash
  chain. Close-is-the-only-mutation, same as the toy and the kernel.

## 4. Where authority comes from: the source pipeline

Training data stops being an anonymous pile. Every corpus shard is a named
source with a declared authority, e.g.:

- curated / textbook-grade: 0.8–0.9
- ordinary web text: 0.4–0.6
- synthetic (teacher-generated): the teacher's authority, docked
- user feedback / RLHF-style signals: varies, declared per program

During an interval, each neuron's pool accumulates its contributing sources
(batch → block attribution is approximate; label it approximate everywhere).
At consolidation, seated authority = a function of the pool: multi-source
agreement raises it, single-source caps it, a known-bad source in the pool
docks it. This is PJD at training scale: pool = evidence, the seating rule =
justification, the seated belief = deed.

Direction (Darren, Oct 26): pools are caller-declared for now. Eventually
the system is self-defined -- it derives its own sources and authorities
from measured outcomes, via the attention engine's feedback loop, rather
than taking declared pools as input.

Correction (Darren, Oct 26): the seating rule is NOT a formulation Darren
owes in advance. Its form is implementation-dependent -- a fixed policy,
if-then program logic, a feedback controller, a learned rule, any of a
million forms -- and the form follows the machine being built. The ARCHE
feedback-controller direction sets the trajectory; the detailed rule is
a build decision made as ARCHE takes shape. The placeholder policy holds
meanwhile, and nothing is blocked waiting on a formulation.

## 5. Snapshots

Per §5.10 discipline: at each consolidation, a snapshot stores the full tensor
state (ordinary checkpoint), and the record stores per-KID value-hashes of that
snapshot. Any past belief can be re-materialized exactly: snapshot + interval
lookup. Snapshots are derived data — the ledger is the truth; snapshots are
readings of it.

## 6. The injury playbook (what the receipts buy, operationally)

1. A source is exposed as poisoned (externally, or by the firewall pot below).
2. **Lineage query**: every belief interval whose pool contains that source,
   across all KIDs. In a vanilla model this query is impossible; here it is a
   WHERE clause. (Darren, Oct 26: interval-level lineage plus working
   repair cover what lineage is actually for. Per-row exactness is not
   a goal; the floor leg of the decline rule stays synthetic-only and
   that is acceptable.)
3. **Null-run**: those beliefs lose epistemic effect — their neurons' gains
   drop toward their pre-injury values (or to null-run behavior per the
   filed §5.8 pattern: recorded in full, sealed, zero epistemic effect).
4. **Flip to the slow trace** (Darren, Oct 26): the moment poison is
   found, the system stops running on the fast trace and operates from
   the slow trace — the record. Live learning halts; behavior runs from
   the last clean consolidated state and its seated beliefs. It does
   NOT stay on fast under a gain mask, because poison would keep
   accumulating in live tensors behind the mask. Fast resumes only
   after repair and release. The bench as of Oct 26 nulls gains but
   does not yet flip traces — that is a known gap, future work, not
   a claimed behavior.
5. **Firewall**: per-source damage pot with bleed (filed §5.8) — a source
   whose contributions keep arriving injured engages the threshold and its
   future writes run as null until it heals below the release line.
6. **Surgical repair**: re-consolidate ONLY the affected neurons from clean
   sources (targeted interval of fine-tuning with other weights frozen).
   Repair cost is proportional to the injury, not the model.
   (Darren, Oct 26: FLOP saving is NOT the goal -- the claim is
   rows/params changed, not backward compute. No sparse implementation
   is owed; the round-2 work-leg pass stands on optimizer writes.)
7. **The deficit remains**: repaired beliefs re-seat at docked authority
   (source authority minus the slow deficit). Behavior returns to baseline
   while seated authority measurably lags — the phenomenon, at model scale,
   as a designed and measured property rather than a surprise.
   (Darren, Oct 26: the absolute confidence bar at repair time is NOT
   the goal -- confidence lagging after repair, then healing as
   authority heals, IS the phenomenon. The trajectory is the claim;
   the absolute bar at repair time is retired as a meter.)
   (Darren, Oct 26: the rules of the tail can change depending on
   deployment -- the law gives the shape (deficit decays, scar heals
   on the hump curve); the deployment sets the tail's parameters
   (decay rate, release line, what counts as healed). The bench's
   tail numbers are one deployment's settings, not constants.)
   (Darren, Oct 26: healing is determined by manipulation of the
   tail -- whether a scar heals at all, and how fast, is set by
   manipulating the tail's parameters.)

## 7. Inference behavior (the advantages, as mechanisms)

- **Discounting**: low-authority paths contribute weakly (gain vector).
- **Abstention**: if the winning answer's support runs predominantly through
  beliefs under an authority floor, the model hedges or declines — the
  cannot-answer ethic, produced by the math rather than a system prompt.
- **Attribution**: any output can be decomposed into the beliefs that fired,
  their authorities, and their sources (receipt-grade explanation; audit use).
- **Silent marking** (Darren, Oct 26): unauthorized data access leaves
  architecture scars -- sealed record events on the access -- without
  ever alerting the attacker. The record is memory, not an alarm:
  the attacker keeps operating unaware they are marked, while
  lineage queries can later reveal the whole intrusion path. Tail
  rules (deployment-set) decide whether such scars heal.
- **Instant honeypots** (Darren, Oct 26): on silent marking, the
  marked actor is set to permanent zero trust -- the intrusion scar
  never heals -- and the architecture restructures for the next
  interaction: the actor's subsequent interactions are routed into
  a honeypot environment that records everything. Blocking would
  signal detection; re-routing preserves the asymmetry, so the
  attacker cannot tell the honeypot from the real system.
- **Fake data pools** (Darren, Oct 26): the honeypot is stocked
  with fake data pools -- plausible-looking data the marked actor
  can find, use, and steal. Everything taken is worthless; everything
  done is recorded. The pools keep the attacker engaged (an empty
  honeypot would give itself away), and anything exfiltrated stays
  traceable by its pool lineage if it surfaces elsewhere.
- **Morphic data integrity** (Darren, Oct 26): integrity rides the
  record, not the shape. Data can morph -- re-encoded, restructured,
  re-routed, faked -- and integrity holds, because identity rides
  the KID and lineage through every transformation. The system never
  judges real vs fake by shape; it judges by lineage.
- **Signal-path regulation** (Darren, Oct 26): the authority/trust signal
  follows the fast trace until it intersects the slow trace, then the
  slow takes over. Fast phase carries impact, slow phase carries
  influence -- one signal throughout, never merged: the handoff is
  seamless in value (the traces are equal at the intersection) and
  total in provenance (live reading becomes record). This switch is
  how the signal path is regulated -- which trace owns the signal
  decides what flows through it and with what weight. RESOLVED
  (Darren, Oct 26): this intersection handoff IS the same switch as
  the poison flip (section 6 step 4) -- one switch, not two. No
  separate live flip battle-test is owed; using the one-signal
  regulation covers the flip.

## 8. Bench build (when and if Darren says build)

- **Phase 0 — Bench**: nanoGPT-class small transformer in PyTorch
  (tens of millions of params), corpus small enough to train in minutes-to-
  hours on one GPU/CPU: a mixed corpus of named sources, including a planted
  "liar" source (confident, fluent, wrong on a controlled fact-set).
- **Phase 1 — KTLinear**: the wrapped layer + record store (SQLite or Postgres)
  + consolidation loop. Vanilla twin trained in lockstep (same seed, same
  data) as the control. Modules, no god files, per standing build doctrine.
- **Phase 2 — Consolidation + authority seating**: source pools, seating rule,
  gain/scar vectors, snapshot/restore, chain verification.
- **Phase 3 — The experiments** (pre-registered before running, his style):
  1. *Cost*: KT vs vanilla — quality delta and speed delta, reported honestly
     whatever they are.
  2. *Poisoning*: mid-training, the liar source teaches a false fact-set.
     Measure: vanilla absorbs it invisibly; KT — firewall engagement,
     lineage quarantine, surgical repair cost vs full retrain cost.
  3. *The law at scale*: after repair, behavior probes return to baseline
     while seated authority on the repaired lineage stays measurably below
     its pre-injury value, decaying on the consolidation timescale. This is
     the headline measurement.
  4. *Abstention*: probes whose only support is the poisoned lineage — KT
     hedges/declines; vanilla answers confidently and wrong.
  5. *C2C proving ground* (Darren, Oct 26): two-model cache transfer,
     staged both ways. Vanilla pair merges cache-style with no record --
     poison flows invisibly, no lineage, no repair. KT pair transfers
     records -- lineage query finds the poison, null-run, surgical
     repair, authority lag visible. This is the proving ground: the
     audit problem from the Cache-to-Cache thread, answered by
     demonstration, not argument.
- **Phase 4 — Rust question**: Python is the concept proof (standing engine
  doctrine). If the bench results hold, the record/consolidation floor is a
  Rust bottom-end candidate under the org rules (construction under
  iamvolshi until it graduates). No step here pre-commits that.

## 9. Known soft spots (honest list)

- Per-neuron source attribution from batches is approximate — gradients mix
  everything. The pool is evidence, not certainty; the paper-trail language
  must say so.
- The seating rule (pool → authority) is a design choice with governance
  weight: whoever grades the sources grades the model's trust. That is a
  PJD question, not a math question, and Darren owns it.
- Consolidation cadence trades record richness against training speed;
  the right interval is an empirical question the bench answers.
- Nothing here shows the discipline survives frontier scale. Bench first,
  claims sized to evidence — always.

## 10. Posture

Stealth (Darren, 2026-10-23 21:04 ET). This plan, the toy, and the
phenomenon talk stay in the dark: no publication draft, no posts, no
applications copy. The filed provisional (64/182,709) covers the record
machine as filed this morning; the KT-neuron mechanism is not in it, and
any filing decision is Darren's, with Greg, later.

## 11. Demo requirement (Darren, 2026-10-26)

When the KT-vs-normal differential is obvious, Darren wants a side-by-side
demo: a KT neuron/model next to a normal neuron/model, same inputs, same
injury, both behaviors visible. The KT side shows what the normal side
cannot: quarantine, surgical repair, authority visibly lagging after
behavior returns, and receipts for every belief. Gated on the differential
being obvious — not before. The vanilla twin harness from Phase 1 is the
rig the demo stands on. Proving ground (Darren, Oct 26): the C2C
cache-transfer audit problem — the demo stages a vanilla KV-style cache
merge against a KATIE-record transfer, same poison, both sides visible.
