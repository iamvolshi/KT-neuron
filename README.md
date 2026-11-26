# The Knowledge Time Neuron

By Darren Hall.

Today's neuron is a stateless weight. It computes, it updates, and it forgets what it was. The Knowledge Time Neuron, the KT neuron or KTN, holds two knowledge points of state inside itself: a fast trace, live and unrecorded, and a slow trace, the sealed append-only record of what it relied on. That state is recent memory of its previous state, not memory held indefinitely. The fast trace forgets; the slow trace is the record. The handoff between them, at their intersection, is the switch. That two-point mechanism is the novel claim.

## The paper

`ktn-paper-draft.md` (and `ktn-paper-draft.pdf`, 20 pages, 6 figures). Status: DRAFT. Not a publication. Not a post. First draft only, built from the author's own record.

The paper presents the neuron, the toy that ran on the night it was conceived, the design for a language model built from it, and the bench that tested the design under real gradient training at small scale. It shows where the mechanism works as specified and where it does not yet win. Those misses stand in the draft as measured.

The paper also sets out the scale logic. A single state gives a single weight. Two states double the weight ability in the same space. Each leg can point to at least 5 other memory points, each a different weight point on the authority param, and the combinations multiply it further, something like 35x. A 40B parameter model in the space of a 4B model is the same idea as an example. The numbers are examples. The logic is the point.

## What is in this repo

- `kt_neuron.py` — the toy. One neuron whose weights and bias are assertions in a K-record. Run it: `python3 kt_neuron.py`.
- `kt_neuron/` — the neuron layer package: KID, record, scar, neuron. 24 tests.
- `kt_bench/` — the bench: KTLinear, small transformer, RecordStore, SnapshotStore, Consolidator, vanilla twin harness.
- `experiments/` — pre-registered experiments and results: Phase 3 (cost, poisoning, the law at scale, abstention, C2C proving ground), Round 3 (signal regulation, tail, tick), and the October 28 three-arm comparison.
- `demo/` — the side-by-side demo: a KT model next to a normal model, same inputs, same injury. See `demo/README.md`.
- `tests/` — 157 test methods.
- `KT-LLM-PLAN.md` — the design record, written October 23, 2026, sealed October 24. It still carries its October 23 STEALTH header as the historical record.
- `KTN-SEAL-2026-10-24.md` — the seal record, hash and date only.
- `WEIGHT-SPACE-PROGRAMMING-RECEIPT-2026-11-05.md` — the priority receipt: MIT CSAIL published the same direction ten days after the author's provisional.
- `ktn-figures/` — the paper's six figures (dark panels, gold lettering) and the script that built them from the bench results.
- `build_ktn_pdf.py` — builds the paper PDF from the draft.

## Run the tests

From this directory:

    python3 -m unittest discover -s tests -v

## History

Conceived October 23, 2026. The toy ran the same night. The bench and the side-by-side demo were built October 26. Held dark under a stealth order from October 23. Opened November 26, 2026, on the author's order, after Stanford open-sourced research on the same direction (HumanLM). The filed provisional, 64/182,709, covers the record machine as filed October 23, 2026. The KT-neuron mechanism is not in it. Any filing decision is the author's, with counsel, later.
