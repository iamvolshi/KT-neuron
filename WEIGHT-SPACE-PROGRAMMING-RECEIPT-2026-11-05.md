# Receipt — Weight-Space Programming vs the KT neuron / KATIE

**Filed: 2026-11-05. Private receipt only. NOT a public claim. KT neuron stays dark per the 2026-10-23 stealth order.**

## Why this note exists

On 2026-11-05 Darren spotted MIT CSAIL's "Sparks of Weight-Space Programming" and read it as
taking KATIE / putting KATIE in the neuron. This note records the dates, what they published,
what predates it on Darren's side, and where the two actually overlap and where they do not —
so the record is sitting there if the self-healing machine ever needs defending.

## Their publication

- **"Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights"**
  Rikiya Takehi, Jayden Teoh, Alex Zhang, Omar Khattab (MIT CSAIL).
- Blog published **2026-11-02**: https://rikiyat.github.io/blog/weight-space-programming/
- X announcement by @rikiyatakehi, **2026-11-03** 4:51 AM, 1M views.
- Full experiment logs: https://github.com/RikiyaT/lm_understand-logs
- Edited checkpoint: https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming
- Not peer-reviewed. Blog post with published logs.

## What they did

- GPT-6 Astra (frontier model) was given Qwen3.5-0.8B's raw weights and asked to make it
  better at math, with no gradients and no training.
- Astra traced a reasoning-loop fault to a direction written by the MLPs of layers 15 and 19
  and subtracted that direction from 64 output columns in each: **131,072 weights, 0.016%**
  of the model.
- Reported results (same sampling settings): MATH-500 **7.9% → 48.1%**, GSM8K 13.8% → 67.1%,
  IFEval 21.0% → 62.3%. Answers hitting the token limit: **93.0% → 33.1%**.
- Independent rebuild by Ahsen Khaliq (Hugging Face) from their logs: **51.0%** on the full
  MATH-500 set.
- Astra also wrote small game-playing networks from scratch that outplayed PPO-trained ones,
  without environment feedback.

## What their own record says (limits)

- The authors' own caveat: "Editing weights could remove an obstacle that hid an ability,
  but it could not add an ability the model lacked." The gain came from removing a repetition
  loop that was hiding math ability already there — not from teaching it math.
- Astra failed on Gemma 3 270M, which did not have the same repetition problem: no loop to
  remove, no gain.
- Matthew Honnibal (spaCy creator) called the write-up "massively misleading" (2026-11-03).

## Darren's side — what predates it

- **2026-10-04:** KATIE named — the Knowledge Authority Time Indexing Engine: an engine that
  indexes knowledge-authority over time, keeping an append-only K-record of what was believed,
  when, and why.
- **2026-10-23:** KT neuron toy built the same night at `~/workspace/kt-neuron/kt_neuron.py`.
  Weights/bias are beliefs in a K-record (KID at birth, authority grid, [k_start,k_end)
  intervals, hash chain, scar in the bias, slow deficit on restored authority).
- **2026-10-23:** Provisional patent FILED — Application **#64/182,709**
  ("Tamper-Evident Append-Only Knowledge Record with Knowledge-Time Indexing"),
  inventor Darren Patrick Hall, $65 fee paid (micro entity).
  Non-provisional deadline: Oct 23, 2027.
- **2026-10-23:** STEALTH set — "no posters.. this is stealth as of now" — no public mention
  of the KT neuron mechanism in copy, posts, or applications. Patent filings excepted
  (provisional is confidential).
- **2026-10-24:** Hash-seal witness email to Bret (hash only, never mechanism) —
  `KTN-SEAL-2026-10-24.md` in this directory.
- Tritemporal field law v1.8, Zenodo https://zenodo.org/records/23198458 (predates all of the above).
- Self-healing machine doctrine (2026-10-04, claim strategy set with Greg Hunt 2026-10-08):
  the ARCHE orchestrator hooked to K-time, so if it senses a deficiency it builds the missing
  code piece itself and hooks it up. Patent strategy: claim the self-healing machine; the math
  and the clock stay public as defensive priority timestamps.

## Timeline

| Date | Event |
|---|---|
| 2026-10-04 | KATIE named; self-healing machine doctrine |
| 2026-10-23 | KT neuron toy built; provisional #64/182,709 filed |
| 2026-10-24 | Hash-seal witness email to Bret |
| 2026-11-02 | MIT CSAIL weight-space programming blog published |
| 2026-11-03 | Takehi's X post (1M views); Honnibal criticism |
| 2026-11-05 | Darren spotted it; this receipt filed |

**Darren's provisional predates their blog by 10 days. His K-time work on Zenodo predates both.**

## Where they overlap

- Weights treated as readable, diagnosable, patchable beliefs — a model inspecting raw weights,
  finding a fault, and fixing it directly without gradients or retraining.
- That is the shape of the KT neuron (weights/bias are beliefs) and of the self-healing machine
  (sense a deficiency, fix it).

## Where they do NOT overlap — what they did not take

- **No K-record.** No append-only history of what was believed, when, and why.
- **No KID.** Nothing is born registered; no identity bound to the record.
- **No authority/trust grid over time.** No trust-weighted belief, no authority collapse on
  a bad input, no scar in the bias, no slow deficit persisting after trust is restored.
- **No belief history.** They patched weights once and reported a score. There is no record
  of the edit as a belief event.
- Their edit removed an obstacle; it did not add knowledge. Their own words.

## Standing position

- No public theft accusation has been made, and none should be made on this record: a public
  claim would force disclosure of the KT neuron mechanism (breaking the 2026-10-23 stealth
  order) and would not land, because their published work contains no K-record.
- This paper helps Darren's position: MIT independently published the same direction ten days
  after his provisional. "MIT published the same direction ten days after my provisional" is
  the line, if the self-healing machine ever needs defending.
- Keep this note with the KTN seal. If the work ever comes out of stealth, this is one of
  the receipts.
