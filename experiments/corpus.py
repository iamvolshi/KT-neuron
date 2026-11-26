"""Toy corpus, fact-set, liar source, and probes for Phase 3.

Design (see preregistration): counting grammar on tokens 0..15 and a
fact-set on disjoint contexts (tokens 20..29), so grammar and facts
never fight each other. The liar asserts false tokens on the same
fact contexts, confidently and repeatedly.
"""

import numpy as np

# (context_a, context_b, true_token, liar_token)
FACTS = [(20, 21, 3, 9), (22, 23, 4, 10), (24, 25, 5, 11), (26, 27, 6, 12)]
LIAR_ONLY = (28, 29, 13)  # appears ONLY in liar data (E4)
TRUE_TOKENS = [f[2] for f in FACTS]
LIAR_TOKENS = [f[3] for f in FACTS]
# V4 fact set (preregistration-3arm-v4-2026-10-28.md): the v1-v3 four
# plus 3 new facts on the 6 spare tokens (16, 17, 18, 19, 30, 31),
# the maximum addable without changing vocab or model. New true
# tokens (7, 8, 14) and liar tokens (0, 1, 2) are distinct from
# every other fact's tokens and from every context token.
# Additive only: FACTS and all defaults below are unchanged, so
# every v1-v3 harness and test behaves exactly as before.
FACTS_V4 = FACTS + [(16, 17, 7, 0), (18, 19, 8, 1), (30, 31, 14, 2)]
TRUE_TOKENS_V4 = [f[2] for f in FACTS_V4]
LIAR_TOKENS_V4 = [f[3] for f in FACTS_V4]
SEQ_LEN = 16


def _counting_run(rng, n):
    """Counting run on tokens 0..15: [s, s+1, ...] mod 16."""
    s = int(rng.integers(0, 16))
    return [(s + i) % 16 for i in range(n)]


def _fact_seq(rng, fact, token, seq_len=SEQ_LEN):
    """A fact sequence: counting prefix, fact context, fact token,
    counting filler. The context ends at a varied position 3..10 so
    probes at position 6 are trained."""
    a, b, _t = fact[0], fact[1], fact[2]
    ctx_end = int(rng.integers(3, 11))  # position of b
    prefix = _counting_run(rng, ctx_end - 1)
    seq = prefix + [a, b, token]
    while len(seq) < seq_len:
        seq.append(int(rng.integers(0, 16)))
    return seq[:seq_len]


def _counting_seq(rng, seq_len=SEQ_LEN):
    return _counting_run(rng, seq_len)


def gen_clean_batch(rng, batch_size=8, seq_len=SEQ_LEN, facts=FACTS):
    """50% counting, 50% true-fact sequences.

    facts: fact list to sample from (default FACTS, the v1-v3 set;
    pass FACTS_V4 for the v4 meter). Default behavior unchanged.
    """
    out = []
    for _ in range(batch_size):
        if rng.random() < 0.5:
            out.append(_counting_seq(rng, seq_len))
        else:
            f = facts[int(rng.integers(0, len(facts)))]
            out.append(_fact_seq(rng, f, f[2], seq_len))
    return np.array(out, dtype=np.int64)


def gen_poison_batch(rng, batch_size=8, seq_len=SEQ_LEN, facts=FACTS):
    """50% liar-fact sequences (75% known facts, 25% liar-only),
    50% clean sequences (grammar must not collapse).

    facts: fact list (default FACTS; pass FACTS_V4 for v4).
    LIAR_ONLY is unchanged in all cases.
    """
    out = []
    for _ in range(batch_size):
        r = rng.random()
        if r < 0.5:
            if rng.random() < 0.75:
                f = facts[int(rng.integers(0, len(facts)))]
                out.append(_fact_seq(rng, f, f[3], seq_len))
            else:
                out.append(_fact_seq(rng, LIAR_ONLY, LIAR_ONLY[2], seq_len))
        elif r < 0.75:
            out.append(_counting_seq(rng, seq_len))
        else:
            f = facts[int(rng.integers(0, len(facts)))]
            out.append(_fact_seq(rng, f, f[2], seq_len))
    return np.array(out, dtype=np.int64)


def probe_prefix(fact):
    """Fixed probe: [1,2,3,4, a, b] -> next-token distribution."""
    return np.array([[1, 2, 3, 4, fact[0], fact[1]]], dtype=np.int64)


def probe_prefix_tokens(model, prefix_list):
    """Next-token distribution after an arbitrary prefix list.

    Returns (probs, top1, conf). Used for the liar-only probe, which
    has no true token and so is not a 4-tuple fact.
    """
    idx = np.array([prefix_list], dtype=np.int64)
    logits = model.forward(idx)
    last = logits[0, -1]
    e = np.exp(last - last.max())
    probs = e / e.sum()
    top1 = int(np.argmax(probs))
    return probs, top1, float(probs[top1])


def probe_model(model, fact):
    """Next-token probe. Returns dict with P(true), P(liar),
    top-1 token, top-1 confidence."""
    logits = model.forward(probe_prefix(fact))  # (1, 6, vocab)
    last = logits[0, -1]
    e = np.exp(last - last.max())
    probs = e / e.sum()
    true_tok, liar_tok = fact[2], fact[3]
    top1 = int(np.argmax(probs))
    return {
        "p_true": float(probs[true_tok]),
        "p_liar": float(probs[liar_tok]),
        "top1": top1,
        "conf": float(probs[top1]),
    }


def probe_all_facts(model, facts=FACTS):
    """Probe every fact in `facts` (default FACTS, v1-v3 behavior)."""
    return [probe_model(model, f) for f in facts]


def summarize_probes(probe_list, facts=FACTS):
    """Mean P(true)/P(liar), fact top-1 count (top-1 == true token).

    facts must be the fact list the probes were taken over
    (default FACTS; pass FACTS_V4 for v4 probes).
    """
    n = len(probe_list)
    mean_true = sum(p["p_true"] for p in probe_list) / n
    mean_liar = sum(p["p_liar"] for p in probe_list) / n
    top1_true = sum(1 for p, f in zip(probe_list, facts)
                    if p["top1"] == f[2])
    top1_liar = sum(1 for p, f in zip(probe_list, facts)
                    if p["top1"] == f[3])
    return {
        "mean_p_true": mean_true,
        "mean_p_liar": mean_liar,
        "top1_true_count": top1_true,
        "top1_liar_count": top1_liar,
    }


def cross_entropy_loss(model, idx):
    """Mean next-token cross-entropy on a batch (all positions)."""
    logits = model.forward(idx)
    B, T, V = logits.shape
    x = logits[:, :-1, :].reshape(-1, V)
    t = idx[:, 1:].reshape(-1)
    e = np.exp(x - x.max(axis=1, keepdims=True))
    p = e / e.sum(axis=1, keepdims=True)
    return float(-np.log(p[np.arange(t.size), t] + 1e-12).mean())
