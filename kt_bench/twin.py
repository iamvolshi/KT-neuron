"""Vanilla twin harness: the lockstep control for later experiments.

build_pair constructs the KT bench model and the vanilla twin from the
same seed, so both start from identical weights: construction order and
init distributions are shared code, and KID generation consumes no RNG.
The harness proves init parity and forward parity; it does not train.
Full twin training and the Phase 3 experiments are NOT this job.
"""

import numpy as np

from .linear import KTLinear, Linear
from .transformer import GPT


def build_pair(config, seed):
    """(KT model, vanilla model) with identical initial weights."""
    kt = GPT(config, np.random.default_rng(seed), KTLinear)
    vn = GPT(config, np.random.default_rng(seed), Linear)
    return kt, vn


def deterministic_batches(seed, n_batches, batch_size, block_size, vocab_size):
    """Fixed token batches: the same data path for both models."""
    rng = np.random.default_rng(seed)
    return [rng.integers(0, vocab_size, size=(batch_size, block_size))
            for _ in range(n_batches)]


def check_init_parity(kt, vn):
    """True iff every shared tensor is exactly equal at init."""
    if not np.array_equal(kt.tok_emb, vn.tok_emb):
        return False
    if not np.array_equal(kt.pos_emb, vn.pos_emb):
        return False
    for (_n1, l1), (_n2, l2) in zip(kt.all_linears(), vn.all_linears()):
        if not np.array_equal(l1.W, l2.W) or not np.array_equal(l1.b, l2.b):
            return False
    return True


def check_forward_parity(kt, vn, idx):
    """True iff both models give exactly equal logits on idx.

    Holds while the KT model's gains are all 1 and scars all 0 -- the
    parity condition. After gains/scars move, the models SHOULD differ:
    that difference is the KT effect, not a bug.
    """
    return np.array_equal(kt.forward(idx), vn.forward(idx))
