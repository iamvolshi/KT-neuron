"""A small nanoGPT-class transformer, numpy only.

The block and model are parameterized by linear class: pass KTLinear
for the KTN bench model, plain Linear for the vanilla twin. Every other
line of code is shared, so the twin is a true lockstep control -- same
shapes, same init distribution, same forward math -- differing only in
the linear layers.

Pre-norm blocks (LayerNorm -> attention -> residual, LayerNorm -> MLP
-> residual), causal multi-head self-attention, GELU MLP, final norm,
untied lm_head. No dropout: the bench must be deterministic.

perturb() is the fast trace: it moves the live tensors and touches no
record. Consolidation (kt_bench.consolidation) is the only thing that
writes beliefs.
"""

import math

import numpy as np

from .linear import KTLinear, Linear


def gelu(x):
    """Tanh-approximation GELU (numpy has no vectorized erf)."""
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) *
                                    (x + 0.044715 * x ** 3)))


def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


class LayerNorm:
    def __init__(self, dim):
        self.w = np.ones(dim)
        self.b = np.zeros(dim)

    def forward(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + 1e-5) * self.w + self.b

    def params(self):
        return [self.w, self.b]


class CausalSelfAttention:
    def __init__(self, n_embd, n_head, rng, linear_cls):
        assert n_embd % n_head == 0, "n_embd must split evenly over heads"
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = linear_cls(n_embd, 3 * n_embd, rng)
        self.out = linear_cls(n_embd, n_embd, rng)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv.forward(x).reshape(B, T, 3, self.n_head, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)  # (3, B, heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = q @ k.transpose(0, 1, 3, 2) / math.sqrt(self.head_dim)
        mask = np.tril(np.ones((T, T), dtype=bool))
        scores = np.where(mask, scores, -1e9)
        att = softmax(scores, axis=-1) @ v  # (B, heads, T, head_dim)
        att = att.transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.out.forward(att)

    def linears(self):
        return [self.qkv, self.out]


class MLP:
    def __init__(self, n_embd, rng, linear_cls):
        self.fc1 = linear_cls(n_embd, 4 * n_embd, rng)
        self.fc2 = linear_cls(4 * n_embd, n_embd, rng)

    def forward(self, x):
        return self.fc2.forward(gelu(self.fc1.forward(x)))

    def linears(self):
        return [self.fc1, self.fc2]


class Block:
    def __init__(self, n_embd, n_head, rng, linear_cls):
        self.ln1 = LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, rng, linear_cls)
        self.ln2 = LayerNorm(n_embd)
        self.mlp = MLP(n_embd, rng, linear_cls)

    def forward(self, x):
        x = x + self.attn.forward(self.ln1.forward(x))
        x = x + self.mlp.forward(self.ln2.forward(x))
        return x

    def linears(self):
        return self.attn.linears() + self.mlp.linears()


class GPTConfig:
    """Bench config: small enough to run on CPU in seconds."""

    def __init__(self, vocab_size=128, block_size=32, n_embd=64,
                 n_head=4, n_layer=2):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer


class GPT:
    """Token + position embeddings, N blocks, final norm, lm_head."""

    def __init__(self, config, rng, linear_cls):
        self.config = config
        self.linear_cls = linear_cls
        C = config.n_embd
        self.tok_emb = rng.normal(0.0, 0.02, size=(config.vocab_size, C))
        self.pos_emb = rng.normal(0.0, 0.02, size=(config.block_size, C))
        self.blocks = [Block(C, config.n_head, rng, linear_cls)
                       for _ in range(config.n_layer)]
        self.ln_f = LayerNorm(C)
        self.lm_head = linear_cls(C, config.vocab_size, rng)
        # Named linears, in construction order, for records/snapshots.
        self.named_linears = []
        for bi, block in enumerate(self.blocks):
            for li, layer in enumerate(block.linears()):
                layer.name = f"block{bi}.linear{li}"
                self.named_linears.append((layer.name, layer))
        self.lm_head.name = "lm_head"
        self.named_linears.append(("lm_head", self.lm_head))

    def forward(self, idx):
        """idx: (B, T) token ids -> (B, T, vocab) logits."""
        B, T = idx.shape
        x = self.tok_emb[idx] + self.pos_emb[:T]
        for block in self.blocks:
            x = block.forward(x)
        x = self.ln_f.forward(x)
        return self.lm_head.forward(x)

    def kt_linears(self):
        """Only the KT rows: (name, KTLinear) for records/snapshots."""
        return [(n, l) for n, l in self.named_linears
                if isinstance(l, KTLinear)]

    def all_linears(self):
        return list(self.named_linears)

    def perturb(self, rng, scale=0.01):
        """Fast trace: move live tensors, touch no record, advance no k."""
        for _name, layer in self.named_linears:
            layer.W += rng.normal(0.0, scale, size=layer.W.shape)
            layer.b += rng.normal(0.0, scale, size=layer.b.shape)

    def param_count(self):
        total = self.tok_emb.size + self.pos_emb.size
        for _name, layer in self.named_linears:
            total += layer.W.size + layer.b.size
        for block in self.blocks:
            total += block.ln1.w.size + block.ln1.b.size
            total += block.ln2.w.size + block.ln2.b.size
        total += self.ln_f.w.size + self.ln_f.b.size
        return int(total)
