"""Manual backprop + Adam for the numpy bench transformer.

The bench is forward-only, so training here means hand-derived
gradients for this exact architecture (embeddings, pre-norm blocks,
causal multi-head attention, GELU MLP, final norm, lm_head), with
KTLinear gain/scar treated as fixed constants during an interval
(gains are set only at consolidation, per the design).

This module is LABELED training: it is real gradient training of a
tiny model, verified by a numerical gradient check before any run.
It is not a claim about frontier training dynamics.

Forward cache replicates kt_bench.transformer math exactly; the
gradient check asserts forward parity with model.forward first.
"""

import math

import numpy as np

from kt_bench.transformer import GPT, GPTConfig, gelu, softmax

SQRT_2_PI = math.sqrt(2.0 / math.pi)


def gelu_prime(x):
    """Derivative of the tanh-approximation GELU."""
    u = SQRT_2_PI * (x + 0.044715 * x ** 3)
    t = np.tanh(u)
    du = SQRT_2_PI * (1.0 + 3.0 * 0.044715 * x ** 2)
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t ** 2) * du


def _gain_scar(layer):
    gain = getattr(layer, "gain", None)
    scar = getattr(layer, "scar", None)
    if gain is None:
        gain = np.ones(layer.out_features)
    if scar is None:
        scar = np.zeros(layer.out_features)
    return gain, scar


def lin_forward(layer, x):
    """y = (x @ W.T + b - scar) * gain. Returns (y, cache)."""
    gain, scar = _gain_scar(layer)
    z = x @ layer.W.T + layer.b - scar
    y = z * gain
    return y, {"x": x, "W": layer.W, "gain": gain, "z": z}


def lin_backward(cache, dy):
    """Returns (dx, dW, db) for the linear above."""
    x, W, gain = cache["x"], cache["W"], cache["gain"]
    dz = dy * gain
    dW = np.einsum("btj,bti->ji", dz, x)
    db = dz.sum(axis=(0, 1))
    dx = dz @ W
    return dx, dW, db


def ln_forward(ln, x):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    xhat = (x - mean) / np.sqrt(var + 1e-5)
    y = xhat * ln.w + ln.b
    return y, {"x": x, "mean": mean, "var": var, "xhat": xhat, "w": ln.w}


def ln_backward(cache, dy):
    xhat, var, w = cache["xhat"], cache["var"], cache["w"]
    inv = 1.0 / np.sqrt(var + 1e-5)
    dxhat = dy * w
    dx = inv * (dxhat - dxhat.mean(axis=-1, keepdims=True)
                - xhat * (dxhat * xhat).mean(axis=-1, keepdims=True))
    dw = (dy * xhat).sum(axis=(0, 1))
    db = dy.sum(axis=(0, 1))
    return dx, dw, db


def attn_forward(attn, x):
    B, T, C = x.shape
    H, D = attn.n_head, attn.head_dim
    qkv, qkv_cache = lin_forward(attn.qkv, x)
    qkv_r = qkv.reshape(B, T, 3, H, D).transpose(2, 0, 3, 1, 4)
    q, k, v = qkv_r[0], qkv_r[1], qkv_r[2]
    scores = q @ k.transpose(0, 1, 3, 2) / math.sqrt(D)
    mask = np.tril(np.ones((T, T), dtype=bool))
    scores_m = np.where(mask, scores, -1e9)
    P = softmax(scores_m, axis=-1)
    att = P @ v
    att_f = att.transpose(0, 2, 1, 3).reshape(B, T, C)
    y, out_cache = lin_forward(attn.out, att_f)
    cache = {"q": q, "k": k, "v": v, "P": P, "mask": mask,
             "qkv_cache": qkv_cache, "out_cache": out_cache,
             "H": H, "D": D}
    return y, cache


def attn_backward(cache, dy):
    q, k, v, P, mask = cache["q"], cache["k"], cache["v"], cache["P"], cache["mask"]
    D = cache["D"]
    d_att_f, dW_out, db_out = lin_backward(cache["out_cache"], dy)
    B, T, C = d_att_f.shape
    H = cache["H"]
    d_att = d_att_f.reshape(B, T, H, D).transpose(0, 2, 1, 3)
    dV = np.einsum("bhqk,bhqd->bhkd", P, d_att)
    dP = np.einsum("bhqd,bhkd->bhqk", d_att, v)
    dS = P * (dP - (dP * P).sum(axis=-1, keepdims=True))
    dS = dS * mask
    dq = np.einsum("bhqk,bhkd->bhqd", dS, k) / math.sqrt(D)
    dk = np.einsum("bhqk,bhqd->bhkd", dS, q) / math.sqrt(D)
    dqkv = np.stack([dq, dk, dV], axis=0)  # (3,B,H,T,D)
    dqkv = dqkv.transpose(1, 3, 0, 2, 4).reshape(B, T, 3 * H * D)
    dx, dW_qkv, db_qkv = lin_backward(cache["qkv_cache"], dqkv)
    grads = {"qkv.W": dW_qkv, "qkv.b": db_qkv, "out.W": dW_out, "out.b": db_out}
    return dx, grads


def mlp_forward(mlp, x):
    h1, fc1_cache = lin_forward(mlp.fc1, x)
    g = gelu(h1)
    y, fc2_cache = lin_forward(mlp.fc2, g)
    return y, {"h1": h1, "fc1_cache": fc1_cache, "fc2_cache": fc2_cache}


def mlp_backward(cache, dy):
    dx_g, dW2, db2 = lin_backward(cache["fc2_cache"], dy)
    dh1 = gelu_prime(cache["h1"]) * dx_g
    dx, dW1, db1 = lin_backward(cache["fc1_cache"], dh1)
    return dx, {"fc1.W": dW1, "fc1.b": db1, "fc2.W": dW2, "fc2.b": db2}


def forward_cache(model, idx):
    """Replicates GPT.forward with caches. Returns (logits, cache)."""
    B, T = idx.shape
    x = model.tok_emb[idx] + model.pos_emb[:T]
    blocks_cache = []
    for block in model.blocks:
        h1, ln1_cache = ln_forward(block.ln1, x)
        att, attn_cache = attn_forward(block.attn, h1)
        x = x + att
        h2, ln2_cache = ln_forward(block.ln2, x)
        mlp_out, mlp_cache = mlp_forward(block.mlp, h2)
        x = x + mlp_out
        blocks_cache.append({"ln1": ln1_cache, "attn": attn_cache,
                             "ln2": ln2_cache, "mlp": mlp_cache})
    xf, lnf_cache = ln_forward(model.ln_f, x)
    logits, lm_cache = lin_forward(model.lm_head, xf)
    cache = {"idx": idx, "blocks": blocks_cache, "lnf": lnf_cache,
             "lm": lm_cache}
    return logits, cache


def loss_and_grads(model, idx):
    """Cross-entropy loss (next-token, all positions) + grads dict.

    Grads are keyed by parameter name (see collect_params for order).
    """
    logits, cache = forward_cache(model, idx)
    B, T, V = logits.shape
    x = logits[:, :-1, :].reshape(-1, V)
    t = idx[:, 1:].reshape(-1)
    e = np.exp(x - x.max(axis=1, keepdims=True))
    p = e / e.sum(axis=1, keepdims=True)
    loss = float(-np.log(p[np.arange(t.size), t] + 1e-12).mean())
    dlogits = np.zeros_like(logits)
    d = (p - np.eye(V)[t]) / t.size
    dlogits[:, :-1, :] = d.reshape(B, T - 1, V)
    grads = {}
    dx, dW, db = lin_backward(cache["lm"], dlogits)
    grads["lm_head.W"], grads["lm_head.b"] = dW, db
    dx, dw, db = ln_backward(cache["lnf"], dx)
    grads["ln_f.w"], grads["ln_f.b"] = dw, db
    for bi in range(len(model.blocks) - 1, -1, -1):
        block = model.blocks[bi]
        bc = cache["blocks"][bi]
        d_mlp, mgrads = mlp_backward(bc["mlp"], dx)
        dx_ln2, dw, db = ln_backward(bc["ln2"], d_mlp)
        grads[f"block{bi}.ln2.w"], grads[f"block{bi}.ln2.b"] = dw, db
        dx = dx + dx_ln2
        d_attn, agrads = attn_backward(bc["attn"], dx)
        dx_ln1, dw, db = ln_backward(bc["ln1"], d_attn)
        grads[f"block{bi}.ln1.w"], grads[f"block{bi}.ln1.b"] = dw, db
        dx = dx + dx_ln1
        for k, v in agrads.items():
            grads[f"block{bi}.attn.{k}"] = v
        for k, v in mgrads.items():
            grads[f"block{bi}.mlp.{k}"] = v
    dtok = np.zeros_like(model.tok_emb)
    np.add.at(dtok, idx, dx)
    dpos = np.zeros_like(model.pos_emb)
    dpos[:T] += dx.sum(axis=0)
    grads["tok_emb"], grads["pos_emb"] = dtok, dpos
    return loss, grads


def collect_params(model):
    """(names, arrays) in a fixed order, matching loss_and_grads keys."""
    names, arrays = ["tok_emb", "pos_emb"], [model.tok_emb, model.pos_emb]
    for bi, block in enumerate(model.blocks):
        names += [f"block{bi}.ln1.w", f"block{bi}.ln1.b"]
        arrays += [block.ln1.w, block.ln1.b]
        for lname, layer in [("attn.qkv", block.attn.qkv),
                             ("attn.out", block.attn.out)]:
            names += [f"block{bi}.{lname}.W", f"block{bi}.{lname}.b"]
            arrays += [layer.W, layer.b]
        names += [f"block{bi}.ln2.w", f"block{bi}.ln2.b"]
        arrays += [block.ln2.w, block.ln2.b]
        for lname, layer in [("mlp.fc1", block.mlp.fc1),
                             ("mlp.fc2", block.mlp.fc2)]:
            names += [f"block{bi}.{lname}.W", f"block{bi}.{lname}.b"]
            arrays += [layer.W, layer.b]
    names += ["ln_f.w", "ln_f.b", "lm_head.W", "lm_head.b"]
    arrays += [model.ln_f.w, model.ln_f.b, model.lm_head.W, model.lm_head.b]
    return names, arrays


class Adam:
    def __init__(self, params, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
        self.params = list(params)
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0

    def step(self, grads, lr_scale=None):
        """grads: list aligned with params. Updates in place.

        lr_scale: optional list aligned with params; each entry a
        scalar or a same-shape array multiplying that param's step
        size. None = exactly the old behavior (all steps x1.0).
        Added for the v2 handoff (3-arm v2 preregistration): a latched
        row's step is scaled by its standing authority (the slow
        number). Gradients are NOT scaled here — Adam normalizes
        per-element gradients (mhat / sqrt(vhat)), so a per-row
        gradient scale cancels out of the update; damping that
        survives Adam must act on the step itself.
        """
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mhat = self.m[i] / (1 - self.b1 ** self.t)
            vhat = self.v[i] / (1 - self.b2 ** self.t)
            s = 1.0 if lr_scale is None else lr_scale[i]
            p[...] = p - self.lr * s * mhat / (np.sqrt(vhat) + self.eps)


def train_step(model, adam, names, idx):
    """One Adam step on a batch. Returns loss."""
    loss, grads = loss_and_grads(model, idx)
    adam.step([grads[n] for n in names])
    return loss


def gradient_check(seed=123):
    """Forward parity + numerical gradient check on a tiny config.

    Returns (forward_max_abs_diff, max_rel_err, n_params_checked).
    The check model uses KTLinear with gains 0.7 / scars 0.05 so the
    gain/scar paths are exercised, not bypassed.
    """
    from kt_bench.linear import KTLinear
    cfg = GPTConfig(vocab_size=8, block_size=4, n_embd=8, n_head=2, n_layer=1)
    rng = np.random.default_rng(seed)
    model = GPT(cfg, rng, KTLinear)
    for _n, layer in model.all_linears():
        layer.gain[:] = 0.7
        layer.scar[:] = 0.05
    idx = rng.integers(0, 8, size=(2, 4))
    logits_cache, _ = forward_cache(model, idx)
    logits_model = model.forward(idx)
    fwd_diff = float(np.abs(logits_cache - logits_model).max())
    names, params = collect_params(model)
    _loss, grads = loss_and_grads(model, idx)
    eps = 1e-5
    max_rel = 0.0
    max_abs = 0.0
    checked = 0
    gate_fail = False
    for name, p in zip(names, params):
        g = grads[name]
        for ix in np.ndindex(p.shape):
            old = p[ix]
            p[ix] = old + eps
            lp, _ = loss_and_grads(model, idx)
            p[ix] = old - eps
            lm, _ = loss_and_grads(model, idx)
            p[ix] = old
            num = (lp - lm) / (2 * eps)
            ana = g[ix]
            abs_diff = abs(ana - num)
            rel = abs_diff / max(1e-8, abs(ana) + abs(num))
            max_rel = max(max_rel, float(rel))
            max_abs = max(max_abs, float(abs_diff))
            # Per-entry gate (amendment A6): rel < 1e-4 OR abs < 1e-9.
            if rel >= 1e-4 and abs_diff >= 1e-9:
                gate_fail = True
            checked += 1
    gate_pass = (fwd_diff < 1e-12) and (not gate_fail)
    return fwd_diff, max_rel, max_abs, checked, gate_pass
