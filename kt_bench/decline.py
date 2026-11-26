"""Support-aware DECLINE (plan §7, stage 4).

Round-1 E4's abstention rule fired only when the post-null winner
itself was quarantined or low-gain, so on the liar-only probe it
hedged (token 26 at 0.189) rather than declining. This rule closes
that gap: it declines when the answer the model WOULD have given
depended predominantly on tainted support AND no clean-supported
answer emerges once that support is removed.

Support, as pre-registered (experiments/preregistration-decline-
2026-10-26.md), is a counterfactual probability drop, not an
activation magnitude and not invented authority:

- Tainted set T = quarantined rows (caller-supplied lineage: the
  round-1 lm_head liar rows + top-20 upstream rows by |dW|,
  labeled APPROXIMATE) union rows whose standing authority, read
  from their own K-record, is strictly below AUTHORITY_FLOOR.
- w_pre = top-1 of the forward with T rows at standing gains;
  P_pre(w) = that forward's probability of w.
- P_post(w) = probability of w in the forward with T rows nulled
  (gain 0.0). fraction_all = (P_pre(w_pre) - P_post(w_pre)) /
  P_pre(w_pre), or 0.0 when P_pre(w_pre) is 0. fraction_q and
  fraction_f null only the quarantined / under-floor subsets.
- The wrapper computes both audit forwards itself and restores
  the caller's incoming gains bit-exactly afterwards; it does NOT
  assume the model arrives already nulled.

DECLINE (frozen criterion) iff T is non-empty AND (fraction_all >
SUPPORT_THRESHOLD OR the pre-null winner's lm_head row is itself
in T) AND the post-null top-1 confidence <= FALLBACK_CONF.
Reason: "quarantined-support" / "floor-breach" / "both", from
which component fires (winner row in the subset, or that subset's
own fraction > threshold); if neither fires but DECLINE fires via
fraction_all alone (joint effect), the reason falls back to
"quarantined-support" when the quarantined subset is non-empty,
else "floor-breach".

A DECLINE returns answer None: no token is returned. A hedge is
status "ANSWER" with confidence <= FALLBACK_CONF and a token
returned. The two are never conflated here.

A model with no KT records (vanilla twin) has nothing to read, so
it returns "ANSWER" with reason None and fraction 0.0; it never
declines by this rule. That is the control, not a gap to paper
over: vanilla answering confidently and wrong is the comparison.

Known bench limitation, stated in the pre-registration: after a
poison consolidation every row's standing authority is the same
interval value, so standing authority alone cannot distinguish
liar rows from clean rows in the main probes. The quarantine set
is the operative liar signal there; the floor leg is exercised
synthetically in tests. Seating policy untouched (placeholder).
No ARCHE feedback controller is designed here.
"""

from dataclasses import dataclass, field

import numpy as np

AUTHORITY_FLOOR = 0.3
SUPPORT_THRESHOLD = 0.5
FALLBACK_CONF = 0.5


@dataclass
class DeclineResult:
    """One inference decision. status is "DECLINE" or "ANSWER"."""

    status: str
    answer: object  # token int when ANSWER, None when DECLINE
    top1_post: int
    confidence_post: float
    top1_pre: int
    confidence_pre: float
    support_fraction: float  # fraction_all
    fraction_quarantined_only: float = 0.0
    fraction_floor_only: float = 0.0
    reason: object = None  # None, "quarantined-support", "floor-breach", "both"
    tainted_rows: list = field(default_factory=list)
    quarantined_rows: list = field(default_factory=list)
    floor_rows: list = field(default_factory=list)
    note: str = ""

    def to_dict(self):
        return {
            "status": self.status,
            "answer": self.answer,
            "top1_post": self.top1_post,
            "confidence_post": self.confidence_post,
            "top1_pre": self.top1_pre,
            "confidence_pre": self.confidence_pre,
            "support_fraction": self.support_fraction,
            "fraction_quarantined_only": self.fraction_quarantined_only,
            "fraction_floor_only": self.fraction_floor_only,
            "reason": self.reason,
            "tainted_rows": [list(r) for r in self.tainted_rows],
            "quarantined_rows": [list(r) for r in self.quarantined_rows],
            "floor_rows": [list(r) for r in self.floor_rows],
            "note": self.note,
        }


def _standing_authority(layer, row):
    rec = layer.records[row]
    standing = rec.standing("row")
    if standing is None:
        return None
    return float(standing["authority"])


def tainted_sets(model, quarantined_rows, floor=AUTHORITY_FLOOR):
    """Split the tainted set into quarantined / under-floor subsets.

    Returns (tainted, quarantined, floor_rows) as sorted lists of
    (layer_name, row). Only rows present in the model's KT layers
    count; supplied rows that do not exist are ignored.
    """
    kt_layers = dict(model.kt_linears()) if hasattr(model, "kt_linears") else {}
    if not kt_layers:
        return [], [], []
    wanted_q = set(quarantined_rows or ())
    quarantined, floor_rows = [], []
    for name, layer in kt_layers.items():
        for row in range(layer.out_features):
            key = (name, row)
            if key in wanted_q:
                quarantined.append(key)
            auth = _standing_authority(layer, row)
            if auth is not None and auth < floor:
                floor_rows.append(key)
    tainted = sorted(set(quarantined) | set(floor_rows))
    return tainted, sorted(quarantined), sorted(floor_rows)


def _probs(model, prefix):
    idx = np.asarray(prefix)
    if idx.ndim == 1:
        idx = idx.reshape(1, -1)
    logits = model.forward(idx)
    last = logits[0, -1]
    e = np.exp(last - last.max())
    probs = e / e.sum()
    top1 = int(np.argmax(probs))
    return probs, top1, float(probs[top1])


def infer_with_decline(model, prefix, quarantined_rows=(),
                       floor=AUTHORITY_FLOOR,
                       support_threshold=SUPPORT_THRESHOLD,
                       fallback_conf=FALLBACK_CONF):
    """Run one probe through the support-aware decline rule.

    quarantined_rows: iterable of (layer_name, row) in the
    quarantined lineage (round-1 set for the main probes).
    Returns a DeclineResult. Incoming gains are restored
    bit-exactly before returning, whatever the decision.
    """
    kt_layers = dict(model.kt_linears()) if hasattr(model, "kt_linears") else {}
    if not kt_layers:
        # Vanilla: no records to read, no quarantine state. Answer.
        probs, top1, conf = _probs(model, prefix)
        return DeclineResult(
            status="ANSWER", answer=top1, top1_post=top1,
            confidence_post=conf, top1_pre=top1, confidence_pre=conf,
            support_fraction=0.0, reason=None,
            note="no-records: vanilla model has no K-records to read")

    tainted, quarantined, floor_rows = tainted_sets(model, quarantined_rows, floor)
    q_set, f_set, t_set = set(quarantined), set(floor_rows), set(tainted)

    # Save incoming gains for every tainted row, then audit.
    saved = {}
    for name, row in tainted:
        saved[(name, row)] = float(kt_layers[name].gain[row])

    def standing_gain(name, row):
        auth = _standing_authority(kt_layers[name], row)
        return float(auth) if auth is not None else saved[(name, row)]

    def set_gains(nulled_subset):
        for name, row in tainted:
            if (name, row) in nulled_subset:
                kt_layers[name].gain[row] = 0.0
            else:
                kt_layers[name].gain[row] = standing_gain(name, row)

    try:
        # Pre-null forward: all tainted rows at standing gains.
        set_gains(set())
        probs_pre, top1_pre, conf_pre = _probs(model, prefix)
        p_pre = float(probs_pre[top1_pre])
        # Post-null forward: all tainted rows nulled.
        set_gains(t_set)
        probs_post, top1_post, conf_post = _probs(model, prefix)
        p_post = float(probs_post[top1_pre])

        def fraction_for(nulled_subset):
            if not nulled_subset or p_pre == 0.0:
                return 0.0
            set_gains(nulled_subset)
            probs_sub, _t, _c = _probs(model, prefix)
            return (p_pre - float(probs_sub[top1_pre])) / p_pre

        fraction_all = (p_pre - p_post) / p_pre if p_pre != 0.0 else 0.0
        fraction_q = fraction_for(q_set)
        fraction_f = fraction_for(f_set)
    finally:
        # Restore incoming gains bit-exactly, whatever happened.
        for (name, row), gain in saved.items():
            kt_layers[name].gain[row] = gain

    winner_row = ("lm_head", top1_pre)
    winner_tainted = winner_row in t_set
    winner_quarantined = winner_row in q_set
    winner_under_floor = winner_row in f_set
    q_fires = winner_quarantined or fraction_q > support_threshold
    f_fires = winner_under_floor or fraction_f > support_threshold

    decline = bool(tainted) and (
        fraction_all > support_threshold or winner_tainted
    ) and conf_post <= fallback_conf

    if not decline:
        reason = None
    elif q_fires and f_fires:
        reason = "both"
    elif q_fires:
        reason = "quarantined-support"
    elif f_fires:
        reason = "floor-breach"
    elif q_set:
        reason = "quarantined-support"  # joint-effect fallback, pre-registered
    else:
        reason = "floor-breach"

    return DeclineResult(
        status="DECLINE" if decline else "ANSWER",
        answer=None if decline else top1_post,
        top1_post=top1_post, confidence_post=conf_post,
        top1_pre=top1_pre, confidence_pre=conf_pre,
        support_fraction=float(fraction_all),
        fraction_quarantined_only=float(fraction_q),
        fraction_floor_only=float(fraction_f),
        reason=reason,
        tainted_rows=tainted, quarantined_rows=quarantined,
        floor_rows=floor_rows,
    )
