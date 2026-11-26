"""Generic seating policy -- PLACEHOLDER, not ARCHE's rule.

Darren, Oct 26: use something basic right now, a generic policy, so the
bench runs end to end before ARCHE's rule exists. This is that policy:
deliberately plain, straight from the plan's own sketch (multi-source
agreement raises, single-source caps, a known-bad source docks).

The real seating rule belongs to ARCHE, the attention engine, acting as
a feedback controller: it reads the record of what the machine believed
and did, measures outcomes against outside checks, and sets the next
round's trust weights. That is Darren's Oct 26 direction, and it is his
to formulate. When it arrives, this module is REPLACED -- not extended.

Pool shape: a dict with "sources" mapping source name -> declared
authority (0.0-1.0), and optional "bad" listing known-bad source names.
An empty or missing pool seats at birth authority 0.5, which keeps
bench consolidation working before any source attribution exists.
Per-row source attribution (which sources taught which rows) is later
work -- the pools here are caller-declared, not measured. Direction
(Darren, Oct 26): eventually the system is self-defined -- it derives
its own sources and authorities from measured outcomes, via ARCHE's
feedback loop, instead of taking caller-declared pools. Caller-declared
is the placeholder until then.
"""

from kt_neuron.record import q

AGREEMENT_BONUS = 0.1
BAD_SOURCE_DOCK = 0.3
AUTHORITY_CAP = 0.9
BIRTH_AUTHORITY = 0.5


def seat_authority(pool, grid: float = 0.1):
    """Generic policy: mean of declared source authorities.

    - No sources declared -> birth authority 0.5.
    - One source -> its own authority (no bonus: single-source caps).
    - Two or more distinct sources -> mean + 0.1 agreement bonus.
    - A known-bad source in the pool -> dock 0.3.
    - Hard cap 0.9, floor 0.0, quantized to the record's grid.
    """
    if not isinstance(pool, dict):
        pool = {}
    sources = pool.get("sources", {}) or {}
    bad = set(pool.get("bad", ()) or ())
    if not sources:
        return BIRTH_AUTHORITY
    vals = [float(a) for a in sources.values()]
    seated = sum(vals) / len(vals)
    if len(sources) >= 2:
        seated += AGREEMENT_BONUS
    if bad & set(sources):
        seated -= BAD_SOURCE_DOCK
    seated = min(seated, AUTHORITY_CAP)
    return q(seated, grid)
