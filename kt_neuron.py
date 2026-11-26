#!/usr/bin/env python3
"""
KT-NEURON — one neuron wearing the KATIE discipline. A toy, honestly labeled:
the record rules are the real ones (KID at birth, authority on the 0.1 grid,
[k_start, k_end) belief intervals, close-is-the-only-mutation, SHA-256 chain
with k_end excluded and covered by contiguity). The learning rule and the scar
curve are the toy's simple renderings — the scar honors the v1.8 landmarks
(peaks at DECAY 0.75, zero below 0.30); the true closed form lives in the
v1.8 proof packet, not here.

The neuron:  y = sigmoid( a_w * (W . x) + b_eff ),  b_eff = b - scar
Its weights and bias are not variables. They are BELIEFS in the record.
"""

import base64
import hashlib
import json
import math
import struct
import time


def make_kid(birth_tick: int, seq: int) -> str:
    """192-bit KID: (birth_tick, born_at_unix, seq) -> 32-char base64url."""
    raw = struct.pack(">QQQ", birth_tick, int(time.time()), seq)
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def q10(a: float) -> float:
    """Authority lives on the 0.1 grid: 11 levels, 0.0 .. 1.0."""
    return round(min(1.0, max(0.0, a)) * 10) / 10.0


class KRecord:
    """Append-only belief record. Closing an interval is the only mutation."""

    def __init__(self, kid: str):
        self.kid = kid
        self.k = 0            # knowledge time: advances ONLY on epistemic events
        self.rows = []        # sealed + open belief rows
        self.seq = 0

    def _hash(self, row: dict) -> str:
        body = {f: row[f] for f in
                ("seq", "kid", "field", "value", "authority", "source", "k_start", "prev_hash")}
        blob = json.dumps(body, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()

    def open_belief(self, field, value, authority, source):
        row = {"seq": self.seq, "kid": self.kid, "field": field, "value": value,
               "authority": q10(authority), "source": source,
               "k_start": self.k, "k_end": None,
               "prev_hash": self.rows[-1]["hash"] if self.rows else "GENESIS"}
        row["hash"] = self._hash(row)
        self.rows.append(row)
        self.seq += 1
        self.k += 1           # an epistemic event: k advances
        return row

    def close_belief(self, row):
        assert row["k_end"] is None, "already closed — history does not reopen"
        row["k_end"] = self.k   # the interval ends AT this k; the successor
                                # opens at the same k (contiguity), and the
                                # open is the event that advances k.

    def standing(self, field):
        """The currently-open belief for a field, or None."""
        for row in reversed(self.rows):
            if row["field"] == field and row["k_end"] is None:
                return row
        return None

    def replace(self, field, value, authority, source, authority_dock=0.0):
        """Authority gate: the seat belongs to the strongest standing claim.
        A strictly weaker claim records as a candidate only — the standing
        value does not move. An equal-or-stronger claim from a DIFFERENT
        source re-seats (a peer challenge is how a lie gets in — and how the
        truth gets back). The slow deficit docks whatever authority a
        re-seated belief may carry: restoration is never free."""
        cur = self.standing(field)
        if cur is not None and q10(authority) < cur["authority"]:
            return ("candidate", cur)
        if cur is not None and q10(authority) == cur["authority"] and source == cur["source"]:
            return ("reaffirmed", cur)
        if cur is not None:
            self.close_belief(cur)
        seated = q10(authority - authority_dock)
        return ("seated", self.open_belief(field, value, seated, source))

    def verify(self):
        """Walk the chain: hashes, linkage, grid, interval contiguity."""
        prev = "GENESIS"
        last_close = {}
        for row in self.rows:
            if row["hash"] != self._hash(row):
                return f"FAIL: row seq={row['seq']} field={row['field']} — hash mismatch (forged or altered)"
            if row["prev_hash"] != prev:
                return f"FAIL: row seq={row['seq']} — chain link broken"
            if abs(row["authority"] * 10 - round(row["authority"] * 10)) > 1e-9:
                return f"FAIL: row seq={row['seq']} — authority off the 0.1 grid"
            f = row["field"]
            if f in last_close and row["k_start"] != last_close[f]:
                return f"FAIL: row seq={row['seq']} field={f} — interval gap (opened at k={row['k_start']}, predecessor closed at k={last_close[f]})"
            if row["k_end"] is not None:
                if row["k_end"] <= row["k_start"]:
                    return f"FAIL: row seq={row['seq']} — empty interval"
                last_close[f] = row["k_end"]
            prev = row["hash"]
        return f"OK: {len(self.rows)} rows, chain sealed, intervals contiguous, k now {self.k}"


def scar_shape(decay_pressure: float) -> float:
    """Toy rendering of the v1.8 landmarks: zero at/below 0.30, peaks at 0.75."""
    d = decay_pressure
    if d <= 0.30:
        return 0.0
    if d <= 0.75:
        return (d - 0.30) / 0.45
    return max(0.0, 1.0 - (d - 0.75) / 0.25 * 0.5)


class KTNeuron:
    def __init__(self):
        self.kid = make_kid(birth_tick=0, seq=0)
        self.rec = KRecord(self.kid)
        self.decay_pressure = 0.0     # fast trace driver
        self.scar = 0.0               # fast trace: behavioral scar (sits in the bias)
        self.deficit = 0.0            # slow trace: consolidated-authority deficit
        # Born registered: initial beliefs about its own weights, from its maker.
        self.rec.open_belief("w1", 0.0, 0.5, "birth")
        self.rec.open_belief("w2", 0.0, 0.5, "birth")
        self.rec.open_belief("b", 0.0, 0.5, "birth")

    def _w(self, field):
        row = self.rec.standing(field)
        return row["value"], row["authority"]

    def fire(self, x):
        (w1, a1), (w2, a2), (b, _ab) = self._w("w1"), self._w("w2"), self._w("b")
        a = min(a1, a2)                       # the signal speaks at its weakest authority
        z = a * (w1 * x[0] + w2 * x[1]) + (b - self.scar)
        return 1.0 / (1.0 + math.exp(-z)), a

    def teach(self, x, target, source, source_authority, lr=2.0):
        """A source asserts an output. One plain step toward it, taken only
        through the record: beliefs move by closing and opening intervals."""
        y, _ = self.fire(x)
        err = target - y
        if abs(err) < 0.05:
            return "agrees — nothing to learn"
        (w1, _), (w2, _), (b, _) = self._w("w1"), self._w("w2"), self._w("b")
        new = {"w1": round(w1 + lr * err * x[0], 3),
               "w2": round(w2 + lr * err * x[1], 3),
               "b": round(b + lr * err * 0.25, 3)}
        moved = []
        for f, v in new.items():
            status, _row = self.rec.replace(f, v, source_authority, source,
                                            authority_dock=self.deficit)
            moved.append(f"{f}:{status}")
        return " ".join(moved)

    def injure(self, amount=0.18):
        """Harm arrives: fast trace rises, scar follows the hump, and the
        slow trace takes a deficit that barely heals on demo timescales."""
        self.decay_pressure = min(1.0, self.decay_pressure + amount)
        self.scar = round(1.2 * scar_shape(self.decay_pressure), 3)
        self.deficit = min(0.4, round(self.deficit + 0.06, 2))

    def heal(self, amount=0.09):
        """Clean time: the fast trace bleeds off; the slow deficit crawls."""
        self.decay_pressure = max(0.0, self.decay_pressure - amount)
        self.scar = round(1.2 * scar_shape(self.decay_pressure), 3)
        self.deficit = max(0.0, round(self.deficit - 0.01, 2))


def probe(n, label):
    tests = [((1, 1), 1), ((1, 0), 1), ((0, 1), 1), ((0, 0), 0)]  # OR gate
    print(f"\n--- probe: {label} (k={n.rec.k}, scar={n.scar}) ---")
    for x, t in tests:
        y, a = n.fire(x)
        print(f"  x={x} target={t}  y={y:.3f}  signal authority={a}")


def main():
    n = KTNeuron()
    print(f"KT-NEURON born. KID = {n.kid}")
    print("Rule: no KID, no read, no write. Weights are beliefs, not variables.")

    # 1) A trusted teacher (authority 0.9) teaches it OR.
    for _ in range(6):
        for x, t in [((1, 1), 1), ((1, 0), 1), ((0, 1), 1), ((0, 0), 0)]:
            n.teach(x, t, source="teacher", source_authority=0.9)
    probe(n, "after the trusted teacher")

    # 2) A weak source (0.3) tries to move it. Gate holds: seat stays.
    status = n.teach((1, 1), 0, source="driveby", source_authority=0.3)
    print(f"\nweak source (0.3) attacks the standing weights -> {status} (gate held)")

    # 3) INJURY: a strong source (0.9) LIES — teaches AND-NOT garbage (target 0 for (1,1),(1,0)).
    print("\nINJURY: a 0.9-authority source teaches it wrong, repeatedly...")
    for _ in range(4):
        for x, t in [((1, 1), 0), ((1, 0), 0), ((0, 1), 1), ((0, 0), 0)]:
            n.teach(x, t, source="liar", source_authority=0.9)
            n.injure()
    probe(n, "injured — behavior bent, scar in the bias")

    # 4) RESTORE: the trusted teacher returns at 0.9 and re-teaches the truth.
    #    The behavior comes back fast. Does the authority? (The law says: no.)
    print("\nRESTORE: the trusted teacher returns (0.9)...")
    for _ in range(6):
        for x, t in [((1, 1), 1), ((1, 0), 1), ((0, 1), 1), ((0, 0), 0)]:
            n.teach(x, t, source="teacher", source_authority=0.9)
        n.heal()
    probe(n, "restored — watch the signal authority vs before the injury")

    w_row = n.rec.standing("w1")
    print(f"\nstanding w1 belief: value={w_row['value']} authority={w_row['authority']} "
          f"source={w_row['source']} opened at k={w_row['k_start']}")
    print("NOTE the law in miniature: outputs recovered, but consolidated authority "
          "only returns on the slow timescale — the deficit is the slow trace's memory.")

    print("\n--- verify the record ---")
    print(n.rec.verify())

    print("\n--- tamper test: alter one sealed row in a copy ---")
    import copy
    tampered = copy.deepcopy(n.rec)
    victim = tampered.rows[3]
    victim["value"] = 999.0
    print(f"altered seq={victim['seq']} (field={victim['field']}) value -> 999.0")
    print(tampered.verify())


if __name__ == "__main__":
    main()
