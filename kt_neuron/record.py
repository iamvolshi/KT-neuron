"""The append-only belief record (K-record) for one KT neuron.

Rules, carried over from the toy without change:

- k (knowledge time) advances ONLY on an epistemic event, which is an
  open_belief. Closing an interval does not advance k.
- Belief intervals are [k_start, k_end). Closing is the only mutation:
  a closed interval is never rewritten, never reopened. The successor
  opens at the same k its predecessor closed at (contiguity).
- Each row is sealed in a SHA-256 prev-hash chain. k_end is deliberately
  excluded from the hashed body; it is covered by the contiguity check
  instead (a forged k_end breaks contiguity or the next row's link).
- Authority lives on the 0.1 grid: 11 levels, 0.0 .. 1.0.
- Authority gate: the seat belongs to the strongest standing claim. A
  strictly weaker claim records as a candidate only and the standing
  value does not move. An equal claim from the same source reaffirms.
  An equal-or-stronger claim from a different source re-seats (a peer
  challenge is how a lie gets in -- and how the truth gets back).
"""

import hashlib
import json


def q10(a: float) -> float:
    """Authority lives on the 0.1 grid: 11 levels, 0.0 .. 1.0."""
    return round(min(1.0, max(0.0, a)) * 10) / 10.0


class KRecord:
    """Append-only belief record. Closing an interval is the only mutation."""

    def __init__(self, kid: str):
        if not kid:
            raise ValueError("no KID, no record: a KID is required at birth")
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
        assert row["k_end"] is None, "already closed -- history does not reopen"
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
        """Seat a new belief for a field under the authority gate.

        Returns (status, row): "candidate" (standing unchanged),
        "reaffirmed" (standing unchanged), or "seated" (new open row).
        A dock subtracts from whatever authority a re-seated belief may
        carry: restoration is never free.
        """
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
                return f"FAIL: row seq={row['seq']} field={row['field']} -- hash mismatch (forged or altered)"
            if row["prev_hash"] != prev:
                return f"FAIL: row seq={row['seq']} -- chain link broken"
            if abs(row["authority"] * 10 - round(row["authority"] * 10)) > 1e-9:
                return f"FAIL: row seq={row['seq']} -- authority off the 0.1 grid"
            f = row["field"]
            if f in last_close and row["k_start"] != last_close[f]:
                return f"FAIL: row seq={row['seq']} field={f} -- interval gap (opened at k={row['k_start']}, predecessor closed at k={last_close[f]})"
            if row["k_end"] is not None:
                if row["k_end"] <= row["k_start"]:
                    return f"FAIL: row seq={row['seq']} -- empty interval"
                last_close[f] = row["k_end"]
            prev = row["hash"]
        return f"OK: {len(self.rows)} rows, chain sealed, intervals contiguous, k now {self.k}"
