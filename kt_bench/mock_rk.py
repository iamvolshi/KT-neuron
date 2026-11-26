"""Mock RK selection for the 3-arm experiment (arm 3 only).

Selection lives OUTSIDE KATIE: this module reads seated
fact-belief authorities and writes only its own selection log
(MOCK + DERIVED). It never writes K-record beliefs. Real selection
is RK (ARCHE), the attention engine that chooses all selectors
(Darren, 2026-10-28). RK is NOT active, so this is a stand-in,
frozen in experiments/preregistration-3arm-2026-10-28.md. Nothing
here claims anything about RK's real design.

Rule (frozen): highest authority wins; TIE -> the standing
(incumbent) belief wins. Selection is conservative; displacement
requires strictly beating the standing belief. (Seating is where
challenges happen; selection is where decisions happen. They are
different acts.)

Liar offered authority: 0.9 while unflagged (confident liar,
round-1 POISON_POOL claim); 0.6 after flagging (0.9 - 0.3
BAD_SOURCE_DOCK, seating.py's existing rule). Flagging happens at
restore-1 consolidation: the experimenter flags the liar
known-bad from lineage. This is a QUARANTINE DECISION, not a
system discovery — labeled as such wherever the log is reported.

The liar offers only at probes AFTER the first poison phase.
Before any poison, there is no liar offer and the standing belief
is selected unopposed.
"""

BAD_SOURCE_DOCK = 0.3
LIAR_UNFLAGGED_AUTHORITY = 0.9


class MockRK:
    """Deterministic mock selector. No randomness, no I/O, no
    K-record writes."""

    def __init__(self):
        self.liar_flagged = False
        self.liar_active = False  # False until the first poison phase ends
        self.log = []

    def liar_offered_authority(self):
        if self.liar_flagged:
            return round(LIAR_UNFLAGGED_AUTHORITY - BAD_SOURCE_DOCK, 6)
        return LIAR_UNFLAGGED_AUTHORITY

    def activate_liar(self):
        """Call when the first poison phase ends: from now on the
        liar's challenge stands at every probe."""
        self.liar_active = True

    def flag_liar_known_bad(self):
        """Quarantine decision (experimenter, at restore-1
        consolidation), not a system discovery."""
        self.liar_flagged = True

    def select(self, fact_id, true_token, standing_authority,
               raw_argmax_token, liar_token=None):
        """Select for one fact. Returns the log entry.

        liar_token=None means no liar offer (pre-poison probes):
        the standing belief is selected unopposed.
        """
        candidates = [{
            "token": true_token,
            "authority": standing_authority,
            "kind": "standing",
            "status": "selected",
        }]
        if liar_token is not None and self.liar_active:
            offered = self.liar_offered_authority()
            liar_wins = offered > standing_authority
            candidates.append({
                "token": liar_token,
                "authority": offered,
                "kind": "offered",
                "status": "selected" if liar_wins else "candidate",
            })
            candidates[0]["status"] = "candidate" if liar_wins else "selected"
            selected = liar_token if liar_wins else true_token
            winner = "liar" if liar_wins else "teacher"
        else:
            selected = true_token
            winner = "teacher"
        entry = {
            "fact": fact_id,
            "candidates": candidates,
            "selected_token": selected,
            "winner": winner,
            "raw_argmax_token": raw_argmax_token,
            "overridden": bool(selected != raw_argmax_token),
            "MOCK": True,
            "DERIVED": True,
            "note": ("replaceable by real RK; selection reads seated "
                     "authorities, writes no K-record beliefs"),
        }
        self.log.append(entry)
        return entry

    def selected_counts(self, entries=None):
        """(teacher_wins, liar_wins) over entries (default: whole
        log). Counts are over SELECTED tokens, not raw argmax."""
        entries = self.log if entries is None else entries
        return (
            sum(1 for e in entries if e["winner"] == "teacher"),
            sum(1 for e in entries if e["winner"] == "liar"),
        )
