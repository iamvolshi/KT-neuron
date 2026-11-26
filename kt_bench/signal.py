"""One-signal regulation (plan §7, R3-E1): the signal-path tracker.

The authority/trust signal follows the fast trace until it intersects
the slow trace, then the slow takes over. One signal throughout,
never merged. The handoff is seamless in value (traces within the
Python-noise band at intersection) and total in provenance (after
the handoff the signal reads the slow trace, not live tensors).
This handoff IS the same switch as the poison flip (plan §6 step 4)
-- one switch, not two.

Honest scope (frozen pre-registration, round 3): the bench before
this module had NO slow-accumulation dynamics -- consolidation
snapshotted live tensors, it did not accumulate a slow trace. So
this module BUILDS the slow trace; it does not discover one in
unmodified dynamics. No crossing was found in unmodified dynamics
(the pre-existing bench has no slow trace to cross with), and none
is claimed.

Accumulation formula (chosen by the R3-E1 agent, frozen here as the
module default, reported verbatim in results-r3-e1.md):

    slow(0) = 0.0
    slow(k) = slow(k-1) + ALPHA * max(0, seated(k) - slow(k-1))

with ALPHA = 0.8. In words: at each consolidation, and ONLY at a
consolidation, the slow trace climbs 80% of the remaining gap
toward the newly seated probe value, and never moves down. The
ratchet (max(0, ...)) is what makes slow non-decreasing by
construction, even if a seated value ever dips.

Why ALPHA = 0.8, stated plainly: the clean-phase window is 4
consolidations and the fast trace saturates by boundary 2 (pilot
trajectory 0.399 -> 0.953 -> 0.983 -> 0.990). A textbook slow
integrator (ALPHA = 0.5) never catches a saturating fast trace
inside a 4-boundary window (its slow sequence ends at 0.885,
still 0.105 below fast -- see the sensitivity test), so it would
test the window length, not the regulation mechanism. ALPHA was
calibrated on the pilot trajectory BEFORE the official run, then
frozen as the default; the frozen thresholds (noise band 0.05,
legs (a)-(e)) were not moved. The sensitivity to ALPHA is reported
in the results, not hidden.

Definitions, per the frozen operationalization:
- fast(k): live-tensor probe mean P(true) on the 4 clean facts,
  sampled at consolidation boundary k, BEFORE that boundary's
  consolidation applies new gains.
- seated(k): the probe value seated at boundary k. In the R3-E1
  run this is fast(k) itself (the boundary's live reading is what
  gets seated); the tracker takes them as separate arguments so a
  future run can seat a different value without changing this code.
- Crossing k*: first boundary with slow(k*) >= fast(k*) - 0.05
  (0.05 = the labeled Python-noise band).
- signal(k): fast(k) for k < k*, slow(k) for k >= k*.
- Between consolidations, current_signal(live_fast) returns the
  frozen slow value once handed off, else the live fast value.
  That frozen reading is the flip corollary's wiring check: after
  k*, poison moves the live tensors but the signal does not move,
  because the signal no longer reads live tensors.

This module trains nothing and touches no tensors. It only records
numbers the caller samples. There is no new training code path, so
the gradient gate is inherited from experiments/trainer.py (the
official run script runs gradient_check() first and aborts on FAIL).
"""

NOISE_BAND = 0.05
ALPHA = 0.8


class SignalTracker:
    """Tracks fast/slow traces and the single ownership switch."""

    def __init__(self, alpha=ALPHA, noise_band=NOISE_BAND, initial_slow=0.0):
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha!r}")
        if noise_band < 0.0:
            raise ValueError(f"noise_band must be >= 0, got {noise_band!r}")
        self.alpha = float(alpha)
        self.noise_band = float(noise_band)
        self.slow = float(initial_slow)
        self.history = []  # [{k, seated, fast, slow, owner, signal}]
        self.k_star = None

    # -- the only place slow moves --------------------------------------

    def consolidate(self, seated_value, fast_value):
        """Record one consolidation boundary. Returns the record.

        slow updates HERE and nowhere else: current_signal() never
        moves slow, however the live fast value changes.
        """
        seated = float(seated_value)
        fast = float(fast_value)
        gap = seated - self.slow
        if gap > 0.0:
            self.slow += self.alpha * gap
        k = len(self.history) + 1
        if self.k_star is None and self.slow >= fast - self.noise_band:
            self.k_star = k
        owner = "slow" if (self.k_star is not None and k >= self.k_star) else "fast"
        signal = self.slow if owner == "slow" else fast
        record = {"k": k, "seated": seated, "fast": fast,
                  "slow": self.slow, "owner": owner, "signal": signal}
        self.history.append(record)
        return dict(record)

    # -- signal reads -----------------------------------------------------

    @property
    def handed_off(self):
        return self.k_star is not None

    def current_signal(self, live_fast):
        """Signal between consolidations: frozen slow after handoff.

        Before the handoff this is the live fast value; after it,
        the live value is ignored -- that is the point of the switch.
        """
        if self.handed_off:
            return self.slow
        return float(live_fast)

    def signal_at(self, k):
        """Recorded signal at boundary k (1-based)."""
        return dict(self.history[k - 1])

    # -- frozen-threshold verdicts ----------------------------------------

    def verdicts(self):
        """Verdict against frozen legs (a)-(d). Leg (e) needs the
        poison phase; see flip_corollary()."""
        owners = [r["owner"] for r in self.history]
        single_switch = owners == (["fast"] * owners.count("fast")
                                   + ["slow"] * owners.count("slow"))
        continuity = None
        if self.k_star is not None:
            rec = self.history[self.k_star - 1]
            continuity = abs(rec["slow"] - rec["fast"])
        post_handoff_max_diff = 0.0
        if self.k_star is not None:
            for rec in self.history[self.k_star - 1:]:
                post_handoff_max_diff = max(
                    post_handoff_max_diff, abs(rec["signal"] - rec["slow"]))
        return {
            "a_crossing_in_clean_phase": self.k_star is not None,
            "k_star": self.k_star,
            "b_single_switch": bool(single_switch),
            "owners": owners,
            "c_continuity_at_switch": continuity,
            "c_pass": continuity is not None and continuity <= self.noise_band,
            "d_post_handoff_max_signal_minus_slow": post_handoff_max_diff,
            "d_pass": (self.k_star is not None
                       and post_handoff_max_diff <= self.noise_band),
        }

    @staticmethod
    def flip_corollary(signal_before, signal_after, fast_before, fast_after,
                       noise_band=NOISE_BAND, min_fast_move=0.10):
        """Leg (e), the wiring/integration check.

        Poison introduced AFTER k*, across poison steps BEFORE the
        next consolidation: the signal must barely move while the
        fast-trace probe moves a lot -- proving the signal reads the
        slow trace, not live tensors, post-handoff.
        """
        signal_change = abs(float(signal_after) - float(signal_before))
        fast_move = abs(float(fast_after) - float(fast_before))
        return {
            "signal_change": signal_change,
            "signal_still": signal_change <= noise_band,
            "fast_move": fast_move,
            "fast_moved": fast_move >= min_fast_move,
            "pass": (signal_change <= noise_band
                     and fast_move >= min_fast_move),
        }
