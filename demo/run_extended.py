"""Extended side-by-side demo (plan §11): KT vs normal, undeniable.

Runs the original E5 C2C scene plus the four extension scenes, in
order, in one console run. Every number comes from a live run of the
existing harnesses — nothing is replayed from a results JSON.

Scenes:
  0. E5 C2C transfer (demo/run_demo.py, untouched) — vanilla inherits
     poison exactly and silently; KT nulls, repairs, shows lineage.
  1. Decline/abstention pair (demo/scene_decline.py) — liar-only
     probe: vanilla answers 13 at 0.983, confident and wrong; KT
     DECLINEs (answer None, reason quarantined-support).
  2. Signal survival (demo/scene_signal.py) — poison after the
     fast->slow crossing: KT's signal stays at 0.982 while the fast
     trace collapses; vanilla's output moves with the poison.
  3. Tail healing (demo/scene_tail.py) — same injury, three tails:
     FAST heals at +5, SLOW at +25, PERMANENT never; vanilla has no
     healing dimension at all.
  4. Tick mutability (demo/scene_tick.py) — FINE/BASELINE/COARSE all
     heal in 25 decay ticks; vanilla has no tick to resize.

Run from ~/workspace/kt-neuron:

    python3 -m demo.run_extended

Expected wall clock is roughly the sum of the scenes (~20 minutes:
E5 ~1 min, decline ~3.5 min, signal ~3 min, tail ~6 min, tick ~6 min).
Each scene also runs standalone (python3 -m demo.scene_<name>).
Console output only; nothing leaves the machine.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demo import run_demo as scene_e5
from demo import scene_decline, scene_signal, scene_tail, scene_tick

SCENES = [
    ("0 — E5 C2C transfer (original scene, untouched)",
     scene_e5.run_demo),
    ("1 — Decline / abstention pair (E4 + decline rule)",
     scene_decline.run_scene),
    ("2 — Signal survival (R3-E1, one-signal regulation)",
     scene_signal.run_scene),
    ("3 — Tail healing (R3-E2, tail controls healing)",
     scene_tail.run_scene),
    ("4 — Tick mutability (R3-E3, shape holds across tick sizes)",
     scene_tick.run_scene),
]


def run_extended():
    """Run all five scenes in order; print timings; return results."""
    started = time.time()
    results = {}
    for title, fn in SCENES:
        print()
        print("#" * 78)
        print(f"# SCENE {title}")
        print("#" * 78)
        t0 = time.time()
        results[title] = fn()
        print(f"[scene wall clock: {time.time() - t0:.1f}s]", flush=True)
    print()
    print("=" * 78)
    print("EXTENDED DEMO COMPLETE — five scenes, same claim five ways:")
    print("vanilla has no record, so it has nothing to query, nothing")
    print("to quarantine, nothing to repair surgically, no signal to")
    print("hold, no scar to heal, and no tick to resize. KT does —")
    print("and each scene shows both sides on the same input/injury.")
    print(f"Total wall clock: {(time.time() - started) / 60:.1f} minutes.")
    print("=" * 78)
    return results


def main():
    run_extended()


if __name__ == "__main__":
    main()
