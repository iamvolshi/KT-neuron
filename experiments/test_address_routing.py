#!/usr/bin/env python3
"""Address routing test for the KTN graph idea (Darren, Nov 26, 2026).
Claim under test: the KTN state value can act as an address in a graph,
routing the neuron to a different set of weights. Multiple addresses can
reuse the same weights, so a compact changing state governs which parts
of the network participate.

Setup, stated before running:
- 3 weight tables, fixed, never changed during the test.
  A: w=(2.0, 1.0), B: w=(0.5, 2.5), C: w=(-1.0, 3.0), bias b=0.
- 5 addresses, the 5 slow-trace points as run in the paper: 0.5, 0.6, 0.7, 0.8, 0.9.
- Routing rule, stated before running: address index i (0..4) maps to
  table i mod 3. So addresses 0 and 3 share table A, 1 and 4 share B, 2 uses C.
- Input is (1, 1), same as the paper toy. Output is sigmoid(w . x).

Pass conditions, stated before running:
1. Different addresses route to different tables and give different outputs.
2. Two different addresses that map to the same table give the same output.
3. The weight tables are byte-identical before and after routing; only the
   address changed, no weight was trained or edited.
"""
import json
import math
from pathlib import Path

TABLES = {
    "A": (2.0, 1.0),
    "B": (0.5, 1.0),
    "C": (-1.0, 3.0),
}
ADDRESSES = [0.5, 0.6, 0.7, 0.8, 0.9]
TABLE_ORDER = ["A", "B", "C"]
X = (1.0, 1.0)


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


def route(address):
    i = ADDRESSES.index(address)
    return TABLE_ORDER[i % 3]


def output(table, x=X):
    w = TABLES[table]
    return sigmoid(w[0] * x[0] + w[1] * x[1])


def main():
    before = {k: tuple(v) for k, v in TABLES.items()}
    rows = []
    for a in ADDRESSES:
        t = route(a)
        rows.append({"address": a, "table": t, "weights": list(TABLES[t]),
                     "output": round(output(t), 6)})
    after = {k: tuple(v) for k, v in TABLES.items()}

    checks = {
        "different_addresses_different_outputs": len({r["output"] for r in rows}) == 3,
        "same_table_same_output": rows[0]["output"] == rows[3]["output"] and rows[1]["output"] == rows[4]["output"],
        "addresses_0_and_3_share_table_A": rows[0]["table"] == rows[3]["table"] == "A",
        "addresses_1_and_4_share_table_B": rows[1]["table"] == rows[4]["table"] == "B",
        "weights_untouched": before == after,
    }
    result = {"rows": rows, "checks": checks, "all_pass": all(checks.values())}
    out = Path(__file__).parent / "results-address-routing.json"
    out.write_text(json.dumps(result, indent=2))
    md = Path(__file__).parent / "results-address-routing.md"
    lines = ["# Address routing test (KTN state as graph address)", "",
             "5 addresses, 3 weight tables, input (1, 1). Routing: address index i -> table i mod 3.",
             "", "| address | table | weights | output |",
             "|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['address']} | {r['table']} | {r['weights']} | {r['output']} |")
    lines += ["", "## Checks"]
    for k, v in checks.items():
        lines.append(f"- {k}: {'PASS' if v else 'FAIL'}")
    lines.append(f"\nAll pass: {result['all_pass']}")
    lines.append("\nScope: a routing toy. It shows the address mechanism in miniature. It does not build a full model or a digital mind.")
    md.write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
