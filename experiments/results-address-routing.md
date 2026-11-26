# Address routing test (KTN state as graph address)

5 addresses, 3 weight tables, input (1, 1). Routing: address index i -> table i mod 3.

| address | table | weights | output |
|---|---|---|---|
| 0.5 | A | [2.0, 1.0] | 0.952574 |
| 0.6 | B | [0.5, 1.0] | 0.817574 |
| 0.7 | C | [-1.0, 3.0] | 0.880797 |
| 0.8 | A | [2.0, 1.0] | 0.952574 |
| 0.9 | B | [0.5, 1.0] | 0.817574 |

## Checks
- different_addresses_different_outputs: PASS
- same_table_same_output: PASS
- addresses_0_and_3_share_table_A: PASS
- addresses_1_and_4_share_table_B: PASS
- weights_untouched: PASS

All pass: True

Scope: a routing toy. It shows the address mechanism in miniature. It does not build a full model or a digital mind.
