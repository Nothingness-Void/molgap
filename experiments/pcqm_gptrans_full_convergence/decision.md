# Terminal decision: full GPTrans-T continuation

IMS job `1548270.ccpbs1` completed six frozen GPTrans-T continuation passes:
158,370 continuation steps and 20,271,360 additional sample presentations.
The selected EMA model reached **0.1037911756 eV** on the already consumed
official-validation role, improving the accepted source checkpoint
(`0.1090123132 eV`) by **0.0052211376 eV**. The sixth evaluation was best.
The stop reason was the six-pass budget with zero stale evaluations, so this
does **not** establish a validation plateau or convergence.

The complete output directory was retrieved on 2026-09-23. Every one of its
15 files matched its independently computed remote SHA256 and byte count.
The local no-inference acceptance rechecked checkpoint and best-bundle hashes,
trace length, prediction hashes, 73,545-row coverage, row/target alignment,
metrics, and the selected bundle identity. See
`results/terminal_reconciliation_20260923.json`.

This is mechanically accepted continuation evidence and a stronger frozen
full-role candidate for a separate comparison decision. Official validation
was used for convergence selection; it is not an untouched estimate.
Test-dev and challenge remain sealed. Any additional passes, official test,
or production promotion require their own contract and decision; the bounded
continuation is terminal.
