# Terminal decision: full K1 continuation

IMS job `1548227.ccpbs1` completed the frozen K1 continuation. Its completion
manifest and no-inference acceptance both report a patience stop after three
non-improving official-validation evaluations: 79,185 continuation steps and
10,135,680 additional sample presentations. The best remained the accepted
source checkpoint at **0.1066722353 eV**. No continuation model bundle was
selected.

The complete output directory was retrieved on 2026-09-23. Every one of its
11 files matched its independently computed remote SHA256 and byte count.
The local no-inference acceptance rechecked checkpoint hash, trace length,
prediction payload hashes, 73,545-row coverage, row/target alignment, metrics,
and stopping identity. See `results/terminal_reconciliation_20260923.json`.

The K1 convergence question is closed under its six-pass/patience-three
contract. The continuation gives no new model gain or full-scale promotion
claim. Official validation was used for the declared convergence selection;
test-dev and challenge remain sealed. No further training is released by this
decision.
