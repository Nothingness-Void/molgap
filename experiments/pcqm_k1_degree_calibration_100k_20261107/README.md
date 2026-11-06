# Degree gate versus unmodified Muon100K

A fresh same-job control isolates whether the accepted degree residual gate adds value to original K1 plus Muon. [Protocol](protocol.md) owns the frozen question and gate; [input](experiment.json) owns authority and runtime inputs; [status](STATUS.md) owns preparation state; [decision](decision.md) stays pending until acceptance.

The implementation is byte-reused from accepted EC owner `codex/exp/k1-ec-gate-100k-20261106`, commit `bf35618fe0191d9f75d190bb3ee96345d8e7f915`, `src/molgap/k1_ec_gate.py`. Only its degree addon is registered here. No EC arm is repeated and no shared trainer/verifier changes are made. This dedicated owner starts from desktop `eca56da4ca5e7a26040714252219fc11318a919f`.
