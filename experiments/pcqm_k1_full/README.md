# Frozen K1 full-role execution

This directory owns the hardened execution contract for the unchanged
`neural_atom_k1` candidate. Historical scale evidence belongs to the server
branch at 36215d9, under experiments/pcqm_k1_scale500k/.

- Scientific and operational freeze: `protocol.md`
- Frozen machine contract: `training_contract.json`
- Pipeline-hardening decision: `decision.md`
- Audit issue-to-fix map: `audit_checklist.json`
- Remote entrypoint: `run.py`
- No-inference terminal acceptance: `accept_training.py`

Source packaging is provided by `package_source_dataset.py`. Authorized full
execution and official-role use are governed by
`../pcqm_k1_gptrans_full_fusion/protocol.md`; follow that experiment's README
for recovery and acceptance records.
