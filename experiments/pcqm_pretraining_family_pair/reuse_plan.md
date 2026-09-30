# Reuse and source provenance

Evidence/source snapshot: server `67a6237895ce94d502991d251f22654df3ab06c7`;
desktop `22af38b37cc60ec1e20ddc93eaff3e467b55007f`. No server merge is proposed.

| Need | Existing owner | Minimal missing adapter |
|---|---|---|
| Local atom/bond/group labels, reductions, masking | `qm9_local_hierarchy.pretrain_local_hierarchy`; atom-aligned fixed-cache sidecar | Provenance-preserving extraction of local reconstruction into a shared callable; dynamic widths and explicit family feature access |
| K1 factory | desktop `src/molgap/qm9_neural_atom.py`; family Spec adapter | Bind exact frozen K1 configuration; no redefinition of model |
| K1 supervised training and resume | `src/molgap/pcqm_k1_variants_runner.py`: train_arm | Reuse accepted `pcqm_k1_variants_runner.train_arm` from its original source commit, adding stage/resume hooks only |
| GPTrans joint factory/objective | `src/molgap/noisy_nodes.py`: make_noisy_nodes_pair_norm_model, _optimizer_step_noisy_nodes | Pretraining feature access and a separate stage; preserve the downstream joint recipe |
| Immutable config / source / local receipts | experiment_spec, experiment_prospective, experiment_package, experiment_launch | Register the reviewed stage/addon and bind actual assets |
| Terminal evidence | experiment_terminal plus research_memory terminal pipeline | Per-arm source/cache/reference/role/cost/trace bindings |
| Kaggle pair launch | Kaggle workload skill and `platforms/kaggle/push_kernel_with_accelerator.py` | Thin experiment bootstrap and per-GPU arguments only |

The newly parameterized chemical-aux wheel lives on
`codex/exp/gptrans-chemical-aux` at `70df7d18`. Its descriptor/fingerprint loss is
not local hierarchical pretraining. Reuse reviewed configuration/checkpoint
patterns only if needed; do not wholesale merge that experiment or replace the
requested pretraining objective with its graph-level targets.

The reused hierarchy loop has hardcoded legacy defaults and model-specific hooks;
its cosine horizon changes with epochs and its legacy checkpoints omit some
RNG/resume identity. It is a scientific implementation reference, not a
launch-ready template. Extend shared primitives and add static/synthetic tests;
model forward/backward and execution cost checks remain remote GPU preflight.

Required checks: local loss/shape/row/mask semantics; masked undirected bonds
share one decision; no padding/virtual-edge labels; stage transition preserves
encoder state and resets only declared state; resume reproduces processed batch
and mask stream; remove head/hooks without changing inference; actual parameter
counts and source inventories; CPU cache full coverage; GPU finite gradients,
memory, throughput and deterministic resume.

Authoritative evidence reviewed on server:
- experiments/pcqm_gap_architecture/results/local_hierarchy_pretraining_seed42/decision.md
- experiments/pcqm_gap_architecture/results/local_hierarchy_allocation10_30_seed42/decision.md
- experiments/pcqm_gap_architecture/results/local_hierarchy_allocation10_30_seed42/acceptance.json
- experiments/pcqm_k1_pair_token_100k/decision.md
- experiments/pcqm_k1_cross_scale_frozen/decision.md
- experiments/pcqm_k1_combined_simplification_100k/decision.md

Desktop evidence: experiments/pcqm_gptrans_noisy_pair_norm_500k/decision.md,
its training_contract.json, and pcqm_scale_transfer_reassessment/decision.md.
Do not inherit superseded representation-collapse claims from an old trajectory
when the later attribution explicitly marks that explanation unproven.
