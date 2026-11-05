# GPTrans propagation / molecular-query readout — four-arm 100K

Desktop-owned question on `codex/exp/gptrans-query-readout-100k-20261105`, based on
verified desktop `5e18aba3`. Frozen executable source: `b886a814`.

| Arm | Full propagation | Dynamic query layers | Parameters | Same-job comparison |
|---|---:|---:|---:|---|
| A `full12` | 12 | 0 | 5,871,201 | A versus D |
| B `full6` | 6 | 0 | 3,127,281 | B versus C |
| C `full6_query6` | 6 | 6 | 5,559,009 | B versus C |
| D `full8_query4` | 8 | 4 | 5,663,073 | A versus D |

Each slot owns its prospective records, frozen input, protocol and eventual
terminal acceptance. [AD](../pcqm_gptrans_query_readout_ad_100k_20261105/protocol.md)
tests quality/cost non-inferiority; [BC](../pcqm_gptrans_query_readout_bc_100k_20261105/protocol.md)
tests whether dynamic relational readout improves truncation. Cross-slot ordering
is descriptive. No model recommendation or replay-readiness claim precedes acceptance.

## Local model verification

`check_model.py` uses the actual family loader and existing optimizer/EMA/RNG
owners on synthetic categorical OGB graphs, without real dataset access.
It completed in 4.067 seconds on CPU, Torch2.7.1+cu126. All four share exact
frozen tensors for retained layers/head and original12-layer DropPath indices.
Finite forward/backward, graph isolation, padding and serialized next-step
model/AdamW/EMA/RNG equality passed. This is not full session/scheduler recovery
or native T4/Torch2.11 qualification; the existing remote barrier owns those.

C/D split-memory absolute gradient sums were1.84955/1.24570, with nonzero atom
memory contributions0.66667/0.36443. All query attention shapes were[2,8,1,6].
Double-precision frozen-memory full-matrix reference errors were at most1.78e-15.
Masked-key perturbations changed valid outputs by zero. Cross-graph perturbations
changed the other graph by zero; changed batch padding differed by <=3.58e-7.
The implementation computes one query row directly, without constructing late
full attention matrices. CPU single-batch times are smoke observations, not T4
speedup evidence. Detailed measurements are in [local_checks.json](local_checks.json).

Only one model module and its catalog registrations extend shared model code.
Family training/session, native qualification, source packaging, Kaggle submitter,
checkpoints, cost/role records, retrieval and strict RML closure are reused.
The input compiler requires explicit reference recipe/data declarations for the
new full6 control; `prepare_inputs.py` drafts those inputs, not a training framework.

Reproduction from this worktree with the primary project interpreter:

```powershell
& 'D:/文档/molgap/.venv/Scripts/python.exe' -c "import sys,runpy; sys.path.insert(0,r'D:/molgap-worktrees/gptrans-query-readout-100k-20261105/src'); sys.argv=['check_model.py','--output','D:/molgap-artifacts/reviews/query-readout-local-check-20261105.json']; runpy.run_path(r'experiments/pcqm_gptrans_query_readout_100k_20261105/check_model.py',run_name='__main__')"
```

## Terminal reconciliation — 2026-11-06

Both version1 kernels completed all60 passes. The existing strict finalizer
closed four trajectories with VERIFIED native qualification, VERIFIED replay,
no exclusions, and STRICT_CAUSAL candidate comparisons within each notebook.
The [AD decision](../pcqm_gptrans_query_readout_ad_100k_20261105/decision.md)
rejects quality non-inferiority; the
[BC decision](../pcqm_gptrans_query_readout_bc_100k_20261105/decision.md)
accepts useful query-readout computation against truncation. Neither adopts a
model or releases500K. The accepted full12 recommendation remains unchanged.
Paired curves, measured process-cost scopes and limitations remain with the
two attribution files beside their decisions.

Retrieval used only receipt-version-pinned manifests and strict dependencies,
not all outputs. Canonical trace per-pass wall fields are null, so mature-pass
timing remains unmeasured; that observed artifact gap was reported to the
infrastructure chat without modifying shared code. No persistent monitoring,
queue trigger or successor was created. Linked positive/negative results remain
on this owner branch pending an adoption/archive disposition.
