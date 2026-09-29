# Configurable GPTrans training objective

The shared implementation is `src/molgap/gptrans_objective.py`. The existing
GPTrans V4 optimizer-step and checkpoint writer accept its explicit adapter.
This is a library extension, not a replacement runner or a submission command.

| JSON parameter | Default | Meaning |
|---|---:|---|
| `auxiliary_hidden_dim` | 32 | Shared chemical head bottleneck width |
| `descriptor_weight` | 0.0 | Weight of mean normalized-descriptor MSE |
| `fingerprint_weight` | 0.0 | Weight of mean path-fingerprint BCE |
| `auxiliary_seed` | 42 | Independent head initialization seed |

Unknown keys, negative/nonfinite weights and invalid dimensions fail closed.
Zero weights disable auxiliary execution and head construction. Configs are
immutable and have a deterministic identity. Separate experiment contracts own
learning rate, batch, exposure and evaluation roles; this API does not alter
their frozen values.

```python
config = GPTransObjectiveConfig.from_dict(json.loads(config_path.read_text()))
model, optimizer, scheduler, ema = _make_training_state(
    initial_state_path, objective_config=config,
)
objective = model._training_objective
# The owner validates an immutable train-only cache and joins by source_index.
# batch.chemical_descriptors: [B, 200]; batch.chemical_fingerprint: [B, 512].
metrics = {}
gap_loss = _optimizer_step(model, optimizer, ema, batch, mean, std,
                          check_finite=True, objective=objective,
                          loss_metrics=metrics)
```

The return value remains normalized Gap L1. Auxiliary component losses and total
loss are detached fields in `metrics`; never log total loss as train Gap MAE.
The head is registered before optimizer/EMA creation, so state capture includes
it. Head initialization restores CPU RNG and does not seed CUDA. A temporary
readout hook captures the existing representation during a loss call and is
removed in `finally`, including on exceptions. Ordinary prediction has no hook
and does not execute the auxiliary head.

Pass `objective=objective` into the existing `_save_checkpoint` along with its
usual arguments. It records configuration, digest, full head state and distinct
loss fingerprint. Before restoring any candidate optimizer/model/EMA state,
call `validate_objective_checkpoint(checkpoint, config)` in addition to the
existing source, recipe and runtime checks. Legacy checkpoints only validate
with the exact default configuration. A candidate cannot resume through the
unchanged frozen-reference `run_training` entrypoint.

Export `export_gap_state_dict(selected_ema_state)` into a fresh original GPTrans
model with strict loading. Do not hand the training model or its unfiltered
state to the frozen production loader. Export equivalence requires GPU preflight.

The opt-in `training_addon=ChemicalTrainingAddon(config, cache)` argument now
threads through the existing `run_preflight` and `run_training` functions.
The thin `experiments/pcqm_gptrans_chemical_aux/run_arm.py` CLI exposes config,
source, dataset, accepted-cache hashes and preflight/train phases. The unchanged
call without an addon remains the frozen reference path. Candidate source/cache
identity is included in runtime, completion and checkpoint checks; exported best
weights exclude the auxiliary head while resumable checkpoints retain it.

`chemical_aux/1` is registered in ExperimentSpec with typed parameters and an
objective SHA equal to the configuration identity. Structural registry success
is not a frozen executable launch. Both arms still require their own Spec v2
prospective records, verified input bindings, exact source package and GPU
qualification before submission.

CPU cache preparation uses `build_labels.py` and the existing atomic writer and
hash helpers. Its JSONL input contains exactly source_index and smiles. The
separately pinned role JSON declares role=train, source_indices, rows_sha256,
dataset_identity and row_identity_semantics. For the V4 adapter, the latter two
are pcqm-fixed100k-v4 and pcqm-fixed100k-v4-source_idx. These are declarations,
not authentication: the input export must be matched to the authoritative
frozen row mapping before assigning them. Never manufacture a role file to
make an unrelated SMILES export pass. Cache failures preserve row reason codes;
no training cache is accepted when any row fails.

Local tests use synthetic SMILES and scalar/tensor loss fixtures, without model
execution. `tests/test_gptrans_objective_remote.py` is opt-in remote CUDA
verification for RNG, gradients, EMA restoration and exported prediction
equivalence; it is not a throughput certificate or a dataset acceptance test.
