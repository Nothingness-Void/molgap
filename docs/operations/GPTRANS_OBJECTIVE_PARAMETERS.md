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

The experiment-owned training addon still must bind label-cache identity and
row joins, pass this adapter through preflight/train/checkpoint/restore, retain
separate epoch metrics, and register its Spec addon. `run_training` remains the
frozen reference entrypoint; adding a config JSON alone does not make it an
auxiliary trainer. No new remote run or replay qualification is claimed here.

Local tests use synthetic SMILES and scalar/tensor loss fixtures, without model
execution. `tests/test_gptrans_objective_remote.py` is opt-in remote CUDA
verification for RNG, gradients, EMA restoration and exported prediction
equivalence; it is not a throughput certificate or a dataset acceptance test.
