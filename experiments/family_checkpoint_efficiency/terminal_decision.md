# Terminal Decision - 2026-10-13

**NO_TRAIN**: accepted synthetic execution diagnostic and shared finite-check
optimization. Adopt the common checker for newly frozen source packages; no
model/registry promotion, quality conclusion or remote release. Historical and
running jobs retain their immutable payloads.

## Measured Result

RTX5060, Torch2.7.1+cu128, CPU threads2; five alternating original/batched repeats
after a warmup for each case. All24 cases completed. Both implementations save
the same payload to the same atomic Torch path; every saved file SHA256 matches.
Complete checkpoint tensors and Python/NumPy/Torch/CUDA RNG remain unchanged.

| GPU state / pending metrics | Check + atomic IO original | Batched | Median time reduction |
|---|---:|---:|---:|
| K1 /512 | 274.35ms | 215.86ms | 21.32% |
| K1 /3905 | 305.66ms | 259.18ms | 15.20% |
| GPTrans /512 | 212.10ms | 189.38ms | 10.71% |
| GPTrans /3905 | 338.31ms | 260.88ms | 22.89% |

At3905 pending metrics, isolated GPU nested validation falls from123.31 to74.88ms
for K1 (39.27%), and97.85 to69.36ms for GPTrans (29.11%). CPU nested validation
falls from59.43 to37.70ms for K1 and61.38 to42.12ms for GPTrans. Raw repeat ranges
are retained: publication timings vary and these medians are not confidence
intervals or a guaranteed lower bound on gains.

Measured worker wall44.585s, process CPU58.828s. GPU allocation wall44.580s
includes CPU cases/setup/IO because RNG capture initializes CUDA; it is not GPU
busy time. Imports, preparation and regression-test costs are outside this
worker measurement; their combined native allocation cost is not established.

## Disposition and Limits

Remove the avoidable per-tensor accelerator synchronization and CPU scalar
check kernels through the existing shared owner. Do not remove independent
writer validation, change sampling/metric reduction/checkpoint cadence or alter
model algebra. Original checks remain only as a frozen diagnostic oracle, not a
live execution option.

The synthetic payload uses actual model tensor shapes and simulated AdamW/EMA
state. It does not execute the complete training session or resume a trained
checkpoint. Publication timing excludes cursor construction and admission/trace
callbacks. No data role, model forward/backward, optimizer update or MAE is used.
No full500K, T4/A100/DCU or default Torch2.11 performance claim follows.

Route reviewed implementation and evidence to `molgap-desktop` under BRANCHES'
accepted-diagnostic/non-promotion rule. Model recommendations and all existing
remote jobs stay unchanged. No automatic successor, full-scale release or
training replay-readiness is granted.

See [raw observations](results/result.json), [analysis](analysis.json),
[acceptance](acceptance.json), [attribution](attribution.md),
[execution audit](audit.md) and [validation scope](validation.md).
