# Linear global exchange did not improve portable regression (2026-09-26)

The seed-42 node-query ELU+1 linear-attention replacement completed the frozen
FP32/BS128 40-epoch contract. Its local backbone, data, target transform and
optimizer exposure matched the immutable K1 reference; remote mechanism and
resume checks passed. Saved-artifact acceptance passed without local inference.
The exact mechanism was closed as `NEGATIVE_UNDER_CONTRACT`: no retraining,
extra seed, 500K training, full run, protected role, or automatic successor.

## Endpoints and trajectory

| Role | K1 MAE | Linear attention MAE | Candidate minus K1 |
|---|---:|---:|---:|
| Fixed100K internal dev, selected endpoint | 0.1413736343 | 0.1417725831 | +0.0003989488 eV |
| Fixed500K internal dev, frozen100K weights | 0.1412533075 | 0.1442067325 | +0.0029534250 eV |

Original-role paired row-bootstrap 95% interval was approximately
[-0.0005091, +0.0013641] eV; it did not establish significant inferiority or any
material improvement. The separately recomputed frozen500K interval was
[+0.0021409, +0.0037787] eV. These are row-level intervals, not seed uncertainty.
Both reused internal development roles had 50,000 rows. The second result was
**not** a model trained on 500K, an official-validation score, or a sealed test.
The prospective 0.003 eV material gate was unchanged and failed.

At matched epochs 9/19/29/39 (zero-based), candidate-minus-K1 dev deltas were
-0.0094303 / -0.0071071 / -0.0001262 / +0.0003989 eV. By epoch 39 the candidate's
normalized training MAE was 0.0741574 versus 0.0772631, while dev was slightly
worse. This supports faster early fitting without retained terminal
generalization, not a proven universal failure of linear attention. Training
metrics use normalized units and must not be directly subtracted from eV dev.

Post-hoc K1-error quintiles showed compensation: the candidate improved the
K1-hardest rows and damaged K1-easiest rows on both roles. Such target-derived
groups are subject to regression-to-the-mean/selection effects; they neither
prove chemical specialization nor authorize routing. No topology/chemical cause
was established. Parameter count fell by 76,608 (about 2.1%) to 3,582,209;
parameter reduction alone did not preserve accuracy.

The failure mode argues against promoting early-checkpoint gains. A later
research proposal would need evidence for a retained late-exposure benefit,
not another ungrounded width, seed or learning-rate retry. This decision did
not release such a proposal.

Full paired distributions, all 40 matched points, and exact source hashes:
[saved-prediction analysis](results/portability_analysis.json).

## Execution and evidence boundaries

Original Kaggle v1 training completed, but the sibling audit failed on missing
`CUBLAS_WORKSPACE_CONFIG`. That failure was preserved separately in
[its diagnosis](results/audit_failure_v1.md). The audit-only recovery reused the
unchanged scientific source and exact selected checkpoints; it did not train.
Original-role maximum prediction discrepancies were 7.63e-6 eV (K1) and
1.43e-6 eV (candidate), well inside the frozen reproduction bound. Forty hashed
5K chunks cover two models by two roles. Official validation/test-dev/challenge
were not read.

Training's native canonical trace retained every observed step, presentation,
metric and checkpoint hash byte-for-byte. Its observed allocated single-worker
device interval was 4,932.522319 seconds, including evaluation/checkpoint IO;
this is not an account-billed-hours claim. Setup, queue and failed-audit device
costs were not measured and were not invented.

The prospective audit RML action bound the original physical Kaggle v1 ID,
not the later recovery kernel. Therefore the original audit action was closed
as infrastructure failure, while the successful recovery was retained as a
separate `retrospective_partial` NO_TRAIN diagnostic, anchored to the earlier
recovery specification/receipt. It was not relabeled as a prospectively bound
training run or strict causal comparison. Training has its own strict/replay
qualification; NO_TRAIN diagnostics cannot be counted as training replay pairs.

Remote checkpoints, raw logs and prediction chunks remained in ignored platform
records. Compact manifests, evidence and decisions were retained in Git.
