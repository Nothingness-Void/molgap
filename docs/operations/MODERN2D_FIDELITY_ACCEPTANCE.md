# Modern2D CPU evidence and fidelity limits

This2026-10-16 record separates mechanical integration, implementation fidelity
and native release. The [machine report](../../experiments/modern2d_model_integration/integration_cpu_report.json)
owns actual test counts, source/package/checkpoint identities and limitations.
The [implementation matrix](MODERN2D_MODEL_INTEGRATION_V1.md) owns paper mapping;
the [feature ABI](MODERN2D_FEATURE_ABI.md) owns input/role boundaries.

## Independent anchors

| Model/configuration | Evidence | Scope/status |
|---|---|---|
| PPGT equations | arXiv2504.12588v3; independent distance-attention forward/gradient, RMS/RRWP/SPE assertions | Equation mechanics only; MODEL_FIDELITY_PENDING |
| PPGT paper-size16x512 | Actual constructor gives34,867,201 parameters versus Table9's17.6M | Unresolved source/config ambiguity; not a budget-compliant candidate |
| ESA author-reference | Commit861a0b12d445ae80cc69b3948af623b4331321d4; MIT raw source and Estimator.forward AST slice | MODEL_FIDELITY_ACCEPTED only for this exact pre-LN FP32 no-padding configuration |
| ESA-small safe self-mask / OGB categories | Same attention/MLP core; explicit input/self/padding adaptation | MODEL_FIDELITY_PENDING for the adapted whole family, not waived by reference parity |

ESA reference: M-M-S-P, hidden256, heads4, gated MLP512/2layers, seeds32,
graph512, pre-LN epsilon1e-8, no dropout, source float mask with diagonal cleared.
Widths/heads/MLP come from the README example; pre-LN is an author-supported
branch, not a claim to reproduce the README post/Admin configuration or a
paper-best PCQM recipe. Two synthetic directed chains have3/17nodes and4/32
edge tokens. The latter tests the author's conditional32-token PMA residual.

Reference code executes in an isolated AST namespace. Imports are removed;
CUDA device constants become CPU. Explicit SDP math substitutes xformers and
silu(gate)*value substitutes flash SwiGLU; candidate uses torch SDPA. Neither
MolGap's model code nor an invented clone stands in for author M/S/P/residuals.
No Lightning, dataset, optimizer or trainer is imported. Raw file hashes and
the upstream method AST are checked. This does not certify GPU kernels.

Token equality is exact. Largest per-layer absolute error was
1.4603137969970703e-6; final Gap-vector errors were1.1920928955078125e-7 and
8.940696716308594e-8. Tolerance was FP32 rtol2e-5/atol3e-6. The accepted scope
excludes author torch-bool inversion, post-LN/Admin, padding, BN, AMP and
published benchmark scores. The raw references are test-only, excluded from
both isolated model source archives; MIT notice stays beside them.
Same-weight parity does not certify the author's constructor RNG sequence or
unused Estimator parameters: its edge path never calls mlp_norm, and its first
output_mlp is overwritten. MolGap omits these dead objects; all reported
parameter counts are the local implementations' complete counts, not a claim
to match the Lightning wrapper's parameter inventory.

## Mechanics and negative cases

- The actual existing FamilyTrainingSession/Core, Kernel, AdamW, scheduler,
  PassBatchSampler and shared evaluator run generated lists only. Batch128,
  four steps, interruption afterstep1, atomic serialized resume and exact
  model/optimizer/scheduler/RNG/cursor/sample counts are checked. No copied loop.
- Read-only live evaluation preserves the resumed model/optimizer/RNG/cursor;
  eV normalization/inverse and ordered source checks remain shared-owner code.
- Models pass finite all-parameter backward, exact state reload, independent
  instances, batch/graph isolation, node/edge permutation and no-file/network/
  CUDA-forward guards. Replacing y/RWSE with NaNs does not affect model output.
- Feature metadata identity/source/order/schema/code/config/role tampering,
  unknown/3D fields, cross-graph edges, protected roles, real/packed datasets,
  accelerator devices, oversized profiles and remote/native entry are denied.
- ESA explicitly rejects empty-edge graphs; no virtual chemical bonds are
  inserted. Per-graph execution avoids padding entirely. PPGT supports isolates.
- Source discovery and the existing immutable source bundler verify six-file
  model closures and isolated Python imports. No upstream trainer is packaged.

## Remaining work, not performed

PPGT source/config reconciliation and independent benchmark reproduction;
ESA-small input/mask fidelity scope; real indexed V2 family loader and accepted
derived identities; authorized graph-size distributions and empty-edge count;
mirror publication; native BS128 memory/throughput/precision/resume certificate;
symmetric prospective tuning and compute approval. CPU timings on two tiny
graphs are not training cost or cross-scale performance evidence.

Native state: NATIVE_PREFLIGHT_NOT_RUN. Scientific state:
NOT_AUTHORIZED_FOR_SCIENTIFIC_TRAINING.100K diagnostics cannot automatically
veto500K. No STQS/uncalibrated early stop, official/Audit access, remote job,
scientific trajectory, MAE ranking or historical conclusion change occurred.

## Regression qualification

New Modern2D tests passed49/49 both in the isolated worktree and after integration
into the primary server checkout. Shared family core/recovery passed294/294.
Common/Cross-Scale/RML batch passed276 with3 historical-checkout hash failures;
primary RML then passed69/69. No receipt or user research edit was rewritten.
Registration/source/lifecycle/sandbox regression passed544, skipped4, failed1.

That remaining legacy K1 inventory assertion was independently reproduced at
the exact pre-task HEAD916e0d4c with baseline source/tests/schemas/policies:
gptrans_screen_workflow.py was present in an unselected legacy package. It is
not a Modern2D regression or a model negative. The legacy inventory was not
silently repaired; the two new isolated model archives both passed. Whole-repo
all-green status is therefore NOT claimed. The first baseline attempt's missing
schema fixture and the incomplete initial monolithic run are retained without
acceptance credit in the machine report. Native-dependent skips grant no gate.

Navigation/layout checks passed33 with2 initial live-document budget failures;
compacting the new delta restored the unchanged120-line cap and both checks
passed on recheck. The final audit covered930documents/3734pointers/28anchors
with zero issues. No old document limit or history was relaxed to get a pass.
