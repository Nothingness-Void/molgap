# Modern2D CPU integration: implementation contract and reuse audit

Frozen on2026-10-16 JST before model implementation. User authority is CPU-only
PPGT-small/ESA-small integration under the supplied second-stage specification.
No real labels, scientific Spec, native job, training or comparison is released.
Implementation uses an isolated `codex/modern2d-cpu` worktree from server HEAD
`916e0d4c9732c92912c6aa328ac8b8cd867c8d5a`; fetched server remote was
`63481324aceb807a3f74a08ba2c62bd3e0dcf9ab`. Desktop changes were not imported.
The source checkout's user README/RML/test modifications remain untouched.

## Prerequisites

Cross-scale [CPU acceptance](CROSS_SCALE_V1_HARDENING_ACCEPTANCE.md) and IMS
[V2 data acceptance](PCQM_UNIFIED_TWO_P_ACCEPTANCE.md) exist separately.
V2 rule is `f1bce8ca0161193341abea04a8920aea425e364348c029b7bd1b219672d1b077`;
master acceptance SHA is
`673bcf10d10cb16d493c01221eac8edaf085918ebcaec77c26bdbdd5072ead65`.
Mirrors are NOT_PUBLISHED. Data acceptance is not an executable input profile.

## Reuse versus gap (M0)

| Capability | Reuse | Exact gap / boundary |
|---|---|---|
| Identity/catalog | Common Spec, experiment_catalog, experiment_execution | New family declarations require reviewed feature/profile identities |
| One-batch CPU | fast_sandbox.run_sandbox, synthetic_batch | Direct disposable model factories are already supported |
| Model ABI | shared_model_adapter.validate_model_output | Normalized FP32 Gap vector; no evaluator duplication |
| Source | source_dependencies, experiment_package, common IO/wire | Explicit model/feature closure; upstream trainer excluded |
| Shared optimization/recovery | FamilyTrainingSession/Core/Kernel/Loaders/Recipe | FamilyRecipe/MODES and family_model_ops are K1/GPTrans-specific |
| Generic graph trainer | graph_training_adapter, graph_screen_training | Hard V1 feature/role/recipe gate; not an indexed V2 or RRWP/mask profile |
| Input | pcqm_topology, accepted V2 member/chunk maps | No real indexed loader is released by this CPU task |
| Fidelity | model_fidelity and scoped reference tests | CPU math agreement is not native/scientific qualification |
| RML/logging/cost | existing owners | No training trajectory or fabricated native certificate |

The [feature/profile proposal](MODERN2D_FEATURE_ABI.md) precedes any formal
registration. The human separately approved CPU-only family/profile support;
no K1/GPTrans identity is borrowed to bypass the Session gate. Existing loops,
optimizers, samplers, resume and submission code are not replaced.

## Paper-to-implementation mapping (M0)

PPGT reference: [arXiv2504.12588v3](https://arxiv.org/html/2504.12588v3),
raw HTML SHA `50ae0c119a2c63610d9505343e575758cfd4f2e33010ed9e8e81f322be803009`.
No author repository/commit/license was verified in bounded paper/title searches:
**UPSTREAM_CODE_NOT_VERIFIED**. Independent equations tests support mechanics,
not full upstream parity or published benchmark replication.

| Source | Planned mechanism | Classification |
|---|---|---|
| Eq3 | AdaRMSNorm, alpha0/beta1, explicit stable FP32 RMS | Mathematical reference; epsilon convention explicit |
| Eq4/5 | key-norm bias, theta before softmax, phi after | Independent direct distance-based reference |
| Eq6 | RRWP I,W,...; zero-degree rows zero after W0 | Deterministic 2D input convention |
| Eq7/8/10 | pre-norm backbone, pair stem FFNs, final norms, real bonds/nonedge zero | Complete core; categorical OGB adaptation explicit |
| Eq9 / C.3 | SPE and log-degree/order, reciprocal-degree/order channels | Included; no RWSE substitution |
| C.2 | distinct stem/backbone initialization | Source-bound implementation choice |
| Table9 | 16x512, heads16, PE24, bases10, pair128, stem FFN8, expansion2 | Structural REFERENCE, not a budget-compliant candidate |

Table9's reported17.6M cannot yet be reconciled with independent standard dense
projections at the listed sizes. Actual counts override approximations; no
parameter sharing or deleted mechanisms will be invented to reproduce it.
Dropout/stochastic-depth and exact activation/epsilon choices stay explicit
configuration; unspecified choices retain fidelity PENDING.

ESA reference: [paper](https://www.nature.com/articles/s41467-025-60252-z),
[author repository](https://github.com/davidbuterez/edge-set-attention), commit
`861a0b12d445ae80cc69b3948af623b4331321d4`, MIT (David Buterez2024).
Downloaded source archive SHA
`7a71e323841cc72e84dd448e4fb32019239a9c8a86692d23d12f771ab09b3dcd`.
Reviewed files: esa/mha.py, masked_layers.py, mlp_utils.py, models.py and LN.

| Source | Planned mechanism | Classification |
|---|---|---|
| models.Estimator.forward | directed edge tokens, both endpoints plus real bond categories | OGB one-hot input adaptation; no optional RWSE/LapPE |
| mha.MAB/SAB/PMA | exact projection/residual-Mish and learnable seeds | Fixed-weight scoped CPU parity |
| masked_layers.*Complete | source LN positions, MLP and conditional residuals | Pre-LN reference; post/Admin path not silently substituted |
| ESA.forward | encoder global input residual, PMA, seed mean, graph projection/Mish | Retained; not replaced by mean edge pooling |
| Estimator unused objects | edge-path mlp_norm not executed; output_mlp overwritten by Linear | Dead objects omitted; same-weight parity does not certify constructor RNG or original inventory |
| upstream masks | float shared-endpoint and boolean-inverted paths differ; diagonal cleared | Known fidelity discrepancy, not silently repaired |
| requested safe masks | self-enabled, no cross-graph/padding attention, reject empty-edge graph | Explicit MolGap adaptation; whole-family fidelity stays PENDING |

CPU reference execution may substitute only mathematically equivalent attention,
SwiGLU and CUDA-to-CPU device plumbing. No Lightning/trainer/optimizer import,
GPU package installation or network access during model execution is allowed.
Component parity does not waive documented whole-model mask differences.

## Predeclared sizing (M3; before any development scores)

PPGT starts deep_narrow16x320/10heads and shallow_wide8x512/16heads, pair128,
PE24/bases10/stem8/expansion2/head3. If over10,493,634 parameters: deep keeps16
blocks and lowers width by32 (heads=width/32); shallow keeps512 and removes one
block per revision. All stem and attention mechanisms stay. Deep is the initial
main candidate; shallow is a structural reserve, not an extra training arm.

ESA starts edge_mixed M-S-M-S-M-P at192/8heads and edge_compact M-M-S-P at256/
8heads; LN pre,32 PMA seeds, gated two-layer MLP at2x hidden, graph projection256.
If over7,317,634: lower width by32 (heads8) without changing layer types/seeds.
Mixed is the initial main; compact is reserve. Combined main cap17,811,268.
Counts include all stem/head/norm/seed parameters. Native activation/throughput
remain unknown. Configurations are not selected using any real development MAE.

## Implementation delivery

The [structural audit](../../experiments/modern2d_model_integration/parameter_audit.json)
records all monotone revisions, exact per-layer counts, weight/buffer and
hypothetical EMA bytes, construction/inference timings and MAC methodology.
PPGT main16x256 has9,447,937 parameters; reserve4x512 has9,584,257.
ESA mixed has5,014,849; compact has5,940,225. Main combined14,462,786 is below
17,811,268. Reserves are not additional authorized arms.

Implementation/negative test outcomes, scoped official ESA parity and unresolved
PPGT fidelity live in the [acceptance record](MODERN2D_FIDELITY_ACCEPTANCE.md).
The [future contract draft](../../experiments/modern2d_model_integration/future_training_contract_draft.md)
is non-executable and grants no native/scientific authority.
