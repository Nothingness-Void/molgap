# Generalization-oriented follow-up research

Reviewed September 27, 2026, against server HEAD
`be13966c59b5ca602724ea83de7021478d7effe6`.

This is a literature-backed proposal, not a protocol, experiment result, RML
trajectory, compute release, or reopening of the three closed relation arms.
No training, inference, protected-role access or remote submission was performed.

## What changed the research priority

The [portability audit](../audit/decision.md) reversed all three original-role
gains with unchanged 100K checkpoints. The
[intervention diagnostic](decision.md) established dependence on the learned
branches, but neither attenuation nor targeted deletion repaired generalization.
Dependence within co-adapted weights is not a net benefit over the trained K1
reference. These observations do not uniquely diagnose oversmoothing,
under-training, coverage shift, or representation interference.

The resulting research question is whether local chemical information can be
preserved with stronger learning constraints or more structured propagation,
instead of adding another unrestricted pair/global return path. The two
directions below are hypotheses, not demonstrated remedies for scale erosion.
K1 remains a reusable screening reference; that does not overturn desktop's
full-scale EdgeState conclusion.

## 1. First information-gain priority: online local reconstruction

**Classification: training-objective experiment, not a new inference architecture.**

Keep the K1 encoder and direct Gap prediction, but retain Gap supervision on
every encoder pass while adding a training-only atom reconstruction head.
Corruption applies to observed training features, not to a new molecule with
an asserted unchanged quantum label. Inference uses clean accepted features
and discards the auxiliary head. No geometry, teacher or external molecules
are involved.

### Evidence and its limits

[Noisy Nodes, ICLR 2022, section 7.1 and Tables 7/15](https://arxiv.org/pdf/2106.07971)
tests categorical corruption plus reconstruction on coordinate-free PCQM4M.
Its 50-layer MPNN comparison improved from 0.1236 to 0.1218 MAE. This is the
older PCQM4M task and a much deeper model, not a MolGap performance forecast;
the paper explicitly notes sensitivity to noise and auxiliary-loss balance.

[GPS++, section 5.2 and Table 3](https://arxiv.org/html/2302.02947)
uses 1% categorical corruption with joint Gap/node/edge losses. Removing node
denoising costs 0.6 meV in its reported final-model ablation, while edge
denoising contributes only 0.1 meV. That ablation includes a different,
geometry-assisted system; it does not prove a 3-meV gain for our pure-2D K1.
It supports testing node supervision first, not stacking all auxiliary tasks.

The local [hierarchy-pretraining decision](../../pcqm_gap_architecture/results/local_hierarchy_pretraining_seed42/decision.md)
closed 20 reconstruction passes followed by 20 Gap passes: learning accelerated,
but the final result did not beat 40 Gap passes. Joint supervision is distinct
because no pass loses its Gap target. The old run used different data/batch
identities and supplies motivation, not a numerical comparator. The closed
[GPS++ directional adapters](../../pcqm_k1_gpspp_local_100k/decision.md)
did not test this training objective either.

### Smallest interpretable proposal

- Retain the accepted clean K1 artifact as the clean endpoint reference.
- Arm A: the unchanged encoder with categorical input corruption and Gap loss.
- Arm B: exactly the same corruption and Gap loss, plus local atom reconstruction.
- A versus clean K1 tests corruption; B versus A isolates the auxiliary objective;
  B versus clean K1 measures the total effect. A is a distinct intervention
  control, not a platform-specific repeat of the clean baseline.
- Match encoder initialization, row order, BS128, FP32, optimizer, schedule and
  supervised exposure. Isolate the augmentation/head RNG from loader/model RNG.
- Freeze corruption, CE reduction and weight in the actual target-normalized
  loss units before observing either candidate. Paper coefficients are not
  interchangeable across different normalizations. Do not add functional-group,
  edge, geometric, dropout or depth changes simultaneously.
- Inspect clean train/dev curves, reconstruction quality and predeclared
  chemical strata. Improving the auxiliary task without portable Gap improvement
  is a negative result for this exact recipe.

The two workers can each use one encoder pass per batch; the small auxiliary
head is removed at inference. Training overhead and extra memory are unmeasured,
not asserted to be zero. This is the lowest implementation/compute-risk proposal,
not the highest promised MAE improvement.

**Release blocker:** the existing causal-purpose allowlist in
`src/molgap/comparison_readiness.py` does not permit a `loss_identity`
intervention. Input corruption also needs an explicit training-transform
identity. Neither change may be hidden in architecture identity or labelled
strictly comparable by copying K1's contract. A reviewed objective-intervention
contract and complete release verification would be required before any run;
this review changed no validator or scientific policy.

## 2. Architecture lead: chemistry-separated local propagation

**Classification: research candidate requiring implementation audit.**

[MMGNN-2D, June 2026, sections 2.2/2.4/3/4](https://arxiv.org/html/2606.20906v1)
processes atom-type-pair subgraphs with shared message-passing weights and
combines their atom representations. The relevant hypothesis is delayed mixing
of chemically different local interactions. Its scaffold-split evidence covers
MoleculeNet classification and solubility-related regression, not PCQM or
quantum Gap. Neither the method nor its feature-attribution examples establish
that it will repair our low-conjugation regression.

This differs from the closed chemistry-conditioned PairToken: local propagation
is separated before aggregation, rather than using chemistry to select one
global relation token. It is not another molecular router or extra expert pool.

### Why not copy the released model blindly

The inspected official repository commit was
`78dcbff3a3d576506434241e4f26c70416453d9e`:

- [Subgraph construction](https://github.com/MathIntelligence/MMGNN/blob/78dcbff3a3d576506434241e4f26c70416453d9e/mmgnn/features/subgraph.py)
  filters edges by allowed atom-type pairs and appends a full-graph view.
  This is not simply the paper's induced two-type subgraph equation.
- [Encoder/readout](https://github.com/MathIntelligence/MMGNN/blob/78dcbff3a3d576506434241e4f26c70416453d9e/mmgnn/models/mpn.py)
  uses a sum-times-max message booster, an atom-sequence bidirectional GRU,
  and mean molecular pooling; the paper gives simpler propagation and sum
  molecular pooling. The sequence GRU raises a permutation-sensitivity question
  requiring verification. No execution-based failure is claimed here.

A faithful reproduction must resolve these differences. A MolGap adaptation
instead must be named as an adaptation and isolate delayed local mixing while
retaining the accepted atom/bond/RWSE inputs and existing global exchanges.
It must not bundle CMPNN, GRU, new readout and topology changes into one causal
claim. Shared weights do not imply shared compute: atom copies, duplicated
edge work and the full-graph view still cost memory and time.

Before freezing an experiment: specify the shared/local state boundary, preserve
every real bond, hash CPU-prepared index sidecars against the unchanged graph
cache, and check permutation equivariance, isolated atoms, disconnected graphs,
rare types and BS128 memory on the actual runtime. No wall-time promise or
parameter count is available yet. This lead is less ready than direction 1.

## Leads screened out of the immediate queue

- **I2-GNN / rooted subgraphs:** the
  [paper's Table 4 and Appendices I/J](https://arxiv.org/html/2210.13978)
  support stronger cycle counting, but QM9 Gap ties NGNN in that table.
  The synthetic counting runtime example is 20.62 ms versus 2.10 ms for MPNN;
  those are not MolGap timings. Greater expressivity is not enough to justify
  this cost before evidence that local structural collisions limit Gap accuracy.
  [GNN-AK SubgraphDrop](https://arxiv.org/html/2110.03753) reduces training
  overhead by sampling, but its full evaluation path still requires auditing.
- **Another normalization/initial-residual recipe:** the
  [ICLR 2025 residual/normalization analysis](https://proceedings.iclr.cc/paper_files/paper/2025/file/6c473e69ba261200dd595d07494c1a73-Paper-Conference.pdf)
  analyzes linearized propagation and offers useful diagnostics. It does not
  demonstrate oversmoothing in our nonlinear K1, and does not reopen the closed
  normalization family.
- **Receiver gates, more triplets/RRWP, extra slots, and branch-strength changes:**
  already covered by local negative/portability evidence; not new directions.

## Shared qualification for any later authorized experiment

Use immutable cross-platform data, seed42, BS128 and FP32. Select checkpoints
only using the declared original internal-dev role. Predeclare one terminal
inference on fixed500K internal dev using those same frozen weights and the
accepted reference predictions. This is a portability screen, not 500K training
or an untouched confirmation set; both development roles have been reused.

A favorable original-role scalar alone cannot release scaling. Freeze the
material-gain and subgroup noninferiority gates before results, require a
favorable portable endpoint, and keep any inconclusive result inconclusive.
No architecture guarantees preservation at 500K/full; later training-scale
validation remains separate. No automatic extra seeds or desktop handoff.

Any eventual run must retain prospective identities, all loss components,
augmentation/RNG state, optimizer-step/exposure trace, runtime, native costs,
role events, atomic checkpoints, aligned predictions and terminal decisions.
Reusable-reference qualification and replay readiness must pass honestly.
This literature note is deliberately not counted as another RML experiment.
