# EdgeState mechanism analysis

Review: 2026-10-01 JST. Scope: persistent real-bond EdgeState and its K1 descendants, pure 2D single-model research. This is a reading-based interpretation, not a canonical result, prospective trajectory, experiment launch, or promotion. No model, gradient probe, training or new prediction ran for this review.

## Source identity and authority

Current desktop source inspected: `316680b0050677bb7887d603887771d0de5a6b61`. Recent desktop commits integrate retained diagnostic and evidence infrastructure and retire archived refs; they do not demonstrate a newly stronger EdgeState architecture.

Use the [existing inventory](evidence_inventory.json) and [literature review](report.md) for the 126 distinct historical trajectory IDs and 15 full-text mechanism papers. Its September30 snapshots remain pinned; the 82 EdgeState-lineage/paired entries are a scope classification, not 82 independent accepted trials. Individual contracts and decisions own their scientific facts. Archive custody at `b1435ba939c39d88b6af55aca04c0b62f82314c3:docs/operations/BRANCH_ARCHIVE_20261001.md` preserves the pretraining tip `8821b5ce` and unreviewed supplement `9f767346`; administrative archival did not strengthen their scientific verdicts.

The newly accepted `pcqm_scale_fit_retained` diagnostic uses GPTrans/NoisyPairNorm factories, not EdgeState/K1. Its same-cohort train/development findings must not be imported as evidence about EdgeState fitting.

## Actual information flow

Owners: `src/molgap/gps.py:290,437`; `pcqm_gap_architecture.py:32`; `qm9_neural_atom.py:120,159,249`. Dependency details below were verified in local PyG 2.7.0. Historical frozen releases need their own dependency/source identity before these details can be attributed to a specific run.

Input consists of OGB atom9 and bond3 categorical fields plus RWSE16. RWSE contains node-wise random-walk return probabilities, not coordinates or arbitrary pair relations. A chemical bond becomes two directed edges with initially equal encoded categories. Later states can differ because source/target projections differ; the same update network is shared between directions within a layer.

For a directed bond u -> v, layer l computes:

```text
c = e_previous + S_l h_u_previous + T_l h_v_previous
e_new = LN_out(e_previous + MLP_l(LN_in(c)))
h_new = node_block(h_previous, e_new)
```

The MLP is Linear -> SiLU -> Dropout -> Linear. Updates use the prior edge, both endpoints and layer-specific parameters. Other neighboring bonds influence this state indirectly through earlier endpoint updates; base EdgeState has no explicit bond-neighbor/triplet operator.

**The final edge update is consumed by the final node block.** It has a computational path to the Gap loss. A claim that the last update is discarded before any node reads it is false for this source. Path existence does not measure actual gradient strength.

The local ResGated message has the schematic form:

```text
m(u -> v) = sigmoid(K[h_v;e] + Q[h_u;e]) * V[h_u;e]
z_v = W_skip h_v + sum_u m(u -> v) + b
```

The gate is a vector over hidden channels; edge content enters both gate and value. Default aggregation is addition, with no neighbor softmax or division by total gate/degree. Thus the precise open aggregation question concerns finite-width joint processing of different incoming messages, not whether edges can carry content at all. Later FFNs/layers already mix aggregate information nonlinearly; adding higher-order terms does not automatically confer a new universal expressivity class.

Dense EdgeState additionally performs all-node attention in each of nine GPS blocks. The local and global branches take the same pre-block node input; the current global attention branch does not directly consume current edge attributes. Subsequent layers can see edge information through updated nodes.

## K1 is an EdgeState descendant

`NeuralAtomEdgeStateGPS` inherits `OGBEdgeStateStructuralGPSWrapper`. It retains nine local blocks, h192/e64, rich categorical inputs and all persistent edge updates. It replaces every-layer dense global attention with one-slot exchanges after layers 3, 6 and 9. A slot is rebuilt from that layer's nodes and learned seed; it is not a persistent slot carried through all layers.

For a single graph, with normalized node states x_i:

```text
a_i = softmax_over_atoms(q(seed) dot k(x_i) / sqrt(64))
z = seed + sum_i a_i v(x_i)
z_tilde = slot_MHA / residual / LN / FFN transformation of z
u = return_projection(z_tilde)
delta_h_i = a_i u
```

The atom softmax is nontrivial even with one slot. Only the internal slot self-attention has a single key and therefore attention probability one. Its query/key projections cannot select another slot; value/output projections and subsequent FFN remain useful trainable paths. This small algebraic redundancy is not evidence of model failure.

In evaluation, one exchange returns an N-by-192 update matrix `a u^T`, rank at most one. Every atom receives the same new vector direction with a different amplitude. Collecting and receiving use the same a_i; there is no independent receiver query. This describes one update, not the rank of the full encoder or its nonlinear Jacobian. Training dropout can also break the observed matrix rank.

The return projection is zero initialized. At exact initialization this branch is identity and blocks upstream branch gradients through that projection; the projection itself can receive gradients. This is an initialization property, not observed lasting gradient starvation.

These restrictions are plausible research questions, not diagnosed causes. Existing receiver, extra-slot, return-gate and relation experiments make a generic instruction to 'remove the bottleneck' poorly justified.

### Final-layer allocation and pooling

The return projection has no bias (`qm9_neural_atom.py:153`). Layer9 returns directly into mean pooling (`:255-257`), with no subsequent local block or normalization. In evaluation, for N>0:

```text
mean(H + delta_H) - mean(H) = sum_i(a_i u)/N = u/N
```

Thus for fixed pre-mixer H, collected slot content and parameters, changing only final-layer return allocation to any distribution of total mass one cannot change the pooled vector. Collection weights still change u. Layers3/6 have subsequent local processing, so return placement matters there; training dropout can also change noise with placement. This is an exact algebraic structure fact up to floating-point rounding, not evidence that large molecules are poorly predicted or that multiplying by N improves accuracy.

At pinned server67a62378, `pcqm_k1_return_allocation_100k` already tested uniform1/N and inverse-softmax return; both preserve unit return mass. Uniform improved1.826meV but did not qualify. `pcqm_k1_slot_processor_100k` collapsed/removes single-slot attention with +0.518/+1.763meV and no promotion. These simplifications are closed, not newly discovered fixes. `pcqm_k1_paper_allocation_100k` changed to four slots with atom-wise slot normalization and regressed2.079meV. That related intervention changes total return scaling, slot number and collection simultaneously; it neither isolates K1's1/N nor proves a pure N multiplier fails. No isolated N-multiplier contract was found in the inspected sources; this is a search limit, not a claim of exhaustive absence.

## What the retained evidence selects

All gains are reference-minus-candidate MAE in meV; positive is better. Rows do not share one universal dataset/recipe and must not be ranked across contracts.

| Question | Retained observation | Interpretation |
|---|---|---|
| Missing chemical input distinctions | Three-seed rich-feature repair +41.950 on legacy 100K; explicit radical/closed-shell graph collisions removed | Strongest historical cause was identifiable input loss; already repaired. |
| Persistent bond memory | Matched PCQM100K StructuralGPS -> EdgeState +5.330 | Positive evidence for preserving this backbone. |
| More dense global communication | Matched500K dense .111349, local-only .107227, K1 .104860 eV | Removing dense global gives +4.122; K1 slot adds +2.367 over local-only. The latter is below the historical 3-meV context. Preserve the exact cross-platform transform caveat in the owning local-ablation decision. |
| Historical capacity changes | Edge96 and deep160x11 -0.394/-0.652 | These change edge width64->96 or node width192->160 with depth9->11. Neither isolates node width192->256; that question is not closed by these results. |
| Broader relations | PairToken +3.044 on selected100K dev, +0.416 with unchanged weights on disjoint500K dev, -0.699 after matched500K training | Portability already fails before further training; exposure alone cannot explain the whole erosion. |
| RRWP / receiver / vector triplet | Original +2.449/+1.866/+1.516; frozen independent-role -3.576/-2.190/-1.942 | GRIT-inspired pair information and receiver extensions have exact closed counterparts. Active-branch knockout dependence is not net training benefit. |
| Final/layer readout | QM9 R5 JK plus final edge summary regressed Gap by4.327 | Readout has already been investigated. This joint multi-target QM9 intervention does not isolate every possible Gap-only head, but makes generic pooling a weak first choice. |
| Retention/PNA/ring state | Historical geometry-GraphState tests -0.088/-0.172/+0.203, small parameter increases but noticeable throughput cost | Closed exact configurations, not untouched ideas. Ring mechanical acceptance also failed a cache binding. Do not treat geometry results as exhaustive pure2D refutations. |
| Pretraining reconstruction | K1 point gain about+0.197, no3-meV gate; formal INCONCLUSIVE due to reference qualification | Successful reconstruction did not establish a stronger Gap predictor. Archived evidence does not release another run. |
| Joint atom reconstruction | Auxiliary term helps the same corruption recipe +3.212/+2.570 on two roles; full recipe fails portable superiority over clean K1 | Reconstruction can repair corruption cost without beating an uncorrupted model. Clean auxiliary supervision remains a separate unanswered question. |

Authoritative desktop pointers: [rich-input cause](../../../experiments/pcqm_edge_state_full/root_cause.md), [matched500K](../../../experiments/pcqm_500k_v4_evidence/final_decision.md), [local/global ablation](../../../experiments/pcqm_500k_v4_evidence/local_ablation_decision.md), [scale reassessment](../../../experiments/pcqm_scale_transfer_reassessment/decision.md), [QM9 readout](../../../experiments/top20_architecture_qm9/edge_state_jk_readout_r5_decision.md). Pinned server sources: `67a6237895ce94d502991d251f22654df3ab06c7:experiments/pcqm_k1_{cross_scale_frozen,relation_resolution_100k,joint_atom_reconstruction_100k}/...`; exact per-arm decisions are in the inventory. No live server state was queried.

## What cannot yet be diagnosed

- Full EdgeState .099638 and K1 .106672 eV are accepted operational scores under different full contracts. Rough nominal exposures were104.7M and30.14M presentations respectively, with optimizer/schedule/selection differences. These do not establish an architectural ceiling for K1.
- Matched500K K1's development lead shrinks from39.793meV at epoch9 to5.527 at59. That demonstrates relative late catch-up, not representation collapse or a universal early-stop policy.
- There are no new activation, gradient, neighbor-information collision or representation measurements in this review. Gate saturation, oversquashing, edge-state redundancy and global rank limitation remain candidate mechanisms, not measured error causes.
- Historical official validation and external test roles were consumed for recorded EdgeState work. They cannot be casually reused as an exploratory diagnostic or described as globally untouched.

## Preferred remaining question

The remaining architecture question worth qualification is **joint aggregation of incoming real-bond messages inside the preserved EdgeState local path**. SSMA is one candidate implementation. It changes the finite-width aggregate using polynomial-like neighbor interactions; it does not add molecular geometry, labels or bonds. See the existing full-text [SSMA review](report.md#additional-lead-ssma-changes-aggregation-on-existing-bonds). This is a low-confidence research lead, not a mature priority for training or an expected leaderboard improvement.

The question must distinguish the proposed joint aggregate from existing persistent wedges, low-rank Hadamard triplets, nonbacktracking memory and PNA statistics. Keep K1's accepted single-slot path fixed. Do not stack a new readout, receiver gate or ring state with this intervention.

K1 stores3,658,817 parameters; the user2x ceiling is7,317,634. Dense EdgeState stores4,771,073; its separate ceiling is9,542,146. A narrow one-layer d64/kappa4 SSMA arithmetic example stores about116,431 additional parameters including fixed affine/bias, compression and 192<->64 projections, approximately3.18% of K1, before any extra normalization/adapter. This is not a constructed or qualified architecture count. The intermediate width1265 per participating message, FFT/product stability, preprocessing and kernel behavior may dominate cost. Frozen matrices count toward storage. No valid-neighbor cap may silently discard bonds; high-degree, disconnected and bondless inputs need an explicit rule.

### Equation-level overlap and implementation hazards

The [SSMA paper](https://arxiv.org/html/2409.19414), equations11/13-14 and38-40, builds coefficients of `product_u(t - q_xu(z))`, yielding direct three/four-neighbor terms for the full coefficient representation. Current `pcqm_gap_architecture.py:220,398-429` already jointly processes two adjacent bonds through persistent wedge MLPs; `pair_triplet_2d.py:97-106` already sums low-rank Hadamard products across paths. SSMA is therefore distinct only in how one same-center neighborhood is jointly aggregated. Multi-layer existing networks can also generate higher-order effects; no universal expressivity improvement follows from the difference.

The [official implementation](https://raw.githubusercontent.com/AlmogDavid/SSMA/master/ssma.py), lines130-136 of the source reviewed October1, uses geometric-mean magnitude with summed phase and an epsilon, rather than the unmodified polynomial product. The paper's complete-coefficient separation guarantee cannot be assigned to this stabilized, compressed network. With attention disabled, kappa4 does not discard neighbors but determines fixed FFT capacity: neighborhoods larger than four incur circular aliasing. With attention enabled, all neighbors are compressed into four weighted slots; this changes the hypothesis and loses the complete-multiset guarantee. Neither option is an unqualified drop-in recipe.

A conditional draft would preserve the full original gated sum and add one narrow zero-initialized joint residual at layer6. Only degree1-4 would use the kappa4 residual; degree0 adds zero, and larger degrees retain the complete original sum. This would be a declared conditional module and could not claim to repair high-degree neighborhoods. Requiring the same joint aggregate for every degree would instead need a different capacity/cost design. Even this draft needs a capacity control using the same projections/compression with independent-message sum; otherwise a gain would not isolate joint mixing. The source audit therefore leaves a real but narrow question and rejects direct nine-layer192-wide or four-attention-slot SSMA adoption.

The cheapest sequence before any training is: mathematical/source overlap review; then, only under a dedicated desktop question and prospective trajectory, qualify permutation invariance, all-neighbor handling, finite numerics, actual storage and native forward/backward cost. If duplicate or unfavorable, close without training. If it survives, a single isolated controlled comparison must assess portable terminal gain under aligned source/recipe/role/exposure; early learning gains do not replace the terminal decision. Exact gates and cost ceilings belong in that future contract, not this review.

Clean Gap training with training-only chemical auxiliary supervision is a secondary question, distinct from archived sequential pretraining and corruption/reconstruction. The available conditional benefit does not prove this clean recipe works, so it should not be bundled with the aggregation intervention. Pooling and global-return changes rank below both until a genuinely new discriminator is identified.

## Learning order for interpreting this review

1. Graphs: atoms, bonds, directed edges and graph batches.
2. Embeddings and vectors: categorical fields, addition, projection and dimensions.
3. Message passing: message -> neighborhood sum -> node update; channel gates versus neighbor softmax.
4. Residual memory: why a bond state persists and how gradients pass through it.
5. Aggregation and attention: finite-width summaries, one-slot collection, receiver amplitudes and rank-one update intuition.
6. Evidence: matched comparisons, same weights/different molecules, steps/presentations, endpoint versus early gain, row uncertainty versus seed uncertainty.
7. Chemical interpretation: conjugation/aromaticity, molecular orbitals and HOMO/LUMO/Gap, then approximate Kohn-Sham labels. Approximate targets can still be learned; the predictor does not need to solve an exact orbital wavefunction.

## Width comparison follow-up

October1 user question: is node width192 too small? Current source read for this follow-up: desktop`f073b66d8f470af9d727b8803dec30ea6c779648`. Width is task/config specific. A model with N atoms carries N node vectors and separate directed bond states before pooling, not just one192-vector throughout. Increasing width expands representational capacity; it supplies no new input observations and does not establish that the current capacity is saturated.

| Paper / source | Task-bound state width | Qualification |
|---|---|---|
| [GRIT, Tables9-10](https://arxiv.org/html/2305.17589) | PCQM4Mv2 hidden256,16 layers; ZINC64 | Node and updated pair states share hidden width in the architecture; full PCQM model16.6M. |
| [GraphGPS, TableA.4](https://arxiv.org/html/2205.12454) | PCQM4Mv2 small304/5 layers; medium384/10 layers; ZINC64 | Parameters6,152,001/19,414,641. Wider/smaller-depth small model is not a width-only comparison with MolGap. |
| [Exphormer, Table8](https://arxiv.org/html/2303.06147) | Peptides-func64, Peptides-struct88, PCQM-Contact64 | PCQM-Contact is a different task from PCQM4Mv2 Gap. |
| [TIGT, Table7](https://arxiv.org/html/2402.02005) | PCQM4Mv2 hidden256/10 layers; ZINC64 | Full PCQM model13M. |
| [ESA official README template](https://raw.githubusercontent.com/davidbuterez/edge-set-attention/main/README.md) | Bond-attention hidden256; pooled graph512; MLP internal512 | This is the documented generic example, not verified paper PCQM settings; exact task width remains unqualified. |
| [CIN, AppendixE.4](https://arxiv.org/pdf/2106.12575) | ZINC hidden128/4 layers; small48/2 layers | Cell states, different task and topology. |
| GSN official [100K](https://raw.githubusercontent.com/gbouritsas/GSN/master/scripts/ZINC_10_runs_100K.py) / [500K](https://raw.githubusercontent.com/gbouritsas/GSN/master/scripts/ZINC_10_runs_500K.py) | ZINC d_out64/150 | Released script versions; paper appendix default128 is another configuration, not one shared width. |
| [PathNN, AppendixD Table8](https://proceedings.mlr.press/v202/michel23a/michel23a.pdf) | ZINC SP/SP+152, AP140 | Path/LSTM representations under500K parameter budget; not PCQM. |
| [D-MPNN, default settings](https://arxiv.org/pdf/1904.01561) | Directed bond hidden300 | Out-of-box default; not a verified tuned PCQM configuration. |
| [RingFormer, AppendixC](https://arxiv.org/html/2412.09030) | hidden512,8 layers across its five datasets | Solar-cell/property tasks, not PCQM. |
| [PNA released paper-100K setup](https://raw.githubusercontent.com/lukecavabarrett/pna/master/realworld_benchmark/README.md) | ZINC no-edge hidden75/out70; with-edge hidden70/out60, edge50 | Source version matters; towers/internal/output widths differ. |
| Physical pooling | No single hidden width assigned in this follow-up | Readout study spans several backbones/tasks; the architecture/task must be specified. |
| Adaptive readouts | No single hidden width assigned in this follow-up | Multiple encoders and readout widths; exact molecular-task configuration not verified here. |
| GraphNorm | No single hidden width intrinsic to the method | Normalization preserves the chosen backbone channel count; specific experiment widths not verified here. |
| KernelReadout | No single hidden width assigned in this follow-up | Backbone width and kernel/readout dimension must be separated; no molecular Gap setting verified here. |

An additional comparison is [Graphormer, AppendixTable7](https://arxiv.org/html/2106.05234): PCQM4M small512/6 layers, standard768/12 layers. This is the earlier PCQM4M benchmark, not PCQM4Mv2. Its edge feature width32 also demonstrates why node width cannot stand in for all state dimensions.

### Static storage arithmetic, not a constructed model

For even node width h, fixed nine layers, e64, K1 slot64/max-allocated-slots4, OGB categorical cardinalities174/13, RWSE16 and current local PyG2.7.0, parameter counts follow:

```text
K1(h)            = 74.5 h^2 + 3755 h + 191489
DenseEdgeState(h)=110.5 h^2 + 3227 h +  78017
```

Owners are `gps.py`, `qm9_neural_atom.py`, OGB feature/encoder definitions and PyG GPSConv/ResGatedGraphConv. Local blocks contribute8h^2 each; dense blocks add4h^2+6h over a local block. RWSE contributesh^2+18h; the scalar head contributes1.5h^2+2h+1. K1 additionally has three mixer contributions194h+37824. These formulas reproduce both registered192 counts. They are source-derived storage estimates, not runtime construction/preflight or performance measurements. Frozen historical runtime differences and future implementation changes require recounting.

| Node width | K1 parameters | Relative | Dense EdgeState parameters | Relative |
|---|---:|---:|---:|---:|
|192|3,658,817|1.000|4,771,073|1.000|
|256|6,035,201|1.650|8,145,857|1.707|
|384|12,618,881|3.449|17,611,073|3.691|

Thus node256 with edge/slot64 fits the user's2x storage ceiling in the audited definitions; node384 does not. Activation memory and native cost are unmeasured. Widening node192->256, edge64->96 and slot64->larger are different interventions and must not be bundled to diagnose one bottleneck.

**Correction to earlier priority interpretation:** the old capacity screen does not refute node-only256. A matched node192/256 capacity control is simpler to interpret than adding SSMA and can distinguish an overlooked capacity explanation before claiming a novel aggregation is necessary. This is an analysis recommendation, not a launch or successor release; active attempts keep their owning protocols. Improved fixed-cohort training fit without a portable terminal development gain would not justify promotion. Missing improvement would reject the tested width/recipe, not prove192 universally sufficient.
