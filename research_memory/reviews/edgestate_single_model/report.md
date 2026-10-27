# EdgeState single-model research review

Review date: 2026-09-30 JST. This is a source-based interpretation and literature review, not a canonical experiment result, prospective trajectory, training release, or promotion decision.

Follow-up source review: 2026-10-01 JST. The original inventory remains the pinned September30 snapshot. The pretraining owner subsequently committed terminal evidence at `8821b5ce`; its October1 fusion supplement is uncommitted local work. See the follow-up below rather than interpreting earlier progress as a waiting-for-training blocker.

Later October1 scope correction: the user narrowed research to EdgeState/K1. Read the [focused mechanism analysis](edgestate_mechanism_analysis.md) for verified update ordering, the final-slot/mean-pooling identity, closed single-slot simplifications and the SSMA overlap/cost audit. Source qualification leaves SSMA a narrow unvalidated question, not a mature training recommendation. The archived pretraining supplement is preserved as unreviewed custody snapshot9f767346; its earlier uncommitted wording below describes the original reading state. Qualification is archived under current routing, with no automatic reopening.

Width follow-up correction: the [task-bound width comparison](edgestate_mechanism_analysis.md#width-comparison-follow-up) verifies that the historical edge96/deep160x11 screens did not isolate node192->256. Do not read the older capacity summary as a refutation of that question. Source-derived counts put node256 within the user's2x parameter ceiling for both K1 and dense EdgeState, with edge/slot widths fixed; no accuracy or runtime measurement follows from this arithmetic.

## Conclusion

There is no presently demonstrated small-parameter module that can be promised to improve the accepted full-scale EdgeState checkpoint. K1 is an EdgeState descendant, and remains an economical research backbone. The evidence does not establish a causal full-scale architecture ranking between K1, dense EdgeState and GPTrans.

Do **not** reopen generic ring state, RRWP-pair, another molecular slot, retention gates, PNA statistics, chemistry-separated local updates, or more triplets merely because a new paper gives them a different name. Several were already tested outside the desktop checkout. The strongest surviving question is whether a different learning constraint or a different finite-width local aggregation can improve *portable terminal performance*, rather than merely improve early fitting or a repeatedly selected development role.

Two additional research leads are examined below: SSMA local aggregation and adversarial latent-feature training. SSMA merits an overlap/cost audit; FLAG is a lower-priority unvalidated objective. Neither is a validated remedy for MolGap. The October1 pretraining review provides no evidence to prioritize its exact recipe for scale-up; remaining strict reference qualification is evidence work. Do not stack new modules on its unqualified endpoints.

## Source coverage and limits

The [inventory](evidence_inventory.json) records 240 snapshot entries representing **126 distinct trajectory IDs**. Every indexed canonical trajectory and linked decision was loaded when present; none of those read targets was missing. This is an index-and-document audit, not independent reacceptance of every historical tensor, checkpoint, scheduler state or raw training log.

| Snapshot | Pinned local identity | Indexed trajectories |
|---|---|---:|
| Desktop | `5cdec4d2` | 62 |
| Pretraining worktree | `44b594cf` | 66 |
| Geometry continuation worktree | `632ae809` | 56 |
| Locally cached server branch | `67a6237895ce94d502991d251f22654df3ab06c7` | 56 |

The server snapshot is a local Git ref, **not a live server-state check**. No remote compute commands were executed. Indexed outcomes are preserved as recorded; newer owning decisions can supersede stale navigation or an older index entry. Direct lineage, paired evidence, GPTrans comparison context and unrelated context are separated in the inventory. A text mention of K1 is not counted as an EdgeState intervention.

The desktop frozen RML check reported current generated outputs. Other branches are not thereby certified as integrated into desktop. No generated RML file was changed. Historical mechanisms absent from the active index were followed through explicit source names and archive history: notably ring hierarchy, PNA statistics, gated edge retention and the QM9 architecture sequence.

Recent desktop commits worth reading are `80274818` (terminal attribution), `7429160e` (geometry RML), `b9b6e6be` (K1 lineage), `a47b945f`/`0cf1319d` (shared topology/training infrastructure), and `5cdec4d2`/`71976042` (expert Oracle). Infrastructure commits do not establish a new model advantage.

## What the local evidence actually supports

Positive gain below means lower MAE than the stated comparator. Units are meV unless eV is explicitly shown. Scores from different roles, datasets or contracts must not be sorted into one model ranking.

| Mechanism / evidence | Observed result | Decision-relevant interpretation |
|---|---|---|
| Rich OGB atom/bond categories | Three-seed legacy 100K mean improved 41.950 meV versus lossy features; radical cases improved much more | The clearest historical improvement came from repairing missing input distinctions. Already incorporated. |
| Persistent real-bond state | Matched PCQM 100K: StructuralGPS 0.143313 eV to EdgeState 0.137983 eV; +5.330 meV, +25.3% parameters | Positive local evidence for the retained bond-state backbone. |
| Wider/deeper backbone | Edge96 and 160×11 lost by 0.394 and 0.652 meV in the architecture search | No justification for spending the 2× budget on blanket width/depth. These tests do not establish a universal scaling law. |
| Static operator swaps | GatedGCN, TransformerConv and GENConv lost by 2.425, 5.000 and 4.793 meV | Their exact replacements are closed. |
| Recurrent graph state | 0.138613 versus 0.137983 eV; 0.630 meV worse | Another generic global state is already covered. |
| Matched desktop 500K local/global ablation | Dense EdgeState 0.111349; local-only 0.107227; sparse-global 0.109116; K1 0.104860 eV | Dense global attention hurt under this contract. K1's slot improved local-only by 2.367 meV, below the 3-meV material context. |
| K1-G and K1-R | Molecule-conditioned slot strength: 3.890 meV worse; separate EdgeState relation slot: 0.147 meV worse | Generic gating or an extra relation slot is not new. |
| Original PairToken | +3.044 meV on original 100K dev; only +0.416 meV with unchanged weights on disjoint fixed500K dev; exact matched60 500K study −0.699 meV | Both molecule-population portability and training-contract changes matter. Do not attribute the whole erosion to long training. |
| PairToken refinements | SPD +1.673; chemistry-conditioned +2.799; one-shot triplet +2.607; adaptive return +1.745 versus K1; none qualified; several lost to original PairToken | Subthreshold gains are useful historical signals, not interchangeable positive promotions. Value multiplication regressed 1.064 meV versus K1. |
| Receiver / vector triplet / RRWP | Original gains +1.866 / +1.516 / +2.449; frozen100K weights on fixed500K dev: −2.190 / −1.942 / −3.576 | GRIT-style RRWP is **already tested**. This is frozen inference on different molecules, not training on 500K. |
| Knockout diagnostic | Removing or halving these branches substantially worsened their own trained checkpoints | Branches were active and co-adapted. Knockout dependence is not the net advantage over separately trained K1. Simple attenuation did not repair the failure. |
| Degree balancing / repeated RWSE injection | Frozen500K dev 0.143424 / 0.142377 versus K1 0.141253 eV | Degree/RWSE renewal is already covered by negative portability evidence. |
| Node-query linear attention | −0.399 meV at original terminal; −2.953 meV on frozen500K dev, despite fewer parameters | Early gains did not persist. Linear complexity did not establish a better predictor. |
| MoSE rooted motif counts | Replacement +1.119; selective residual +2.046; context gate −0.527; hidden BN −1.273 versus K1 | Static motif information, selective gates and normalization already have dispositions. A GSN proposal must distinguish its information from this family. |
| Functional groups / conjugated hyperedges | Group token +1.542; persistent conjugated state +1.269; neither qualified | Merely naming chemically meaningful groups is insufficient. Some subgroup gains occurred where the new path was inactive, preventing direct mechanism attribution. |
| RingState64 historical experiment | +0.203 meV, +58,040 parameters, throughput 0.8492×; declared acceptance failed on omitted cache field | The later ring-GraphState run **did train and close**. Early ring-cache documents saying GPU deferred are superseded. It is neither an accepted win nor an untouched research idea. |
| PNA statistics historical experiment | 0.172 meV worse; +58,946 parameters; throughput −24.3%, peak memory +23.8% | Mean/max/min/std/degree adapter already failed its geometry/GraphState contract. Not a faithful reproduction of every PNA configuration. |
| Gated edge retention historical experiment | 0.088 meV worse; +77,472 parameters; throughput −7.6% | A cheap retention gate has already been tested. Preserve its closed decision; endpoint data do not prove all bond memory is sufficient. |
| GPS++ local directional adapters | Sender-only −1.593; bidirectional −0.206 versus K1, +23.4% parameters | Better fitting did not establish improved development performance. Exact family closed. |
| Chemistry-separated local update | Atom-pair −2.464; bond-type −1.580 | Delayed local mixing, suggested by MMGNN, is also already tested. |
| New motif hierarchy / MetaGIN2D | Motif addition −0.342, interval crosses zero; MetaGIN2D −22.814 | Do not reopen those exact adaptations through CIN/RingFormer terminology. |
| Joint corruption/reconstruction | Reconstruction helped versus identical corruption by 3.212 / 2.570 on original / fixed500K dev, but full recipe did not establish portable superiority over clean K1 | Distinguish the helpful conditional auxiliary objective from a failed total recipe. This does not answer clean auxiliary supervision without corruption. |
| New 60-epoch geometry continuation | K1 distance-angle −0.0033; GPTrans distance +0.857 versus own pure-2D references | Completed matched exposure; no material single-model geometry win. |
| Geometry versus pure-2D fixed blend | +0.365530, 95% paired-row interval [−0.033628, +0.767960] | No reliable blend-over-blend gain. Existing pure-2D architecture complementarity explains most of the within-pair blend benefit. |
| Prediction-only expert router | Soft gate only +0.279 versus crossfit global blend; hard gate worse; Oracle is label-aware | Oracle headroom is not deployable specialist evidence. Already proposed shared-backbone specialist heads remain conditional hypotheses. |

### Authority pointers for this table

Desktop sources: [scale reassessment](../../../experiments/pcqm_scale_transfer_reassessment/decision.md), [matched500K decision](../../../experiments/pcqm_500k_v4_evidence/final_decision.md), [residual reconciliation](../../../experiments/pcqm_k1_residual_reconciliation/decision.md), [full rich-feature cause](../../../experiments/pcqm_edge_state_full/root_cause.md), [expert Oracle](../../../experiments/pcqm_expert_oracle_feasibility/decision.md). Other exact desktop decisions are enumerated in the inventory.

Geometry authority: `C:/Users/17449/.codex/worktrees/geometry-v4-500k/molgap/experiments/pcqm_geometry_v4_500k_pair/fixed_blend_scale_decision.md`. This later review recovered the exact K1 reference predictions from `D:/文档/molgap-exp/molgap-500k-v4-evidence/`; the earlier missing-reference assessment is superseded. The immutable cost discrepancy of 0.971 seconds and replay binding work remain bookkeeping limitations, not a reason for more training.

Server authority at pinned `67a62378`: `experiments/pcqm_k1_relation_resolution_100k/{protocol.md,arms/rrwp_pair/decision.md,audit/decision.md,diagnostic/decision.md}`; `pcqm_k1_cross_scale_frozen/decision.md`; `pcqm_k1_joint_atom_reconstruction_100k/decision.md`; `pcqm_k1_chem_local_100k/decision.md`; `pcqm_motif_hierarchy_100k/attempt_v3/decision.md`; `pcqm_metagin_2d_100k/attempt_v2/decision.md`; individual MoSE, conjugated, PairToken and portability decisions in the inventory.

Historical direct sources: `git show 9054ba8c:experiments/pcqm_gap_architecture/results/ring_graphstate_seed42/decision.md`; `git show 9006ef5e:experiments/pcqm_gap_architecture/results/{pna_statistics_graphstate_seed42,edge_retention_graphstate_seed42}/decision.md`; `git show 285e1dc:experiments/pcqm_gap_architecture/results/recurrent_graph_state_seed42/decision.md`. Their historical causal narratives are not adopted beyond what retained comparator/trace evidence identifies.

### Full-scale comparison is not an architecture experiment

Accepted official-validation checkpoint scores: EdgeState 0.099638 eV; K1 0.106672; GPTrans initially 0.109012, then 0.103791. EdgeState selected after roughly 104.7M nominal sample presentations; K1/GPTrans initial exposure was 20M, and their continuations reached roughly 30.14M/40.27M. Optimization and selection semantics also differ. Use EdgeState as the accepted operational reference; do not claim these numbers prove its architecture intrinsically dominates K1/GPTrans. Official validation has been consumed. October1 role correction: the [reference index](../../../models/REFERENCE_INDEX.md) records external test-role consumption by the older EdgeState submission; those roles must not be described as globally untouched. This literature review and the pretraining experiment consumed no protected role.

The old scale diagnostic's claim of proven representation collapse and its 7.8-meV projected gain are superseded by the later reassessment. The actual joint Noisy Nodes/PairNorm 500K gain was 2.056 meV. The evidence supports gain erosion, not a unique mechanistic diagnosis.

## Fifteen full-text mechanism references

These are **15 mechanism papers compatible with a 2D-input investigation**, not 15 distinct pure-2D papers from the top 15 leaderboard entries. The [official leaderboard](https://ogb.stanford.edu/docs/lsc/leaderboards/#pcqm4mv2) mixes geometry, ensembles, repeated submissions and much larger models. Published validation-only numbers are not official test-dev submissions. Training with 3D supervision is a separate qualification from inference without coordinates.

Each entry reports methods and an ablation or limitation, rather than relying on an abstract. Geometry-based examples within general papers are marked and not used as strict-2D evidence.

| # | Paper; full-text locations | Mechanism / actual gate | Evidence, cost and local disposition |
|---:|---|---|---|
| 1 | [GRIT](https://arxiv.org/html/2305.17589), §3.1–3.3, Tables4–7, App.B.6; official PCQM config | Off-diagonal normalized walk probabilities; learned pair states modulate Q/K additively and multiplicatively and write values into atoms. Degree scaling is not a sigmoid gate. | PCQM val 0.0859, 16.6M. ZINC RRWP→RWSE: .059→.081; sparse variant .066. Dense pair activation remains quadratic. Already has local RRWP negative portability; no reopening. |
| 2 | [GraphGPS](https://arxiv.org/html/2205.12454), §3, Tables3–5, App.A/B | Parallel local message passing and global attention plus RWSE/PE; local gated messages and global softmax do different work. | PCQM10% Transformer .1159, Performer .1142, no-global .1213. Performer 83s/epoch versus Transformer61s on A100. Existing backbone; asymptotic linear attention does not guarantee speed. |
| 3 | [Exphormer](https://arxiv.org/html/2303.06147), §3/5, Tables3/5, App.D | Real edges, fixed-degree random expander edges and virtual nodes in sparse attention; edge-type embeddings. | No PCQM Gap result. Peptides-Struct removing expander improves .2643→.2481. O(E+N) interactions under fixed-degree assumptions; small-molecule benefit not established. |
| 4 | [TIGT](https://arxiv.org/html/2402.02005), §3, Tables3/4, App.B/C | Real-bond and cycle-clique message paths; graph sum-pool→MLP→sigmoid channel gate. | PCQM val .0826 at13M; ZINC .057→.059 without graph gate, .064 with single path. Cycle clique contributes O(b²); no PCQM module-isolation ablation. Ring/motif negatives lower priority. |
| 5 | [ESA](https://www.nature.com/articles/s41467-025-60252-z), Methods Eq.1–11, PCQM and layer-order/cost analyses; official code | Bond tokens contain endpoints; masked softmax attention between bonds sharing atoms, alternating with dense bond attention; PMA pooling. | Claimed PCQM val .0235/400epochs is anomalous and cannot set an expected gain. Public loader/entry/config inconsistencies prevent exact reproduction qualification; no direct label leakage found. Dense E×E masks remain. |
| 6 | [CIN / CW Networks](https://arxiv.org/html/2106.12575), §4/5, App.B/E.4–5 | Persistent atom/bond/chordless-ring cells; boundary and upper-incidence aggregation followed by GIN-like MLP, not an obligatory confidence gate. | ZINC CIN-small no rings .174→.094; roughly1.9× epoch time versus GIN-small. No PCQM. Bond–ring incidence differs from old atom–ring bottleneck, but distinction alone does not justify a new run. |
| 7 | [GSN](https://arxiv.org/html/2006.09252), §3/5.3/5.5, App.C.3 | Fixed subgraph automorphism-orbit counts attached to atoms/edges; unchanged ordinary message passing. | ZINC100K MPNN .209 versus GSN .115; preprocessing can be expensive. Different cycle sets from SymmSSSR; must distinguish from closed local MoSE rather than advertise all counting as new. |
| 8 | [PathNN](https://arxiv.org/html/2306.05955), §3/4, App.B–E | Reverse-path LSTM uses input/forget/output gates with topological root distance; simple-path aggregation. | ZINC AP .090, SP .104, SP+ .131, versus CIN .079; AP4 ~501K and4.38s versus4-layerGIN1.97s. Paths can grow exponentially; weak fit for the compute constraint. |
| 9 | [D-MPNN](https://arxiv.org/html/1904.01561), §3.2–3.6/4.2/5.4–5.5 | Directed bond states exclude immediate reverse edges; shared recurrence, hard topological exclusion. | Direction advantage often not statistically significant; headline results combine tuning/descriptors/ensembles. Local QM9 nonbacktracking and PCQM directed-triplet tests overlap substantially. |
| 10 | [RingFormer](https://arxiv.org/html/2412.09030), Eq.1–10, Tables2–7, implementation appendix and official layer code | Atom↔ring GINE; relation-aware ring attention puts neighbor and relation into both K and V; ring virtual node, layer fusion and JK/readout. | CEPDB PCE .244→.189; Gap figure is multitask, not PCQM. d512,L8, no parameter matching; removing ring virtual node can improve .1886→.1860. Old hierarchy already covers the main concept. |
| 11 | [PNA](https://arxiv.org/html/2004.05718), §2/3/5.2, App.D/K | Mean/std/min/max crossed with identity/amplifying/attenuating degree scalers; towers constrain parameter growth. | Own ZINC PNA .188 versus sum .288; no-scalers .247. Different patience is disclosed. No PCQM and local statistics adapter negative; not a fresh recommendation. |
| 12 | [Physical pooling](https://arxiv.org/html/2207.13779), §2.2–2.3/3.3, target tables | Sum for extensive quantities, mean/max for intensive ones; pooling is a target assumption, not a learned confidence gate. | QM9 1-GNN Gap sum .1322, mean .1336, max .1283. Geometry-based MXMNet examples are excluded from strict-2D support. No universal instruction to replace mean with sum. |
| 13 | [Adaptive readouts](https://papers.nips.cc/paper_files/paper/2022/file/7caf9d251b546bc78078b35b4a6f3b7e-Paper-Conference.pdf), neural-readout methods and experiment/appendix tables | SetTransformer self-attention and seed-query PMA; MLP/GRU variants can depend on atom order. | QM9 multioutput R² is not Gap MAE; some larger/deeper models diverge. Existing learned-query/JK/RepSet evidence lowers priority, though some local comparisons were not isolated pooling ablations. |
| 14 | [GraphNorm](https://proceedings.mlr.press/v139/cai21e/cai21e.pdf), §3–5, Eq.8 | Graphwise h−αμ with learnable mean retention, variance normalization, γ/β. Three scalars per channel. | Cheap but no PCQM/Gap remedy established; local normalization failures do not prove this exact method fails, yet no diagnosed normalization defect justifies priority. |
| 15 | [Kernel readout](https://www.ijcai.org/proceedings/2024/0277.pdf), §3.2 Eq.4–9, §4.1–4.3 | Channelwise Gaussian similarity to centers and multiple pooling heads. | Classification/clustering evidence, no quantum Gap or PCQM. Another nonlinear pooling module lacks a diagnosed local need. |

ESA qualification note: inspected official commit `861a0b12d445ae80cc69b3948af623b4331321d4`. The PCQM loader returns seven objects while the graph-level entry unpacks six; loader-created graph metadata does not match fields the entry expects; README's fixed400-epoch prescription differs from early-stopped best-checkpoint defaults. This establishes a public reproduction gap, **not proof that the paper is wrong**. The [implementation](https://github.com/davidbuterez/edge-set-attention/tree/861a0b12d445ae80cc69b3948af623b4331321d4) and [parameter issue](https://github.com/davidbuterez/edge-set-attention/issues/1) are supporting records.

Excluded from strict-2D evidence: GoMS fragment geometry, Uni-Mol+, GPS++ geometry-assisted headline results, and methods that require 3D teachers even if final inference is coordinate-free. General architecture ideas can still be discussed as adaptations, with that distinction explicit.

## Additional lead: SSMA changes aggregation on existing bonds

[SSMA, NeurIPS2024](https://arxiv.org/html/2409.19414), §3–4.4/5 and App.B/C/D.3/E, mixes neighbor features through FFT-domain products, inverse FFT and low-rank compression. It changes finite-width aggregation without adding graph edges. It is not PNA's collection of first-order statistics; nevertheless, the existing wedge/low-rank multiplicative triplet code already supplies related interactions, so an equation-level overlap audit is required.

Parameter-matched ZINC100K: GIN .252→.222 and PNA .192→.172. The more relevant GraphGPS+RW20 result is only .102±.004→.100±.003. **No QM9 or PCQM Gap evidence**; no large expected MolGap gain follows.

The [released code](https://raw.githubusercontent.com/AlmogDavid/SSMA/master/ssma.py) stores a dense frozen affine map. For d64 and κ4, intermediate width1265 yields80,960 fixed weights plus roughly9,303 compression weights before biases. d192 yields734,400 fixed weights per layer. Frozen tensors must count toward model storage. Sparse constant construction or a few narrow layers could control parameters; FFT activations and stability remain unmeasured. These are design estimates, not instantiated counts.

An arithmetic example adds192→64 and64→192 projections to that narrow layer, bringing stored weights to roughly0.12M, approximately3.3% of K1. This shows room within the user's2× ceiling, not the final count of an implementable model. κ4 is not an accepted chemistry degree cap: higher-degree and isolated atoms, selection versus full-neighborhood semantics, and constant-map storage must be resolved without silently discarding accepted bonds. Use one bounded local intervention rather than nine unqualified dense192-wide SSMA layers.

This lead is worth source inspection only if it can be expressed as a bounded local replacement, preserve permutation invariance and rare high-degree/bondless molecules, and distinguish from the closed triplet information path. Otherwise stop as duplicate or unfavorable cost before training.

## Additional lead: FLAG is a learning constraint, not new information

[FLAG](https://arxiv.org/html/2010.09891), §2–5, Algorithm1, Tables4/6/7, perturbs continuous atom embeddings in the loss-increasing gradient direction, accumulates model gradients across M perturbations, then performs one optimizer update. It differs from random categorical corruption and reconstruction, adds no inference parameters, and changes no graph edges.

There is **no quantum Gap evidence**. MolPCBA GIN-VN AP improves27.03→28.34, but MolHIV GCN-VN decreases75.99→75.45. Default M3 needs three forward/backward passes per step; the paper's near-equal-time fast variant reduces epochs. It does not preserve equal exposure at free cost. No bandwidth gain is established.

[Graphormer](https://arxiv.org/html/2106.05234), §4/AppendixB, uses FLAG for downstream molecular classification, **not its PCQM4M experiment**. Official [PCQM script](https://github.com/microsoft/Graphormer/blob/main/examples/property_prediction/pcqv1.sh) uses ordinary graph regression/L1; [HIV script](https://github.com/microsoft/Graphormer/blob/main/examples/property_prediction/hiv_pre.sh) enables pretrained FLAG. The PCQM embedding-dropout ablation is adverse, warning against assuming more perturbation helps.

The [FLAG molecular runner](https://github.com/devnkong/FLAG/blob/b507e6286797b8a726d594c38fd69c54cd1b19ef/ogb/graphproppred/mol/main_pyg.py) hardcodes a300-wide perturbation and passes classification BCE in FLAG training, despite declaring a regression criterion. Do not copy it as a Gap recipe. Latent smoothness is a hypothesis; label-preserving physical meaning is not guaranteed by preserving the graph. This ranks below SSMA for further source investigation and below completion of existing pretraining.

## Parameter and communication budget

| Exact model | Existing parameters | User's 2× ceiling |
|---|---:|---:|
| K1-v4 | 3,658,817 | 7,317,634 |
| Full dense EdgeState | 4,771,073 | 9,542,146 |
| GPTrans-T | 5,246,817 | 10,493,634 |

Do not use EdgeState's ceiling to silently grant K1 more than its own 2× budget. Count frozen stored tensors as well as trainable tensors, and report train-only heads separately.

K1 already offers sparse local bond processing plus three node↔one-slot exchanges. Interaction count is O(E+N) per such exchange at fixed dimensions; nine dense GPS attention layers involve O(N²) interactions. This is a communication-volume argument, **not measured GPU bandwidth**. Narrow global summaries can discard receiver-specific structure, but restoring it has already failed portable performance in some exact local interventions.

The local ring/PNA examples demonstrate why few new parameters do not imply low cost: 1.6% extra parameters cost15–24% throughput. Report activation memory, CPU cache size, epoch/device cost and full convergence exposure as separate axes. Paper hardware cannot be converted to local T4/DCU/A100 savings.

## A decision path that does not repeat closed work

1. The pretraining terminal has been reviewed in the October1 follow-up below. Its exact reconstruction recipe did not establish material endpoint improvement; strict reference qualification remains pending. Preserve its disposition and retrieve existing evidence rather than authorizing another run. Training-only heads do not enlarge inference.
2. Review the already testing fixed fusion against the correct original pure-2D pair. Prediction-only Oracle remains an upper bound, not a teacher available at inference. Existing specialist-head plans are conditional and are not claimed as a new discovery here.
3. For a subsequent standalone research question, investigate SSMA's bounded same-neighborhood mixing only if its overlap/cost audit passes. FLAG remains a secondary idea requiring independent evidence of relevant embedding sensitivity and an honest training-cost contract. Neither presently warrants a claim of likely material improvement. Do not add RRWP, ring state, another gate and pooling simultaneously.
4. The cheapest decision-relevant prerequisite is source/contract review: exact mechanism difference, required information, parameter/storage arithmetic and valid comparison identity. It can terminate NO_TRAIN. Do not launch diagnostics merely to create research activity.
5. Only after a separate prospective contract is authorized should runtime qualification, a bounded diagnostic or training occur. Preserve the existing experiment-local materiality rules; 3meV is not a universal physical constant or measured seed variance.
6. A candidate must retain a favorable terminal tail and portable benefit under its predeclared contract. Original-dev gains and branch-knockout sensitivity alone do not release scaling. Frozen500K inference is a molecule-population audit, not 500K retraining or a sealed test. Training-seed uncertainty remains separate from row bootstrap.

The most informative falsifier is not “does it lower training loss?” It is “does the isolated change improve the accepted comparator at matched terminal exposure without sacrificing performance on the predeclared comparison populations?” Existing records do not uniquely identify oversmoothing, undertraining, capacity shortage or a chemical subclass cause. Those explanations remain `insufficient_evidence` where fixed-cohort metrics or causal comparators are absent.

## Pretraining terminal follow-up, 2026-10-01

Authority: [terminal decision](C:/Users/17449/.codex/worktrees/hierarchy-exposure/molgap/experiments/pcqm_pretraining_family_pair/terminal_decision.md), its per-arm acceptance/V5 records, frozen protocol and raw reconstruction/downstream traces. Supplement: [retained fusion review](C:/Users/17449/.codex/worktrees/hierarchy-exposure/molgap/experiments/pcqm_pretraining_family_pair/fusion_comparison.md) and its source-bound JSON. The supplement is uncommitted other-agent work; reading it does not integrate or promote it.

The selected same-row endpoints fail the predeclared material point gates. K1's retained paired-row interval crosses zero, as does the pretrained-versus-unpretrained fixed-blend interval. Most observed fusion gain is already present in the unpretrained pair. The evidence does not support prioritizing this exact pretraining allocation for scale-up or treating family complementarity as a pretraining achievement. Both formal scientific outcomes remain **INCONCLUSIVE** because strict historical trace/runtime/reference qualification is incomplete; numerical point-gate failure is already decidable. Missing qualification cannot improve the observed point estimates by itself.

Both reconstruction stages learned their declared tasks: atom/bond/group losses fell, all ten passes completed, stage records report optimizer/RNG resets and restored initial Gap heads. GPTrans EMA is constructed after pretraining. There is no observed disabled-pretraining path or pretraining-contaminated EMA to repair. K1's selected downstream endpoint precedes the last two passes; GPTrans selects the last EMA endpoint. Neither behavior establishes overfitting or an exposure shortage without matched reference curves and fixed-cohort metric semantics.

A **source-derived alternative explanation**, not an established failure cause, is an easy local reconstruction shortcut. The actual reused training loop masks directed bond-feature rows independently. Conditional on masking one direction, its opposite direction remains visible with probability about0.85 under the ordinary independent0.15 sampling rule, ignoring the rare forced-nonempty correction. Message passing can therefore access redundant bond categories. This predicts that excellent bond reconstruction need not require the longer-range electronic relationships relevant to Gap. Actual learned exploitation, importance to endpoint quality, functional-group imbalance and the relative contributions of the three objectives remain unmeasured. This observation does not release paired-mask retries or another pretraining experiment.

The runtime qualification helper instead uses paired undirected masks. The frozen protocol explicitly declares independent directed training masks; the observed difference is a qualification-coverage limit, not a claim that execution violated its training contract. Existing stage tests cover hooks/head/RNG behavior and the helper's mask symmetry, not an empirical attribution of this possible shortcut.

Downstream supervised exposure matches the family-specific references, but the extra unsupervised updates do not match total compute. Assigned-arm wall times cannot be relabeled GPU busy time or billing. No inference parameters were added. Remaining reference/readiness reconciliation does not authorize scratch reference retraining, longer adaptation or specialist heads. The revised research priority is an isolated, demonstrably distinct mechanism or objective only after evidence review; SSMA remains a low-confidence source-audit candidate, with no new support supplied by these pretraining results.

## Beginner reading route

Read atom/bond categorical embeddings → message passing and neighborhood aggregation → residual connections and persistent edge state → sum/mean/softmax and pooling → random-walk diagonal versus pair encoding → matched exposure/selection/row bootstrap → auxiliary and adversarial training objectives. Quantum topics can follow: π conjugation and aromaticity → molecular orbitals/HOMO/LUMO → approximate Kohn–Sham target definition. Exact molecular orbitals need not be solved by the model; approximate labels can still define a meaningful prediction task.

One useful question after each paper is: **What information enters that our model does not have, where is it preserved, what gate actually controls it, and which matched ablation shows net benefit after accounting for capacity and training?** If none can be answered, the paper is background rather than a next experiment.
