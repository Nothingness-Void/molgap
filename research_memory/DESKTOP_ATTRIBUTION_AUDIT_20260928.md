# Desktop RML attribution audit — 2026-09-28

This audit uses the committed `molgap-desktop` RML snapshot at
`7429160e`. The index contains 52 trajectories across 36 experiment
directories; 47 trajectories across 32 directories have desktop ownership
and are in scope here. Multiple arms, retries, reference records, and
no-train actions are separate trajectories, not independent module tests.
Historical and server-owned trajectories are outside this audit. No remote
state, server branch, training, checkpoint inference, or protected role was
accessed. A locally uncommitted pair-memory terminal draft is not substituted
for canonical RML.

An endpoint answers **whether** a candidate cleared its frozen gate.
A paired intervention estimates **which change** produced an effect under
one contract. A root-cause claim such as underfitting, exposure insufficiency,
representation harm, or overfitting needs measurements that separate those
alternatives. Online training metrics, EMA development metrics, different
training cohorts, and row bootstrap have different meanings. Missing
comparators and traces remain missing.

## Newly inspected retained artifacts

| Closed question | What the local evidence narrows down | Remaining causal ambiguity |
| --- | --- | --- |
| [Centered logits](../experiments/pcqm_gptrans_centered_logits_100k_kaggle1_pair/results/posthoc_attribution.md) | The 0.598 meV endpoint gain covers fewer than half of rows; much of it tracks an output shift. The very large early EMA lead contracts to near zero. Online training favors the candidate more than development. | The frozen 60-epoch result is sub-threshold; the trace does not prove overfitting or that more epochs would help. It lacks live-development observations and remains replay-ineligible. |
| [RWSE16 plus local edge](../experiments/pcqm_gptrans_local_inductive_bias_100k_kaggle1/results/posthoc_attribution_attempt_002.md) | The small 1.082 meV gain survives descriptive offset correction and agrees between terminal live and EMA development. Its relative lead shrinks late while training fits better; the extra path costs 40.4% more measured T4 interval time. | A one-seed online-train/development contrast does not prove overfitting or a generic local-path ceiling. |
| [Feature denoising](../experiments/pcqm_gptrans_feature_denoising_100k/results/posthoc_attribution.md) | The auxiliary objective was learned; bond reconstruction improves over atom-only under the *executed* recipe. The decisive blocker is the observed source/optimizer mismatch with the frozen contract. | The intended contracted mechanism cannot be accepted or assigned a causal failure mode from this run. |
| [K1 PairToken value decoupling](../experiments/pcqm_k1_pair_value_100k/results/posthoc_attribution.md) | The candidate was still improving at its last frozen epoch, so a horizon limit cannot be excluded. Both predeclared point gates nevertheless failed. | The exact K1 reference predictions and deterministic resume evidence are absent; no paired mechanism or subgroup attribution is possible. |

The new paired diagnostics verify accepted prediction SHA256, finite values,
exact 50,000 ordered internal-development source indices, equal targets, and
sealed protected-role flags before calculating descriptive statistics. Their
same-role offset fits and target strata are explicitly post-hoc and do not
replace the owning scientific decisions.

## Coverage and remaining questions

The following table covers every desktop-owned RML directory in this snapshot.
`Effect` means an intervention's endpoint was measured under its stated
contract, not that its inner learning mechanism is known. `Diagnostic`
means a dedicated audit has bounded alternatives. `Blocked` identifies the
evidence needed before a stronger conclusion. `Reference/operations` records
do not each require a new module postmortem.

| Experiment directory or related group | Coverage | Decision-relevant limit |
| --- | --- | --- |
| `pcqm_500k_v4_evidence` | Effect + diagnostic | Matched local/global ablation attributes dense-attention cost and smaller K1-slot benefit; one seed and reused role limit generalization. |
| `pcqm_distance_angle_500k` | Effect | Combined geometry helped its legacy paired baseline; distance versus angle contributions and V5 comparability remain unknown. |
| `pcqm_edge_state_full` | Effect + diagnostic | Rich-feature repair is large; the full-scale architecture rank reversal is not explained by that repair alone. |
| `pcqm_geometry_component_attribution` | Diagnostic | Existing prediction-only NO_TRAIN audit identifies the missing strict 2D component comparator. |
| `pcqm_geometry_reliability_gate` | Diagnostic | Cross-fitted scalar fusion is exploratory on a previously selected development role. |
| `pcqm_geometry_transfer_500k` | Blocked | Geometry-model blend is positive context, but lacks matching V4 runtime/reference qualification. |
| `pcqm_gine_expert` | Diagnostic | Local 1M continuation isolated source-ordered BatchNorm-statistics drift in failed streaming trials; specialist transfer is a separate question. |
| `pcqm_gptrans_100k_transfer_control` | Diagnostic | Same checkpoints retain most joint gain on a second previously used cohort, ruling out cohort composition alone; train-size, horizon, selection and seed remain entangled. |
| `pcqm_gptrans_500k_frozen_readout` | Diagnostic | Frozen mean-atom head did not consistently beat a similarly sized virtual-only head; end-to-end readout bottleneck remains unproved. |
| `pcqm_gptrans_centered_logits_100k` | No scientific result | Canceled single arm produced no accepted metric. |
| `pcqm_gptrans_centered_logits_100k_kaggle1_pair` | Effect + new diagnostic | Weak endpoint, near-zero late relative lead, mixed rows and output shift; no exposure or overfitting proof. |
| `pcqm_gptrans_feature_denoising_100k` | Blocked + new diagnostic | Frozen source/optimizer mismatch prevents strict scientific attribution despite observed trace behavior. |
| `pcqm_gptrans_full_convergence` | Reference/operations | Continuation improved at the last budgeted evaluation; no plateau or equal-compute architecture conclusion. |
| `pcqm_gptrans_geometry_channels_100k_kaggle1` | Effect + diagnostic | Same-job angle increment hurt; invalid geometry and EMA lag explain only part. Distance-only versus pure 2D is contextual cross-job evidence. |
| `pcqm_gptrans_local_inductive_bias_100k_kaggle1` | Effect + new diagnostic | Small expensive local-edge gain, shrinking relative lead, stronger training fit; cause of transfer limit not isolated. |
| `pcqm_gptrans_memory_value_100k_kaggle1_pair` | Reference/operations | The proposed redundant baseline arms closed NO_TRAIN before submission. |
| `pcqm_gptrans_noisy_nodes_100k`, `pcqm_gptrans_noisy_nodes_500k` | Effect + diagnostic | Material 100K gain became sub-threshold at 500K; matched traces show early-to-late contraction, not a proven representation collapse. |
| `pcqm_gptrans_noisy_pair_norm_100k`, `pcqm_gptrans_noisy_pair_norm_500k` | Effect + diagnostic | Joint gain contracted and is sub-additive with isolated 500K effects; data diversity versus horizon remains unresolved. |
| `pcqm_gptrans_pair_memory_dual_100k_kaggle1` | Pending | Committed RML has two ACTIVE submitted arms. A local terminal draft is uncommitted; finish independent acceptance before causal postmortem or replay claim. |
| `pcqm_gptrans_pair_norm_500k` | Effect + diagnostic | Isolated 1.140 meV gain is real under its 500K pair, sub-threshold; joint mechanism and seed variance remain unresolved. |
| `pcqm_gptrans_t_100k_v4` | Reference/operations | Frozen desktop reference, not an unexplained new-module failure. |
| `pcqm_k1_full_convergence` | Reference/operations | Continuation failed to improve under its LR restart/patience contract; alternative schedules were not tested. |
| `pcqm_k1_gptrans_full_fusion` | Diagnostic | Fixed averaging explains most fusion gain; calibrated scalar adds little and still trails EdgeState on consumed official validation. |
| `pcqm_k1_pair_value_100k` | Blocked + new diagnostic | Last-epoch improvement is observed, but accepted row-paired K1 reference and deterministic resume are missing. |
| `pcqm_k1_residual_reconciliation` | Diagnostic | K1/GPTrans errors are complementary; simple graph descriptors explain little and do not authorize routing. |
| `pcqm_route_b` | Effect | Frozen specialist fusion passed its own role gate; it is not a comparable Track B V4 architecture mechanism test. |
| `pcqm_scale_transfer_attribution` | Diagnostic | Two early transfer examples motivated a stricter funnel; they do not establish a universal scale law. |
| `pcqm_scale_transfer_diagnostic` | Superseded interpretation | Exposure mapping did not prove representation collapse; its projected 500K gain failed and was corrected by the later reassessment. |
| `pcqm_scale_transfer_reassessment` | Diagnostic | Matched 500K traces establish late catch-up and gain erosion; unique data size, exposure, optimizer, selection and seed effects are still unidentified. |
| `pcqm_xian_determinism` | Blocked runtime | Gradient fingerprints diverged across processes despite matching forward results; the first divergent backward operation is not identified. |

## Before the next module

1. Finish canonical pair-memory terminal acceptance and any saved-artifact
   failure analysis. Do not count the uncommitted draft or fill absent trace
   fields by derivation.
2. For a future module, freeze a falsifier that distinguishes at least one
   plausible failure mode before running it: implementation/contract fault,
   optimization or exposure limit, training-fit versus development transfer,
   calibration shift, or added-path harm. Preserve a same-contract reference,
   fixed-role train/development observations, exposure coordinates and cost.
3. After terminal acceptance, write an attribution disposition beside the
   experiment decision before proposing another unrelated module. State what
   is measured, what is only consistent with the evidence, what is ruled out,
   and the cheapest missing discriminator. If the discriminator is unavailable,
   close as unidentified rather than inventing a cause or submitting a
   replacement.

This audit changes no frozen decision, RML trajectory outcome, or model
recommendation.
