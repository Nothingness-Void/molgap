# Curve and mechanism coverage review

Date: 2026-09-30. Saved-record synthesis only. No training, inference, remote
polling, new role access, candidate release or historical outcome revision.

## Scope and alignment

The [numeric extract](curve_mechanism_review.json) binds five trace files by
SHA256: immutable K1 plus MetaGIN2D, motif K1, atom-pair local and bond-type
local. All 40 observations per trace were checked for unique epoch, identical
optimizer step and sample presentations, and learning-rate difference at most
1e-12. Other scientific-contract matches inherit the owning accepted decisions.
This is a focused review, not a claim to have reaccepted every RML trajectory.

Epochs 9/19/29/39 are zero-based and correspond to 999,680 / 1,999,360 /
2,999,040 / 3,998,720 sample presentations. Compare raw live-model development
MAE in eV, not each model's best-so-far envelope. Training MAE uses normalized
target units; it is never directly subtracted from development eV.
The K1 reference is historical-partial: matching available observations does
not reconstruct missing stochastic state or estimate training-seed variance.

## What changed over exposure

Candidate minus K1 development MAE (eV); positive means worse:

| Candidate | ~1M | ~2M | ~3M | ~4M | Interpretation under this contract |
|---|---:|---:|---:|---:|---|
| MetaGIN2D | +0.030672 | +0.019777 | +0.019631 | +0.023065 | Persistently worse development fit despite stronger late training fit |
| Motif K1 | +0.005687 | -0.003671 | -0.000794 | +0.000342 | Mid-course gain eroded; no retained endpoint advantage |
| Atom-pair local | +0.004375 | -0.001075 | +0.002280 | +0.002641 | Brief gain, then stronger train fit with worse development |
| Bond-type local | -0.001122 | +0.004925 | +0.002241 | +0.001580 | Weak early signal did not persist |

At epoch39 K1 normalized train MAE was 0.077263; MetaGIN 0.062004, motif
0.075742, atom-pair 0.072665 and bond-type 0.074220. MetaGIN therefore was not
simply unable to fit the training role at the endpoint. That observation does
not identify whether regularization, inductive bias or the adaptation caused
the generalization deficit. Its selected epoch34 endpoint differs from this
matched epoch39 comparison; similarly atom-pair selected epoch35. Neither
comparison should silently replace the other.

The accepted [linear-attention curve](../../pcqm_k1_linear_attention_100k/decision.md)
provides a separate corroborating sequence: -0.009430/-0.007107/-0.000126/
+0.000399 eV at the same four epochs. The [joint reconstruction study](../../pcqm_k1_joint_atom_reconstruction_100k/decision.md)
also lost much of its lead over clean K1 between epochs19 and39. Its positive
contrast against *corruption-only* remained, so the auxiliary loss helped that
recipe; it did not prove superiority over clean K1.

## Three distinct failure classes

1. **Exposure-dependent relative catch-up.** Motif and linear exchange were
   better at intermediate checkpoints but not at the frozen terminal budget.
   A short screen can favor learning speed rather than retained accuracy.
   This does not justify changing selection epochs after seeing the curves.
2. **Cross-population instability before any extra training.** The
   [relation portability audit](../../pcqm_k1_relation_resolution_100k/audit/decision.md)
   changed only evaluation molecules with frozen 100K weights. Receiver,
   triplet and RRWP positives reversed. Additional optimizer steps or parameter
   exposure cannot explain that particular reversal. Population coverage and
   repeated-selection optimism remain plausible, not uniquely proven.
3. **Matched larger-scale generalization deficit.** In the
   [PairToken scale attribution](../../pcqm_k1_pair_token_scale_attribution/results/round3_decision.md),
   PairToken fit train better at matched epoch48 while losing to matched K1
   on development. Deleting its relation branch was harmful: the branch was
   used, but dependence is not incremental benefit over a trained comparator.

Do not pool these classes into one causal label such as insufficient exposure,
too many parameters, or universal slot compression. Different full-scale
recipes/roles are excluded from this aligned curve table. Existing full-run
desktop conclusions remain contextual here; no desktop jobs were inspected.

## Mechanism coverage and remaining questions

| Component | Retained tested interventions | Evidence boundary | Still unanswered; not a release |
|---|---|---|---|
| Local persistent bonds | [Storage/read normalization](../../pcqm_k1_edge_memory_100k/decision.md), [directional GPS++ adapters](../../pcqm_k1_gpspp_local_100k/decision.md), [chemistry partitions](../../pcqm_k1_chem_local_100k/decision.md) | Active modifications; gains absent or below gate; stronger train fit often failed to generalize | Whether task-useful relation directions change across scales, rather than merely their RMS |
| Global summary/content | [K1-G/K1-R](../../pcqm_k1_variants_100k/decision.md), [slot processor removal](../../pcqm_k1_slot_processor_100k/decision.md), linear attention and motif exchange | More flexibility was not consistently useful; simplification had small positive evidence | Relation between preserved node diversity and downstream prediction sensitivity |
| Return/addressing | [Uniform/inverse return](../../pcqm_k1_return_allocation_100k/decision.md), [shared selectors](../../pcqm_k1_readout_selector_100k/decision.md) | Uniform return directional below gate; inverse/tied choices did not establish a win | Which content direction matters; NOT permission for another scalar or return-only gate |
| Pair/topology representation | PairToken, receiver-pair, triplet and RRWP in the linked scale/portability decisions | Original-role benefits were not reliable across populations/scales | A relation effect that survives both later exposure and different molecules |
| Final readout | RepSet in the readout/selector decision | No material gain; interval included zero | No evidence identifying readout as the bottleneck |
| Composition | [Two simplifications](../../pcqm_k1_combined_simplification_100k/decision.md), [edge/slot interactions](../../pcqm_k1_edge_slot_interaction_100k/decision.md) | Partial or absent additivity; material gate failed | No authorization to combine all favorable-looking modules |
| Training objective | Joint atom reconstruction with continuous Gap supervision | Benefit conditional on corruption; complete recipe not portable over clean K1 | Clean-input auxiliary supervision was not isolated by that experiment, but remains only an untested question |
| Independent backbone | [MetaGIN2D](../../pcqm_metagin_2d_100k/attempt_v2/decision.md) | Exact adaptation failed broadly; does not falsify the paper family | Implementation/inductive-bias differences require separate fidelity analysis |

This map distinguishes an exact failed intervention from a whole rejected
research family. It is not a claim that all possible mechanisms were tested.
GPTrans author-alignment already has a separate owner; do not duplicate it.

## Statistical cautions and selection consequence

Hard/easy subsets defined from K1 absolute errors are target-selected. Another
model can appear to help the hardest and hurt the easiest partly through
selection/regression-to-the-mean. They do not establish chemical specialization
or an inference-time routing rule. The [structure-based residual follow-up](../../pcqm_k1_cross_scale_frozen/structural_residual_decision.md)
did not find a stable portable subgroup that resolves this limitation.

The finite reused development roles and single seed do not support a new
promotion threshold calibrated from these selected winners. Row bootstrap
does not measure run-to-run training variance. No old threshold was changed.

The review selected **no new training candidate**. It supported finishing the
already authorized representation diagnostic as a distinct measurement, then
checking whether rank/sensitivity changes align with paired residual changes.
If they do not, close that explanation instead of inventing another gate.
If they do, propose one new intervention only after checking this coverage map
and freezing its own comparison, budget, role and terminal obligations.

For future proposals, an intermediate-checkpoint gain alone is insufficient:
report the entire matched-exposure trend and the predeclared terminal result.
Where separately authorized, frozen500K inference distinguishes population
instability from retraining effects; it is not proof of 500K-trained superiority.
This is an interpretation rule, not an automatic new early-stop or release gate.
