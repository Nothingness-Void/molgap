# Deep Reading Batch 1: PCQM4Mv2, HOMO--LUMO Gap, Pretraining, Teachers, and Delta Learning

Date: 2026-09-07

This record is the first detailed-reading batch after the broader literature,
leaderboard, and public-asset audits. It is deliberately narrower than a
bibliography: every card below records a primary-source claim, the actual data
and geometry contract, the useful ablations or negative evidence, public
reproducibility assets, and the transfer decision for MolGap.

The project database contract is unchanged. Track B remains the official
PCQM4Mv2 data and Track A remains the repaired-2M PubChemQC corpus. External
datasets are not silently merged. The existing ETKDG train/inference rule also
remains unchanged. No experiment, pretrained initialization, remote job, or
production change was authorized by this reading batch.

## Evidence levels used in this record

- **A**: direct PCQM4Mv2/HOMO--LUMO evidence or an official benchmark/code
  artifact that can be checked against the project contract.
- **B**: a primary method result with a credible transfer mechanism, but a
  different geometry, target, split, scale, or downstream task blocks direct
  admission.
- **C**: useful design evidence or a discovery lead, but insufficient public
  provenance for a possible experiment.

The metric values below are not interchangeable. Paper validation, official
test-dev, challenge test, single-model, and ensemble numbers are kept separate.

## 1. Benchmark contract: OGB-LSC / PCQM4M / PCQM4Mv2

**Primary reading.** [OGB-LSC paper](https://arxiv.org/html/2103.09430) and
[the official PCQM4Mv2 specification](https://ogb.stanford.edu/docs/lsc/pcqm4mv2/).

**Question and actual task.** The benchmark defines graph regression of the DFT
HOMO--LUMO gap in eV from a molecular graph, with MAE as the metric. The
original PCQM4M has 3,803,453 graphs; the paper explains that PCQM4Mv2 adds
DFT-calculated 3D structures for training molecules while keeping validation
and test inputs 2D-only. The paper also documents the practical constraint:
inference, including preprocessing, should remain below 0.1 s per molecule and
the 377,423-molecule test set must fit the 12-hour challenge window.

**Data and labels.** The task is not an experimental optical gap. It is a
PubChemQC DFT HOMO--LUMO gap target, reported in eV. The original paper says
the released graph uses atom and bond features derived from SMILES and that the
split is by PubChem CID at 80/10/10. The v2 update corrected the graph/3D
mismatch and exposed DFT geometry only for training roles; validation and test
geometry is intentionally unavailable.

**What the benchmark actually teaches.** The challenge winners were large and
deep, used global information flow, and used 3D structure to supervise the
model. This is evidence that global communication, scale, and privileged
geometry can matter. It is not evidence that a DFT conformer may be used by a
MolGap model at inference. It also explains why the project keeps official
validation/test-dev sealed and treats DFT geometry as a distinct role.

**MolGap disposition.** **A, contract anchor.** Use this paper to reject any
candidate that changes the target, silently mixes PCQM4M with PCQM4Mv2, uses
sealed-role geometry, or reports a non-comparable split. The project’s ETKDG
route is a deliberate deployment contract, not the original challenge’s only
possible training setup.

## 2. Graphormer: global topology without explicit 3D input

**Primary reading.** [Graphormer paper](https://arxiv.org/html/2106.05234) and
[official implementation](https://github.com/microsoft/Graphormer).

**Question and mechanism.** Graphormer asks whether a standard Transformer can
be made graph-aware by placing graph statistics in the attention computation.
It adds centrality encodings, shortest-path distance as an attention bias, and
edge encodings aggregated along the shortest path. A virtual graph token gives
the model a global readout. The paper also proves that the construction can
simulate common message-passing models and can be more expressive than 1-WL in
the relevant settings.

**PCQM evidence.** The paper reports a PCQM4M result on the original large
benchmark and states that Graphormer improves over mainstream GNN baselines by
more than 10% relative. This is an original-graph result, not automatically a
PCQM4Mv2/ETKDG result. Later PCQM4Mv2 systems use the same family of global
distance/edge biases, which makes Graphormer an important mechanism baseline.

**Ablations and limits.** The useful ablation is architectural rather than a
pretraining result: centrality, shortest-path bias, and edge-path encoding are
separate sources of graph information. The main cost is dense attention and
all-pairs structural preprocessing. A strong Graphormer score therefore does
not establish that full global attention is necessary; GPS++ later tests that
assumption directly.

**Reproducibility.** The paper and Microsoft repository are public. The
repository is a reference implementation, not a MolGap drop-in: its dependency
versions, preprocessing, parameter scale, and official PCQM role handling must
be audited before execution.

**MolGap disposition.** **A, reference only.** Graphormer validates the value
of explicit global structural biases, but the active GraphState line has
already tested global/local allocation under the project’s own budget. Do not
reopen it as a generic architecture transplant.

## 3. GPS / GraphGPS / GPS++: global attention is not automatically necessary

**Primary reading.** [GraphGPS paper](https://arxiv.org/abs/2205.12454),
[GPS++ paper](https://arxiv.org/abs/2302.02947),
[GraphGPS code](https://github.com/rampasek/GraphGPS), and the
[PCQM4Mv2 GPS++ code and checkpoints](https://github.com/graphcore/ogb-lsc-pcqm4mv2).

**Question and mechanism.** GPS combines local message passing with a global
attention or global positional mechanism. GPS++ specifically studies the
PCQM4Mv2 scale problem and couples a message-passing network with Transformer
components rather than assuming that every pair of atoms should interact at
every layer.

**Direct PCQM evidence.** The official PCQM4Mv2 repository publishes several
model sizes and validation expectations, roughly 0.090, 0.082, and 0.077 eV for
the public 11M, 22M, and 44M configurations. Its challenge record also uses a
112-model ensemble to reach a much lower test-challenge number. The important
comparison is the ablation: the paper reports that nearly all performance can
be retained without global self-attention, while message passing remains
competitive and a larger no-3D model can be more accurate. These are not
single-run MolGap claims; they are evidence about information-flow efficiency.

**Negative evidence and resource lesson.** Dense global attention is expensive
and not guaranteed to beat a well-designed local path. The public configs and
checkpoints are useful because they expose realistic depth/width/resource
choices, but the strongest challenge score is an ensemble and is not a fair
single-model comparison to a 12-hour A100 candidate.

**Reproducibility.** Code and checkpoints are public. The repository uses its
own dataset loader, feature schema, and training conventions; importing the
checkpoint would also violate the project’s random-initialization architecture
screen.

**MolGap disposition.** **A, upper-bound/reference.** This is the strongest
public evidence for keeping local state and global readout separate. It
supports the project’s GraphState design and the decision not to spend the
remaining architecture budget on an unbounded dense-attention transplant.

## 4. Transformer-M: supervised 2D/3D training with 2D-only inference

**Primary reading.** [Transformer-M paper](https://arxiv.org/html/2210.01765)
and [official implementation](https://github.com/lsj2408/Transformer-M).

**Question and mechanism.** Transformer-M uses separate 2D and 3D channels in
one Transformer. It can accept either modality, and the channels can be
enabled or disabled according to the deployment setting. The crucial PCQM
setup jointly trains with 2D and 3D information while evaluating the deployed
path with the 2D channel only.

**Direct result.** The paper reports 0.0787 eV on its PCQM4Mv2 validation
comparison for a 47.1M-parameter model, against approximately 0.0864 for its
Graphormer comparison and 0.0858 for GraphGPS-base. The paper attributes the
gain to joint 2D/3D supervised training, not merely to a new 2D attention
block.

**What is and is not proven.** This is a strong teacher/privileged-training
precedent. It does not prove that an ETKDG-only model improves, because the
training signal includes target-associated 3D information and the target Gap
is part of the supervised objective. It also uses a much larger model than
the active screen.

**MolGap disposition.** **A for precedent, B for direct transfer.** Keep it as
evidence for a future separately authorized teacher/geometry track. It is not a
fair random-init architecture candidate under the current contract and must
not be described as one.

## 5. ViSNet and the PCQM challenge teacher/student experiment

**Primary reading.** [ViSNet paper](https://arxiv.org/html/2210.16518),
[PCQM challenge report](https://ogb.stanford.edu/paper/neurips2022/pcqm4mv2_ViSNet.pdf),
and [challenge ensemble report](https://arxiv.org/html/2211.12791).

**Question and mechanism.** ViSNet uses a runtime-generated geometric graph
(RGC) and scalar/vector interactions. Angles, dihedrals, and improper angles
are computed with linear-complexity geometric operations. The model is
designed to encode directional information without building a full tensor
product network.

**Direct PCQM evidence.** The challenge report gives a vanilla ViSNet result
of 0.0216 eV MAE on an 8:1:1 held-out split of the training data when trained
with optimized coordinates. It then addresses deployment without optimized
coordinates: freeze a teacher trained on optimized structures, train a student
on RDKit-generated structures, and align graph representations with InfoNCE or
L1 loss. The report also finds that removing runtime angle/dihedral features
hurts; dihedral terms contribute slightly more, and improper angles matter
more for larger molecules.

**Critical contract detail.** PCQM4Mv2 supplies optimized geometry for training
roles but not validation/test. RDKit-generated coordinates are therefore a
student/deployment workaround, not proof of ETKDG consistency. The challenge
report’s teacher is a direct example of privileged geometry distillation, but
its teacher geometry is not the current MolGap inference geometry.

**MolGap disposition.** **A for teacher evidence, B for implementation.** The
most transferable part is frozen node-level or graph-level geometric
representation alignment. A legal MolGap adaptation must regenerate both
teacher and student geometry with the project’s ETKDG construction, or obtain
a separately approved geometry contract. The optimized-geometry checkpoint
cannot be imported as an ordinary warm start.

## 6. TGT: triplet interaction and learned distance geometry

**Primary reading.** [TGT paper](https://arxiv.org/html/2402.04538) and
[official code](https://github.com/shamim-hussain/tgt).

**Question and mechanism.** TGT adds third-order triplet interaction to pair
channels, moving the model toward 2-WL-like expressivity. Its pipeline has
three stages: a 2D graph predicts binned/clipped all-pair distances, a task
network is trained on noisy 3D structures, and the task model is fine-tuned on
predicted distances. The full distance matrix determines a geometry up to
reflection but not chirality.

**PCQM results.** The paper reports approximately 68.6 meV validation and
69.8 meV test-dev for TGT-At, with RDKit assistance reaching about 67.1 meV
validation and 68.3 meV test-dev. The comparison table lists Uni-Mol+ around
69.3/70.5 meV and GPS++ around 77.8/72.0 meV, but these are different model,
split, and geometry contracts.

**Ablation evidence.** In the smaller ablation, the baseline is about 85.1
meV; adding RDKit coordinates, denoising, local smoothing, source dropout,
DFT pretraining, and a DFT distance predictor yields successive improvements.
The reported full ablation also attributes a substantial gain to triplet
aggregation/attention. The paper states a complexity of at least
approximately O(N^2.37), higher than O(N^2), as a practical limitation.

**MolGap disposition.** **A for mechanism evidence, B for admission.** TGT
supports pair-distance denoising, triplet interaction, and learned geometry as
real mechanisms. However, the architecture is expensive, uses privileged DFT
distance/geometry supervision, and overlaps with the project’s already closed
distance/torsion routes. Keep it as a design reference, not an automatic
successor job.

## 7. Uni-Mol+: geometry refinement from cheap conformers toward DFT geometry

**Primary reading.** [Uni-Mol+ paper](https://www.nature.com/articles/s41467-024-51321-w),
[Uni-Mol code](https://github.com/deepmodeling/Uni-Mol/), and
[released assets](https://zenodo.org/records/12670462).

**Question and mechanism.** Uni-Mol+ starts from cheap RDKit conformers and
uses a two-track atom/pair Transformer to refine the representation toward an
equilibrium geometry before property prediction. The implementation generates
eight initial conformers per molecule with RDKit ETKDG followed by MMFF94
optimization, uses a fallback flat structure when conformer generation fails,
and averages the eight Gap predictions at inference.

**Direct PCQM evidence.** The reported model uses DFT equilibrium structure as
the refinement target and the later TGT comparison lists roughly 69.3 meV
validation and 70.5 meV test-dev for Uni-Mol+. The paper reports an 18-layer
model as its best configuration and an 11.4% improvement over its previous
reference in the stated comparison.

**What is transferable.** The useful design is not the exact checkpoint; it is
the separation between a cheap conformer proposal, a geometry correction
module, and property prediction. It also provides a concrete conformer-ensemble
failure policy and a measurable multi-conformer cost.

**Contract risk.** The training objective uses DFT-optimized structure, while
inference begins with RDKit/MMFF conformers and then refines them. That is a
privileged geometry route rather than the current same-ETKDG training/inference
contract. Eight conformers and an 18-layer 77M-scale model also challenge the
12-hour budget.

**MolGap disposition.** **B.** Use as geometry-teacher/refinement evidence
only. A future same-database experiment would need a CPU-only ETKDG cost and
failure audit before any GPU screen; it must state whether refinement predicts
geometry, representation, or Gap and must compare one frozen candidate at a
time.

## 8. GraphMVP: 2D/3D cross-view pretraining without 3D at downstream inference

**Primary reading.** [GraphMVP paper](https://arxiv.org/html/2110.07728) and
[official code](https://github.com/chao1224/graphmvp).

**Question and mechanism.** GraphMVP treats the same molecule as a 2D graph and
a 3D geometry. It combines a contrastive objective, where matched 2D/3D views
are positives, with a generative variational representation reconstruction
objective. The latter uses a stop-gradient target and a KL-style regularizer.
The stated goal is to let 3D geometry improve the 2D representation while
discarding 3D at downstream inference.

**Evidence and limitation.** The paper demonstrates improvements on several
2D downstream tasks, not a direct PCQM4Mv2 Gap architecture comparison. The
3D view is privileged during pretraining and the released code is built around
older PyTorch/PyG versions. Therefore, the paper validates the role separation
idea, not the final MolGap metric.

**ETKDG adaptation.** A legal adaptation can replace the privileged DFT view
with an ETKDG-generated view, but that is a new hypothesis. The exact positive
pair, coordinate normalization, noise policy, and downstream student input
must use the same ETKDG implementation as inference.

**MolGap disposition.** **B.** Retain as the cleanest conceptual template for
geometry-to-2D pretraining. No experiment is admitted from the paper without a
same-geometry cache and a role manifest.

## 9. GeoSSL-DDM: geometry denoising expressed as pair-distance denoising

**Primary reading.** [GeoSSL-DDM paper](https://arxiv.org/html/2206.13602).

**Question and mechanism.** GeoSSL introduces 3D coordinate denoising as a
mutual-information proxy. GeoSSL-DDM uses an SE(3)-invariant score-matching
derivation to turn coordinate denoising into denoising pairwise distances at
multiple noise levels. The paper explicitly decomposes the coordinate score
into pair-distance scores, so the learned quantity has a clear physical
interpretation as a sum of pairwise pseudo-forces.

**Evidence.** The study evaluates 22 geometric downstream tasks and reports
improvements over nine pretraining baselines. Its pretraining/downstream
settings are QM9, MD17, LBA, and LEP rather than the official PCQM4Mv2 Gap
screen. It is therefore not direct evidence for a PCQM score.

**Negative and implementation evidence.** The method requires all-pair
geometric distances, SE(3)-compatible score construction, and a noise schedule.
It is more expensive than a purely local graph pretext task. It also assumes a
geometry sample around a conformer/local energy minimum; using DFT equilibrium
coordinates versus ETKDG coordinates changes the meaning of the denoising
task.

**MolGap disposition.** **B.** This is a credible source for a future
ETKDG-only distance-denoising pretraining objective and for the bottom-fusion
distance representation. It is not evidence that the current architecture
screen should be reopened.

## 10. Unified 2D and 3D Pre-Training of Molecular Representations

**Primary reading.** [UnifiedMolPretrain paper](https://arxiv.org/html/2207.08806)
and [official repository](https://github.com/teslacool/UnifiedMolPretrain).

**Question and mechanism.** The model jointly processes 2D and 3D information
in one network. Its pretraining tasks are masked atom/coordinate reconstruction,
2D-to-3D conformation generation, and 3D-to-2D graph generation. It uses a
permutation-invariant coordinate loss for symmetric substructures and a
roto-translation invariant loss for conformation generation.

**Data and result boundary.** The paper pretrains on about 3.38M PCQM4Mv2
molecules and randomly splits that pretraining collection 95/5. Its reported
downstream tests are six MoleculeNet tasks, OGB-molpcba, and toxicity tasks;
it does not establish a direct official PCQM4Mv2 Gap score for the pretrained
model. The model can accept 2D-only molecules by randomly initializing
coordinates, but that compatibility is not the same as a stable ETKDG
deployment path.

**MolGap disposition.** **B.** The symmetry-aware reconstruction loss and the
2D/3D task separation are useful implementation ideas. The PCQM data use,
however, is pretraining evidence and not a permission to use DFT geometry or
to add a new dataset role to MolGap.

## 11. Frad: fractional denoising of dihedrals and coordinates

**Primary reading.** [Frad paper](https://www.nature.com/articles/s42256-024-00900-z),
[official code](https://github.com/fengshikun/FradNMI), and
[released pretrained models](https://zenodo.org/records/12697467).

**Question and mechanism.** Frad argues that ordinary isotropic coordinate
denoising under-covers molecular conformational variation and learns an
isotropic rather than chemically directional force field. It introduces a
hybrid noise strategy over dihedral angles and coordinates, then derives a
fractional denoising objective that decouples the angle-noise and coordinate-
noise parts while preserving the force-field interpretation.

**Evidence.** The primary paper reports state-of-the-art results on 9/12 QM9
targets and 7/8 MD17 targets. Its released training configurations include
PCQM4Mv2 pretraining and the paper/data availability statements confirm public
code, PCQM data, and pretrained models. This is strong reproducibility evidence
for a denoising method, but not a direct PCQM4Mv2 Gap leaderboard result.

The expanded NMI study separates two versions of the method: RN perturbs
rotatable-bond torsions and VRN also perturbs bond lengths, bond angles, and
non-rotatable torsions. It reports matched QM9 HOMO/LUMO/Gap values of
`15.3/13.7/27.8` meV for RN and `17.9/13.8/27.7` meV for VRN, against
`17.7/14.3/31.8` meV for coordinate denoising. Its key causal detail is the
intermediate/final construction `x_eq -> x_med -> x_fin`: the model predicts
only the final coordinate-Gaussian component `x_fin - x_med`, not the entire
hybrid displacement. The paper's robustness control uses RDKit Distance
Geometry plus MMFF rather than the project's ETKDG cache, so it is a
robustness signal, not geometry equivalence.

**Contract risk.** The public pipeline expects generated coordinates and its
historical environment is tied to older Torch/PyG versions. The published
geometry route is not yet proven to be the project’s exact ETKDG construction
at both train and inference.

**MolGap disposition.** **B.** A future ETKDG-only torsion/coordinate denoising
screen could borrow the objective, but the active torsion route is already
closed and a new screen would need a genuinely distinct hypothesis plus a
fresh control. The checkpoint is not an admissible warm start without an
explicit contract and hash/forward-pass audit.

For the full RN/VRN theory, ring/degree handling, public artifact hashes, and
the separate ICML/NMI contract audit, see the [continued denoising deep-reading
card](deep_reading_denoising_continuation_2026-09-07.md#6-fractional-denoising-frad-chemical-aware-noise-without-breaking-the-score-objective).

## 12. Self-Conditioned Denoising (SCD): a recent PCQ pretraining result

**Primary reading.** [SCD paper](https://arxiv.org/html/2603.17196v1),
[official code](https://github.com/TyJPerez/SelfConditionedDenoisingAtoms), and
[PCQ checkpoint](https://huggingface.co/Ty-Perez/ct-scd-pcq).

**Question and mechanism.** SCD computes a clean-geometry embedding and uses
that embedding to condition denoising of a corrupted geometry. The model then
uses one pass for downstream prediction. The paper implements conditional
normalization/gating and compares conditional TorchMD-Net (CT) and conditional
geometric TorchMD-Net (CGT).

**Direct PCQ evidence.** The paper uses 3.378M PCQ molecules with B3LYP/6-31G*
DFT geometry for pretraining. On QM9 downstream evaluation, the same-PCQ
comparison reports Gap MAE 24.5 meV for CT-SCD versus 31.8 meV for ordinary
coordinate denoising and 36.1 meV for its no-pretraining baseline; CGT-SCD is
reported at 19.7 meV in the comparison table. These are downstream QM9
measurements, not PCQM4Mv2 official validation/test-dev results.

**Useful ablations.** The paper reports gains across most QM9 targets and
compares SCD with Frad and SliDe. It also finds that a random 10% PCQ slice
(about 300k molecules) captures 97% of the full-PCQ pretraining gain for the
reported QM9 tasks. The paper’s speed table shows CT at roughly 9.2M parameters
and about 802 samples/s training in its benchmark, while the geometric CGT is
roughly twice as large and about five times slower; these are not MolGap
throughput numbers but are useful budget evidence.

**Contract risk.** The public PCQ checkpoint expects the paper’s coordinate
pipeline and a different downstream geometry contract. Directly loading it
would mix DFT-pretraining geometry with MolGap ETKDG inference. The method can
be adapted by generating the clean and corrupted coordinates from ETKDG, but
that adaptation has not been shown by the paper.

**MolGap disposition.** **B, highest-priority future pretraining lead.** The
paper, code, and checkpoint are sufficient for an evidence-only smoke/hash
audit. They are not sufficient to authorize a run or to claim a benefit under
MolGap until ETKDG-only preprocessing and a fresh random-init control are
defined.

## 13. Denoise-and-Distill (D&D): 3D geometry teacher to 2D student

**Primary reading.** [AAAI paper page](https://ojs.aaai.org/index.php/AAAI/article/view/31986),
[paper PDF](https://ojs.aaai.org/index.php/AAAI/article/download/31986/34141).

**Question and mechanism.** D&D pretrains a 3D conformer denoiser, freezes the
3D teacher, and distills its representation into a deployable 2D student. It
provides graph-level alignment by mean pooling (D&D-GRAPH) and atom-level
one-to-one alignment (D&D-NODE). The PCQM collection is used for the
pretraining/distillation source, and the paper states that PCQM Gap labels are
not used in the denoising pretraining.

**Evidence.** Across its MoleculeNet evaluation, the paper reports gains on 9/10
tasks and average improvements of about 4.6% for classification and 18.6% for
regression. Those are downstream transfer results, not a direct PCQM Gap
leaderboard result. The node-level alignment is the more relevant design for
MolGap because graph atoms and conformer atoms have an explicit identity map.

**Hard blocker.** D&D pairs the student with DFT lowest-energy conformers. A
student distilled from DFT-teacher states and deployed from ETKDG would violate
the current train/inference geometry rule. The paper also has no official
checkpoint or code artifact identified in this audit.

**MolGap disposition.** **B.** Keep as the main teacher/student protocol
template. A legal adaptation would use ETKDG for both teacher and student, or
would require a separately approved privileged-geometry contract. Teacher
property outputs should not be used unless generated out-of-fold with a strict
role manifest.

## 14. Atom-level quantum-property pretraining versus molecular Gap pretraining

**Primary reading.** [Fallani et al., Journal of Cheminformatics (2025)](https://link.springer.com/article/10.1186/s13321-025-00970-0).

**Question and comparison.** This study compares Graphormer pretraining on
atom-level quantum properties, on a molecular HOMO--LUMO gap, and on atom
masking. It uses a public atom-resolved QM dataset with charges, NMR shielding,
and electrophilic/nucleophilic Fukui indices, while the molecular pretraining
uses PCQM4Mv2 HLG. The input is explicitly 2D only even though the source data
also contain geometry.

**Actual evidence.** The downstream evaluation is 22 TDC ADMET tasks, not
PCQM4Mv2 Gap. The paper reports that atom-level pretraining supplies the best
model or a tie on most of the benchmark tasks and performs better than HLG
pretraining on its larger internal microsomal-clearance dataset. It also
analyzes retained pretraining information, receptive-field sensitivity, and
the alignment of attention-rollout modes with graph-Laplacian eigenmodes.

**Interpretation.** The paper does not prove that charges are always better
than Gap labels. It does provide a controlled warning: a molecular target can
be too global or too task-specific to provide the most transferable local
representation. It also provides a concrete evaluation pattern for a future
electronic auxiliary teacher: compare local quantum properties, molecular Gap,
and masking under the same backbone and downstream protocol.

**MolGap risks.** The atom-level source has a different functional/basis and
potential molecular overlap risk. Importing its labels would be an external
teacher/data role, not database augmentation. Charges, Fukui indices, and NMR
shielding are not interchangeable with the PCQM B3LYP/6-31G* target.

**MolGap disposition.** **B.** Retain as evidence for a conditional local-
electronic teacher route. Before any experiment, require source license,
canonical identity/conformer overlap audit, theory metadata, and a teacher-only
role. Do not add the source to the main training database.

## 15. DelFTa and the delta-learning pattern

**Primary reading.** [DelFTa code](https://github.com/josejimenezluna/delfta),
[documentation](https://delfta.readthedocs.io/en/latest/), and
[delta-QML paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC9093086/).

**Question and mechanism.** DelFTa compares direct learning with delta
learning against a cheap GFN2-xTB baseline. The final prediction is

```text
b(x)       = cheap baseline property
r(x)       = learned residual y(x) - b(x)
y_hat(x)   = b(x) + r(x)
```

The paper reports that delta learning improves most QMugs endpoints when the
low-level baseline is correlated with the higher-level reference. The public
toolbox supports HOMO, LUMO, Gap, dipoles, charges, and bond orders and is
therefore a useful implementation reference.

**Contract mismatch.** The released target is omegaB97X-D/def2-SVP on QMugs,
not the PCQM B3LYP/6-31G* Gap. The geometry and charge/spin conventions also
do not establish the MolGap ETKDG contract. A released checkpoint must not be
treated as a PCQM predictor.

**Database-preserving adaptation.** A future MolGap delta audit could compute a
low-fidelity proxy for the existing PCQM molecules from the same ETKDG
coordinates and learn only the PCQM residual on the allowed train split. The
first gate must be CPU-only and measure proxy coverage/failure rate, MAE and
correlation to PCQM Gap, residual variance, stratified residual behavior, cost,
and train/inference coordinate identity. A learned baseline must use out-of-
fold outputs; same-row in-sample predictions are leakage.

**MolGap disposition.** **B, conditional method lead.** DelFTa is strong
evidence for the delta pattern and public code, but it does not authorize a
new experiment. The residual gate must pass before a paired seed-42 direct-vs-
delta screen is written.

## 16. DeMol: recent dual atom/bond model with target-label pretraining

**Primary reading.** [DeMol paper](https://arxiv.org/html/2603.00568) and the
[author-linked repository](https://github.com/LiuYunqing/DeMol).

**Question and mechanism.** DeMol uses separate atom-centric and bond-centric
graphs with atom--bond and bond--bond interaction blocks. It adds masked atom
prediction, coordinate recovery, and bond prediction as auxiliary objectives.
The paper’s PCQM experiment states that Gap prediction is included in
pretraining and that evaluation does not use an additional fine-tuning stage.

**Reported result and comparability.** The paper reports about 0.0603 eV on its
PCQM4Mv2 validation comparison for a 186M-parameter model, ahead of the listed
TGT-At comparison. It is a large-model and target-label-pretraining result,
not a random-init architecture comparison. The author-linked GitHub URL did
not expose a usable implementation or checkpoint during this audit.

The appendix makes the scale and contract concrete: 12 layers, 768-dimensional
atom and bond states, 128 Gaussian kernels, AdamW with peak learning rate
`2e-4`, batch size `1024`, 1.5M steps, 150K warmup, EMA `0.999`, about seven
days on eight A6000 GPUs, and an ablation-level inference estimate of about
34 ms/molecule versus about 28 ms for Transformer-M. The model uses PCQM
training 3D structures, explicit pair distances, bond midpoints, bond angles
and dihedrals, and a covalent-radii threshold of `1.15` for its auxiliary bond
target. Its reported no-pretraining/pretraining QM9 Gap comparison is
`30.4` versus `26.8` meV, but that is a downstream transfer result, not a
second PCQM Gap result. The official author-page link
[`github.com/LiuYunqing/DeMol`](https://github.com/LiuYunqing/DeMol) returned
404 during this audit, which is stronger negative reproducibility evidence than
merely finding no link in the paper.

**MolGap disposition.** **C for direct use, B for design reading.** The
atom/bond interaction idea is partly covered by the existing EdgeState line,
and the lack of a usable artifact prevents reproducibility. The target-label
pretraining result can be monitored, but it cannot justify a current remote
screen.

## 17. DGT: dual graph Transformer as a design reference

**Primary reading.** [DGT code](https://github.com/zhangsy-ryan/DGT) and
[paper](https://www.nature.com/articles/s41467-026-75005-9).

**Question and mechanism.** DGT maintains atom and bond graphs separately and
uses cross-level interactions, with optional 3D descriptors. The public code
and paper provide a concrete atom/bond implementation and QM9 ablations.

**Evidence boundary.** The available primary evidence does not establish a
direct PCQM4Mv2 Gap result under the MolGap ETKDG and 12-hour contract. The
bond-state mechanism overlaps substantially with the existing persistent
EdgeState/GraphState experiments. Reimplementing the whole model would not
isolate a new causal information-flow hypothesis.

**MolGap disposition.** **B for design reference, not a queue item.** A
specific atom--bond interaction could be reconsidered only if it is formulated
as a new, non-overlapping hypothesis with a parameter/throughput gate.

## 18. MoleculeSDE / GraphMVPv2: data-space 2D--3D diffusion pretraining

**Primary reading.** [MoleculeSDE paper](https://arxiv.org/html/2305.18407),
[official repository](https://github.com/chao1224/MoleculeSDE),
[Geom3D model collection](https://github.com/chao1224/Geom3D), and the
[released checkpoint tree](https://huggingface.co/chao1224/MoleculeSDE/tree/main).

**Question and mechanism.** MoleculeSDE asks whether 2D--3D mutual information
can be estimated in the input space rather than only by matching compressed
representations. It combines an EBM-NCE contrastive term with two generative
SDE objectives: topology-to-conformation and conformation-to-topology. The
first is an SE(3)-equivariant, reflection-antisymmetric diffusion built from
local frames; the second is SE(3)-invariant because it generates discrete atom
and bond structure. The paper explicitly describes the three objectives as one
contrastive and two generative components.

**Data and implementation evidence.** The paper uses about 3.4M paired
PCQM4Mv2 topologies and conformations for pretraining. The repository names
the expected raw files (`data.csv`, `data.csv.gz`,
`pcqm4m-v2-train.sdf`, and its archive) and exposes a pretraining command.
Its reference environment is Python 3.7, PyTorch 1.9.1, PyG 2.0.2, and OGB
1.2.1. The repository is MIT-licensed and links both a checkpoint index and a
Hugging Face model tree; the checkpoint index maps VE/VP variants to manuscript
tables. This is stronger implementation evidence than a paper-only proposal,
but it is an old dependency stack and no local smoke test has been run here.

**Numerical evidence and its boundary.** In the paper's QM9 ablation using
110K/10K/11K train/validation/test, the Gap MAE is 44.64 meV for GraphMVP VRR,
41.84 meV for SDE-VE, and 42.75 meV for SDE-VP. HOMO is 27.32/25.79/25.84
meV and LUMO is 22.50/21.63/21.52 meV for the same three rows. The paper also
reports the generative-only PCQM-pretrained model beating the VRR baseline on
the listed QM9/ADMET comparisons. These are QM9 and transfer results, not a
direct PCQM4Mv2 Gap validation or official test-dev score.

**Geometry contract.** The released pretraining path consumes the PCQM paired
3D conformations, while MolGap requires ETKDG for every train/inference graph
that uses geometry. Therefore the published checkpoint cannot be dropped into
the current contract. A legal future adaptation would be a new ETKDG-only
pretraining protocol, with a no-pretraining control and an explicit ablation
for the 2D-only inference path; it would not be a continuation of the fresh
random-init architecture screen.

**MolGap disposition.** **B, high-value objective and code reference; no
current experiment.** The strongest transferable idea is direct denoising of
geometry/topology rather than a representation-space proxy. The evidence does
not establish a current-database Gap gain, and the published geometry is not
ETKDG-consistent.

## 19. MoleculeJAE: joint trajectory auto-encoding

**Primary reading.** [NeurIPS 2023 paper](https://papers.neurips.cc/paper_files/paper/2023/hash/acddda9cd6f310689f7657f947705a99-Abstract-Conference.html)
and [full paper](https://arxiv.org/html/2312.03475).

**Question and mechanism.** MoleculeJAE models a joint trajectory for 2D bond
connections and 3D conformations. Its objective is a weighted sum of trajectory
score matching and a contrastive surrogate. An encoder sees the original
molecule and conditions a second encoder on the noised molecule; an equivariant
decoder produces a 3D vector score and a 2D invariant bond score. This is a
more explicit “original structure plus noisy structure” teacher signal than a
single masked-coordinate loss.

**Data and results.** Pretraining uses 3.4M PCQM4Mv2 molecules with paired 2D
topology and 3D geometry, using SchNet for the 3D backbone and a bond-focused
2D GNN. On the QM9 110K/10K/11K split, the random-init Gap/HOMO/LUMO rows are
44.13/27.64/22.55 meV, while MoleculeJAE reports 42.73/25.95/21.55 meV. The
paper reports improvement over the listed baselines on 9 of 12 QM9 quantum
tasks and 15 of 20 geometry-related tasks overall. Its contrastive-loss
ablation is important: the reported $λ_2=0.01$ row is the selected setting,
whereas $λ_2=1$ worsens Gap/HOMO/LUMO to 45.45/28.23/23.67 meV. This is
evidence that the extra contrastive term is sensitive, not a free improvement.

**Evidence boundary.** The reported quantum results are QM9, not PCQM4Mv2
Gap. The paper's public record did not expose an author-linked implementation
or checkpoint during this audit; the paper therefore supplies a method and
ablation reference, not a ready-to-run artifact. Its coordinate pretraining
also uses paired PCQM 3D structures rather than MolGap ETKDG coordinates.

**MolGap disposition.** **B/C, method reference only.** If revisited after
architecture selection, the safe hypothesis is a small ETKDG-only trajectory
denoising auxiliary objective, with $λ_2$ treated as a separately gated
hyperparameter and with a matched no-contrastive control. It is not evidence
for a direct target improvement and is not authorized now.

## 20. MoleBlend: atom-relation-level modality blending

**Primary reading.** [ICLR 2024 paper](https://arxiv.org/html/2307.06235)
and [official implementation](https://github.com/YudiZh/MoleBlend).

**Question and mechanism.** MoleBlend treats atom relationships as the common
anchor between 2D and 3D. It randomly blends shortest-path distance, bond-edge
type, and 3D Euclidean-distance relation vectors into one attention-bias
matrix, then predicts the modality-specific relation matrices. A noisy-node
coordinate-denoising term is added as regularization. The reported backbone is
a 12-layer, 768-dimensional, 32-head Transformer with 128 Gaussian kernels;
pretraining uses a 2:2:6 SPD:edge:3D blending ratio, batch 4096, peak LR
`1e-5`, 1M steps, and 100K warmup.

**Evidence and reproducibility.** The paper states that pretraining uses
3.37M paired PCQM4Mv2 molecules and evaluates 2D, 3D, and PCQM Gap tasks. Its
public code provides a pretraining shell, a pretrained-model link, and a
four-A100 training note, but pins an obsolete PyTorch 1.7.1/PyG 1.6.3 stack.
The linkable main tables expose the QM9 result, where MoleBlend reports Gap
34.75 meV, HOMO 21.47 meV, and LUMO 19.23 meV, plus the MoleculeNet results.
The HTML version does not expose a directly checkable PCQM4Mv2 Gap number or
split/role table, despite mentioning that task in the experimental setup.

**MolGap disposition.** **B for relation-level pretraining, C for direct PCQM
performance.** The useful design lesson is to align pairwise channels before
fusion rather than only align pooled molecule embeddings. The published
pretraining coordinate path is PCQM geometry, not ETKDG; the training scale
and legacy environment are also far outside the current architecture screen.
No MoleBlend score is admitted as a MolGap comparator, and no database or
checkpoint is imported.

## 21. FlexMol: paired-to-unpaired multimodal pretraining

**Primary reading.** [FlexMol paper](https://arxiv.org/html/2510.07035) and
[official repository](https://github.com/tewiSong/FlexMol).

**Question and mechanism.** FlexMol uses a two-stage pipeline. Stage 1 learns
from paired 2D/3D PCQM4Mv2 data with separate encoders, shared attention
parameters, InfoNCE alignment, and 2D-to-3D/3D-to-2D feature decoders. Stage 2
continues on single-modality Uni-Mol data and uses the decoder to reconstruct
the missing modality. In addition to masked atom and coordinate recovery, the
paper adds shortest-path-distance prediction as a 2D self-supervised target.

**Reported scope.** Stage 1 uses about 3.4M paired PCQM molecules and the
paper explicitly says the PCQM Gap label is not used in self-supervised
pretraining. Stage 2 uses a separate 2M single-modality subset from the
Uni-Mol data. The reported downstream tables are MoleculeNet, QM9, and
conformation generation; no direct PCQM4Mv2 Gap result is exposed. The public
code records the PCQM SDF MD5 and exposes stage commands, preprocessing, and
checkpoint boundaries. The paper reports 112M parameters, about 7 GPU-hours
per Stage-1 epoch, 28 ms inference, and 24 GB x2 peak memory for the full
model.

**MolGap disposition.** **B, teacher/missing-modality reference.** It provides
a concrete way to test whether a 2D student can inherit information from a
3D branch without requiring 3D at deployment. However, the published recipe
uses PCQM DFT coordinates plus external Uni-Mol data, so it cannot be copied
under the current ETKDG-only and unchanged-database contract. A future version
would need ETKDG for every geometry view and a separate same-database control.

## 22. Cross-paper synthesis

### 18.1 What has the strongest evidence

1. **Global information is useful but dense attention is not mandatory.**
   Graphormer establishes structural attention biases; GPS++ shows that local
   message passing can retain most performance at PCQM scale. This supports
   the project’s GraphState/local-global allocation framing.
2. **Geometry is useful, but the geometry role is causal.** ViSNet, TGT,
   Uni-Mol+, GraphMVP, GeoSSL-DDM, Frad, and SCD all use a geometry signal.
   Their results cannot be transferred without tracking whether the geometry
   is DFT-optimized, RDKit/MMFF, ETKDG, noisy, predicted, or teacher-only.
3. **Teacher/student separation is real evidence, not just a slogan.** ViSNet’s
   optimized-teacher/RDKit-student experiment and D&D’s frozen 3D teacher
   provide two independent precedents. Neither permits a DFT teacher to be
   silently paired with ETKDG deployment.
4. **Self-supervised denoising is the most reproducible new method family.**
   Frad, SCD, MoleculeSDE, and MoleBlend have public code or checkpoints;
   SCD also has a public PCQ checkpoint and a same-PCQ downstream table.
   MoleculeSDE and MoleculeJAE add trajectory/data-space denoising, while
   MoleBlend adds pair-relation prediction. Their direct value for MolGap is
   still conditional on ETKDG adaptation and a fresh target-matched control.
5. **Local electronic teachers may be more transferable than another Gap head.**
   The atom-level pretraining study compares local QM descriptors with HLG
   pretraining under a common Graphormer protocol. This is evidence for a
   possible teacher question, not evidence for merging an external database.
6. **Delta learning is a physical hypothesis, not a residual/fusion alias.**
   DelFTa supports the idea only when the cheap baseline is correlated with the
   target and is available at inference. A same-PCQM proxy must pass a CPU
   residual gate before training.
7. **Multimodal pretraining evidence is not interchangeable with target
   evidence.** MoleBlend, MoleculeSDE, MoleculeJAE, and FlexMol all use paired
   PCQM geometry in some stage, but the strongest published numerical results
   are on QM9, MoleculeNet, or conformation generation. A paper mentioning
   PCQM Gap in its setup is not enough to admit a PCQM Gap number.

### 18.2 What should not be counted as evidence for the current screen

- target-labeled pretraining or joint supervised 2D/3D training presented as
  a pure architecture gain;
- DFT geometry at training paired with ETKDG/RDKit geometry at inference;
- a challenge ensemble compared with a single model;
- QM9, QMugs, ADMET, MD17, or other target/geometry contracts relabeled as
  PCQM4Mv2 evidence;
- a repository README without a paper, fixed revision, data provenance, and
  independently checkable metrics;
- an external quantum database merged into the official training database;
- a same-row learned baseline prediction used to define a delta target.
- a QM9 Gap/HOMO/LUMO result relabeled as PCQM4Mv2 Gap evidence;
- a PCQM pretraining checkpoint treated as ETKDG-compatible without rerunning
  the exact geometry preprocessing.

### 18.3 Highest-value evidence-only follow-ups

These are reading/asset audits, not experiment authorization:

1. Verify the SCD repository revision, license, checkpoint hash, coordinate
   preprocessing, and one local forward pass without changing MolGap code.
2. Read the full D&D implementation status and determine whether any official
   code/checkpoint has appeared; otherwise retain the paper-only card.
3. Reproduce the exact public DelFTa baseline metadata and estimate whether a
   same-PCQM ETKDG proxy is computationally feasible before writing a protocol.
4. Audit the atom-level QM source for identity overlap, functional/basis,
   charge/spin, and license before considering a teacher-only role.
5. Keep a separate table of paper validation, official test-dev, and ensemble
   results; never use the lowest number alone to select a MolGap candidate.
6. For MoleculeSDE/JAE/MoleBlend/FlexMol, first audit geometry construction and
   inference modality; do not infer that paired PCQM pretraining proves a
   2D-only or ETKDG-only Gap gain.

## 23. Evidence gate for a future possible experiment

No method from this record becomes a candidate until its card has all of the
following:

1. primary paper, code revision, checkpoint URL/hash if applicable, and license;
2. exact target, units, functional, basis, charge/spin, geometry source, and
   split;
3. canonical molecule/conformer overlap and role audit;
4. a clear ETKDG train/inference decision or a separately approved geometry
   contract;
5. an independent fresh random-init GraphState control on the frozen internal
   split;
6. a single causal hypothesis and a numerical material-gain/resource gate;
7. CPU cache acceptance before GPU use, atomic checkpoints, and retrievable
   output chunks;
8. a seed-42 result treated as a promising signal only, with seeds 43/44
   requiring a separate budget decision.

## Primary-source index

- [OGB-LSC / PCQM4M paper](https://arxiv.org/html/2103.09430)
- [Official PCQM4Mv2 specification](https://ogb.stanford.edu/docs/lsc/pcqm4mv2/)
- [Graphormer](https://arxiv.org/html/2106.05234) and
  [code](https://github.com/microsoft/Graphormer)
- [GraphGPS](https://arxiv.org/abs/2205.12454) and
  [code](https://github.com/rampasek/GraphGPS)
- [GPS++](https://arxiv.org/abs/2302.02947) and
  [PCQM code/checkpoints](https://github.com/graphcore/ogb-lsc-pcqm4mv2)
- [Transformer-M](https://arxiv.org/html/2210.01765) and
  [code](https://github.com/lsj2408/Transformer-M)
- [ViSNet](https://arxiv.org/html/2210.16518) and
  [PCQM challenge report](https://ogb.stanford.edu/paper/neurips2022/pcqm4mv2_ViSNet.pdf)
- [TGT](https://arxiv.org/html/2402.04538) and
  [code](https://github.com/shamim-hussain/tgt)
- [Uni-Mol+](https://www.nature.com/articles/s41467-024-51321-w),
  [code](https://github.com/deepmodeling/Uni-Mol/), and
  [models](https://zenodo.org/records/12670462)
- [GraphMVP](https://arxiv.org/html/2110.07728) and
  [code](https://github.com/chao1224/graphmvp)
- [GeoSSL-DDM](https://arxiv.org/html/2206.13602)
- [UnifiedMolPretrain](https://arxiv.org/html/2207.08806) and
  [code](https://github.com/teslacool/UnifiedMolPretrain)
- [Frad](https://www.nature.com/articles/s42256-024-00900-z),
  [code](https://github.com/fengshikun/FradNMI), and
  [models](https://zenodo.org/records/12697467)
- [SCD](https://arxiv.org/html/2603.17196v1),
  [code](https://github.com/Ty-Perez/SelfConditionedDenoisingAtoms), and
  [checkpoint](https://huggingface.co/Ty-Perez/ct-scd-pcq)
- [D&D](https://ojs.aaai.org/index.php/AAAI/article/view/31986)
- [Atom-level quantum pretraining](https://link.springer.com/article/10.1186/s13321-025-00970-0)
- [DelFTa](https://github.com/josejimenezluna/delfta) and
  [delta-QML paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC9093086/)
- [DeMol](https://arxiv.org/html/2603.00568) and
  [author-linked repository](https://github.com/LiuYunqing/DeMol)
- [DGT](https://www.nature.com/articles/s41467-026-75005-9) and
  [code](https://github.com/zhangsy-ryan/DGT)
- [MoleculeSDE](https://arxiv.org/html/2305.18407),
  [official code](https://github.com/chao1224/MoleculeSDE),
  [Geom3D](https://github.com/chao1224/Geom3D), and
  [released checkpoints](https://huggingface.co/chao1224/MoleculeSDE/tree/main)
- [MoleculeJAE](https://arxiv.org/html/2312.03475) and
  [NeurIPS record](https://papers.neurips.cc/paper_files/paper/2023/hash/acddda9cd6f310689f7657f947705a99-Abstract-Conference.html)
- [MoleBlend](https://arxiv.org/html/2307.06235) and
  [official code](https://github.com/YudiZh/MoleBlend)
- [FlexMol](https://arxiv.org/html/2510.07035) and
  [official code](https://github.com/tewiSong/FlexMol)
