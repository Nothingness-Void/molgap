# Deep Reading Continuation: 3D-PGT, AniDS, 3D-EMGP, Mol-MFFGE, and PVD

Date: 2026-09-07

This continuation reads several denoising/pretraining lines that are easy to
miss when the search is limited to the newest papers. Together they cover
direct PCQM4Mv2 3D pretraining, learned anisotropic noise, energy/force-
motivated equivariant objectives, task-aware pseudo-force fields, standard
vector denoising, and chemical-aware fractional denoising. None changes the
MolGap database or authorizes a run.

## 1. 3D-PGT: automated fusion of geometric pretext tasks

**Primary sources.** The KDD 2023 paper is
[Automated 3D Pre-Training for Molecular Property
Prediction](https://arxiv.org/html/2306.07812). The authors release an
[MIT-licensed implementation](https://github.com/LARS-research/3D-PGT).

**Problem.** The authors want a 2D graph predictor to inherit useful geometric
information without calculating a 3D conformer at downstream inference. They
split local geometry into three pretext tasks—bond length, bond angle, and
dihedral angle—and use a GPS-style graph transformer as the encoder.

**Method.** Each pretext head predicts a local geometric descriptor from graph
embeddings. The paper does not simply average the losses. It treats the
weights of the three tasks as a bi-level problem and uses a surrogate total
energy objective to search those weights. The motivation is that a low-energy
conformer should encode physically meaningful geometry; the surrogate is used
to select the pretraining mixture rather than to become the downstream target.
The paper also studies multiple conformers and backbone search.

**Direct PCQM evidence.** On the paper's PCQM4Mv2 validation comparison,
3D-PGT reports `0.0762 eV` with `42.6M` parameters. The same table lists GPS
without 3D pretraining at `0.0858`, while its ablation reports GPS `0.0764`
and the searched/fused version `0.0762`. The paper states that it uses the
default OGB train/validation split, pretrains on `3.37M` DFT structures, and
keeps the hidden test set for the challenge. It also reports that automated
fusion improves over average fusion (`0.0762` versus `0.0781`) and over each
single pretext task (`0.0807` bond length, `0.0811` angle, `0.0813` dihedral).

These numbers are real primary-source PCQM validation evidence, but they are
not directly comparable to the MolGap screen: the model is much larger than
the bounded GraphState candidate, the pretraining coordinates are DFT
equilibrium structures, and the reported comparison is a paper validation
table rather than a fresh result under the project's 100K/10K role contract.

**What transfers.** The strongest transferable idea is the *task-weight gate*.
If a future ETKDG-only pretraining contract has several geometric targets,
their weights should be selected by an auditable train-role proxy or fixed
ablation, not tuned on sealed Gap labels. A smaller implementation could use
bond-distance bins plus angle/dihedral summaries, but it must remain distinct
from the already tested torsion-state architecture and use the same ETKDG
construction at pretraining and inference.

**What does not transfer for free.** The total-energy surrogate is not an
available MolGap label under the existing role contract. The DFT conformer is
not an ETKDG conformer. The paper's 42.6M GPS model and old PyG 2.0.1 stack
also exceed the bounded implementation target. Copying the reported `0.0762`
into the project SOTA table would confuse paper validation with the official
benchmark and with the active internal split.

**Disposition: B for direct PCQM evidence, C for a present candidate.** Keep
3D-PGT as the clearest historical proof that geometry pretraining can help a
2D PCQM predictor. A possible future MolGap question is a width-reduced,
ETKDG-only pretext ablation after architecture selection, but it requires a
separate protocol and a fresh random-init control.

## 2. AniDS: anisotropic, molecule-conditioned noise

**Primary sources.** The NeurIPS 2025 paper is
[Learning 3D Anisotropic Noise Distributions Improves Molecular Force Field
Modeling](https://arxiv.org/html/2510.22123), and the authors release the
[official AniDS code](https://github.com/ZeroKnighting/AniDS).

**Question.** DenoiseVAE learns atom-specific scalar noise scales. AniDS asks
whether molecular motion is directional as well as atom-specific: bond-aligned
and perpendicular displacement directions need not have the same variance.

**Mechanism.** AniDS uses a structure-aware noise generator that produces an
atom-specific full covariance correction to an isotropic base distribution.
The construction is designed to remain positive semidefinite and SO(3)-
equivariant. The paper compares additive and subtractive covariance updates,
and its analysis emphasizes that the noise generator can suppress motion along
rigid bond directions while allowing more freedom in flexible directions.
This is a refinement of the adaptive-noise family, not a new target predictor.

**Data and results.** The paper pretrains on `3,746,619` PCQM4Mv2 molecules
with DFT equilibrium structures and explicitly states that PCQM property
values are not used for that pretraining. It then fine-tunes on MD17 and OC22
force/energy tasks. On the reported MD17 experiments, AniDS improves the
average force result by about `8.9%`; on OC22 it reports about `6.2%` average
force improvement. The public configuration records a 4-A100 pretraining
run of about 32 GPU-hours, 40,000 steps, a prior sigma of `0.1`, and KL
coefficient `1.0`. The code has a PCQM pretraining command and stores weights
under the configured log directory.

**MolGap implication.** The evidence supports a narrow design progression:
fixed isotropic denoising -> learned scalar noise (DenoiseVAE) -> learned
directional noise (AniDS). It does not support the claim that AniDS improves
HOMO/LUMO/Gap, because its downstream evidence is force-field modeling, not a
direct PCQM Gap evaluation. Its 129M Equiformer-V2-scale model also conflicts
with the bounded architecture budget.

**Contract audit.** The PCQM pretraining coordinates are DFT equilibrium
structures, while MolGap requires ETKDG for both training and inference. The
full covariance generator is also more expensive and may be numerically
fragile on failed or degenerate ETKDG conformers. Any legal adaptation would
need a compact covariance parameterization, a coordinate-validity test, an
identical ETKDG construction at both stages, and an explicit scalar-noise
control. It cannot reuse PCQM DFT coordinates merely because they are present
in the same database.

**Disposition: B for adaptive-noise design, C for direct MolGap admission.**
AniDS should inform a future denoising ablation only after the simpler
DenoiseVAE/ETKDG audit is complete. No full Equiformer/AniDS port belongs in
the current screen.

## 3. 3D-EMGP: energy-motivated equivariant pretraining

**Primary sources.** The AAAI 2023 paper is
[Energy-Motivated Equivariant Pretraining for 3D Molecular
Graphs](https://ojs.aaai.org/index.php/AAAI/article/view/25978), with a public
[MIT implementation and checkpoints](https://github.com/jiaor17/3D-EMGP).

**Method.** 3D-EMGP uses an equivariant energy-based model. Its node-level
pretraining objective predicts force-like information through a position
denoising loss, with a Riemann--Gaussian construction intended to preserve
E(3) invariance. It adds a graph-level noise-scale prediction task. The model
is pretrained without property labels on GEOM-QM9 and fine-tuned for QM9 and
MD17; the public fine-tuning script exposes `gap`, `homo`, and `lumo` among the
QM9 properties.

**Evidence.** The paper's evidence is controlled QM9/MD17 pretraining and
ablation, not PCQM4Mv2. The repository has a visible license, checkpoints,
configuration files, and a complete old-stack training path around Python
3.7.10, PyTorch 1.7, and PyG 1.6.3. This makes it useful for understanding
force/noise-scale objectives, but not a direct MolGap result.

**What transfers.** The graph-level noise-scale head is a useful alternative
to a coordinate-only denoising loss: it asks the encoder to estimate how
uncertain or flexible the corruption is for the molecule. A compact ETKDG
version could predict a bounded scalar corruption scale as an auxiliary task,
but its value must be measured against DenoiseVAE and a no-pretraining control.

**What does not transfer.** GEOM-QM9 is not the official PCQM4Mv2 database,
and the model's 3D geometry, stack, and equivariant backbone differ from
MolGap. The existence of a checkpoint does not justify importing it or
claiming a PCQM improvement.

**Disposition: B for objective design, C for direct target evidence.** Retain
the graph-level noise-scale task as a historical precursor to DenoiseVAE and
AniDS. Do not open a separate 3D-EMGP experiment.

## 4. Mol-MFFGE: task-aware pseudo-force-field pretraining

**Primary sources.** The Pattern Recognition article is
[Meta-learning of pseudo force field generation and estimation for enhancing
3D molecular property prediction](https://www.sciencedirect.com/science/article/pii/S0031320325001918),
and the authors release the [official Mol-MFFGE implementation](https://github.com/Yufei-Luo/Mol-MFFGE).

### Method

Mol-MFFGE starts from an energy-based denoising model and asks whether a
task-agnostic noise distribution is too far from the downstream property task.
It adds a learnable noise-transformation module, jointly learns noise generation
and noise estimation with the downstream objective, and formulates the task-aware
pretraining as a bi-level meta-learning problem. The outer optimization adjusts
task weights and noise-transformation parameters to reduce downstream loss; the
inner optimization updates the molecular representation model under the weighted
auxiliary and property losses.

The paper's physical interpretation is that estimating the learned perturbation
resembles estimating a pseudo force field. This is not a claim that the model
recovers a real force field; the pseudo field is a self-supervised representation
target whose noise distribution is adapted to the downstream task.

### Evidence and code

The paper evaluates QM9, revised MD17, and SPICE. The public repository exposes
separate task-agnostic denoising and task-aware adaptation configurations, data
preprocessing for GEOM-QM9 and SPICE, property choices including `homo`, `lumo`,
and `delta`, and comparisons with other pretraining methods. The code has only
one visible commit and no released checkpoint was independently verified in the
audited surface; it is therefore implementation evidence, not a drop-in model.

The source does not provide a direct PCQM4Mv2 Gap result under the MolGap split,
and its conformers come from GEOM/SPICE rather than the project's ETKDG cache.
The QM9 `homo/lumo/delta` task names are useful for method tracing, but their
values cannot be compared with PCQM B3LYP/6-31G* Gap MAE.

### What can transfer

- Adapt the denoising noise distribution to the downstream task only through a
  train-role objective; do not tune noise parameters on validation/test-dev Gap.
- Treat auxiliary-task weighting as a meta-learning object and report a fixed-
  weight control, an adapted-weight control, and a no-pretraining control.
- Keep the pseudo-force-field interpretation separate from a real low-fidelity
  quantum proxy. Mol-MFFGE is not Δ-learning and does not supply a physical
  residual target.

**Disposition: B for task-aware pretraining design, C for direct MolGap evidence.**
It is a stronger conceptual extension of the scalar/adaptive-noise line than a
new architecture candidate. Any ETKDG adaptation would require its own protocol,
CPU geometry/throughput estimate, and a fresh same-contract GraphState control.

## 5. Pre-training via Denoising (PVD): the clean PCQM pretraining baseline

**Primary sources.** The ICLR 2023 paper is
[Pre-training via Denoising for Molecular Property
Prediction](https://arxiv.org/html/2206.00133). The authors release an
[MIT-licensed official implementation](https://github.com/shehzaidi/pre-training-via-denoising)
with a documented PCQM4Mv2 command, a retrievable checkpoint
(`checkpoints/denoised-pcqm4mv2.ckpt`), and a separate QM9 fine-tuning path.
This is stronger implementation evidence than a paper-only denoising claim,
but it is still not a same-contract MolGap result.

### Method and physical interpretation

PVD takes an equilibrium structure with atomic numbers and coordinates,
perturbs every position with independent Gaussian noise,
`p_tilde = p + sigma * epsilon`, and trains an equivariant vector head to
predict the injected noise with an MSE loss. The model therefore predicts the
corruption vector rather than the clean coordinates. The paper derives this
objective from denoising-score matching: if the equilibrium structures are
treated as centers of a Gaussian mixture, the learned score is an approximate
force field around the observed local energy minima. This is an interpretation
of the self-supervised objective, not a DFT force label and not a delta target.

The paper also makes an important translation-invariance point. A global
translation cannot be inferred from a molecule's relative geometry, so the
noise target is considered in the mean-centered subspace. An ETKDG adaptation
must preserve this detail; otherwise the head can spend capacity on an
unidentifiable center-of-mass component and the reported loss is not the same
objective.

The authors test the objective on two backbones. GNS/GNS-TAT uses an
edge-and-node message-passing model with a graph-level decoder, while the
TorchMD-NET implementation applies gated equivariant blocks to scalar/vector
features and uses the vector features for noise prediction. The latter matters
for MolGap because it is a cleaner precedent for attaching a denoising head to
an existing equivariant encoder than importing the full GNS-TAT architecture.

### PCQM upstream and downstream evidence

The upstream dataset is PCQM4Mv2: the paper states `3,378,606` organic
molecules with DFT-equilibrium 3D structures, and explicitly says that the
available property labels are not used for denoising. The main downstream
evidence is QM9. In the GNS-TAT table, pretraining plus Noisy Nodes improves
HOMO from `17.3` to `14.9` meV, LUMO from `17.1` to `14.7` meV, and Gap from
`25.7` to `22.0` meV relative to the same GNS-TAT+Noisy-Nodes family without
PCQM pretraining. The reported values average three seeds, but they are QM9
results, not PCQM4Mv2 Gap validation.

The more isolated TorchMD-NET ablation is especially useful: random-init
HOMO/LUMO are `22.0 +/- 0.6` and `18.7 +/- 0.4` meV; adding the downstream
Noisy Nodes auxiliary task gives `18.1 +/- 0.1` and `15.6 +/- 0.1`; adding
PCQM denoising pretraining gives `15.6 +/- 0.1` and `13.2 +/- 0.2`. The
official README rounds the last pair to `15.5` and `13.2` meV. The gain is
therefore not evidence that “any auxiliary loss” is sufficient: the paper
separates random initialization, downstream denoising, and upstream
pretraining.

The transfer analysis is as important as the headline result. Increasing the
upstream PCQM sample count generally helps before saturation, and pretraining
helps at every tested QM9 downstream size, with larger benefit when downstream
labels are scarce. Conversely, PCQM pretraining offers no validation benefit
for OC20 IS2RE, where the element and structure distribution is very
different; same-dataset OC20 pretraining mainly accelerates convergence. This
is direct evidence that upstream/downstream compatibility, not database size
alone, controls transfer.

Two further controls constrain the interpretation. On DES15K, TorchMD-NET
interaction-energy MAE changes from `0.721` to `0.406` kcal/mol after PCQM
pretraining, so the representation can transfer beyond QM9. With the
pretrained backbone frozen and only a simple decoder trained for QM9 HOMO,
the paper reports about `40` meV, versus more than `100` meV for a randomly
initialized frozen backbone. Full fine-tuning remains better, so freezing is
an informative representation probe, not the recommended downstream recipe.

### Reproducible configuration and cost

The official `ET-PCQM4MV2.yaml` is a TorchMD-NET denoising configuration: a
5.0-Angstrom cutoff, 32 maximum neighbors, 8 layers, 8 attention heads,
256-dimensional embeddings, batch size 70, 400,000 steps, position-noise
scale `0.04`, denoising-only training, and `float32` precision. The README
documents three RTX 2080 Ti GPUs for pretraining and two for fine-tuning. The
paper's GNS-TAT appendix uses a related but not identical setup: 300,000
gradient steps, 512-dimensional vertex/edge latents, dynamic batches, EMA
decay `0.9999`, Gaussian position noise `0.02`, and a `0.75` atom-type mask
probability with coefficient `4.0`. These must not be silently collapsed into
one “PVD hyperparameter” record.

The official QM9 fine-tuning config is also explicit: 110,000 training and
10,000 validation molecules, 300,000 steps, 5.0-Angstrom cutoff, 8 layers,
8 heads, batch size 128, standardization enabled, position-noise scale
`0.005`, and denoising weight `0.1`. It is a useful recipe for ablation
shape, but it does not define a valid MolGap split or ETKDG path.

### MolGap contract audit

PVD is a high-quality *same-database pretraining precedent*, not a legal
initialization for the current screen. The released and paper-described path
uses DFT-equilibrium PCQM coordinates; MolGap requires ETKDG for both training
and inference. The source also pretrains on the full PCQM structure pool and
then transfers to QM9, whereas MolGap has explicit Track A/Track B roles and
architecture-discovery restrictions. “The labels are unused” prevents target
label leakage, but it does not by itself authorize using all coordinate rows
or a released checkpoint in the current role contract.

What is safe to borrow later is narrow: mean-centered vector-noise prediction,
the separate no-pretraining/auxiliary/pretraining controls, the frozen
backbone probe, and the practice of publishing upstream-size and
upstream/downstream compatibility curves. Any ETKDG version would need a
fresh same-contract random-init control and a fixed train-role manifest. It
must not import the checkpoint, DFT coordinates, or QM9 metrics as MolGap
evidence.

**Disposition: B for direct PCQM self-supervised pretraining evidence, C for a
current experiment.** PVD is the cleanest completed reference for the causal
question “does structural pretraining add beyond an auxiliary denoising loss?”
It strengthens the evidence base for a future post-selection ETKDG audit, but
does not reopen the current architecture screen.

## 6. Fractional Denoising (Frad): chemical-aware noise without breaking the score objective

**Primary sources.** The ICML 2023 formulation is
[Fractional Denoising for 3D Molecular Pre-training](https://arxiv.org/html/2307.10683)
with an [official MIT codebase](https://github.com/fengshikun/Frad). The expanded
Nature Machine Intelligence 2024 study is
[Pre-training with Fractional Denoising to Enhance Molecular Property
Prediction](https://arxiv.org/html/2407.11086), with the
[FradNMI implementation](https://github.com/fengshikun/FradNMI),
[Zenodo pretrained models](https://zenodo.org/records/12697467), and public
[source data for the figures](https://doi.org/10.6084/m9.figshare.25902679.v1).
The two papers should not be collapsed: the later paper adds VRN noise,
broader downstream tasks, inaccurate-conformation robustness, and a more
complete efficiency/ablation packet.

### Why ordinary coordinate denoising is limited

PVD-style isotropic coordinate noise must use a small standard deviation to
avoid chemically implausible structures such as distorted aromatic rings. Frad
argues that this creates two linked problems: samples remain close to one
equilibrium and the induced force field is isotropic, although real molecules
have rigid rings/double bonds and flexible single-bond torsions. Increasing
coordinate noise alone does not solve the sampling problem without violating
those constraints.

Frad therefore separates the sampling distribution from the part that is
denoised. Starting from an equilibrium conformation `x_eq`, it first applies a
chemical-aware noise (CAN) to obtain `x_med`, then adds coordinate Gaussian
noise (CGN) to obtain `x_fin`. The encoder sees `x_fin`, but the noise head
predicts only `x_fin - x_med`, the CGN component. This is the “fractional” part:
the model recovers a fraction of the total injected perturbation. The theorem
requires the final conditional noise to be isotropic Gaussian; the earlier CAN
distribution may be arbitrary. Consequently, CAN changes the sampled
conformation distribution and the induced anisotropic force field without
destroying the score-matching equivalence.

### Chemical-aware noise variants and implementation details

The expanded paper exposes two CAN variants:

- **RN (rotation noise):** perturb the torsion angles of rotatable bonds with
  Gaussian noise. The PCQM pretraining setting uses rotatable-bond torsion
  standard deviation `2` and CGN standard deviation `0.04`.
- **VRN (vibration-and-rotation noise):** perturb bond lengths, bond angles,
  non-rotatable torsions, and rotatable torsions independently. The reported
  scales are `0.058`, `0.129`, `0.18`, and `1`, respectively, followed by CGN
  standard deviation `0.04`.

The implementation searches rotatable single bonds with RDKit and includes
hydrogens in that search. To avoid dependent angle perturbations, it selects
one edge when an atom has degree greater than two and does not perturb bonds,
angles, or torsions inside rings. These are not cosmetic details: they keep
CAN's degrees of freedom from exceeding the Cartesian coordinates and make
the claimed noise independence closer to the theoretical setup.

The model follows TorchMD-NET with a scalar/vector equivariant Transformer and
an MLP noise head. The public PCQM recipe uses AdamW, batch size `70`, `10,000`
warmup steps, maximum learning rate `4e-4`, cosine decay over `400,000`
steps, and the RN/VRN scales above. The NMI repository pins an older but
complete environment (Python 3.8, PyTorch 1.13.1+cu116, PyG 2.3.0); the
paper's efficiency table reports about `14 h 56 min` on one A100 40 GB for
Frad pretraining. That timing is hardware-specific and is not a MolGap job
estimate.

### Results and what the ablations actually establish

The upstream is the same `3,378,606`-molecule PCQM4Mv2 structure pool, with
labels unused. On QM9, the matched TorchMD-NET coordinate-denoising baseline
reports HOMO/LUMO/Gap `17.7/14.3/31.8` meV. Frad(RN) reports
`15.3/13.7/27.8` meV and Frad(VRN) `17.9/13.8/27.7` meV. The paper reports
new best results on 9 of 12 QM9 targets and improvement over the same
TorchMD-NET backbone on 11 targets; the Gap values are QM9, not PCQM4Mv2
validation/test-dev evidence.

The pretraining ablation is the most relevant control: Frad beats coordinate
denoising on all six listed QM9 tasks under aligned architecture,
optimization, and Noisy-Nodes settings. The fine-tuning ablation is separate:
traditional Noisy Nodes fails to converge on conformation-sensitive MD17
force prediction, while decoupling the property input from the noisy input
and using Frad noise restores convergence and improves Aspirin force MAE from
`0.2141` without Noisy Nodes to `0.2087` in the listed setting. This does not
prove a Gap benefit, but it demonstrates why a noisy auxiliary input can be
wrong for a geometry-sensitive label.

The force and coverage controls support the proposed mechanism rather than
just reporting downstream scores. On aspirin, hybrid noise gives higher
force correlation than coordinate noise when sampling farther from
equilibrium; the paper selects `sigma=2, tau=0.04` because larger torsion
noise improves sampling but increases the local linearization error. In the
NMI robustness study, replacing DFT pretraining conformers with RDKit
Distance Geometry plus MMFF increases absolute downstream error, yet Frad
still beats training from scratch and can even beat coordinate denoising
pretrained on accurate conformers. That is evidence of robustness to an
inaccurate upstream geometry, not evidence that RDKit+MMFF and ETKDG are
interchangeable.

### MolGap contract audit

Frad is methodologically distinct from the already closed random-init
torsion-state route: torsion changes are a pretraining sampling view, and the
downstream property model receives the uncorrupted conformation in the
NMI/Frad-Noisy-Nodes design. Nevertheless, the source cannot enter the current
architecture screen or be used as a checkpoint. Its main PCQM pretraining
uses DFT equilibrium coordinates; its inaccurate-coordinate control uses RDKit
Distance Geometry plus MMFF, not a verified MolGap ETKDG cache; and all
reported frontier-orbital numbers are QM9. The released environment is also
legacy and the public weights are external artifacts whose input contract has
not been rewritten for ETKDG.

The safe borrowing boundary is narrow: treat CAN as a pretraining view over
an existing ETKDG conformer, preserve the `x_eq -> x_med -> x_fin` provenance,
regress only the final coordinate-noise component, and compare against a
coordinate-denoising/no-pretraining control. Before any run, the audit must
also freeze rotatable-bond definitions, ring exclusions, angle-selection rules,
noise scales, coordinate hashes, and the train-role manifest. It must not
reuse the Frad checkpoint, DFT coordinates, RDKit+MMFF conformers, or QM9
scores as MolGap evidence.

**Disposition: B for a complete chemical-aware denoising reference, C for a
current experiment.** Frad is the strongest existing justification for a
future ETKDG-only *pretraining* ablation if a separate budget and protocol are
opened. It is not a reason to reopen the closed torsion architecture and does
not authorize a database change or a pretrained initialization now.

## 7. SliDe: force-consistent bond/angle/torsion pretraining with random slicing

**Primary sources.** The ICLR 2024 paper is
[Sliced Denoising: A Physics-Informed Molecular Pre-Training
Method](https://arxiv.org/html/2311.02124). The authors release an
[official MIT implementation](https://github.com/fengshikun/SliDe), with
PCQM4Mv2 preparation instructions and downloadable QM9/MD17 pretrained models.

### Method and objective

SliDe replaces the coarse isotropic coordinate-noise energy used by Coord and
the rotatable-bond treatment used by Frad with a quadratic classical
intramolecular energy over three relative-coordinate families:

- bond lengths;
- bond angles;
- torsion angles.

The stiffness and torsion-period parameters are taken from Open Force Field
2.0.0 (Sage), so different atom/bond types receive different noise scales.
The resulting BAT noise is sampled as independent Gaussian perturbations in
the three relative-coordinate families. Electrostatic and van der Waals
terms are omitted, and the periodic torsion energy is locally approximated by
a quadratic Taylor expansion. This is a deliberately local physical model,
not a claim that the denoising target is a full DFT force field.

The force target is the Cartesian gradient of this relative-coordinate energy.
Rather than explicitly forming the expensive Cartesian-to-BAT Jacobian, SliDe
uses Gaussian random slicing: it projects the model output and the force
target onto `N_v` random Cartesian directions, and estimates each Jacobian
projection with a finite coordinate difference. The paper proves equivalence
to force-field regression only in the limits `sigma -> 0` and `N_v ->
infinity`; the finite implementation is therefore an estimator with a
measurable bias/variance trade-off.

The model is GET, a TorchMD-NET-like equivariant Transformer with an additional
edge update. Edge features combine bond-length embeddings with angle and
torsion embeddings; updated edge features then affect the vertex attention.
This is the architecture contribution, while BAT noise and random slicing are
the pretraining-objective contribution. They should be kept separate in any
MolGap adaptation.

### Evidence and ablations

The force-field audit uses 1,000 randomly selected PCQM4Mv2 molecules and
compares the learned force estimate with B3LYP/6-31G DFT forces. The reported
Pearson correlation is `0.616(0.047)` for Coord, `0.631(0.046)` for Frad, and
`0.895(0.071)` for SliDe. This is unusually useful evidence for the denoising
mechanism, but it is a force-label audit on a sample, not a Gap prediction
result.

For the QM9 downstream task, the matched table reports HOMO/LUMO/Gap in meV:

| pretraining | HOMO | LUMO | Gap |
|---|---:|---:|---:|
| Coord | 17.7 | 14.3 | 31.8 |
| Frad | 15.3 | 13.7 | 27.8 |
| SliDe | 13.6 | 12.3 | 26.2 |

The regularization ablation is important: scratch is `17.6/16.7/31.3`,
SliDe without the auxiliary coordinate regularizer is `15.0/14.8/27.7`, and
SliDe with it is `13.6/12.3/26.2`. Thus the paper does not isolate BAT/random
slicing from every downstream Noisy-Nodes choice, but it does show that the
reported gain is not simply “pretraining versus scratch.” The paper also
reports GET versus TorchMD-NET on MD17, with GET improving the SliDe Aspirin
force MAE from `0.2045` to `0.1740` and Benzene from `0.1810` to `0.1691`.

The PCQM pretraining pool contains about `3.4M` equilibrium structures and is
used without PCQM property labels. The released recipe records batch `128`,
AdamW, `10,000` warmup steps, maximum learning rate `4e-4`, cosine cycle
`240,000`, `N_v=128`, finite-difference scale `sigma=0.001`, and coordinate
regularization scale `tau=0.04`. The repository reports experiments on eight
A100 40 GB GPUs. Its environment is legacy (Python 3.8, PyTorch 1.13.1,
PyG 2.3.0), but the code, license, config values, and model links are
inspectable.

### MolGap contract audit

SliDe is stronger than a generic torsion proposal because it keeps torsion
perturbation inside a label-free pretraining view and gives a force-consistency
control. It is not evidence for reopening the closed torsion-state
architecture. It also has four direct blockers for the present screen:

1. its main PCQM coordinates are the supplied DFT equilibrium conformers, not
   the project's ETKDG cache;
2. its downstream frontier-orbital numbers are QM9, not PCQM4Mv2 Gap;
3. its BAT parameters come from an external force field and the paper omits
   long-range terms, so the objective is a physical prior rather than a target
   theory replacement;
4. its GET backbone and `N_v=128` slicing loop are not a bounded GraphState
   architecture comparison.

A legal future adaptation would use the existing ETKDG conformer as `x_0`,
derive BAT parameters deterministically from the declared bond/angle/torsion
definitions, freeze `sigma`, `N_v`, ring/degeneracy handling and the
coordinate-difference validity policy, and compare (i) random init, (ii)
coordinate denoising, and (iii) SliDe-style BAT denoising at the same student
capacity. The force audit must remain a train-role diagnostic; it cannot use
sealed Gap labels to choose `N_v` or `sigma`.

**Disposition: B for physics-informed pretraining and code, C for a present
MolGap experiment.** SliDe is a high-priority future ETKDG-only pretraining
reference, but it does not change the database, geometry rule, current
architecture queue, or experiment authorization.

## Cross-paper synthesis

| Source | Direct PCQM Gap evidence | Transferable idea | Blocking mismatch | Safe classification |
|---|---|---|---|---|
| 3D-PGT | `0.0762` validation MAE, 42.6M, DFT pretraining | Multi-task geometry and auditable loss-weight selection | DFT coordinates, large GPS, paper validation protocol | B / future contract |
| AniDS | PCQM used for label-free pretraining; no PCQM Gap result | Directional, atom-specific covariance noise | 129M force-field model, DFT coordinates, MD17/OC22 target | B / design reference |
| 3D-EMGP | no direct PCQM result | E(3)-invariant force/noise-scale objectives | GEOM-QM9, old stack, 3D downstream tasks | B / historical objective |
| Mol-MFFGE | no direct PCQM result; QM9/MD17/SPICE only | Task-aware noise transformation and bi-level auxiliary-loss weighting | GEOM/SPICE conformers, external targets, no independently verified checkpoint | B / task-aware denoising reference |
| PVD | PCQM4Mv2 pretraining; QM9 HOMO/LUMO/Gap and DES15K transfer; no same-contract PCQM Gap result | Mean-centered vector denoising, upstream-size/compatibility analysis, frozen-backbone probe | DFT-equilibrium coordinates, full-PCQM upstream role, TorchMD/GNS stack, no ETKDG MolGap comparison | B / direct-PCQM pretraining reference |
| Frad | PCQM4Mv2 pretraining; QM9 HOMO/LUMO/Gap and force/robustness transfer; no same-contract PCQM Gap result | Chemical-aware RN/VRN sampling, fractional CGN target, CAN/coordinate decoupling, and noisy-input separation | DFT main coordinates, RDKit+MMFF robustness control, legacy TorchMD stack, QM9 downstream and closed torsion route | B / chemical-aware pretraining reference |
| SliDe | PCQM4Mv2 label-free pretraining; 1,000-molecule DFT-force audit; QM9 HOMO/LUMO/Gap and MD17 downstream results | BAT bond/angle/torsion noise, random-sliced Jacobian estimator, GET edge-angle/torsion update, force-correlation gate | DFT-equilibrium coordinates, QM9 rather than PCQM Gap downstream, external OpenFF parameters, legacy GET stack | B / future ETKDG-only pretraining reference |

The main conclusion is not that every stronger denoising paper should be
stacked. It is that adaptive noise is a coherent line with increasing
complexity, and only the smallest contract-preserving step can identify a
causal gain. The sequence should be audited from scalar to directional noise;
it should not jump directly to a 129M equivariant model.

## Evidence gate

Before any source in this batch can become a possible MolGap experiment, the
record must include the exact code revision and license, coordinate-generation
method, split and role manifest, atom ordering, checkpoint hash, and a local
forward smoke test. A pretraining candidate must use ETKDG at both training
and inference or receive a separately approved geometry contract. The direct
3D-PGT number is recorded as paper evidence only; it is not a reason to alter
the active random-init screen.

No experiment, database replacement, checkpoint initialization, or remote job
was performed while writing this continuation.

## Primary-source index

- [3D-PGT paper](https://arxiv.org/html/2306.07812) and [MIT code](https://github.com/LARS-research/3D-PGT)
- [AniDS paper](https://arxiv.org/html/2510.22123) and [official code](https://github.com/ZeroKnighting/AniDS)
- [3D-EMGP paper](https://ojs.aaai.org/index.php/AAAI/article/view/25978) and [MIT code](https://github.com/jiaor17/3D-EMGP)
- [Mol-MFFGE paper](https://www.sciencedirect.com/science/article/pii/S0031320325001918) and [official code](https://github.com/Yufei-Luo/Mol-MFFGE)
- [PVD paper](https://arxiv.org/html/2206.00133), [official code/checkpoint](https://github.com/shehzaidi/pre-training-via-denoising), and [PCQM configuration](https://github.com/shehzaidi/pre-training-via-denoising/blob/main/examples/ET-PCQM4MV2.yaml)
- [Frad ICML paper](https://arxiv.org/html/2307.10683), [Frad code](https://github.com/fengshikun/Frad), [Frad NMI paper](https://arxiv.org/html/2407.11086), [FradNMI code](https://github.com/fengshikun/FradNMI), [Zenodo weights](https://zenodo.org/records/12697467), and [source data](https://doi.org/10.6084/m9.figshare.25902679.v1)
- [SliDe paper](https://arxiv.org/html/2311.02124), [official code](https://github.com/fengshikun/SliDe), and [ICLR proceedings record](https://proceedings.iclr.cc/paper_files/paper/2024/hash/4a1d69d1f64c6b6df105b15984ca527a-Abstract-Conference.html)
- [Official PCQM4Mv2 specification](https://ogb.stanford.edu/docs/lsc/pcqm4mv2/)
