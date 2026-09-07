# Deep Reading Batch: Adaptive Denoising, 3D Re-mask Decoding, 3D Tokens, and Spectral Teachers

Date: 2026-09-07

This batch extends the MolGap evidence reserve with four primary-source reads
that were not part of the earlier ledger. The shared question is whether a
3D pretraining signal, a richer representation of local structure, or an
electronic teacher can improve HOMO--LUMO Gap prediction while retaining the
existing PCQM4Mv2 database and the project-level ETKDG train/inference rule.

The records are evidence only. No source below changes the active database,
the random-initialized architecture screen, the accepted GraphState anchor, or
the remote-run queue.

## Evidence labels

- **A**: primary paper and public implementation are both available, with a
  clearly described method/data role.
- **B**: strong primary evidence or public code exists, but the target,
  geometry, split, or downstream role differs from MolGap.
- **C**: useful design signal, but the source is not directly comparable and
  should not receive experiment compute without a separate contract.

An A or B label means that the source is worth reading or auditing. It does
not mean that its reported score is a MolGap score.

## 1. DenoiseVAE: molecule-adaptive noise distributions

**Primary sources.** The work is the ICLR 2025 paper
[DenoiseVAE: Learning Molecule-Adaptive Noise Distributions for Denoising-based
3D Molecular Pre-training](https://proceedings.iclr.cc/paper_files/paper/2025/hash/37e9e62294ff6607f6f7c170cc993f2c-Abstract-Conference.html).
The [conference PDF](https://openreview.net/attachment?id=ym7pr83XQr&name=pdf)
contains the PCQM4Mv2 appendix table, and the authors expose a public
[DenoiseVAE repository](https://github.com/liuyurou1/DenoiseVAE).

**Question.** Ordinary coordinate denoising fixes one noise distribution for
all molecules. DenoiseVAE asks whether the noise should be learned from the
molecule itself, because the physically plausible displacement scale differs
between rigid and flexible local environments.

**Mechanism.** A Noise Generator maps a clean equilibrium conformation to an
atom-specific Gaussian distribution. Reparameterized samples are added to the
coordinates and passed to a Denoising Module. The two modules are trained with
a denoising reconstruction term plus a KL term to a prior. The KL term is not
decorative: without it, the generator can collapse toward nearly zero noise.
The paper also provides an O(3)-invariance argument for the isotropic Gaussian
construction. After pretraining, the Noise Generator is not the downstream
prediction head; the denoising representation is transferred to property
prediction.

**Direct PCQM evidence.** Appendix A.7 reports a PCQM4Mv2 validation result of
`0.0777 +/- 0.0005` MAE with a reported `1.44M`-parameter model. The same
appendix compares against much larger graph-transformer references. This is
the strongest direct-PCQM number found in this batch because it is explicitly
reported for the PCQM4Mv2 Gap task rather than inferred from QM9. It is still
only a paper-reported validation result until the exact split, coordinate
source, checkpoint, and evaluation script are independently reproduced.

**What the paper supports.** The result supports a bounded hypothesis that
the *noise distribution* can be more important than adding another generic
message-passing block. The ablations also provide a concrete starting point:
the paper reports a prior scale around `sigma=0.1` and a KL weight near `1` as
strong settings, while the code exposes the data, configuration, and training
layout needed for a smoke audit.

**Repository audit (fixed evidence).** The public repository has two visible
commits, no visible license file, and no released checkpoint or checkpoint
manifest on its top-level page. The default
[`pretrain_denoisevae.yml`](https://raw.githubusercontent.com/liuyurou1/DenoiseVAE/main/config/pretrain_denoisevae.yml)
points to `datasets/GEOM/blocks`, not to the PCQM directory. The pretraining
script loads `GEOMDataset` block files and writes a local checkpoint path, so
the paper's PCQM table is not a turnkey invocation of the default repository
configuration.

The PCQM conversion script is a more serious contract mismatch. It adds
hydrogens, calls RDKit `EmbedMolecule`, then calls
`MMFFOptimizeMolecule`, removes hydrogens, and for train rows aligns that
generated conformer against a DFT molblock before storing both `input_pos` and
`label_pos` in LMDB. This is not the project's ETKDG construction, and the
stored DFT position makes the train-role data path require an explicit role
audit even though the paper states that PCQM labels are not used in
pretraining. The legacy environment is also Python 3.7 / PyTorch 1.7 / PyG
1.6.3. Finally, the public
[`denoise_prednoise.py`](https://raw.githubusercontent.com/liuyurou1/DenoiseVAE/main/denoisevae/models/denoise_prednoise.py)
returns an undefined `noneed` variable, while the training script expects a
three-value return. The released code therefore fails the direct-run gate as
published, independently of the geometry mismatch.

**Disposition: B for the paper/method, C for code reproducibility.** The
`0.0777 +/- 0.0005` remains a primary-source validation claim and a useful
adaptive-noise hypothesis, but it cannot be treated as a runnable checkpoint,
an ETKDG result, or a current leaderboard number. A future audit would need a
new ETKDG-only implementation, a repaired and pinned code revision, a split
and role manifest, and a fresh random-init GraphState control. The safe
scientific question is not “can the released DenoiseVAE checkpoint be pasted
into GraphState?”; it is “does molecule-adaptive denoising help when both
pretraining and downstream inference use the permitted ETKDG construction?”
That question still needs a separately approved budget.

## 2. 3D-GSRD: selective re-mask decoding

**Primary sources.** The paper is
[3D-GSRD: 3D Molecular Graph Auto-Encoder with Selective Re-mask
Decoding](https://arxiv.org/abs/2510.16780), with a public
[NeurIPS 2025 implementation](https://github.com/WuChang0124/3D-GSRD).

**Question and mechanism.** Masked graph modeling becomes ambiguous in 3D:
the decoder must receive enough 2D structure to reconstruct masked atoms, but
that same 2D information can leak the answer and prevent the encoder from
learning 3D information. 3D-GSRD uses Selective Re-mask Decoding to remove
only 3D-relevant information from the encoder representation while preserving
the 2D graph context. It couples this with a 3D relational Transformer and a
structure-independent decoder.

**Evidence.** The primary abstract reports new best results on 7 of 8 MD17
targets. The official repository has a PCQM4Mv2 pretraining script and a QM9
fine-tuning script whose task argument includes `homo`, `lumo`, and `gap`.
The repository also pins an executable environment around Python 3.8,
PyTorch 2.4.1, CUDA 12.1, PyG extensions, Lightning, RDKit, and OpenBabel.
This is real implementation evidence, but the public README does not expose a
direct PCQM4Mv2 Gap validation/test-dev number comparable to MolGap.

**What is transferable.** The cleanest idea is decoder-side leakage control:
if a future pretraining task reconstructs coordinates, distances, or local
3D states, the decoder must not receive a shortcut that makes the encoder
irrelevant. Selective re-masking can also be used as an ablation against the
simpler “mask all coordinates” objective.

**Contract mismatch.** The published route uses PCQM-style 3D pretraining and
reports mainly QM9/MD17 transfer. It does not establish ETKDG-to-ETKDG
training/inference consistency, a direct official PCQM Gap result, or a bounded
parameter/12-hour MolGap screen. The current MolGap graph path already has
strict geometry ownership, so importing the full 3D-ReTrans stack would be a
new architecture and a new pretraining contract rather than a small patch.

**Disposition: B for decoder design, C for direct MolGap candidacy.** Keep the
selective re-mask rule as a design reference. Do not allocate a PCQM screen or
rewrite the active encoder on the basis of MD17 gains alone.

## 3. 3D-MolT5: discrete 3D tokens for a molecule-text model

**Primary sources.** The ICLR 2025 paper is
[3D-MolT5: Leveraging Discrete Structural Information for Molecule-Text
Modeling](https://arxiv.org/html/2406.05797); the authors release an
[Apache-2.0 implementation and model/data instructions](https://github.com/QizhiPei/3D-MolT5).

**Question and mechanism.** 3D-MolT5 maps local 3D structure around each atom
to hashed E3FP-like discrete tokens, aligns those tokens with 1D SELFIES, and
trains a T5 encoder--decoder. Its pretraining tasks include 1D denoising,
joint 1D+3D denoising, 3D-to-1D translation, 3D-to-text, and text-to-1D
translation. The useful abstraction is not the language model itself; it is a
way to turn continuous local geometry into a discrete auxiliary view that can
be masked, aligned, and reconstructed.

**Data and results.** The paper uses about `3.377M` DFT-calculated 3D
structures from PCQM4Mv2 for the joint denoising and 3D-to-1D tasks, then uses
additional PubChem SELFIES/text and molecule--text pairs. Its property table
is on a PubChemQC dataset rather than the official OGB PCQM4Mv2 leaderboard:
the specialist 3D-MolT5 row reports `0.08 eV` for HOMO, LUMO, and H-L Gap with
100% valid numerical answers. Its ablation reports a PubChemQC Gap MAE of
`0.0791` with 3D information versus `0.0968` without it. These are useful
signals that a discrete 3D view can matter, but neither number is a MolGap
PCQM4Mv2 validation result.

**What is transferable.** A compact version could predict masked local
geometry descriptors—distance bins, angle/dihedral bins, or hashed local
frames—from the existing graph representation. This would be a discrete
auxiliary task, not a language-model transplant. The paper also demonstrates
that multiple pretraining tasks need explicit ablations; otherwise the gain
cannot be assigned to 3D tokens rather than scale or text data.

**Contract mismatch.** 3D-MolT5 mixes PCQM pretraining with external PubChem,
PubMed, C4, and molecule-text data, and its downstream table uses a
PubChemQC split with a text-generation validity metric. It also depends on
3D coordinates that are not shown to be MolGap ETKDG coordinates. External
text and molecule sources cannot be silently added to the MolGap database.

**Disposition: B for discrete-geometry pretraining, C for direct score.**
Retain the masked local-geometry idea as a post-selection auxiliary-task
reference. Do not import the T5 checkpoint, external text corpus, or PubChemQC
property score into the current database or leaderboard.

## 4. MolSpectra: multi-modal electronic spectra as a teacher

**Primary sources.** The ICLR 2025 paper is
[MolSpectra: Pre-training 3D Molecular Representation with Multi-modal Energy
Spectra](https://arxiv.org/html/2502.16284), with a public
[source repository](https://github.com/AzureLeon1/MolSpectra).

**Question and mechanism.** MolSpectra asks whether a 3D representation can
learn more useful electronic structure when denoising is paired with actual
energy spectra. Its SpecFormer processes UV--Vis, IR, and Raman spectra using
masked patch reconstruction. A contrastive loss aligns the spectral embedding
with a 3D denoising encoder. The method has two stages: coordinate denoising
on PCQM4Mv2, followed by spectral pretraining on QM9Spectra. Spectra are used
only during pretraining; they are not required by the downstream predictor.

**Electronic evidence.** The QM9Spectra calculations use B3LYP/def-TZVP,
frequency analysis, and TD-DFT. On QM9, the reported Gap MAE is `26.8 meV`
for MolSpectra versus `31.8 meV` for coordinate denoising; HOMO and LUMO are
`15.5` and `13.1 meV` versus `17.7` and `14.7 meV`. Removing masked-patch
reconstruction and contrastive alignment degrades Gap to `31.2 meV`, while
removing IR or Raman also worsens the result. These controlled ablations make
the source valuable as an electronic-teacher design, even though they are
QM9 results.

**What is transferable.** The strongest idea is an electronic auxiliary
teacher with an explicit modality ablation: a frozen or separately trained
teacher can supply a representation of spectra, charges, orbital descriptors,
or other local electronic information, while the deployed Gap model receives
only the allowed MolGap input. A useful MolGap version would require a named,
deduplicated source at a declared theory level and would keep the teacher
outside the official validation/test-dev roles.

**Contract mismatch and reproducibility.** QM9Spectra is not the current
B3LYP/6-31G* PCQM target: it uses B3LYP/def-TZVP and spectral/TD-DFT labels.
The source repository exposes code and a processed QM9S archive, but the
repository page does not expose a clear software license or a PCQM Gap
checkpoint. The 3D coordinate path and external spectral data therefore need
an identity, geometry, theory, and license audit before any teacher query.

**Disposition: B for electronic teacher design, C for direct MolGap score.**
This is a stronger rationale for an electronic auxiliary channel than for
another generic graph layer, but it is not permission to add QM9Spectra or to
change the current labels.

## Cross-paper synthesis

| Route | Strongest evidence | Main unresolved MolGap gate | Safe role now |
|---|---|---|---|
| DenoiseVAE | Paper directly reports PCQM4Mv2 Gap validation `0.0777 +/- 0.0005`; public code is available but not directly runnable as published | ETKDG train/inference path, exact split/config, checkpoint/license, undefined `noneed` return, cost | Paper/method B; code C; evidence-only audit lead, no initialization |
| 3D-GSRD | Public PCQM pretraining and QM9 `gap` fine-tune path; 3D masked-decoder design | No direct PCQM Gap score; DFT/3D contract and budget | Selective re-mask design reference |
| 3D-MolT5 | PCQM 3D pretraining plus PubChemQC Gap ablation `0.0791` vs `0.0968` without 3D | PubChemQC/text setting, external data, discrete-token geometry | Compact masked-geometry auxiliary-task idea |
| MolSpectra | Controlled QM9 electronic-spectrum ablations; Gap `26.8` vs `31.8` meV coordinate baseline | B3LYP/def-TZVP QM9S teacher, license/identity/geometry audit | Electronic teacher/auxiliary-objective reference |

The ranking is deliberately not a ranking of claimed MolGap performance. The
only direct PCQM Gap number in this batch is DenoiseVAE's paper appendix
number, and even that number is not accepted as comparable until the contract
audit is complete. The other three sources provide transferable mechanisms or
public engineering assets, not official PCQM leaderboard evidence.

## Evidence gate before any possible experiment

For DenoiseVAE, the minimum evidence packet would contain the exact paper/code
revision, license status, checkpoint hash if any, repaired-forward patch or
replacement implementation, PCQM split manifest, coordinate-construction
record, target-label usage, parameter/memory estimate, and a local forward
smoke test. A candidate implementation must use ETKDG for both pretraining and
inference or receive a separately approved geometry contract. The public
repository's MMFF path and undefined return variable are blockers, not tasks
that can be silently fixed inside an experiment run.

For an electronic teacher such as MolSpectra, the packet must additionally
contain the source theory/basis, canonical identity overlap with PCQM4Mv2 and
Track A, conformer identity, license, teacher-only role, and a frozen-source
manifest. No external spectra or labels may be appended to the existing
database merely because they are publicly downloadable.

No experiment, database replacement, checkpoint initialization, or remote job
was performed while writing this batch.

## Primary-source index

- [DenoiseVAE proceedings page](https://proceedings.iclr.cc/paper_files/paper/2025/hash/37e9e62294ff6607f6f7c170cc993f2c-Abstract-Conference.html), [paper PDF](https://openreview.net/attachment?id=ym7pr83XQr&name=pdf), and [code](https://github.com/liuyurou1/DenoiseVAE)
- [3D-GSRD paper](https://arxiv.org/abs/2510.16780) and [code](https://github.com/WuChang0124/3D-GSRD)
- [3D-MolT5 paper](https://arxiv.org/html/2406.05797) and [code](https://github.com/QizhiPei/3D-MolT5)
- [MolSpectra paper](https://arxiv.org/html/2502.16284) and [code](https://github.com/AzureLeon1/MolSpectra)
- [Official PCQM4Mv2 specification](https://ogb.stanford.edu/docs/lsc/pcqm4mv2/)
