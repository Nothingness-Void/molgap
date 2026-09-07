# Deep Reading: Sparse Cardinality Attention and Probabilistic Pretraining (2026-09-08)

This note audits two newly located primary sources that add genuinely different
pretraining or graph-attention mechanisms to the MolGap reserve. Both are read
from the full primary text. Neither changes the permitted database, the ETKDG
train/inference contract, the active random-initialized screen, or experiment
authorization.

The evidence rule is strict: a claim is recorded only when it is exposed by the
paper or an independently inspectable public artifact. A method result on
MoleculeNet, OGB tasks other than PCQM4Mv2, ADME, or another theory/geometry
contract is not a MolGap result.

## Evidence map

| Source | Primary evidence | Artifact status | MolGap status |
|---|---|---|---|
| [Cardinality-Preserving Attention Channels for Graph Transformers in Molecular Property Prediction](https://arxiv.org/html/2602.02201) | Full arXiv HTML, v5, 17 February 2026; method, ablations, five-seed results, compute, limitations, and data/split protocol are exposed. | The paper says code and artifacts are in a reproducibility package and gives reserved DOI `10.5281/zenodo.18622116`; the DOI landing page was not independently retrievable during this audit and no public repository is linked in the paper record. | B/C method reference; no direct PCQM Gap result and no artifact import. |
| [Probabilistic Contrastive Pretraining for Multi-task ADME Property Prediction](https://arxiv.org/html/2606.11508) | Full arXiv HTML, v1, 9 June 2026; equations, variants, corpus ablations, statistical tests, limitations, and frozen-probe appendix are exposed. | The paper states that an anonymized code package was supplied for review and will be de-anonymized at publication; no public project repository or checkpoint was found in the primary record. | B/C algorithm reference; no direct PCQM Gap result and no external corpus/checkpoint import. |

## 1. CardinalGraphFormer: query-conditioned cardinality preservation

### What the paper claims

CardinalGraphFormer combines a Graphormer-style structural bias with a sparse
shortest-path support and a query-conditioned cardinality-preserving attention
(CPA) channel. The paper's central distinction is between a normalized
softmax weighted average, which can lose support multiplicity, and an aligned
unnormalized support sum whose magnitude retains information about the current
support size.

The paper is explicitly 2D: it uses heavy-atom graph topology, RDKit atom and
edge features, shortest-path distances, and degree bins. It does not use 3D
coordinates or conformers. The full benchmark list is MoleculeNet (ESOL,
Lipophilicity, BBBP, Tox21, ClinTox), OGB `ogbg-molhiv`/`ogbg-molpcba`, and
four TDC ADMET tasks; PCQM4Mv2/HOMO--LUMO Gap is not one of the reported
downstream tasks.

### Exact mechanism

For node (i), the support is

`S(i) = {j : SPD(i,j) <= K}`,

with the main experiments using `K=3`, including the node itself. The attention
logit contains query/key similarity, shortest-path bias, direct-bond edge bias,
and a key-side degree-bin bias. The proposed per-head output is

`softmax_attention(i) + sigmoid(W_g q_i) * sum_{j in S(i)} v_j`.

The second sum uses the same `K`-hop support as the normalized attention. This
alignment is causal in the paper: replacing it with a global sum is reported to
reduce `ogbg-molhiv` AUC from `0.819` to `0.811`, while the aligned version is
`0.819 +/- 0.007`.

The paper provides a useful theoretical boundary. Under replication-like
conditions, softmax attention can map different support cardinalities to the
same weighted average, while the unnormalized sum distinguishes them when the
mean and gate are nonzero. The 1-WL expressivity result is an existence result,
not a guarantee that optimization learns an injective map. The authors also
state that degree-normalizing the sum removes the intended cardinality signal.

### Self-supervised objective and controls

The model uses two pretraining signals:

1. two graph views made by subgraph sampling plus node/edge dropout, with
   per-view shortest-path and degree recomputation, followed by a mean-pooled
   projection and NT-Xent at `tau=0.2`;
2. independent masking of 15% of nodes and 15% of edges, decoded by shared
   categorical/continuous reconstruction heads.

The total loss is `L_mask + 0.5 L_contrast`. The paper compares the full model
against an objective-matched `SparseGraphormer-K3` with identical depth, width,
support, biases, pretraining corpus, optimizer, and fine-tuning protocol. It
also includes explicit-size, scalar-support-size, learned-scaling, and learned
temperature controls, which is important because it tests whether the gain is
only a size shortcut.

### Reported evidence

- The main OGB comparison gives `0.819 +/- 0.007` AUC versus `0.802 +/- 0.009`
  for the no-CPA sparse baseline on `ogbg-molhiv`, and `0.304 +/- 0.002` AP
  versus `0.294 +/- 0.002` on `ogbg-molpcba`.
- On the four TDC tasks, the reported CPA improvements over the matched sparse
  baseline are `-0.024` MAE for Caco2, `+0.028` AUC for hERG, `+0.013`
  Spearman for hepatocyte clearance, and `+0.007` Spearman for microsome
  clearance; the paper marks these comparisons significant under its
  Holm-corrected paired-bootstrap procedure.
- The authors report five seeds (`42`--`46`) and size-shift controls. The
  benefits persist when graph size is shifted, while size-only baselines recover
  only part of the gain.
- `K=3` is near-global for the paper's drug-like distribution: median coverage
  is above 95%. The meaningful computational saving appears mainly for larger
  graphs. The full corpus has roughly 28M molecules after filtering; pretraining
  is reported as 29 days on four A100 80GB GPUs, approximately 2,784 GPU-hours.

### Data, leakage, and artifact boundary

The pretraining corpus is a filtered/deduplicated mixture of a 40M ZINC20
sample and ChEMBL35, reduced to about 28M molecules. The paper reports removal
of about 20K benchmark molecules by scaffold matching, not only exact identity.
This is stronger than exact deduplication, but it is still an external
drug-like corpus with Lipinski filtering and does not establish a PCQM role.

The paper's reproducibility statement is evidence that a package existed for the
submission, not evidence that a retrievable, hashable package is available to
MolGap. At audit time the reserved Zenodo landing page could not be fetched, so
the source is not treated as a public code asset or a runnable baseline.

### MolGap disposition

The portable idea is narrow: an unnormalized, query-gated support sum can be a
local information channel complementary to normalized attention, and its
support must be exactly the same as the normalized path. This is relevant to
the project's local-first graph reasoning, but the paper does not show a
HOMO/LUMO/Gap result, uses external ZINC/ChEMBL pretraining, and has no
ETKDG-dependent evidence.

**Disposition: B/C method-only reference, not a current experiment.** A future
same-database version would have to derive all pretraining views from the
permitted PCQM train role, preserve the existing graph and ETKDG cache roles,
and compare CPA against both no-CPA and explicit-support-size controls. It
would be a separately authorized post-K3 pretraining/architecture question,
not a claim that the paper's ADMET numbers transfer to Gap.

## 2. Contrastive KERMT: probabilistic graph-to-SMILES pretraining

### What the paper claims

Contrastive KERMT combines KERMT's atom-context, bond-context, and functional
group prediction tasks with a graph-to-SMILES contrastive mutual-information
machine (cMIM). The graph encoder receives a 2D molecular graph; a variational
latent code is sampled from the graph; and an autoregressive character-level
SMILES decoder reconstructs the canonical SMILES.

The distinctive algorithmic claim is not merely “add contrastive loss.” The
paper writes reconstruction, latent-density regularization, contrastive
discrimination, and chemistry-specific targets as log-probability factors in
one latent-variable objective with unit scalar coefficients. The authors
explicitly warn that unit coefficients do not imply equal gradient magnitudes:
token sums, latent dimensions, and task reductions still have different scales.

### Exact objective

The graph-to-SMILES A-MIM part is

`-mean_i [ log p(s_i | z_i) + 0.5*(log q(z_i | g_i) + log p(z_i)) ]`,

where `q` is a diagonal Gaussian posterior and `p(z)` is a standard-normal
prior. cMIM adds a batch-conditioned probability in which the matched sample
has the fixed cosine self-similarity `sim(z_i,z_i)=1`, while other batch members
provide mismatched negatives. It therefore does not require two augmented
positive molecular views. Chemistry targets are then appended as native
log-probability factors; in this paper they are atom-context, bond-context, and
functional-group heads inherited from KERMT.

The paper compares three matched variants:

| Variant | Active pretraining signal | Pretraining-only modules | Downstream backbone |
|---|---|---|---|
| KERMT | atom/bond/functional-group vocabulary tasks | vocabulary heads | same 56.95M KERMT backbone |
| cMIM-only | graph-to-SMILES plus cMIM | posterior, decoder, contrastive branch | same KERMT backbone |
| Contrastive KERMT | both of the above | all pretraining-only modules | same KERMT backbone |

The decoder and posterior/contrastive modules are discarded before downstream
fine-tuning. This is a clean way to separate representation initialization
from inference-time model capacity.

### Configuration and evidence

- The KERMT encoder has hidden size 800, six message-passing/attention layers,
  four heads, one multi-task block, PReLU activations, and dropout 0.1.
- cMIM uses temperature `tau=0.1`, in-batch negatives, reparameterized
  diagonal-Gaussian latents, and posterior variance clipped below `1e-6`.
- The fixed 11M corpus variants run for 100 epochs with 20 warmup epochs; the
  208M variants run for four or six epochs. The paper reports a global batch of
  1,024 on eight A100 GPUs and roughly three to eight weeks of pretraining,
  depending on corpus scale.
- Contrastive KERMT is best on 3/4 Biogen, 5/9 ExpansionRX, and 14/25
  ChEMBL-MT endpoints under the paper's Tukey-HSD comparison. Its matched
  within-family result is the important causal evidence: cMIM helps when added
  to KERMT, while cMIM-only is weak and is significantly worse on one benchmark.
- Latent-neighborhood diagnostics report improvements in SMILES, Morgan
  fingerprint, and measured-property neighborhoods. Frozen linear probes also
  favor combined Contrastive KERMT over KERMT on the two source embeddings.

The headline `+7.6%/+9.9%/+9.5%` gains are conditional averages over endpoints
with statistically significant differences, not all-endpoint macro improvements.
The paper discloses this in its appendix; it should not be compared directly
with a MolGap MAE.

### Data and limitations

The source corpora are ZINC15/ChEMBL mixtures, optional Biogen molecules,
MolMIM-generated Biogen-seeded molecules, ExpansionRX, and ChEMBL-MT. Some
adaptation configurations include downstream validation/test molecules without
assay labels. The authors correctly classify those settings as label-free
transductive corpus adaptation rather than strict molecule-holdout
pretraining. The paper's evaluation is ADME-only and uses one KERMT backbone;
it also acknowledges the substantial compute cost.

The paper says an anonymized code package was supplied with the submission and
will be made public on de-anonymization. The audited primary HTML does not link
to a public repository or checkpoint, so no executable artifact is admitted.

### MolGap disposition

The most portable part is the objective design, not the external corpus: a
same-database graph can provide canonical-SMILES reconstruction and deterministic
atom/bond/fragment targets, while cMIM can shape global latent neighborhoods
without augmentation-defined positive pairs. The distinction between cMIM-only
and combined local-plus-global pretraining is also a valuable control lesson.

A MolGap adaptation would need to:

1. use only canonicalized molecules from the permitted PCQM4Mv2 or repaired-2M
   train role, with a frozen identity manifest;
2. avoid PCQM target labels in the pretraining objective unless a separate
   supervised source-task protocol is explicitly approved;
3. keep the current ETKDG path unchanged for every geometry-dependent model;
4. compare scratch, local-source-task-only, cMIM-only, and combined objectives
   under equal budgets, with frozen probes and all official roles sealed; and
5. report token-length, latent-dimension, and task-reduction normalization,
   because “unit weighting” is not a gradient-scale guarantee.

**Disposition: B/C algorithm reference, no current experiment.** It is a
database-preserving post-selection hypothesis only; the paper's ADME gains,
external corpora, and anonymized code do not justify a current pretraining run.

## 3. Adjacent algorithm and public-code leads kept outside the ledger

These sources were checked because they are relevant to teacher or pretraining
mechanisms, but they are not molecular PCQM evidence and are therefore not
counted as rows in the molecular coverage ledger.

### GCKD: decoupled graph-structure distillation

The [Information Sciences paper](https://www.sciencedirect.com/science/article/pii/S0020025525003135)
and [public repository](https://github.com/zanchenyi/GCKD-main) describe a
teacher-embedding graph, graph-contrastive learning of a GNN bridge, and reuse
of that bridge during student distillation. The paper's stated experiments are
CIFAR-100 teacher/student image models, and the README exposes a legacy
Python 3.6/PyTorch 1.10/DGL 1.0.1 environment. This is useful as a conceptual
warning against jointly learning a powerful relation extractor and student,
but it has no molecular target, PCQM result, or chemistry-valid positive-pair
definition. **Keep as an adjacent distillation idea only; no code import.**

### MultiPUFFIN: multimodal domain-informed fusion

The [full arXiv paper](https://arxiv.org/html/2603.00857) combines SMILES,
2D graph, 3D conformer, auxiliary conditions, and gated cross-modal fusion for
thermophysical properties, using a 500K PubChem pretraining corpus and nine
experimental property tasks. It provides a useful multimodal/gated-fusion
design vocabulary, but no direct PCQM/HOMO/LUMO/Gap result, no verified public
code in the primary record, and external experimental/geometry roles. **Keep
as an adjacent reference, not a MolGap candidate or data source.**

### Source-task pretraining listings without primary full text

The May/June 2026 listings for [Supervised Source-Task Pretraining](https://www.researchgate.net/publication/405238198_Supervised_Source-Task_Pretraining_for_Low-Data_Electrochemical_Molecular_Property_Prediction)
and [Adaptive Source-Task Expert Routing](https://www.researchgate.net/publication/407275195_Adaptive_Source-Task_Expert_Routing_for_Supervised_Molecular_Pretraining_in_Low-Data_Electrochemical_Property_Prediction)
provide abstracts describing QM9 HOMO/LUMO/Gap/u0 source tasks and electrochemical
transfer. They do not provide primary full text or a retrievable code/data
package in the audited surface. They remain **unadmitted leads**, not evidence
for an experiment or a ledger row.

## 4. Cross-paper conclusion

The two full primary reads support two distinct, testable ideas:

| Idea | Evidence-backed portable part | Why it is not admitted now |
|---|---|---|
| Support-cardinality channel | Same-support unnormalized aggregation can complement normalized attention and must be tested against explicit size controls. | No PCQM Gap result, external pretraining corpus, unavailable package, and 2D-only contract. |
| Probabilistic global pretraining | Graph-to-SMILES reconstruction plus in-batch mismatched negatives can complement local atom/bond/fragment targets; combined beats cMIM-only in the reported matched ADME ablation. | No PCQM Gap result, external/transductive corpus variants, anonymized code, and very high compute. |

Both reinforce a project-level rule: isolate the information channel before
stacking it. Neither supports importing external labels, pretrained weights,
ZINC/ChEMBL rows, MMFF/DFT conformers, or ADME numbers into the current MolGap
database or architecture claim.

**No experiment was started, no database was changed, and no checkpoint or
external data was downloaded.**

## Primary sources

- [Cardinality-Preserving Attention Channels, arXiv HTML](https://arxiv.org/html/2602.02201)
- [Cardinality-Preserving Attention Channels, arXiv record](https://arxiv.org/abs/2602.02201)
- [Probabilistic Contrastive Pretraining, arXiv HTML](https://arxiv.org/html/2606.11508)
- [Probabilistic Contrastive Pretraining, arXiv record](https://arxiv.org/abs/2606.11508)
- [GCKD paper](https://www.sciencedirect.com/science/article/pii/S0020025525003135) and [repository](https://github.com/zanchenyi/GCKD-main)
- [MultiPUFFIN](https://arxiv.org/html/2603.00857)
