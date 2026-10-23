# Message normalization evidence review

Dated 2026-10-23. This review selects a bounded K1 mechanism contrast. The owning
protocol and prospective records define execution, roles and budget authority.

## Evidence and remaining question

Targeted RML/reference-index and repository searches found no prior named
MsgNorm, Batch Renorm, GRED or NeuralWalker training trial. An absent name does
not prove an absent equivalent mechanism; the comparison below distinguishes
the proposed intervention from retained normalization and topology work.

The [normalization disposition](../pcqm_k1_norm_mechanisms_100k/terminal_decision_20261023.md)
retains a 1.645meV Post-Graph RMS advantage over Post-Node RMS, below its 3meV
gate; adaptive RMS did not improve its final endpoint. These change feature
normalization after residual addition, rather than relative neighbor-message
amplitude before self/root addition. The later
[qualification supplement](../pcqm_k1_norm_mechanisms_100k/qualification_20261023.md)
establishes strict same-run comparisons and four replay memberships without
changing those scientific outcomes. Graph scope remains a possible future
question; it does not identify a message-scale defect.

The [frozen BN diagnostic](../pcqm_k1_bn_mechanism_a100/terminal_decision.md)
shows sensitivity to buffer estimation in one retained predictor. It does not
establish a training-time cause or a benefit from replacing BN. The
[extended LN decision](../pcqm_k1_norm_exposure_100k/terminal_decision.md)
constrains simple exposure explanations and documents optimization sensitivity.
Neither record measures an imbalance between K1 node and neighbor-message norms.

## Five routes

| Route and primary source | Mechanism and disposition for this question |
|---|---|
| [MsgNorm / DeeperGCN](https://arxiv.org/abs/2006.07739) | Normalize each aggregated message direction and scale it by the receiving node norm and scalar s. A small intervention at the existing K1 aggregation seam; select fixed versus learned s. This adapts MsgNorm, not the whole DeeperGCN architecture, generalized aggregator or preactivation recipe. Published evidence does not establish a PCQM Gap gain for this K1 adaptation. |
| [Batch Renormalization](https://arxiv.org/abs/1702.03275) | Running-statistic corrections target minibatch dependence and train/inference mismatch. Plausible separate question, but correction clipping/schedules and running buffers introduce additional choices. The local BN diagnostic does not establish that BRN will help. |
| [GRED](https://arxiv.org/html/2312.01538v3) | Group nodes by shortest distance and encode hop summaries with a diagonal linear recurrence. This changes information access and requires distance preprocessing. ZINC12K MAE 0.077 versus GraphGPS 0.070 in the paper is no evidence of PCQM Gap superiority; defer from the normalization contrast. |
| [NeuralWalker](https://arxiv.org/abs/2406.03386), [official implementation](https://github.com/BorgwardtLab/NeuralWalker) | Sample walks, encode their sequences and aggregate features into nodes, with optional local/global message passing. A distinct representation route with sampling, coverage, cost and resume identities to qualify. PCQM-contact in the official experiment list is a different task from PCQM4Mv2 Gap. |
| [ESA](https://www.nature.com/articles/s41467-025-60252-z), [official implementation](https://github.com/davidbuterez/edge-set-attention) | Alternate connectivity-masked and global edge attention, followed by attention pooling. A broader architecture change with dense edge-attention cost and task-specific reproduction qualification to resolve; defer from this bounded K1 pair. Headline scores do not provide a matched K1 reference. |

## Selected discriminator

Hypothesis: allowing each local layer to learn its neighbor-to-node amplitude
ratio can improve development error relative to fixed unit-ratio MsgNorm under
one frozen K1 recipe. This is a prospective hypothesis, not an observed scale
imbalance or a diagnosed explanation of earlier model errors.

Both new arms use `m_scaled = s * ||x_i||_2 * m_i / max(||m_i||_2, epsilon)`
over feature channels. Arm A fixes s=1; arm B learns one scalar per local layer,
initialized to 1. Their initial forward functions match; the learned arm adds
nine scalar parameters for nine local convolutions. This contrast is simpler
to interpret than changing node/graph statistics scope simultaneously. The
fixed arm is a new intervention reference, not a retraining of unchanged K1.

Apply the operator to the summed neighbor aggregate before ResGatedGraphConv's
self/root projection and bias. Preserve the separate root branch, outer block
residuals and downstream normalization. Bind the corresponding original node
input and define zero-message/isolated-node behavior in the implementation.

The cheapest authorized discriminator is the paired fixed-versus-learned
100K screen with identical common initialization, rows, shuffle, recipe and
selection semantics. Retain selected/final scale parameters and native cost with
the ordinary endpoint evidence; a complete scale trajectory is not promised.
A scalar moving away from 1 is evidence of learning,
not evidence of improved generalization; the paired endpoint decides the latter.

The [historical strict K1 reference](../v5_legacy_evidence_migration/k1_v4_100k_reference/reference_bundle.json)
is tied to its exact 40-pass contract and runtime identities. Historical placement
requires qualification of the applicable common identities. The primary pair
tests learned versus fixed MsgNorm; it cannot alone establish that MsgNorm beats
unchanged K1, and no baseline retraining is inferred from missing comparability.
Row bootstrap describes conditional row uncertainty, not seed variability.
Positive screening does not authorize adoption, 500K/full training or protected
role consumption.
